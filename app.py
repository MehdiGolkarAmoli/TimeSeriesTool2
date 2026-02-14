"""
Sentinel-2 Time Series Building Classification
VERSION 06 - GEE Cloud Masking + Gap-Filling + Python Patch Validation
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, box
import rasterio
from patchify import patchify, unpatchify
import datetime
import torch
import math
import ee
import tempfile
import requests
import time
import warnings
import sys
import base64
import json
import subprocess
import zipfile
import io
from datetime import date
from PIL import Image
from io import BytesIO

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Examining the path of torch.classes")

import streamlit as st

st.set_page_config(
    layout="wide",
    page_title="Building Classification Time Series v06",
    page_icon="🏗️"
)

import folium
from folium import plugins
from streamlit_folium import st_folium
import segmentation_models_pytorch as smp

# =============================================================================
# CONSTANTS
# =============================================================================
SPECTRAL_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
PATCH_SIZE = 224
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2
DOWNLOAD_TIMEOUT = 120
CHUNK_SIZE = 8192
MIN_BAND_FILE_SIZE = 10000
MIN_MULTIBAND_FILE_SIZE = 100000
CLOUD_PROB_THRESHOLD = 50
CDI_THRESHOLD = -0.5
MAX_MASKED_PERCENT_FOR_GAPFILL = 30
STATUS_NO_DATA = "no_data"
STATUS_SKIPPED = "skipped"
STATUS_COMPLETE = "complete"
STATUS_REJECTED = "rejected"

# =============================================================================
# Session State Initialization
# =============================================================================
defaults = {
    'drawn_polygons': [], 'last_drawn_polygon': None, 'ee_initialized': False,
    'model_loaded': False, 'model': None, 'device': None,
    'classification_thumbnails': [], 'processing_complete': False,
    'processed_months': {}, 'current_temp_dir': None, 'downloaded_images': {},
    'valid_patches_mask': None, 'valid_months': {}, 'pdf_report': None,
    'month_analysis_results': {}, 'failed_downloads': [],
    'analysis_complete': False, 'download_complete': False,
    'cloud_free_collection': None, 'processing_params': None,
    'selected_region_index': 0, 'processing_in_progress': False,
    'processing_config': None, 'probability_maps': {},
    'change_detection_result': None, 'processing_log': [],
    'sentinel2_zip_bytes': None, 'sentinel2_download_ready': False,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# =============================================================================
# Helper Functions
# =============================================================================
def normalized(img):
    min_val = np.nanmin(img)
    max_val = np.nanmax(img)
    if max_val == min_val:
        return np.zeros_like(img)
    return (img - min_val) / (max_val - min_val + 1e-5)


def add_log_entry(message, level="INFO", include_timestamp=True):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.processing_log.append({'timestamp': timestamp, 'level': level, 'message': message})
    if include_timestamp:
        print(f"[{timestamp}] [{level}] {message}")


def clear_log():
    st.session_state.processing_log = []
    st.session_state.sentinel2_zip_bytes = None
    st.session_state.sentinel2_download_ready = False


# =============================================================================
# Sentinel-2 ZIP Download
# =============================================================================
def create_sentinel2_zip(downloaded_images):
    """Create a ZIP file containing all downloaded Sentinel-2 GeoTIFF images."""
    if not downloaded_images:
        return None
    try:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for month_name in sorted(downloaded_images.keys()):
                file_path = downloaded_images[month_name]
                if os.path.exists(file_path):
                    arcname = f"sentinel2_{month_name}.tif"
                    zip_file.write(file_path, arcname)

            # Add metadata
            metadata_lines = [
                "Sentinel-2 Downloaded Images Metadata",
                "=" * 50,
                f"Download Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"Number of Images: {len(downloaded_images)}",
                f"Bands: {', '.join(SPECTRAL_BANDS)}",
                f"Cloud Masking: prob > {CLOUD_PROB_THRESHOLD}, CDI < {CDI_THRESHOLD}",
                "", "Monthly Images:", "-" * 30,
            ]
            for month_name in sorted(downloaded_images.keys()):
                file_path = downloaded_images[month_name]
                if os.path.exists(file_path):
                    size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    try:
                        with rasterio.open(file_path) as src:
                            metadata_lines.append(
                                f"  {month_name}: {src.width}x{src.height}, {src.count} bands, "
                                f"{size_mb:.1f} MB, CRS: {src.crs}, Bounds: {src.bounds}"
                            )
                    except:
                        metadata_lines.append(f"  {month_name}: {size_mb:.1f} MB")
            zip_file.writestr("metadata.txt", "\n".join(metadata_lines))

        zip_buffer.seek(0)
        return zip_buffer.getvalue()
    except Exception as e:
        add_log_entry(f"Failed to create ZIP: {e}", "ERROR")
        return None


def show_sentinel2_download_section():
    """Display the Sentinel-2 image download section in the UI."""
    if not st.session_state.downloaded_images:
        return

    st.divider()
    st.subheader("📥 Download All Sentinel-2 Images")

    num_images = len(st.session_state.downloaded_images)
    total_size = 0
    for path in st.session_state.downloaded_images.values():
        if os.path.exists(path):
            total_size += os.path.getsize(path)
    total_size_mb = total_size / (1024 * 1024)

    st.info(f"📦 **{num_images} Sentinel-2 images** available ({total_size_mb:.1f} MB total)")

    with st.expander("📋 Image Details", expanded=False):
        for month_name in sorted(st.session_state.downloaded_images.keys()):
            path = st.session_state.downloaded_images[month_name]
            if os.path.exists(path):
                size_mb = os.path.getsize(path) / (1024 * 1024)
                try:
                    with rasterio.open(path) as src:
                        st.write(f"  • **{month_name}**: {src.width}×{src.height} px, {src.count} bands, {size_mb:.1f} MB")
                except:
                    st.write(f"  • **{month_name}**: {size_mb:.1f} MB")

    col1, col2 = st.columns([1, 2])

    with col1:
        if st.button("📦 Prepare ZIP for Download", key="prepare_s2_zip", type="primary"):
            with st.spinner("📦 Creating ZIP file..."):
                zip_bytes = create_sentinel2_zip(st.session_state.downloaded_images)
                if zip_bytes:
                    st.session_state.sentinel2_zip_bytes = zip_bytes
                    st.session_state.sentinel2_download_ready = True
                    zip_size_mb = len(zip_bytes) / (1024 * 1024)
                    st.success(f"✅ ZIP ready! ({zip_size_mb:.1f} MB)")
                else:
                    st.error("❌ Failed to create ZIP file")

    with col2:
        if st.session_state.sentinel2_download_ready and st.session_state.sentinel2_zip_bytes:
            zip_size_mb = len(st.session_state.sentinel2_zip_bytes) / (1024 * 1024)
            st.download_button(
                label=f"⬇️ Download All Sentinel-2 Images ({zip_size_mb:.1f} MB)",
                data=st.session_state.sentinel2_zip_bytes,
                file_name=f"sentinel2_images_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip",
                key="download_s2_zip"
            )


# =============================================================================
# File Validation
# =============================================================================
def validate_geotiff_file(file_path, expected_bands=1):
    try:
        if not os.path.exists(file_path):
            return False, "File does not exist"
        file_size = os.path.getsize(file_path)
        min_size = MIN_BAND_FILE_SIZE if expected_bands == 1 else MIN_MULTIBAND_FILE_SIZE
        if file_size < min_size:
            return False, f"File too small ({file_size} bytes)"
        with rasterio.open(file_path) as src:
            if src.count < expected_bands:
                return False, f"Wrong band count ({src.count}, expected {expected_bands})"
        return True, "File is valid"
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def validate_band_file(band_file_path, band_name):
    return validate_geotiff_file(band_file_path, expected_bands=1)


# =============================================================================
# Model Functions
# =============================================================================
@st.cache_data
def download_model_from_gdrive(gdrive_url, local_filename):
    try:
        correct_file_id = "1_8jOOSXnELA-xOGW0DKgRMo6RvnJYV5_"
        st.info("Downloading model...")
        try:
            import gdown
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
            import gdown
        methods = [
            f"https://drive.google.com/uc?id={correct_file_id}",
            f"https://drive.google.com/file/d/{correct_file_id}/view",
            correct_file_id
        ]
        for method in methods:
            try:
                gdown.download(method, local_filename, quiet=False, fuzzy=True)
                if os.path.exists(local_filename) and os.path.getsize(local_filename) > 1024:
                    st.success("Model downloaded!")
                    return local_filename
            except:
                if os.path.exists(local_filename):
                    os.remove(local_filename)
        return None
    except:
        return None


@st.cache_resource
def load_model(model_path):
    try:
        device = torch.device('cpu')
        model = smp.UnetPlusPlus(
            encoder_name='timm-efficientnet-b8', encoder_weights='imagenet',
            in_channels=12, classes=1, decoder_attention_type='scse'
        ).to(device)
        loaded_object = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(loaded_object, dict) and 'model_state_dict' in loaded_object:
            model.load_state_dict(loaded_object['model_state_dict'])
        elif isinstance(loaded_object, dict):
            model.load_state_dict(loaded_object)
        else:
            return None, None
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None


# =============================================================================
# Earth Engine
# =============================================================================
@st.cache_resource
def initialize_earth_engine():
    try:
        ee.Initialize()
        return True, "Earth Engine initialized"
    except:
        try:
            base64_key = os.environ.get('GOOGLE_EARTH_ENGINE_KEY_BASE64')
            if base64_key:
                key_json = base64.b64decode(base64_key).decode()
                key_data = json.loads(key_json)
                key_file = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
                with open(key_file.name, 'w') as f:
                    json.dump(key_data, f)
                credentials = ee.ServiceAccountCredentials(key_data['client_email'], key_file.name)
                ee.Initialize(credentials)
                os.unlink(key_file.name)
                return True, "Authenticated with Service Account"
            else:
                ee.Authenticate()
                ee.Initialize()
                return True, "Authenticated"
        except Exception as auth_error:
            return False, f"Auth failed: {str(auth_error)}"


def get_utm_zone(longitude):
    return math.floor((longitude + 180) / 6) + 1

def get_utm_epsg(longitude, latitude):
    zone_number = get_utm_zone(longitude)
    return f"EPSG:326{zone_number:02d}" if latitude >= 0 else f"EPSG:327{zone_number:02d}"


# =============================================================================
# GEE Cloud Masking
# =============================================================================
def create_cloud_free_collection(aoi, extended_start, extended_end, cloudy_pixel_percentage=10):
    s2_sr = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
             .filterBounds(aoi).filterDate(extended_start, extended_end)
             .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloudy_pixel_percentage))
             .select(SPECTRAL_BANDS + ['SCL']))
    s2_cloud_prob = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
                     .filterBounds(aoi).filterDate(extended_start, extended_end))
    join_filter = ee.Filter.equals(leftField='system:index', rightField='system:index')
    joined = ee.Join.saveFirst('cloud_probability').apply(
        primary=s2_sr, secondary=s2_cloud_prob, condition=join_filter)

    def add_cloud_band(feature):
        img = ee.Image(feature)
        return img.addBands(ee.Image(img.get('cloud_probability')))

    s2_joined = ee.ImageCollection(joined.map(add_cloud_band))

    def mask_cloud_and_shadow(img):
        cloud_prob = img.select('probability')
        cdi = ee.Algorithms.Sentinel2.CDI(img)
        is_cloud = cloud_prob.gt(CLOUD_PROB_THRESHOLD).And(cdi.lt(CDI_THRESHOLD))
        kernel = ee.Kernel.circle(radius=20, units='meters')
        cloud_dilated = is_cloud.focal_max(kernel=kernel, iterations=2)
        masked = img.updateMask(cloud_dilated.Not())
        scaled = masked.select(SPECTRAL_BANDS).multiply(0.0001).clip(aoi)
        return scaled.copyProperties(img, ['system:time_start'])

    return s2_joined.map(mask_cloud_and_shadow)


# =============================================================================
# Download Functions
# =============================================================================
def download_band_with_retry(image, band, aoi, output_path, scale=10):
    try:
        region = aoi.bounds().getInfo()['coordinates']
    except Exception as e:
        return False, f"AOI bounds error: {e}"

    temp_path = output_path + '.tmp'
    if os.path.exists(temp_path):
        os.remove(temp_path)
    if os.path.exists(output_path):
        is_valid, msg = validate_band_file(output_path, band)
        if is_valid:
            return True, "cached"
        os.remove(output_path)

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            url = image.select(band).getDownloadURL({
                'scale': scale, 'region': region, 'format': 'GEO_TIFF', 'bands': [band]
            })
            response = requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT)
            if response.status_code == 200:
                if 'text/html' in response.headers.get('content-type', ''):
                    raise Exception("GEE rate limit")
                downloaded_size = 0
                with open(temp_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)
                if downloaded_size < MIN_BAND_FILE_SIZE:
                    raise Exception(f"File too small ({downloaded_size} bytes)")
                is_valid, msg = validate_band_file(temp_path, band)
                if is_valid:
                    os.replace(temp_path, output_path)
                    return True, "success"
                else:
                    if os.path.exists(temp_path): os.remove(temp_path)
                    raise Exception(f"Validation failed: {msg}")
            else:
                raise Exception(f"HTTP {response.status_code}")
        except requests.exceptions.Timeout:
            last_error = "Timeout"
        except requests.exceptions.ConnectionError:
            last_error = "Connection error"
        except Exception as e:
            last_error = str(e)

        for f in [output_path, temp_path]:
            if os.path.exists(f):
                try: os.remove(f)
                except: pass
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAY_BASE ** (attempt + 1))
    return False, last_error


def download_composite(composite, aoi, output_path, month_name, scale=10, status_placeholder=None):
    try:
        if os.path.exists(output_path):
            is_valid, msg = validate_geotiff_file(output_path, expected_bands=len(SPECTRAL_BANDS))
            if is_valid:
                if status_placeholder: status_placeholder.info(f"✅ {month_name} cached")
                return output_path
            else:
                os.remove(output_path)

        temp_dir = os.path.dirname(output_path)
        bands_dir = os.path.join(temp_dir, f"bands_{month_name}")
        os.makedirs(bands_dir, exist_ok=True)

        band_files, failed_bands = [], []
        for i, band in enumerate(SPECTRAL_BANDS):
            band_file = os.path.join(bands_dir, f"{band}.tif")
            if status_placeholder:
                status_placeholder.text(f"📥 {month_name}: {band} ({i+1}/{len(SPECTRAL_BANDS)})...")
            success, error_msg = download_band_with_retry(composite, band, aoi, band_file, scale)
            if success:
                band_files.append(band_file)
            else:
                failed_bands.append(f"{band}: {error_msg}")

        if failed_bands:
            st.error(f"❌ {month_name}: Failed bands - {'; '.join(failed_bands)}")
            return None

        if len(band_files) == len(SPECTRAL_BANDS):
            if status_placeholder: status_placeholder.text(f"📦 {month_name}: Merging bands...")
            with rasterio.open(band_files[0]) as src:
                meta = src.meta.copy()
            meta.update(count=len(band_files))
            with rasterio.open(output_path, 'w', **meta) as dst:
                for i, band_file in enumerate(band_files):
                    with rasterio.open(band_file) as src:
                        dst.write(src.read(1), i + 1)
            is_valid, msg = validate_geotiff_file(output_path, expected_bands=len(SPECTRAL_BANDS))
            if is_valid:
                return output_path
            else:
                if os.path.exists(output_path): os.remove(output_path)
        return None
    except Exception as e:
        st.error(f"❌ {month_name}: Download exception - {str(e)}")
        return None


def download_monthly_image_v06(aoi, cloud_free_collection, month_info, temp_dir,
                                scale=10, status_placeholder=None):
    month_name = month_info['month_name']
    month_index = month_info['month_index']
    origin = month_info['origin']
    output_file = os.path.join(temp_dir, f"sentinel2_{month_name}.tif")

    add_log_entry(f"Processing month: {month_name}")

    if os.path.exists(output_file):
        is_valid, msg = validate_geotiff_file(output_file, expected_bands=len(SPECTRAL_BANDS))
        if is_valid:
            add_log_entry(f"{month_name}: Using cached file")
            if status_placeholder: status_placeholder.info(f"✅ {month_name} cached")
            return output_file, STATUS_COMPLETE, "Cached"
        else:
            os.remove(output_file)

    try:
        origin_date = ee.Date(origin)
        month_start = origin_date.advance(month_index, 'month')
        month_end = origin_date.advance(ee.Number(month_index).add(1), 'month')
        month_middle = month_start.advance(15, 'day')

        if status_placeholder: status_placeholder.text(f"📥 {month_name}: Analyzing...")

        monthly_images = cloud_free_collection.filterDate(month_start, month_end)
        image_count = monthly_images.size().getInfo()
        add_log_entry(f"{month_name}: Found {image_count} cloud-free images")

        if image_count == 0:
            add_log_entry(f"{month_name}: REJECTED - No images", "WARNING")
            return None, STATUS_NO_DATA, "No images available"

        if status_placeholder:
            status_placeholder.text(f"📥 {month_name}: Composite from {image_count} images...")

        def create_valid_mask(img):
            valid = img.select('B4').mask().And(img.select('B11').mask())
            return ee.Image(1).updateMask(valid).unmask(0).toInt()

        frequency = monthly_images.map(create_valid_mask).sum().toInt().rename('frequency')
        composite = monthly_images.median()

        masked_stats = frequency.eq(0).reduceRegion(
            reducer=ee.Reducer.sum(), geometry=aoi, scale=10, maxPixels=1e13)
        total_stats = frequency.gte(0).reduceRegion(
            reducer=ee.Reducer.sum(), geometry=aoi, scale=10, maxPixels=1e13)

        masked_count = ee.Number(masked_stats.get('frequency')).getInfo()
        total_count = ee.Number(total_stats.get('frequency')).getInfo()

        if total_count == 0:
            return None, STATUS_NO_DATA, "No valid pixels"

        masked_percent = (masked_count / total_count) * 100
        add_log_entry(f"{month_name}: Masked: {masked_percent:.5f}%")

        if masked_percent > MAX_MASKED_PERCENT_FOR_GAPFILL:
            return None, STATUS_SKIPPED, f"Masked {masked_percent:.5f}% > {MAX_MASKED_PERCENT_FOR_GAPFILL}%"

        if masked_percent == 0:
            if status_placeholder: status_placeholder.text(f"📥 {month_name}: 0% masked, downloading...")
            path = download_composite(composite, aoi, output_file, month_name, scale, status_placeholder)
            if path:
                return path, STATUS_COMPLETE, "Complete (0% masked)"
            return None, STATUS_REJECTED, "Download failed"

        # GAP-FILL
        add_log_entry(f"{month_name}: Gap-filling ({masked_percent:.5f}% masked)")
        if status_placeholder: status_placeholder.text(f"📥 {month_name}: Gap-filling...")

        gap_mask = frequency.eq(0)
        month_middle_millis = month_middle.millis()

        m1_past_start = origin_date.advance(ee.Number(month_index).subtract(1), 'month')
        m1_past_end = month_start
        m1_future_start = month_end
        m1_future_end = origin_date.advance(ee.Number(month_index).add(2), 'month')

        m1_past = cloud_free_collection.filterDate(m1_past_start, m1_past_end)
        m1_future = cloud_free_collection.filterDate(m1_future_start, m1_future_end)
        all_candidates = m1_past.merge(m1_future)

        def add_time_distance(img):
            diff = ee.Number(img.get('system:time_start')).subtract(month_middle_millis).abs()
            return img.set('time_distance', diff)

        sorted_images = all_candidates.map(add_time_distance).sort('time_distance', True)
        candidate_count = sorted_images.size().getInfo()

        if candidate_count == 0:
            return None, STATUS_REJECTED, f"No gap-fill candidates"

        closest_mosaic = sorted_images.mosaic().select(SPECTRAL_BANDS)
        has_closest = closest_mosaic.select('B4').mask()
        fill_from_closest = gap_mask.And(has_closest)
        still_masked = gap_mask.And(has_closest.Not())
        filled_composite = composite.unmask(closest_mosaic.updateMask(fill_from_closest))

        fill_source = (ee.Image.constant(0).clip(aoi).toInt8()
                       .where(fill_from_closest, 1).where(still_masked, 2).rename('fill_source'))
        still_masked_result = fill_source.eq(2).reduceRegion(
            reducer=ee.Reducer.sum(), geometry=aoi, scale=10, maxPixels=1e13).get('fill_source')
        still_masked_count = ee.Number(ee.Algorithms.If(
            ee.Algorithms.IsEqual(still_masked_result, None), 0, still_masked_result)).getInfo()

        if still_masked_count == 0:
            if status_placeholder: status_placeholder.text(f"📥 {month_name}: Gap-filled, downloading...")
            path = download_composite(filled_composite, aoi, output_file, month_name, scale, status_placeholder)
            if path:
                return path, STATUS_COMPLETE, f"Complete after gap-fill (was {masked_percent:.5f}%)"
            return None, STATUS_REJECTED, "Download failed after gap-fill"
        else:
            pct = (still_masked_count / total_count) * 100
            return None, STATUS_REJECTED, f"{pct:.5f}% still masked after gap-fill"

    except Exception as e:
        add_log_entry(f"{month_name}: ERROR - {str(e)}", "ERROR")
        return None, STATUS_NO_DATA, f"Error: {str(e)}"


# =============================================================================
# RGB Thumbnail
# =============================================================================
def generate_rgb_thumbnail(image_path, month_name, max_size=256):
    try:
        with rasterio.open(image_path) as src:
            red, green, blue = src.read(4), src.read(3), src.read(2)
        rgb = np.nan_to_num(np.stack([red, green, blue], axis=-1), nan=0.0)

        def percentile_stretch(band, lower=2, upper=98):
            valid = band[band > 0]
            if len(valid) == 0: return np.zeros_like(band, dtype=np.uint8)
            p_low, p_high = np.percentile(valid, lower), np.percentile(valid, upper)
            if p_high <= p_low: p_high = p_low + 0.001
            return (np.clip((band - p_low) / (p_high - p_low), 0, 1) * 255).astype(np.uint8)

        rgb_uint8 = np.stack([percentile_stretch(rgb[:,:,i]) for i in range(3)], axis=-1)
        pil_img = Image.fromarray(rgb_uint8, mode='RGB')
        h, w = pil_img.size[1], pil_img.size[0]
        if h > max_size or w > max_size:
            scale = max_size / max(h, w)
            pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        return pil_img
    except:
        return None


# =============================================================================
# Patch Validity
# =============================================================================
def check_patch_validity(patch, nodata_threshold_percent=0):
    if np.any(np.isnan(patch)): return False
    if np.all(patch == 0): return False
    if (np.sum(patch == 0) / patch.size) * 100 > nodata_threshold_percent: return False
    if patch.ndim == 3:
        for b in range(patch.shape[-1]):
            if np.all(patch[:, :, b] == 0): return False
    return True


def get_patch_validity_mask(image_path, patch_size=224, nodata_threshold_percent=0):
    try:
        with rasterio.open(image_path) as src:
            img_data = src.read()
        img = np.moveaxis(img_data, 0, -1)
        h, w, c = img.shape
        new_h = int(np.ceil(h / patch_size) * patch_size)
        new_w = int(np.ceil(w / patch_size) * patch_size)
        if h != new_h or w != new_w:
            padded = np.zeros((new_h, new_w, c), dtype=img.dtype)
            padded[:h, :w, :] = img
            img = padded
        patches = patchify(img, (patch_size, patch_size, c), step=patch_size)
        nh, nw = patches.shape[0], patches.shape[1]
        mask = np.zeros((nh, nw), dtype=bool)
        for i in range(nh):
            for j in range(nw):
                mask[i, j] = check_patch_validity(patches[i, j, 0], nodata_threshold_percent)
        return mask, (h, w), (nh, nw)
    except:
        return None, None, None


def find_common_valid_patches(downloaded_images, nodata_threshold_percent=0):
    st.info("🔍 Analyzing patch validity across all months...")
    month_names = sorted(downloaded_images.keys())
    if not month_names:
        st.error("❌ No downloaded images!")
        return None, None, None

    # Step 1: Validate dimensions
    st.write("**Step 1: Validating image dimensions...**")
    dimensions = {}
    for mn in month_names:
        try:
            with rasterio.open(downloaded_images[mn]) as src:
                dimensions[mn] = {'height': src.height, 'width': src.width, 'bands': src.count}
        except Exception as e:
            st.error(f"❌ Cannot read {mn}: {e}")
            return None, None, None

    ref_dim = dimensions[month_names[0]]
    mismatched = [f"{mn}: {d['height']}x{d['width']}" for mn, d in dimensions.items()
                  if d['height'] != ref_dim['height'] or d['width'] != ref_dim['width']]
    if mismatched:
        st.error(f"❌ DIMENSION MISMATCH! Ref: {ref_dim['height']}x{ref_dim['width']}, Mismatched: {', '.join(mismatched)}")
        return None, None, None
    st.success(f"✅ All {len(month_names)} images: {ref_dim['height']}x{ref_dim['width']} ({ref_dim['bands']} bands)")

    h, w = ref_dim['height'], ref_dim['width']
    nh = int(np.ceil(h / PATCH_SIZE))
    nw = int(np.ceil(w / PATCH_SIZE))
    total = nh * nw
    st.write(f"**Step 2: Patch grid**: {nh}×{nw} = **{total} patches**")

    # Step 3: Validity per month
    st.write("**Step 3: Calculating validity...**")
    progress = st.progress(0)
    masks, counts = {}, {}
    for idx, mn in enumerate(month_names):
        vm, _, _ = get_patch_validity_mask(downloaded_images[mn], PATCH_SIZE, nodata_threshold_percent)
        if vm is not None and vm.shape == (nh, nw):
            masks[mn] = vm
            counts[mn] = np.sum(vm)
        progress.progress((idx + 1) / len(month_names))
    progress.empty()

    if not masks:
        st.error("❌ Could not analyze any months!")
        return None, None, None

    # Step 4: Reference mask
    max_count = max(counts.values())
    ref_month = [mn for mn, c in counts.items() if c == max_count][0]
    ref_mask = masks[ref_month]
    st.info(f"📊 Max valid patches: **{max_count}/{total}** ({100*max_count/total:.1f}%)")

    # Step 5: Filter months
    st.write("**Step 5: Filtering months...**")
    valid_months, excluded = {}, {}
    for mn, mask in masks.items():
        if np.all(mask[ref_mask]):
            valid_months[mn] = downloaded_images[mn]
            st.write(f"   ✅ {mn}: {counts[mn]}/{total} - **INCLUDED**")
        else:
            missing = np.sum(ref_mask & ~mask)
            excluded[mn] = {'valid': counts[mn], 'missing': missing}
            st.write(f"   ❌ {mn}: {counts[mn]}/{total} - **EXCLUDED** (missing {missing})")

    st.divider()
    if not valid_months:
        st.error("❌ No months match reference mask!")
        return None, None, None
    st.success(f"✅ **{len(valid_months)}/{len(masks)}** months valid")
    if excluded:
        with st.expander(f"🚫 Excluded ({len(excluded)})"):
            for mn, info in excluded.items():
                st.write(f"  • {mn}: {info['valid']} patches, missing {info['missing']}")
    return ref_mask, (h, w), valid_months


# =============================================================================
# Classification
# =============================================================================
def classify_image_with_mask(image_path, model, device, month_name, valid_mask, original_size):
    try:
        with rasterio.open(image_path) as src:
            img_data = src.read()
        img = np.moveaxis(img_data, 0, -1)
        h, w, c = img.shape
        if original_size and (h != original_size[0] or w != original_size[1]):
            st.error(f"❌ {month_name}: Dimension mismatch!")
            return None, None, 0

        new_h = int(np.ceil(h / PATCH_SIZE) * PATCH_SIZE)
        new_w = int(np.ceil(w / PATCH_SIZE) * PATCH_SIZE)
        nh, nw = new_h // PATCH_SIZE, new_w // PATCH_SIZE
        if valid_mask.shape != (nh, nw):
            st.error(f"❌ {month_name}: Patch grid mismatch!")
            return None, None, 0

        if h != new_h or w != new_w:
            padded = np.zeros((new_h, new_w, c), dtype=img.dtype)
            padded[:h, :w, :] = img
            img = padded

        patches = patchify(img, (PATCH_SIZE, PATCH_SIZE, c), step=PATCH_SIZE)
        classified = np.zeros((nh, nw, PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
        prob_patches = np.zeros((nh, nw, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
        valid_count = 0

        for i in range(nh):
            for j in range(nw):
                if not valid_mask[i, j]: continue
                patch = normalized(patches[i, j, 0])
                tensor = torch.tensor(np.moveaxis(patch, -1, 0), dtype=torch.float32).unsqueeze(0)
                with torch.inference_mode():
                    pred = torch.sigmoid(model(tensor)).cpu().squeeze().numpy()
                classified[i, j] = (pred > 0.5).astype(np.uint8) * 255
                prob_patches[i, j] = pred
                valid_count += 1

        result = unpatchify(classified, (new_h, new_w))[:original_size[0], :original_size[1]]
        result_prob = unpatchify(prob_patches, (new_h, new_w))[:original_size[0], :original_size[1]]
        return result, result_prob, valid_count
    except:
        return None, None, 0


def get_valid_patch_bounds(valid_mask, patch_size=224, original_size=None):
    if valid_mask is None or not np.any(valid_mask): return None
    rows = np.where(np.any(valid_mask, axis=1))[0]
    cols = np.where(np.any(valid_mask, axis=0))[0]
    if len(rows) == 0 or len(cols) == 0: return None
    r0, r1 = rows[0] * patch_size, (rows[-1] + 1) * patch_size
    c0, c1 = cols[0] * patch_size, (cols[-1] + 1) * patch_size
    if original_size:
        r1 = min(r1, original_size[0])
        c1 = min(c1, original_size[1])
    return (r0, r1, c0, c1)


def create_pixel_mask_from_patches(valid_mask, patch_size=224, target_size=None):
    if valid_mask is None: return None
    pm = np.repeat(np.repeat(valid_mask, patch_size, axis=0), patch_size, axis=1)
    if target_size: pm = pm[:target_size[0], :target_size[1]]
    return pm


def generate_thumbnails(image_path, classification_mask, month_name, valid_mask=None, original_size=None, max_size=256):
    try:
        crop_bounds = get_valid_patch_bounds(valid_mask, PATCH_SIZE, original_size) if valid_mask is not None else None
        pixel_mask = create_pixel_mask_from_patches(valid_mask, PATCH_SIZE, original_size) if valid_mask is not None else None

        with rasterio.open(image_path) as src:
            rgb = np.nan_to_num(np.stack([src.read(4), src.read(3), src.read(2)], axis=-1), nan=0.0)

        def pstretch(band):
            valid = band[band > 0]
            if len(valid) == 0: return np.zeros_like(band, dtype=np.uint8)
            pl, ph = np.percentile(valid, 2), np.percentile(valid, 98)
            if ph <= pl: ph = pl + 0.001
            return (np.clip((band - pl) / (ph - pl), 0, 1) * 255).astype(np.uint8)

        rgb8 = np.stack([pstretch(rgb[:,:,i]) for i in range(3)], axis=-1)
        if pixel_mask is not None:
            rgb8 = np.where(np.stack([pixel_mask]*3, axis=-1), rgb8, 0)
        if crop_bounds:
            r0, r1, c0, c1 = crop_bounds
            rgb8 = rgb8[r0:r1, c0:c1, :]
            classification_mask = classification_mask[r0:r1, c0:c1]

        pil_rgb = Image.fromarray(rgb8, mode='RGB')
        pil_cls = Image.fromarray(classification_mask.astype(np.uint8))
        h, w = pil_rgb.size[1], pil_rgb.size[0]
        if h > max_size or w > max_size:
            s = max_size / max(h, w)
            pil_rgb = pil_rgb.resize((int(w*s), int(h*s)), Image.LANCZOS)
            pil_cls = pil_cls.resize((int(w*s), int(h*s)), Image.NEAREST)

        return {
            'rgb_image': pil_rgb, 'classification_image': pil_cls,
            'month_name': month_name, 'original_size': classification_mask.shape,
            'cropped_size': classification_mask.shape,
            'building_pixels': int(np.sum(classification_mask > 0)),
            'total_pixels': int(classification_mask.shape[0] * classification_mask.shape[1])
        }
    except Exception as e:
        st.warning(f"Thumbnail error for {month_name}: {e}")
        return None


# =============================================================================
# MAIN PROCESSING PIPELINE
# =============================================================================
def process_timeseries(aoi, start_date, end_date, model, device,
                       cloudy_pixel_percentage=10, scale=10, nodata_threshold_percent=5,
                       resume=False):
    try:
        if not resume:
            clear_log()

        add_log_entry(f"{'RESUME' if resume else 'START'}: Pipeline initiated")
        add_log_entry(f"Date range: {start_date} to {end_date}")

        if st.session_state.current_temp_dir is None or not os.path.exists(st.session_state.current_temp_dir):
            st.session_state.current_temp_dir = tempfile.mkdtemp()
        temp_dir = st.session_state.current_temp_dir

        start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        total_months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)

        st.info(f"📅 Processing {total_months} months | 📁 {temp_dir}")

        extended_start = (start_dt - datetime.timedelta(days=31)).strftime('%Y-%m-%d')
        extended_end = (end_dt + datetime.timedelta(days=31)).strftime('%Y-%m-%d')

        # PHASE 1
        st.header("Phase 1: Cloud-Free Collection")
        st.info(f"☁️ Cloud mask: prob > {CLOUD_PROB_THRESHOLD}, CDI < {CDI_THRESHOLD}")
        cloud_free_collection = create_cloud_free_collection(aoi, extended_start, extended_end, cloudy_pixel_percentage)

        # PHASE 2
        st.header("Phase 2: Download & Gap-Fill")
        downloaded_images = {}
        month_statuses = {}

        month_infos = []
        current_range_months = set()
        for mi in range(total_months):
            year = start_dt.year + (start_dt.month - 1 + mi) // 12
            month = (start_dt.month - 1 + mi) % 12 + 1
            mn = f"{year}-{month:02d}"
            month_infos.append({'month_name': mn, 'month_index': mi, 'origin': start_date})
            current_range_months.add(mn)

        st.info(f"📅 Expected: {month_infos[0]['month_name']} to {month_infos[-1]['month_name']} ({len(month_infos)} months)")

        # Check cached
        if resume and st.session_state.downloaded_images:
            for mn, path in st.session_state.downloaded_images.items():
                if mn not in current_range_months: continue
                if os.path.exists(path):
                    ok, _ = validate_geotiff_file(path, len(SPECTRAL_BANDS))
                    if ok:
                        downloaded_images[mn] = path
                        month_statuses[mn] = {'status': STATUS_COMPLETE, 'message': 'Cached'}
            if downloaded_images:
                st.info(f"🔄 Found {len(downloaded_images)} cached downloads")

        if resume and st.session_state.month_analysis_results:
            for mn, si in st.session_state.month_analysis_results.items():
                if mn in current_range_months and mn not in month_statuses:
                    month_statuses[mn] = si

        months_to_process = [m for m in month_infos
                             if m['month_name'] not in downloaded_images
                             and m['month_name'] not in month_statuses]

        if months_to_process:
            st.info(f"📥 {len(months_to_process)} months to process")
            progress = st.progress(0)
            status_text = st.empty()

            for idx, mi in enumerate(months_to_process):
                mn = mi['month_name']
                path, status, message = download_monthly_image_v06(
                    aoi, cloud_free_collection, mi, temp_dir, scale, status_text)

                month_statuses[mn] = {'status': status, 'message': message}
                st.session_state.month_analysis_results[mn] = {'status': status, 'message': message}

                icon = {"no_data": "⚫", "skipped": "🟡", "complete": "🟢", "rejected": "🔴"}.get(status, "❓")
                st.write(f"{icon} **{mn}**: {status} - {message}")

                if path:
                    downloaded_images[mn] = path
                    st.session_state.downloaded_images[mn] = path
                progress.progress((idx + 1) / len(months_to_process))
            progress.empty()
            status_text.empty()

        # Summary
        st.divider()
        sc = {s: sum(1 for ms in month_statuses.values() if ms['status'] == s)
              for s in [STATUS_NO_DATA, STATUS_SKIPPED, STATUS_COMPLETE, STATUS_REJECTED]}
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("✅ Complete", sc[STATUS_COMPLETE])
        c2.metric("🔴 Rejected", sc[STATUS_REJECTED])
        c3.metric("🟡 Skipped", sc[STATUS_SKIPPED])
        c4.metric("⚫ No Data", sc[STATUS_NO_DATA])

        st.session_state.downloaded_images = downloaded_images
        st.session_state.month_analysis_results = month_statuses

        if not downloaded_images:
            st.error("❌ No images downloaded!")
            return []

        st.success(f"✅ Downloaded {len(downloaded_images)}/{total_months} months")

        # =====================================================================
        # SENTINEL-2 DOWNLOAD SECTION - Before patch validation & classification
        # =====================================================================
        show_sentinel2_download_section()

        # =====================================================================
        # PHASE 3: Patch Validation
        # =====================================================================
        st.header("Phase 3: Patch Validation")
        add_log_entry("PHASE 3: Patch Validation")

        ref_mask, original_size, valid_months = find_common_valid_patches(
            downloaded_images, nodata_threshold_percent)

        if ref_mask is None or not valid_months:
            st.error("❌ Patch validation failed!")
            return []

        st.session_state.valid_patches_mask = ref_mask
        st.session_state.valid_months = valid_months

        # =====================================================================
        # PHASE 4: Classification
        # =====================================================================
        st.header("Phase 4: Classification")
        add_log_entry("PHASE 4: Classification")

        thumbnails = []
        progress = st.progress(0)
        status_text = st.empty()
        sorted_months = sorted(valid_months.keys())

        for idx, mn in enumerate(sorted_months):
            image_path = valid_months[mn]
            status_text.text(f"🔬 Classifying {mn} ({idx+1}/{len(sorted_months)})...")
            add_log_entry(f"Classifying {mn}")

            result, prob_map, valid_count = classify_image_with_mask(
                image_path, model, device, mn, ref_mask, original_size)

            if result is not None:
                st.session_state.probability_maps[mn] = prob_map
                thumb = generate_thumbnails(image_path, result, mn, ref_mask, original_size)
                if thumb:
                    thumbnails.append(thumb)
                    st.write(f"✅ **{mn}**: {valid_count} patches, "
                             f"{thumb['building_pixels']}/{thumb['total_pixels']} building pixels "
                             f"({100*thumb['building_pixels']/max(thumb['total_pixels'],1):.1f}%)")
            else:
                st.warning(f"⚠️ {mn}: Classification failed")
            progress.progress((idx + 1) / len(sorted_months))

        progress.empty()
        status_text.empty()

        st.session_state.classification_thumbnails = thumbnails
        st.session_state.processing_complete = True
        add_log_entry(f"COMPLETE: {len(thumbnails)} months classified")

        return thumbnails

    except Exception as e:
        st.error(f"❌ Pipeline error: {str(e)}")
        add_log_entry(f"PIPELINE ERROR: {str(e)}", "ERROR")
        import traceback
        st.error(traceback.format_exc())
        return []


# =============================================================================
# STREAMLIT UI
# =============================================================================
def main():
    st.title("🏗️ Building Classification Time Series v06")
    st.markdown("*GEE Cloud Masking + Gap-Filling + Patch Validation*")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # Earth Engine
        st.subheader("🌍 Earth Engine")
        if not st.session_state.ee_initialized:
            if st.button("Initialize Earth Engine"):
                with st.spinner("Initializing..."):
                    success, msg = initialize_earth_engine()
                    if success:
                        st.session_state.ee_initialized = True
                        st.success(msg)
                    else:
                        st.error(msg)
        else:
            st.success("✅ Earth Engine ready")

        # Model
        st.subheader("🤖 Model")
        if not st.session_state.model_loaded:
            if st.button("Load Model"):
                with st.spinner("Loading model..."):
                    model_path = "model.pth"
                    if not os.path.exists(model_path):
                        model_path = download_model_from_gdrive("", model_path)
                    if model_path:
                        model, device = load_model(model_path)
                        if model:
                            st.session_state.model = model
                            st.session_state.device = device
                            st.session_state.model_loaded = True
                            st.success("✅ Model loaded")
                        else:
                            st.error("Failed to load model")
                    else:
                        st.error("Failed to download model")
        else:
            st.success("✅ Model ready")

        st.divider()

        # Date range
        st.subheader("📅 Date Range")
        start_date = st.date_input("Start Date", value=date(2020, 1, 1))
        end_date = st.date_input("End Date", value=date(2020, 12, 1))

        st.subheader("🔧 Parameters")
        cloudy_pct = st.slider("Max Cloud %", 1, 50, 10)
        scale = st.selectbox("Scale (m)", [10, 20, 30], index=0)
        nodata_thresh = st.slider("Nodata Threshold %", 0, 20, 5)

    # Main area - Map
    st.header("🗺️ Select Area of Interest")
    st.write("Draw a rectangle on the map:")

    m = folium.Map(location=[25.0, 45.0], zoom_start=6)
    draw = plugins.Draw(
        export=False,
        draw_options={
            'polyline': False, 'polygon': True, 'circle': False,
            'marker': False, 'circlemarker': False, 'rectangle': True
        }
    )
    draw.add_to(m)
    map_data = st_folium(m, width=800, height=500, key="main_map")

    # Handle drawn features
    if map_data and map_data.get('last_active_drawing'):
        drawing = map_data['last_active_drawing']
        if drawing.get('geometry', {}).get('type') in ['Polygon', 'Rectangle']:
            coords = drawing['geometry']['coordinates'][0]
            st.session_state.last_drawn_polygon = coords
            st.success(f"✅ Area selected with {len(coords)} vertices")

    # Process button
    st.divider()
    col1, col2, col3 = st.columns(3)

    with col1:
        process_btn = st.button("🚀 Process", type="primary",
                                disabled=not (st.session_state.ee_initialized and
                                              st.session_state.model_loaded and
                                              st.session_state.last_drawn_polygon))
    with col2:
        resume_btn = st.button("🔄 Resume",
                               disabled=not st.session_state.downloaded_images)
    with col3:
        if st.button("🗑️ Clear All"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    if process_btn or resume_btn:
        coords = st.session_state.last_drawn_polygon
        if coords:
            aoi = ee.Geometry.Polygon([coords])
            sd = start_date.strftime('%Y-%m-%d')
            ed = end_date.strftime('%Y-%m-%d')

            thumbnails = process_timeseries(
                aoi, sd, ed,
                st.session_state.model, st.session_state.device,
                cloudy_pct, scale, nodata_thresh,
                resume=resume_btn
            )

            if thumbnails:
                st.header("📊 Results")
                for thumb in thumbnails:
                    st.subheader(f"📅 {thumb['month_name']}")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(thumb['rgb_image'], caption="RGB", use_container_width=True)
                    with c2:
                        st.image(thumb['classification_image'], caption="Classification", use_container_width=True)
                    bpct = 100 * thumb['building_pixels'] / max(thumb['total_pixels'], 1)
                    st.write(f"Building coverage: **{bpct:.2f}%** ({thumb['building_pixels']:,} pixels)")
        else:
            st.warning("⚠️ Please draw an area on the map first!")

    # Show download section if images exist but processing not yet started
    if st.session_state.downloaded_images and not st.session_state.processing_complete:
        show_sentinel2_download_section()

    # Show download section after processing is complete
    if st.session_state.processing_complete and st.session_state.downloaded_images:
        st.divider()
        st.header("📥 Downloads")
        show_sentinel2_download_section()


if __name__ == "__main__":
    main()
