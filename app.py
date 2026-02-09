"""
Sentinel-2 Time Series Building Classification
VERSION 07 - FIXED: Interior NaN patch issue caused by incomplete tile coverage

FIX APPLIED (3 locations in download_monthly_image_v06):
═══════════════════════════════════════════════════════════════════════════════
1. frequency image: .unmask(0).clip(aoi) ensures pixels NOT covered by ANY
   Sentinel-2 tile get frequency=0, not silently absent.
2. composite download (0% masked path): .unmask(0).clip(aoi) before download.
3. filled_composite download (gap-fill path): .unmask(0).clip(aoi) before download.

ROOT CAUSE: When no S2 tile covers part of AOI in a month, those pixels were
absent from the frequency image (not zero). reduceRegion only evaluated
existing pixels, so masked_percent was 0% even with missing coverage.

ALGORITHM:
═══════════════════════════════════════════════════════════════════════════════
GEE SERVER-SIDE:
1. Filter images by CLOUDY_PIXEL_PERCENTAGE metadata (<10%)
2. Join S2_SR with S2_CLOUD_PROBABILITY
3. Apply cloud mask: cloudProb > 50 AND cdi < -0.5, dilate 20m
4. For each month M:
   a. Create median composite from cloud-free collection
   b. Count masked pixels (frequency == 0) OVER ENTIRE AOI (FIXED)
   c. Status logic:
      - image_count == 0        → "no_data"   → SKIP
      - masked_percent > 30%    → "skipped"   → SKIP
      - masked_percent == 0%    → "complete"  → DOWNLOAD ✅
      - 0% < masked <= 30%      → TRY GAP-FILL from M-1, M+1

PYTHON CLIENT-SIDE (after download):
5. Validate all downloaded images have SAME dimensions
6. Check patch validity (NaN/zeros) for each month
7. Find the month with MAXIMUM valid patches (reference)
8. EXCLUDE months that don't have ALL reference patches valid
9. Classify only valid months using reference patch mask
═══════════════════════════════════════════════════════════════════════════════
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
from datetime import date
from PIL import Image
from io import BytesIO

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Examining the path of torch.classes")

import streamlit as st

st.set_page_config(
    layout="wide",
    page_title="Building Classification Time Series v07",
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
_defaults = {
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
    'monthly_component_images': {}, 'monthly_image_counts': {},
    'change_timing_map': None, 'log_pdf_bytes': None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# =============================================================================
# Normalization function
# =============================================================================
def normalized(img):
    min_val = np.nanmin(img)
    max_val = np.nanmax(img)
    if max_val == min_val:
        return np.zeros_like(img)
    return (img - min_val) / (max_val - min_val)


# =============================================================================
# Logging Helper Functions
# =============================================================================
def add_log_entry(message, level="INFO", include_timestamp=True):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = {'timestamp': timestamp, 'level': level, 'message': message}
    if 'processing_log' not in st.session_state:
        st.session_state.processing_log = []
    st.session_state.processing_log.append(entry)
    if include_timestamp:
        print(f"[{timestamp}] [{level}] {message}")


def clear_log():
    st.session_state.processing_log = []
    st.session_state.monthly_component_images = {}
    st.session_state.monthly_image_counts = {}
    st.session_state.change_timing_map = None
    st.session_state.log_pdf_bytes = None


# =============================================================================
# File Validation Functions
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
# Model Download Functions
# =============================================================================
@st.cache_data
def download_model_from_gdrive(gdrive_url, local_filename):
    try:
        correct_file_id = "1_8jOOSXnELA-xOGW0DKgRMo6RvnJYV5_"
        st.info(f"Downloading model...")
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
                    st.success(f"Model downloaded!")
                    return local_filename
            except:
                if os.path.exists(local_filename):
                    os.remove(local_filename)
        return None
    except Exception as e:
        return None


# =============================================================================
# Model Loading Function
# =============================================================================
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
# Earth Engine Authentication
# =============================================================================
@st.cache_resource
def initialize_earth_engine():
    try:
        ee.Initialize()
        return True, "Earth Engine initialized"
    except Exception:
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


# =============================================================================
# Helper Functions
# =============================================================================
def get_utm_zone(longitude):
    return math.floor((longitude + 180) / 6) + 1

def get_utm_epsg(longitude, latitude):
    zone_number = get_utm_zone(longitude)
    return f"EPSG:326{zone_number:02d}" if latitude >= 0 else f"EPSG:327{zone_number:02d}"


# =============================================================================
# GEE CLOUD MASKING (Server-Side)
# =============================================================================
def create_cloud_free_collection(aoi, extended_start, extended_end, cloudy_pixel_percentage=10):
    s2_sr = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
             .filterBounds(aoi)
             .filterDate(extended_start, extended_end)
             .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloudy_pixel_percentage))
             .select(SPECTRAL_BANDS + ['SCL']))
    s2_cloud_prob = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
                     .filterBounds(aoi)
                     .filterDate(extended_start, extended_end))
    join_filter = ee.Filter.equals(leftField='system:index', rightField='system:index')
    joined = ee.Join.saveFirst('cloud_probability').apply(
        primary=s2_sr, secondary=s2_cloud_prob, condition=join_filter)

    def add_cloud_band(feature):
        img = ee.Image(feature)
        cloud_prob_img = ee.Image(img.get('cloud_probability'))
        return img.addBands(cloud_prob_img)

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
        st.error(f"❌ Failed to get AOI bounds: {e}")
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
                content_type = response.headers.get('content-type', '')
                if 'text/html' in content_type:
                    last_error = "GEE rate limit (HTML response)"
                    raise Exception(last_error)
                downloaded_size = 0
                with open(temp_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)
                if downloaded_size < MIN_BAND_FILE_SIZE:
                    last_error = f"File too small ({downloaded_size} bytes)"
                    raise Exception(last_error)
                is_valid, msg = validate_band_file(temp_path, band)
                if is_valid:
                    os.replace(temp_path, output_path)
                    return True, "success"
                else:
                    last_error = f"Validation failed: {msg}"
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    raise Exception(last_error)
            else:
                last_error = f"HTTP {response.status_code}"
                raise Exception(last_error)
        except requests.exceptions.Timeout:
            last_error = "Timeout"
        except requests.exceptions.ConnectionError:
            last_error = "Connection error"
        except Exception as e:
            if last_error is None:
                last_error = str(e)
        for f in [output_path, temp_path]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    pass
        if attempt < MAX_RETRIES - 1:
            wait_time = RETRY_DELAY_BASE ** (attempt + 1)
            time.sleep(wait_time)
    return False, last_error


def download_composite(composite, aoi, output_path, month_name, scale=10, status_placeholder=None):
    try:
        if os.path.exists(output_path):
            is_valid, msg = validate_geotiff_file(output_path, expected_bands=len(SPECTRAL_BANDS))
            if is_valid:
                if status_placeholder:
                    status_placeholder.info(f"✅ {month_name} cached")
                return output_path
            else:
                st.warning(f"⚠️ {month_name}: Cached file invalid ({msg}), re-downloading...")
                os.remove(output_path)
        temp_dir = os.path.dirname(output_path)
        bands_dir = os.path.join(temp_dir, f"bands_{month_name}")
        os.makedirs(bands_dir, exist_ok=True)
        band_files = []
        failed_bands = []
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
            if status_placeholder:
                status_placeholder.text(f"📦 {month_name}: Merging bands...")
            with rasterio.open(band_files[0]) as src:
                meta = src.meta.copy()
            meta.update(count=len(band_files))
            with rasterio.open(output_path, 'w', **meta) as dst:
                for i, band_file in enumerate(band_files):
                    with rasterio.open(band_file) as src:
                        dst.write(src.read(1), i+1)
            is_valid, msg = validate_geotiff_file(output_path, expected_bands=len(SPECTRAL_BANDS))
            if is_valid:
                return output_path
            else:
                st.error(f"❌ {month_name}: Merged file validation failed: {msg}")
                if os.path.exists(output_path):
                    os.remove(output_path)
        return None
    except Exception as e:
        st.error(f"❌ {month_name}: Download exception - {str(e)}")
        return None


def download_component_images_for_log(monthly_images, aoi, month_name, image_count, max_images=10):
    component_rgbs = []
    try:
        actual_count = min(image_count, max_images)
        image_list = monthly_images.toList(actual_count)
        for i in range(actual_count):
            try:
                img = ee.Image(image_list.get(i))
                img_date = ee.Date(img.get('system:time_start')).format('YYYY-MM-dd').getInfo()
                rgb_img = img.select(['B4', 'B3', 'B2'])
                region = aoi.bounds().getInfo()['coordinates']
                url = rgb_img.getThumbURL({
                    'region': region, 'dimensions': 256, 'format': 'png',
                    'min': 0, 'max': 0.3
                })
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    img_pil = Image.open(BytesIO(response.content))
                    img_array = np.array(img_pil.convert('RGB'))
                    component_rgbs.append({'date': img_date, 'rgb': img_array})
            except Exception as e:
                add_log_entry(f"{month_name}: Failed to get component image {i+1}: {e}", "WARNING")
                continue
        return component_rgbs
    except Exception as e:
        add_log_entry(f"{month_name}: Failed to download component images: {e}", "WARNING")
        return []


# =============================================================================
# MONTHLY IMAGE DOWNLOAD WITH FIX v07
# =============================================================================
def download_monthly_image_v06(aoi, cloud_free_collection, month_info, temp_dir,
                                scale=10, status_placeholder=None):
    """
    Download a single monthly composite with gap-filling.

    ═══════════════════════════════════════════════════════════════════════════
    FIX v07 (3 changes marked with "# *** FIX v07 ***"):
    1. frequency: .unmask(0).clip(aoi) so uncovered pixels = 0, not absent
    2. composite download: .unmask(0).clip(aoi) before download
    3. filled_composite download: .unmask(0).clip(aoi) before download
    ═══════════════════════════════════════════════════════════════════════════

    Returns: (output_path, status, message)
    """
    month_name = month_info['month_name']
    month_index = month_info['month_index']
    origin = month_info['origin']
    output_file = os.path.join(temp_dir, f"sentinel2_{month_name}.tif")
    add_log_entry(f"Processing month: {month_name}", "INFO")

    # Check cache first
    if os.path.exists(output_file):
        is_valid, msg = validate_geotiff_file(output_file, expected_bands=len(SPECTRAL_BANDS))
        if is_valid:
            add_log_entry(f"{month_name}: Using cached file", "INFO")
            if status_placeholder:
                status_placeholder.info(f"✅ {month_name} using cached file")
            return output_file, STATUS_COMPLETE, "Cached"
        else:
            add_log_entry(f"{month_name}: Cache invalid ({msg}), re-processing", "WARNING")
            if status_placeholder:
                status_placeholder.warning(f"⚠️ {month_name} cache invalid, re-processing...")
            os.remove(output_file)

    try:
        origin_date = ee.Date(origin)
        month_start = origin_date.advance(month_index, 'month')
        month_end = origin_date.advance(ee.Number(month_index).add(1), 'month')
        month_middle = month_start.advance(15, 'day')

        if status_placeholder:
            status_placeholder.text(f"📥 {month_name}: Analyzing...")

        # Filter to current month
        monthly_images = cloud_free_collection.filterDate(month_start, month_end)
        image_count = monthly_images.size().getInfo()
        st.session_state.monthly_image_counts[month_name] = image_count
        add_log_entry(f"{month_name}: Found {image_count} cloud-free images", "INFO")

        # CHECK 1: No images
        if image_count == 0:
            add_log_entry(f"{month_name}: REJECTED - No images available", "WARNING")
            return None, STATUS_NO_DATA, "No images available"

        if status_placeholder:
            status_placeholder.text(f"📥 {month_name}: Creating composite from {image_count} images...")

        # Download component images for logging
        try:
            add_log_entry(f"{month_name}: Downloading {image_count} component image previews at 100m", "INFO")
            component_rgbs = download_component_images_for_log(monthly_images, aoi, month_name, image_count)
            if component_rgbs:
                st.session_state.monthly_component_images[month_name] = component_rgbs
                add_log_entry(f"{month_name}: Captured {len(component_rgbs)} component image previews", "INFO")
        except Exception as e:
            add_log_entry(f"{month_name}: Failed to capture component images: {e}", "WARNING")

        # Create frequency map and median composite
        def create_valid_mask(img):
            return ee.Image(1).updateMask(img.select('B4').mask()).unmask(0).toInt()

        # *** FIX v07 *** : .unmask(0).clip(aoi) ensures pixels NOT covered
        # by ANY Sentinel-2 tile get frequency=0 instead of being absent.
        # Without this, reduceRegion only sees pixels where tiles exist,
        # reporting 0% masked even when part of AOI has no coverage.
        frequency = (monthly_images.map(create_valid_mask)
                     .sum()
                     .unmask(0)       # *** FIX v07: absent pixels → 0 ***
                     .clip(aoi)       # *** FIX v07: full AOI coverage ***
                     .toInt()
                     .rename('frequency'))

        composite = monthly_images.median()

        # Calculate masked pixel statistics - now correctly over ENTIRE AOI
        masked_stats = frequency.eq(0).reduceRegion(
            reducer=ee.Reducer.sum(), geometry=aoi, scale=10, maxPixels=1e13)
        total_stats = frequency.gte(0).reduceRegion(
            reducer=ee.Reducer.sum(), geometry=aoi, scale=10, maxPixels=1e13)

        masked_count = ee.Number(masked_stats.get('frequency')).getInfo()
        total_count = ee.Number(total_stats.get('frequency')).getInfo()

        if total_count == 0:
            add_log_entry(f"{month_name}: REJECTED - No valid pixels", "WARNING")
            return None, STATUS_NO_DATA, "No valid pixels"

        masked_percent = (masked_count / total_count) * 100
        add_log_entry(f"{month_name}: Masked pixels: {masked_percent:.2f}% ({masked_count}/{total_count})", "INFO")

        # CHECK 2: Too many masked (> 30%)
        if masked_percent > MAX_MASKED_PERCENT_FOR_GAPFILL:
            add_log_entry(f"{month_name}: SKIPPED - Masked {masked_percent:.1f}% > {MAX_MASKED_PERCENT_FOR_GAPFILL}%", "WARNING")
            return None, STATUS_SKIPPED, f"Masked {masked_percent:.1f}% > {MAX_MASKED_PERCENT_FOR_GAPFILL}%"

        # CHECK 3: No masked pixels - ready to download
        if masked_percent == 0:
            add_log_entry(f"{month_name}: Complete (0% masked), starting download at {scale}m", "INFO")
            if status_placeholder:
                status_placeholder.text(f"📥 {month_name}: Complete (0% masked), downloading...")

            # *** FIX v07 ***: unmask + clip composite before download to ensure
            # full AOI coverage with explicit zeros where no data exists
            download_img = composite.unmask(0).clip(aoi)

            path = download_composite(download_img, aoi, output_file, month_name, scale, status_placeholder)
            if path:
                add_log_entry(f"{month_name}: Download SUCCESSFUL", "INFO")
                return path, STATUS_COMPLETE, "Complete (0% masked)"
            else:
                add_log_entry(f"{month_name}: Download FAILED", "ERROR")
                return None, STATUS_REJECTED, "Download failed"

        # GAP-FILL: 0% < masked <= 30%
        add_log_entry(f"{month_name}: Starting gap-fill ({masked_percent:.1f}% masked)", "INFO")
        if status_placeholder:
            status_placeholder.text(f"📥 {month_name}: Gap-filling ({masked_percent:.1f}% masked)...")

        gap_mask = frequency.eq(0)
        month_middle_millis = month_middle.millis()

        # M-1 and M+1 ranges
        m1_past_start = origin_date.advance(ee.Number(month_index).subtract(1), 'month')
        m1_past_end = month_start
        m1_future_start = month_end
        m1_future_end = origin_date.advance(ee.Number(month_index).add(2), 'month')

        m1_past_images = cloud_free_collection.filterDate(m1_past_start, m1_past_end)
        m1_future_images = cloud_free_collection.filterDate(m1_future_start, m1_future_end)
        all_candidates = m1_past_images.merge(m1_future_images)

        def add_time_distance(img):
            img_time = ee.Number(img.get('system:time_start'))
            time_diff = img_time.subtract(month_middle_millis).abs()
            return img.set('time_distance', time_diff)

        images_with_distance = all_candidates.map(add_time_distance)
        sorted_images = images_with_distance.sort('time_distance', True)
        candidate_count = sorted_images.size().getInfo()
        add_log_entry(f"{month_name}: Found {candidate_count} gap-fill candidates from adjacent months", "INFO")

        if candidate_count == 0:
            add_log_entry(f"{month_name}: REJECTED - No gap-fill candidates", "WARNING")
            return None, STATUS_REJECTED, f"No gap-fill candidates, {masked_percent:.1f}% still masked"

        # Create mosaic (closest pixel first)
        closest_mosaic = sorted_images.mosaic().select(SPECTRAL_BANDS)
        has_closest = closest_mosaic.select('B4').mask()
        fill_from_closest = gap_mask.And(has_closest)
        still_masked = gap_mask.And(has_closest.Not())
        filled_composite = composite.unmask(closest_mosaic.updateMask(fill_from_closest))

        # Check remaining masked pixels
        fill_source = (ee.Image.constant(0).clip(aoi).toInt8()
                       .where(fill_from_closest, 1)
                       .where(still_masked, 2)
                       .rename('fill_source'))

        still_masked_result = fill_source.eq(2).reduceRegion(
            reducer=ee.Reducer.sum(), geometry=aoi, scale=10, maxPixels=1e13
        ).get('fill_source')

        still_masked_count = ee.Number(ee.Algorithms.If(
            ee.Algorithms.IsEqual(still_masked_result, None), 0, still_masked_result
        )).getInfo()

        # CHECK 4: After gap-fill
        if still_masked_count == 0:
            add_log_entry(f"{month_name}: Gap-fill SUCCESSFUL, starting download", "INFO")
            if status_placeholder:
                status_placeholder.text(f"📥 {month_name}: Gap-filled successfully, downloading...")

            # *** FIX v07 ***: unmask + clip filled composite before download
            download_img = filled_composite.unmask(0).clip(aoi)

            path = download_composite(download_img, aoi, output_file, month_name, scale, status_placeholder)
            if path:
                add_log_entry(f"{month_name}: Download SUCCESSFUL after gap-fill", "INFO")
                return path, STATUS_COMPLETE, f"Complete after gap-fill (was {masked_percent:.1f}%)"
            else:
                add_log_entry(f"{month_name}: Download FAILED after gap-fill", "ERROR")
                return None, STATUS_REJECTED, "Download failed after gap-fill"
        else:
            still_masked_pct = (still_masked_count / total_count) * 100
            add_log_entry(f"{month_name}: REJECTED - {still_masked_pct:.1f}% still masked after gap-fill", "WARNING")
            return None, STATUS_REJECTED, f"{still_masked_pct:.1f}% still masked after gap-fill"

    except Exception as e:
        add_log_entry(f"{month_name}: ERROR - {str(e)}", "ERROR")
        return None, STATUS_NO_DATA, f"Error: {str(e)}"


# =============================================================================
# RGB Thumbnail Generation
# =============================================================================
def generate_rgb_thumbnail(image_path, month_name, max_size=256):
    try:
        with rasterio.open(image_path) as src:
            red = src.read(4); green = src.read(3); blue = src.read(2)
            rgb = np.stack([red, green, blue], axis=-1)
            rgb = np.nan_to_num(rgb, nan=0.0)
            def percentile_stretch(band, lower=2, upper=98):
                valid = band[band > 0]
                if len(valid) == 0: return np.zeros_like(band, dtype=np.uint8)
                p_low, p_high = np.percentile(valid, lower), np.percentile(valid, upper)
                if p_high <= p_low: p_high = p_low + 0.001
                return (np.clip((band - p_low) / (p_high - p_low), 0, 1) * 255).astype(np.uint8)
            rgb_uint8 = np.zeros_like(rgb, dtype=np.uint8)
            for i in range(3):
                rgb_uint8[:, :, i] = percentile_stretch(rgb[:, :, i])
            pil_img = Image.fromarray(rgb_uint8, mode='RGB')
            h, w = pil_img.size[1], pil_img.size[0]
            if h > max_size or w > max_size:
                scale = max_size / max(h, w)
                pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            return pil_img
    except:
        return None


# =============================================================================
# PYTHON-SIDE PATCH VALIDITY CHECKING
# =============================================================================
def check_patch_validity(patch, nodata_threshold_percent=0):
    if np.any(np.isnan(patch)): return False
    if np.all(patch == 0): return False
    zero_percent = (np.sum(patch == 0) / patch.size) * 100
    if zero_percent > nodata_threshold_percent: return False
    if patch.ndim == 3:
        for band_idx in range(patch.shape[-1]):
            if np.all(patch[:, :, band_idx] == 0): return False
    return True


def get_patch_validity_mask(image_path, patch_size=224, nodata_threshold_percent=0):
    try:
        with rasterio.open(image_path) as src:
            img_data = src.read()
        img_for_patching = np.moveaxis(img_data, 0, -1)
        h, w, c = img_for_patching.shape
        new_h = int(np.ceil(h / patch_size) * patch_size)
        new_w = int(np.ceil(w / patch_size) * patch_size)
        if h != new_h or w != new_w:
            padded_img = np.zeros((new_h, new_w, c), dtype=img_for_patching.dtype)
            padded_img[:h, :w, :] = img_for_patching
            img_for_patching = padded_img
        patches = patchify(img_for_patching, (patch_size, patch_size, c), step=patch_size)
        n_patches_h, n_patches_w = patches.shape[0], patches.shape[1]
        validity_mask = np.zeros((n_patches_h, n_patches_w), dtype=bool)
        for i in range(n_patches_h):
            for j in range(n_patches_w):
                validity_mask[i, j] = check_patch_validity(patches[i, j, 0], nodata_threshold_percent)
        return validity_mask, (h, w), (n_patches_h, n_patches_w)
    except:
        return None, None, None


def find_common_valid_patches(downloaded_images, nodata_threshold_percent=0):
    st.info("🔍 Analyzing patch validity across all months...")
    month_names = sorted(downloaded_images.keys())
    if len(month_names) == 0:
        st.error("❌ No downloaded images to analyze!")
        return None, None, None

    # STEP 1: Validate dimensions
    st.write("**Step 1: Validating image dimensions...**")
    dimensions = {}
    for month_name in month_names:
        try:
            with rasterio.open(downloaded_images[month_name]) as src:
                dimensions[month_name] = {'height': src.height, 'width': src.width, 'bands': src.count}
        except Exception as e:
            st.error(f"❌ Cannot read {month_name}: {e}")
            return None, None, None
    first_month = month_names[0]
    reference_dim = dimensions[first_month]
    mismatched = [f"{mn}: {d['height']}x{d['width']}" for mn, d in dimensions.items()
                  if d['height'] != reference_dim['height'] or d['width'] != reference_dim['width']]
    if mismatched:
        st.error(f"❌ **DIMENSION MISMATCH!** Ref ({first_month}): {reference_dim['height']}x{reference_dim['width']}")
        st.error(f"Mismatched: {', '.join(mismatched)}")
        return None, None, None
    st.success(f"✅ All {len(month_names)} images: {reference_dim['height']}x{reference_dim['width']} ({reference_dim['bands']} bands)")

    # STEP 2: Patch grid
    h, w = reference_dim['height'], reference_dim['width']
    n_patches_h = int(np.ceil(h / PATCH_SIZE))
    n_patches_w = int(np.ceil(w / PATCH_SIZE))
    total_patches = n_patches_h * n_patches_w
    st.write(f"**Step 2: Patch grid**: {n_patches_h} x {n_patches_w} = **{total_patches} patches**")

    # STEP 3: Validity per month
    st.write("**Step 3: Calculating validity for each month...**")
    progress_bar = st.progress(0)
    month_validity_masks = {}
    month_valid_counts = {}
    for idx, month_name in enumerate(month_names):
        validity_mask, _, _ = get_patch_validity_mask(downloaded_images[month_name], PATCH_SIZE, nodata_threshold_percent)
        if validity_mask is None:
            st.warning(f"⚠️ Could not analyze {month_name}")
            progress_bar.progress((idx + 1) / len(month_names))
            continue
        if validity_mask.shape != (n_patches_h, n_patches_w):
            st.error(f"❌ {month_name}: Patch grid mismatch!")
            return None, None, None
        month_validity_masks[month_name] = validity_mask
        month_valid_counts[month_name] = np.sum(validity_mask)
        progress_bar.progress((idx + 1) / len(month_names))
    progress_bar.empty()
    if not month_validity_masks:
        st.error("❌ Could not analyze any months!")
        return None, None, None

    # STEP 4: Reference mask
    st.write("**Step 4: Finding reference mask (maximum valid patches)...**")
    max_valid_count = max(month_valid_counts.values())
    max_months = [mn for mn, c in month_valid_counts.items() if c == max_valid_count]
    st.info(f"📊 Maximum valid patches: **{max_valid_count}/{total_patches}** ({100*max_valid_count/total_patches:.1f}%)")
    st.write(f"   Months with max patches: {', '.join(max_months)}")
    reference_mask = month_validity_masks[max_months[0]]

    # STEP 5: Filter months
    st.write("**Step 5: Filtering months that match reference mask...**")
    valid_months = {}
    excluded_months = {}
    for month_name, mask in month_validity_masks.items():
        if np.all(mask[reference_mask] == True):
            valid_months[month_name] = downloaded_images[month_name]
            st.write(f"   ✅ {month_name}: {month_valid_counts[month_name]}/{total_patches} - **INCLUDED**")
        else:
            missing = np.sum(reference_mask & ~mask)
            excluded_months[month_name] = {'valid_count': month_valid_counts[month_name], 'missing_count': missing}
            st.write(f"   ❌ {month_name}: {month_valid_counts[month_name]}/{total_patches} - **EXCLUDED** (missing {missing})")

    st.divider()
    if not valid_months:
        st.error("❌ No months match the reference mask!")
        return None, None, None
    st.success(f"✅ **{len(valid_months)}/{len(month_validity_masks)}** months have all {max_valid_count} valid patches")
    if excluded_months:
        with st.expander(f"🚫 Excluded Months ({len(excluded_months)})", expanded=True):
            for mn, info in excluded_months.items():
                st.write(f"  • {mn}: had {info['valid_count']} patches, missing {info['missing_count']}")
    return reference_mask, (h, w), valid_months


def classify_image_with_mask(image_path, model, device, month_name, valid_mask, original_size):
    try:
        with rasterio.open(image_path) as src:
            img_data = src.read()
        img_for_patching = np.moveaxis(img_data, 0, -1)
        h, w, c = img_for_patching.shape
        if original_size is not None:
            if h != original_size[0] or w != original_size[1]:
                st.error(f"❌ {month_name}: Dimension mismatch!")
                return None, None, 0
        new_h = int(np.ceil(h / PATCH_SIZE) * PATCH_SIZE)
        new_w = int(np.ceil(w / PATCH_SIZE) * PATCH_SIZE)
        n_patches_h, n_patches_w = new_h // PATCH_SIZE, new_w // PATCH_SIZE
        if valid_mask.shape != (n_patches_h, n_patches_w):
            st.error(f"❌ {month_name}: Patch grid mismatch!")
            return None, None, 0
        if h != new_h or w != new_w:
            padded = np.zeros((new_h, new_w, c), dtype=img_for_patching.dtype)
            padded[:h, :w, :] = img_for_patching
            img_for_patching = padded
        patches = patchify(img_for_patching, (PATCH_SIZE, PATCH_SIZE, c), step=PATCH_SIZE)
        classified_patches = np.zeros((n_patches_h, n_patches_w, PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
        probability_patches = np.zeros((n_patches_h, n_patches_w, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
        valid_count = 0
        for i in range(n_patches_h):
            for j in range(n_patches_w):
                if not valid_mask[i, j]: continue
                patch = patches[i, j, 0]
                patch_normalized = normalized(patch)
                patch_tensor = torch.tensor(np.moveaxis(patch_normalized, -1, 0), dtype=torch.float32).unsqueeze(0)
                with torch.inference_mode():
                    prediction = model(patch_tensor)
                    prediction = torch.sigmoid(prediction).cpu()
                pred_np = prediction.squeeze().numpy()
                classified_patches[i, j] = (pred_np > 0.5).astype(np.uint8) * 255
                probability_patches[i, j] = pred_np
                valid_count += 1
        reconstructed = unpatchify(classified_patches, (new_h, new_w))[:original_size[0], :original_size[1]]
        reconstructed_prob = unpatchify(probability_patches, (new_h, new_w))[:original_size[0], :original_size[1]]
        return reconstructed, reconstructed_prob, valid_count
    except:
        return None, None, 0


def get_valid_patch_bounds(valid_mask, patch_size=224, original_size=None):
    if valid_mask is None or not np.any(valid_mask): return None
    valid_rows = np.any(valid_mask, axis=1)
    valid_cols = np.any(valid_mask, axis=0)
    row_indices = np.where(valid_rows)[0]
    col_indices = np.where(valid_cols)[0]
    if len(row_indices) == 0 or len(col_indices) == 0: return None
    r0, r1 = row_indices[0] * patch_size, (row_indices[-1] + 1) * patch_size
    c0, c1 = col_indices[0] * patch_size, (col_indices[-1] + 1) * patch_size
    if original_size:
        r1 = min(r1, original_size[0]); c1 = min(c1, original_size[1])
    return (r0, r1, c0, c1)


def create_pixel_mask_from_patches(valid_mask, patch_size=224, target_size=None):
    if valid_mask is None: return None
    pixel_mask = np.repeat(np.repeat(valid_mask, patch_size, axis=0), patch_size, axis=1)
    if target_size: pixel_mask = pixel_mask[:target_size[0], :target_size[1]]
    return pixel_mask


def generate_thumbnails(image_path, classification_mask, month_name, valid_mask=None, original_size=None, max_size=256):
    try:
        crop_bounds = pixel_mask = None
        if valid_mask is not None:
            crop_bounds = get_valid_patch_bounds(valid_mask, PATCH_SIZE, original_size)
            pixel_mask = create_pixel_mask_from_patches(valid_mask, PATCH_SIZE, original_size)
        with rasterio.open(image_path) as src:
            red, green, blue = src.read(4), src.read(3), src.read(2)
            rgb = np.nan_to_num(np.stack([red, green, blue], axis=-1), nan=0.0)
            def ps(band, lo=2, hi=98):
                v = band[band > 0]
                if len(v) == 0: return np.zeros_like(band, dtype=np.uint8)
                pl, ph = np.percentile(v, lo), np.percentile(v, hi)
                if ph <= pl: ph = pl + 0.001
                return (np.clip((band - pl) / (ph - pl), 0, 1) * 255).astype(np.uint8)
            rgb8 = np.zeros_like(rgb, dtype=np.uint8)
            for i in range(3): rgb8[:, :, i] = ps(rgb[:, :, i])
        if pixel_mask is not None:
            rgb8 = np.where(np.stack([pixel_mask]*3, axis=-1), rgb8, 0)
        if crop_bounds:
            r0, r1, c0, c1 = crop_bounds
            rgb8 = rgb8[r0:r1, c0:c1, :]; classification_mask = classification_mask[r0:r1, c0:c1]
        pil_rgb = Image.fromarray(rgb8, mode='RGB')
        pil_class = Image.fromarray(classification_mask.astype(np.uint8))
        h, w = pil_rgb.size[1], pil_rgb.size[0]
        if h > max_size or w > max_size:
            s = max_size / max(h, w)
            pil_rgb = pil_rgb.resize((int(w*s), int(h*s)), Image.LANCZOS)
            pil_class = pil_class.resize((int(w*s), int(h*s)), Image.NEAREST)
        bp = np.sum(classification_mask > 0); tp = classification_mask.shape[0] * classification_mask.shape[1]
        return {'rgb_image': pil_rgb, 'classification_image': pil_class, 'month_name': month_name,
                'original_size': classification_mask.shape, 'cropped_size': classification_mask.shape,
                'building_pixels': bp, 'total_pixels': tp}
    except Exception as e:
        st.warning(f"Error generating thumbnails for {month_name}: {e}")
        return None


# =============================================================================
# MAIN PROCESSING PIPELINE
# =============================================================================
def process_timeseries(aoi, start_date, end_date, model, device,
                       cloudy_pixel_percentage=10, scale=10, nodata_threshold_percent=5,
                       resume=False):
    try:
        if not resume: clear_log()
        add_log_entry(f"{'RESUME' if resume else 'START'}: Pipeline initiated", "INFO")
        add_log_entry(f"Date range: {start_date} to {end_date}", "INFO")
        add_log_entry(f"Cloud %: {cloudy_pixel_percentage}, Scale: {scale}m, Nodata %: {nodata_threshold_percent}", "INFO")

        if st.session_state.current_temp_dir is None or not os.path.exists(st.session_state.current_temp_dir):
            st.session_state.current_temp_dir = tempfile.mkdtemp()
        temp_dir = st.session_state.current_temp_dir
        start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        total_months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)
        add_log_entry(f"Total months: {total_months}", "INFO")
        st.info(f"📅 Processing {total_months} months | 📁 {temp_dir}")

        extended_start = (start_dt - datetime.timedelta(days=31)).strftime('%Y-%m-%d')
        extended_end = (end_dt + datetime.timedelta(days=31)).strftime('%Y-%m-%d')

        # PHASE 1
        add_log_entry("PHASE 1: Creating cloud-free collection", "INFO")
        st.header("Phase 1: Cloud-Free Collection")
        st.info(f"☁️ Cloud mask: prob > {CLOUD_PROB_THRESHOLD}, CDI < {CDI_THRESHOLD}")
        cloud_free_collection = create_cloud_free_collection(aoi, extended_start, extended_end, cloudy_pixel_percentage)
        add_log_entry("Cloud-free collection created", "INFO")

        # PHASE 2
        add_log_entry("PHASE 2: Download & Gap-Fill", "INFO")
        st.header("Phase 2: Download & Gap-Fill")
        downloaded_images = {}
        month_statuses = {}

        month_infos = []
        current_range_months = set()
        for mi in range(total_months):
            y = start_dt.year + (start_dt.month - 1 + mi) // 12
            m = (start_dt.month - 1 + mi) % 12 + 1
            mn = f"{y}-{m:02d}"
            month_infos.append({'month_name': mn, 'month_index': mi, 'origin': start_date})
            current_range_months.add(mn)

        st.info(f"📅 Expected: {month_infos[0]['month_name']} to {month_infos[-1]['month_name']} ({len(month_infos)} months)")

        # Resume: check cached downloads
        if resume and st.session_state.downloaded_images:
            for mn, p in st.session_state.downloaded_images.items():
                if mn not in current_range_months: continue
                if os.path.exists(p):
                    ok, _ = validate_geotiff_file(p, len(SPECTRAL_BANDS))
                    if ok:
                        downloaded_images[mn] = p
                        month_statuses[mn] = {'status': STATUS_COMPLETE, 'message': 'Cached'}
            if downloaded_images:
                add_log_entry(f"RESUME: {len(downloaded_images)} cached", "INFO")
                st.info(f"🔄 Found {len(downloaded_images)} cached downloads")
                for mn in sorted(downloaded_images.keys()):
                    st.write(f"🟢 **{mn}**: complete (cached)")

        # Resume: restore statuses
        if resume and st.session_state.month_analysis_results:
            for mn, si in st.session_state.month_analysis_results.items():
                if mn not in current_range_months: continue
                if mn not in month_statuses:
                    month_statuses[mn] = si
                    s = si.get('status', '?')
                    icon = {"no_data":"⚫","skipped":"🟡","complete":"🟢","rejected":"🔴"}.get(s, "❓")
                    st.write(f"{icon} **{mn}**: {s} (cached) - {si.get('message','')}")

        # Process remaining
        to_process = [m for m in month_infos if m['month_name'] not in downloaded_images and m['month_name'] not in month_statuses]
        if to_process:
            add_log_entry(f"Processing {len(to_process)} remaining months", "INFO")
            st.info(f"📥 {len(to_process)} months to process")
            pb = st.progress(0); st_text = st.empty()
            for idx, mi in enumerate(to_process):
                mn = mi['month_name']
                if mn in month_statuses:
                    pb.progress((idx+1)/len(to_process)); continue
                path, status, message = download_monthly_image_v06(
                    aoi=aoi, cloud_free_collection=cloud_free_collection, month_info=mi,
                    temp_dir=temp_dir, scale=scale, status_placeholder=st_text)
                month_statuses[mn] = {'status': status, 'message': message}
                st.session_state.month_analysis_results[mn] = {'status': status, 'message': message}
                icon = {"no_data":"⚫","skipped":"🟡","complete":"🟢","rejected":"🔴"}.get(status, "❓")
                st.write(f"{icon} **{mn}**: {status} - {message}")
                if path:
                    downloaded_images[mn] = path
                    st.session_state.downloaded_images[mn] = path
                pb.progress((idx+1)/len(to_process))
            pb.empty(); st_text.empty()

        # Summary
        st.divider()
        sc = {s: sum(1 for ms in month_statuses.values() if ms['status']==s)
              for s in [STATUS_NO_DATA, STATUS_SKIPPED, STATUS_COMPLETE, STATUS_REJECTED]}
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("✅ Complete", sc[STATUS_COMPLETE])
        c2.metric("🔴 Rejected", sc[STATUS_REJECTED])
        c3.metric("🟡 Skipped", sc[STATUS_SKIPPED])
        c4.metric("⚫ No Data", sc[STATUS_NO_DATA])
        st.session_state.failed_downloads = [mn for mn, ms in month_statuses.items() if ms['status'] != STATUS_COMPLETE]
        st.session_state.month_analysis_results = month_statuses
        if not downloaded_images:
            st.error("❌ No images downloaded!"); return []
        st.success(f"✅ Downloaded {len(downloaded_images)}/{total_months} months")

        # PHASE 3
        st.header("Phase 3: Patch Validity Analysis")
        valid_mask, original_size, valid_months = find_common_valid_patches(downloaded_images, nodata_threshold_percent)
        if valid_mask is None or valid_months is None: return []
        st.session_state.valid_patches_mask = valid_mask
        st.session_state.valid_months = valid_months
        if st.session_state.processed_months:
            st.info("🔄 Clearing thumbnail cache...")
        st.session_state.processed_months = {}
        excluded_count = len(downloaded_images) - len(valid_months)
        if excluded_count > 0:
            st.warning(f"⚠️ {excluded_count} months excluded due to missing patches")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.imshow(valid_mask, cmap='RdYlGn', vmin=0, vmax=1)
        ax.set_title(f"Reference Valid Patches ({np.sum(valid_mask)} patches)")
        st.pyplot(fig); plt.close()

        # PHASE 4
        st.header("Phase 4: Classification")
        st.info(f"🧠 Classifying **{len(valid_months)}** months (excluded {excluded_count})")
        thumbnails = []; probability_maps = {}
        mns = sorted(valid_months.keys())
        pb = st.progress(0); st_text = st.empty()
        for idx, mn in enumerate(mns):
            if mn in st.session_state.processed_months:
                thumbnails.append(st.session_state.processed_months[mn])
                if mn in st.session_state.probability_maps:
                    probability_maps[mn] = st.session_state.probability_maps[mn]
                pb.progress((idx+1)/len(mns)); continue
            st_text.text(f"🧠 {mn} ({idx+1}/{len(mns)})...")
            mask, prob_map, vc = classify_image_with_mask(valid_months[mn], model, device, mn, valid_mask, original_size)
            if mask is not None:
                probability_maps[mn] = prob_map
                st.session_state.probability_maps[mn] = prob_map
                thumb = generate_thumbnails(valid_months[mn], mask, mn, valid_mask=valid_mask, original_size=original_size)
                if thumb:
                    thumb['valid_patches'] = vc
                    ms = month_statuses.get(mn, {})
                    thumb['gap_filled'] = 'gap-fill' in ms.get('message', '').lower()
                    thumbnails.append(thumb)
                    st.session_state.processed_months[mn] = thumb
            pb.progress((idx+1)/len(mns))
        pb.empty(); st_text.empty()
        st.success(f"✅ Classified {len(thumbnails)} months!")

        # Filter haze/snow
        if len(thumbnails) >= 2:
            bpcts = [(t['building_pixels']/t['total_pixels'])*100 for t in thumbnails]
            med = np.median(bpcts)
            filtered = []
            for t in thumbnails:
                pct = (t['building_pixels']/t['total_pixels'])*100
                if pct < (med - 8.0):
                    print(f"🚫 Excluded {t['month_name']}: {pct:.1f}% (median={med:.1f}%)")
                    st.session_state.valid_months.pop(t['month_name'], None)
                    st.session_state.probability_maps.pop(t['month_name'], None)
                else:
                    filtered.append(t)
            thumbnails = filtered
        return thumbnails
    except Exception as e:
        st.error(f"Error: {str(e)}")
        import traceback; st.error(traceback.format_exc())
        return []


# =============================================================================
# Comprehensive PDF Log Generation
# =============================================================================
def generate_comprehensive_log_pdf():
    try:
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.colors as mcolors
        add_log_entry("Starting PDF log generation", "INFO")
        buffer = BytesIO()
        with PdfPages(buffer) as pdf:
            # PAGE 1: SUMMARY
            fig, ax = plt.subplots(figsize=(11.69, 8.27)); ax.axis('off')
            ax.text(0.5, 0.95, 'Building Classification Processing Log',
                   fontsize=22, fontweight='bold', ha='center', va='top', transform=ax.transAxes)
            ax.text(0.5, 0.89, f'Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                   fontsize=12, ha='center', va='top', transform=ax.transAxes)
            config = st.session_state.get('processing_config', {})
            if config:
                y = 0.80
                ax.text(0.1, y, 'Processing Configuration:', fontsize=14, fontweight='bold', va='top', transform=ax.transAxes); y -= 0.05
                ax.text(0.1, y, f"  Date Range: {config.get('start_date','N/A')} to {config.get('end_date','N/A')}", fontsize=10, va='top', transform=ax.transAxes); y -= 0.04
                ax.text(0.1, y, f"  Max Cloud %: {config.get('cloudy_pct','N/A')}", fontsize=10, va='top', transform=ax.transAxes); y -= 0.04
                ax.text(0.1, y, f"  Patch Nodata %: 0 (fixed)", fontsize=10, va='top', transform=ax.transAxes)
            ma = st.session_state.get('month_analysis_results', {})
            if ma:
                y -= 0.06
                ax.text(0.1, y, 'Monthly Analysis Summary:', fontsize=14, fontweight='bold', va='top', transform=ax.transAxes); y -= 0.05
                sc = {}
                for ms in ma.values(): sc[ms.get('status','?')] = sc.get(ms.get('status','?'), 0) + 1
                ax.text(0.1, y, f"  Total: {len(ma)}", fontsize=10, va='top', transform=ax.transAxes); y -= 0.04
                for s, c_name in [('complete','green'),('skipped','orange'),('rejected','red'),('no_data','gray')]:
                    ax.text(0.1, y, f"  {s.title()}: {sc.get(s,0)}", fontsize=10, va='top', transform=ax.transAxes, color=c_name); y -= 0.04
            cr = st.session_state.get('change_detection_result', {})
            if cr:
                stats = cr.get('stats', {}); y -= 0.06
                ax.text(0.1, y, 'Change Detection Summary:', fontsize=14, fontweight='bold', va='top', transform=ax.transAxes); y -= 0.05
                ax.text(0.1, y, f"  Change pixels: {stats.get('change_pixels',0):,}", fontsize=10, va='top', transform=ax.transAxes); y -= 0.04
                ax.text(0.1, y, f"  Change rate: {stats.get('change_percentage',0):.4f}%", fontsize=10, va='top', transform=ax.transAxes)
            pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

            # MONTHLY DETAIL PAGES
            ma = st.session_state.get('month_analysis_results', {})
            ic = st.session_state.get('monthly_image_counts', {})
            ci = st.session_state.get('monthly_component_images', {})
            for mn in sorted(ma.keys()):
                info = ma[mn]; status = info.get('status','?'); message = info.get('message','')
                sc = {'complete':'green','skipped':'orange','rejected':'red','no_data':'gray'}.get(status,'black')
                comps = ci.get(mn, []); nc = len(comps); ncols = 4; nrows = max(1, (nc+ncols-1)//ncols)
                fig = plt.figure(figsize=(11.69, min(4+nrows*2.5, 8.27)))
                ha = fig.add_axes([0.05, 0.85, 0.9, 0.12]); ha.axis('off')
                ha.text(0.5, 0.9, f'Month: {mn}', fontsize=16, fontweight='bold', ha='center', va='top', transform=ha.transAxes)
                ha.text(0.5, 0.5, f'Status: {status.upper()}', fontsize=14, ha='center', va='top', transform=ha.transAxes, color=sc)
                ha.text(0.5, 0.2, f'Reason: {message}', fontsize=10, ha='center', va='top', transform=ha.transAxes)
                ha.text(0.5, -0.1, f'Component Images: {ic.get(mn,0)}', fontsize=10, ha='center', va='top', transform=ha.transAxes)
                if comps:
                    for idx2, comp in enumerate(comps[:12]):
                        r, c = idx2//ncols, idx2%ncols
                        l = 0.05+c*0.24; b = 0.7-(r+1)*0.22
                        if b < 0.05: break
                        axi = fig.add_axes([l, b, 0.22, 0.18])
                        axi.imshow(comp['rgb']); axi.set_title(comp['date'], fontsize=8); axi.axis('off')
                pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

            # PATCH VALIDITY PAGE
            vm = st.session_state.get('valid_patches_mask', None)
            if vm is not None:
                fig, ax = plt.subplots(figsize=(11.69, 8.27))
                im = ax.imshow(vm, cmap='RdYlGn', vmin=0, vmax=1)
                ax.set_title(f'Patch Validity ({np.sum(vm)} valid / {vm.size} total)', fontsize=14, fontweight='bold')
                cb = plt.colorbar(im, ax=ax, shrink=0.6); cb.set_ticks([0,1]); cb.set_ticklabels(['Invalid','Valid'])
                ax.set_xticks(np.arange(-0.5, vm.shape[1], 1), minor=True)
                ax.set_yticks(np.arange(-0.5, vm.shape[0], 1), minor=True)
                ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
                pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

            # CHANGE DETECTION PAGES
            ct = st.session_state.get('change_timing_map', None)
            cr = st.session_state.get('change_detection_result', {})
            if ct is not None and cr:
                stats = cr.get('stats', {}); sm = stats.get('months', [])
                fig, ax = plt.subplots(figsize=(11.69, 8.27))
                nm = len(sm)
                if nm > 1:
                    cols = plt.cm.tab20(np.linspace(0, 1, nm))
                    cwb = np.vstack([[0,0,0,1], cols]); cmap = mcolors.ListedColormap(cwb)
                    ax.imshow(ct, cmap=cmap, vmin=0, vmax=nm)
                    ax.set_title('Change Detection Timing Map', fontsize=14, fontweight='bold')
                    from matplotlib.patches import Patch
                    ll = ['No Change'] + sm[1:]
                    lc = [cwb[0]] + [cwb[i+1] for i in range(nm-1)]
                    le = [Patch(facecolor=lc[i], label=ll[i]) for i in range(min(len(ll), 15))]
                    ax.legend(handles=le, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8)
                ax.axis('off'); plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
                tbm = stats.get('transition_by_month', {})
                for mi, month in enumerate(sm[1:], start=1):
                    cnt = tbm.get(month, 0)
                    if cnt == 0: continue
                    bm = (ct == mi).astype(np.uint8) * 255
                    fig, ax = plt.subplots(figsize=(11.69, 8.27))
                    ax.imshow(bm, cmap='gray', vmin=0, vmax=255)
                    ax.set_title(f'Changes in {month} ({cnt:,} pixels)', fontsize=14, fontweight='bold')
                    ax.axis('off')
                    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

            # PROCESSING LOG PAGES
            pl = st.session_state.get('processing_log', [])
            if pl:
                epp = 40; npages = (len(pl)+epp-1)//epp
                for pn in range(npages):
                    si, ei = pn*epp, min((pn+1)*epp, len(pl))
                    fig, ax = plt.subplots(figsize=(11.69, 8.27)); ax.axis('off')
                    ax.text(0.5, 0.98, f'Processing Log ({pn+1}/{npages})', fontsize=14, fontweight='bold',
                           ha='center', va='top', transform=ax.transAxes)
                    y = 0.93
                    for entry in pl[si:ei]:
                        ts, lv, msg = entry.get('timestamp',''), entry.get('level','INFO'), entry.get('message','')
                        co = {'INFO':'black','WARNING':'orange','ERROR':'red'}.get(lv, 'black')
                        if len(msg) > 80: msg = msg[:77] + '...'
                        ax.text(0.02, y, f'[{ts}] [{lv}] {msg}', fontsize=7, va='top',
                               transform=ax.transAxes, color=co, family='monospace')
                        y -= 0.022
                        if y < 0.02: break
                    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
        buffer.seek(0)
        add_log_entry("PDF log generation complete", "INFO")
        return buffer.getvalue()
    except Exception as e:
        add_log_entry(f"Error generating PDF: {e}", "ERROR")
        return None


# =============================================================================
# Change Detection Algorithm
# =============================================================================
def analyze_building_transition(probability_maps, non_building_thr=0.2, building_thr=0.8,
                                non_building_duration=2, building_duration=2):
    add_log_entry(f"Starting change detection analysis", "INFO")
    sorted_months = sorted(probability_maps.keys())
    if len(sorted_months) < (non_building_duration + building_duration):
        add_log_entry(f"Not enough months", "ERROR")
        return None, None, {"error": f"Need {non_building_duration + building_duration} months, got {len(sorted_months)}"}
    add_log_entry(f"Analyzing {len(sorted_months)} months: {sorted_months[0]} to {sorted_months[-1]}", "INFO")
    first_map = probability_maps[sorted_months[0]]
    height, width = first_map.shape
    n_times = len(sorted_months)
    data = np.zeros((n_times, height, width), dtype=np.float32)
    for i, month in enumerate(sorted_months):
        data[i] = probability_maps[month]
    results = np.zeros((height, width), dtype=np.uint8)
    transition_timing = np.zeros((height, width), dtype=np.int16)
    for y in range(height):
        for x in range(width):
            ps = data[:, y, x]
            if np.all(ps == 0): continue
            if ps[0] > non_building_thr: continue
            if ps[-1] < building_thr: continue
            tp = None; invalid = False
            for t in range(1, len(ps)):
                if ps[t-1] > non_building_thr and ps[t] <= non_building_thr:
                    invalid = True; break
                elif ps[t-1] >= building_thr and ps[t] < building_thr:
                    invalid = True; break
                if ps[t-1] <= non_building_thr and ps[t] >= building_thr and tp is None:
                    tp = t
            if tp is not None and not invalid:
                nb_dur = tp; b_dur = len(ps) - tp
                if nb_dur >= non_building_duration and b_dur >= building_duration:
                    results[y, x] = 1; transition_timing[y, x] = tp
    change_pixels = np.sum(results == 1)
    total_pixels = np.sum(data[0] > 0)
    change_pct = (change_pixels / total_pixels * 100) if total_pixels > 0 else 0
    tbm = {}
    for i, month in enumerate(sorted_months):
        if i > 0: tbm[month] = int(np.sum(transition_timing == i))
    add_log_entry(f"Change detection: {change_pixels} pixels ({change_pct:.2f}%)", "INFO")
    for month, cnt in tbm.items():
        if cnt > 0: add_log_entry(f"  {month}: {cnt} transitions", "INFO")
    stats = {'change_pixels': int(change_pixels), 'total_pixels': int(total_pixels),
             'change_percentage': float(change_pct), 'n_months': n_times, 'months': sorted_months,
             'first_month': sorted_months[0], 'last_month': sorted_months[-1],
             'transition_by_month': tbm}
    st.session_state.change_timing_map = transition_timing
    return results, transition_timing, stats


def generate_rgb_from_sentinel(image_path, max_size=None):
    try:
        with rasterio.open(image_path) as src:
            red, green, blue = src.read(4), src.read(3), src.read(2)
            rgb = np.nan_to_num(np.stack([red, green, blue], axis=-1), nan=0.0)
            def ps(band, lo=2, hi=98):
                v = band[band > 0]
                if len(v) == 0: return np.zeros_like(band, dtype=np.uint8)
                pl, ph = np.percentile(v, lo), np.percentile(v, hi)
                if ph <= pl: ph = pl + 0.001
                return (np.clip((band - pl) / (ph - pl), 0, 1) * 255).astype(np.uint8)
            rgb8 = np.zeros_like(rgb, dtype=np.uint8)
            for i in range(3): rgb8[:, :, i] = ps(rgb[:, :, i])
            return rgb8
    except:
        return None


# =============================================================================
# Display Thumbnails
# =============================================================================
def get_image_download_data(image_path, month_name):
    try:
        with open(image_path, 'rb') as f: return f.read()
    except: return None


def display_thumbnails(thumbnails, valid_months=None):
    if not thumbnails: return
    # Log PDF button
    if valid_months:
        st.subheader("📋 Processing Log")
        c1, c2 = st.columns([1, 3])
        with c1:
            if st.button("📄 Generate Log PDF", type="primary"):
                with st.spinner("Generating..."):
                    pb = generate_comprehensive_log_pdf()
                    if pb: st.session_state.log_pdf_bytes = pb; st.success("✅ Done!")
        with c2:
            if st.session_state.get('log_pdf_bytes'):
                st.download_button("⬇️ Download Log PDF", data=st.session_state.log_pdf_bytes,
                    file_name=f"log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", mime="application/pdf")
        st.divider()
    mode = st.radio("Display:", ["Side by Side", "Classification", "RGB"], horizontal=True)
    st.divider()
    if mode == "Side by Side":
        for i in range(0, len(thumbnails), 2):
            cols = st.columns(4)
            for j in range(2):
                idx = i + j
                if idx < len(thumbnails):
                    t = thumbnails[idx]; pct = (t['building_pixels']/t['total_pixels'])*100
                    sfx = " (filled)" if t.get('gap_filled') else ""
                    cols[j*2].image(t['rgb_image'], caption=f"{t['month_name']} RGB{sfx}")
                    if valid_months and t['month_name'] in valid_months:
                        d = get_image_download_data(valid_months[t['month_name']], t['month_name'])
                        if d: cols[j*2].download_button(f"⬇️ {t['month_name']}", data=d,
                            file_name=f"sentinel2_{t['month_name']}_12bands.tif", mime="image/tiff",
                            key=f"dl_sbs_{t['month_name']}")
                    cols[j*2+1].image(t['classification_image'], caption=f"{t['month_name']} ({pct:.1f}%)")
    else:
        key = 'classification_image' if mode == "Classification" else 'rgb_image'
        for row in range((len(thumbnails)+3)//4):
            cols = st.columns(4)
            for c in range(4):
                idx = row*4+c
                if idx < len(thumbnails):
                    t = thumbnails[idx]; pct = (t['building_pixels']/t['total_pixels'])*100
                    cap = f"{t['month_name']} ({pct:.1f}%)" if mode == "Classification" else t['month_name']
                    if t.get(key): cols[c].image(t[key], caption=cap)
                    if mode == "RGB" and valid_months and t['month_name'] in valid_months:
                        d = get_image_download_data(valid_months[t['month_name']], t['month_name'])
                        if d: cols[c].download_button("⬇️", data=d,
                            file_name=f"sentinel2_{t['month_name']}_12bands.tif", mime="image/tiff",
                            key=f"dl_rgb_{t['month_name']}")



# =============================================================================
# Main Application
# =============================================================================
def main():
    tab1, tab2 = st.tabs(["🏗️ Classification", "🔍 Change Detection"])
    with tab1: main_classification_tab()
    with tab2: change_detection_tab()


def main_classification_tab():
    st.title("🏗️ Building Classification v07 (Fixed)")
    st.markdown("""
    **v07 Fix**: `frequency.unmask(0).clip(aoi)` ensures incomplete Sentinel-2 tile coverage
    is detected, preventing interior NaN patches in downloaded images.

    | Status | Condition | Download? |
    |--------|-----------|-----------|
    | `no_data` | No images | ❌ |
    | `skipped` | masked > 30% | ❌ |
    | `complete` | masked == 0% | ✅ |
    | `rejected` | masked > 0% after gap-fill | ❌ |
    """)
    ee_ok, ee_msg = initialize_earth_engine()
    if not ee_ok: st.error(ee_msg); st.stop()
    st.sidebar.success(ee_msg)

    st.sidebar.header("🧠 Model")
    model_path = "best_model_version_Unet++_v02_e7.pt"
    if not os.path.exists(model_path):
        if not download_model_from_gdrive("", model_path): st.stop()
    if not st.session_state.model_loaded:
        model, device = load_model(model_path)
        if model:
            st.session_state.model = model; st.session_state.device = device
            st.session_state.model_loaded = True; st.sidebar.success("✅ Model loaded")
        else: st.stop()
    else: st.sidebar.success("✅ Model loaded")

    st.sidebar.header("⚙️ Parameters")
    cloudy_pct = st.sidebar.slider("Max Cloud % (metadata)", 0, 50, 10, 5,
        help="GEE: Filter by CLOUDY_PIXEL_PERCENTAGE metadata",
        disabled=st.session_state.processing_in_progress)
    nodata_pct = 0

    st.sidebar.header("🗂️ Cache Status")
    ci = []
    if st.session_state.month_analysis_results: ci.append(f"📊 {len(st.session_state.month_analysis_results)} analyzed")
    if st.session_state.downloaded_images: ci.append(f"📥 {len(st.session_state.downloaded_images)} downloaded")
    if st.session_state.processed_months: ci.append(f"🧠 {len(st.session_state.processed_months)} classified")
    for info in ci: st.sidebar.success(info)
    if not ci: st.sidebar.info("No cached data")
    if st.session_state.failed_downloads:
        st.sidebar.warning(f"❌ Failed: {', '.join(st.session_state.failed_downloads)}")
    if st.session_state.processing_in_progress:
        st.sidebar.error("⏳ Processing...")

    if st.sidebar.button("🗑️ Clear All Cache", disabled=st.session_state.processing_in_progress):
        for key in ['processed_months','downloaded_images','classification_thumbnails',
                    'valid_patches_mask','valid_months','current_temp_dir','month_analysis_results',
                    'failed_downloads','analysis_complete','download_complete',
                    'processing_params','processing_config','pdf_report',
                    'probability_maps','change_detection_result',
                    'processing_log','monthly_component_images','monthly_image_counts',
                    'change_timing_map','log_pdf_bytes']:
            if key in st.session_state:
                if isinstance(st.session_state[key], dict): st.session_state[key] = {}
                elif isinstance(st.session_state[key], list): st.session_state[key] = []
                else: st.session_state[key] = None
        st.session_state.processing_complete = False
        st.session_state.processing_in_progress = False
        clear_log(); st.rerun()

    if st.session_state.processing_in_progress:
        if st.sidebar.button("🛑 Stop Processing", type="primary"):
            add_log_entry("Stopped by user", "WARNING")
            st.session_state.processing_in_progress = False
            st.session_state.processing_config = None
            st.warning("⚠️ Stopped. Resume later."); st.rerun()

    # Region Selection
    st.header("1️⃣ Region")
    if not st.session_state.processing_in_progress:
        m = folium.Map(location=[35.6892, 51.3890], zoom_start=8)
        plugins.Draw(export=True, position='topleft', draw_options={
            'polyline': False, 'rectangle': True, 'polygon': True,
            'circle': False, 'marker': False, 'circlemarker': False}).add_to(m)
        folium.TileLayer(tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
                         attr='Google', name='Satellite').add_to(m)
        folium.LayerControl().add_to(m)
        mk = f"region_map_{len(st.session_state.drawn_polygons)}_{st.session_state.processing_complete}"
        map_data = st_folium(m, width=800, height=500, key=mk)
        if map_data and map_data.get('last_active_drawing'):
            geom = map_data['last_active_drawing'].get('geometry', {})
            if geom.get('type') == 'Polygon':
                dp = Polygon(geom['coordinates'][0])
                st.session_state.last_drawn_polygon = dp
                centroid = dp.centroid
                lat_c = centroid.y
                km_lat = 111.32; km_lon = 111.32 * math.cos(math.radians(lat_c))
                coords = geom['coordinates'][0]; n = len(coords) - 1
                area = 0.0
                for i in range(n):
                    x1 = coords[i][0]*km_lon; y1 = coords[i][1]*km_lat
                    x2 = coords[(i+1)%n][0]*km_lon; y2 = coords[(i+1)%n][1]*km_lat
                    area += x1*y2 - x2*y1
                area = abs(area) / 2.0
                st.success(f"✅ Region — **{area:.2f} km²**")
        if st.button("💾 Save Region"):
            if st.session_state.last_drawn_polygon:
                dup = any(e.equals(st.session_state.last_drawn_polygon) for e in st.session_state.drawn_polygons)
                if not dup:
                    st.session_state.drawn_polygons.append(st.session_state.last_drawn_polygon)
                    st.success("✅ Saved!"); st.rerun()
                else: st.warning("⚠️ Already saved")
            else: st.warning("⚠️ Draw first")
    else: st.info("🔒 Map locked during processing")

    if st.session_state.drawn_polygons:
        st.subheader("📍 Saved Regions")
        for i, p in enumerate(st.session_state.drawn_polygons):
            c1, c2, c3 = st.columns([3, 1, 1])
            ct = p.centroid
            c1.write(f"**Region {i+1}**: ~{p.area * 111 * 111:.2f} km² | Center: ({ct.y:.4f}, {ct.x:.4f})")
            c2.write(f"UTM Zone {get_utm_zone(ct.x)}")
            if c3.button("🗑️", key=f"del_{i}", disabled=st.session_state.processing_in_progress):
                st.session_state.drawn_polygons.pop(i)
                if st.session_state.selected_region_index >= len(st.session_state.drawn_polygons):
                    st.session_state.selected_region_index = max(0, len(st.session_state.drawn_polygons) - 1)
                st.rerun()

    # Date
    st.header("2️⃣ Time Period")
    c1, c2 = st.columns(2)
    start = c1.date_input("Start (inclusive)", value=date(2024, 1, 1), disabled=st.session_state.processing_in_progress)
    end = c2.date_input("End (exclusive)", value=date(2025, 1, 1), disabled=st.session_state.processing_in_progress,
                        help="This month is NOT included")
    if start >= end: st.error("Invalid dates"); st.stop()
    months = (end.year - start.year) * 12 + (end.month - start.month)
    first_month = f"{start.year}-{start.month:02d}"
    last_year = start.year + (start.month - 1 + months - 1) // 12
    last_month_num = (start.month - 1 + months - 1) % 12 + 1
    last_month = f"{last_year}-{last_month_num:02d}"
    st.info(f"📅 **{months} months**: {first_month} → {last_month} (end date {end.strftime('%Y-%m-%d')} excluded)")

    # Process
    st.header("3️⃣ Process")
    selected_polygon = None
    if st.session_state.drawn_polygons:
        region_options = [f"Region {i+1} (~{p.area*111*111:.2f} km²)" for i, p in enumerate(st.session_state.drawn_polygons)]
        if st.session_state.selected_region_index >= len(st.session_state.drawn_polygons):
            st.session_state.selected_region_index = 0
        selected_idx = st.selectbox("🎯 Select Region", range(len(region_options)),
            format_func=lambda i: region_options[i], index=st.session_state.selected_region_index,
            disabled=st.session_state.processing_in_progress, key="region_selector")
        st.session_state.selected_region_index = selected_idx
        selected_polygon = st.session_state.drawn_polygons[selected_idx]
        ct = selected_polygon.centroid
        st.success(f"✅ Region {selected_idx+1} | ~{selected_polygon.area*111*111:.2f} km² | UTM {get_utm_zone(ct.x)}")
    elif st.session_state.last_drawn_polygon is not None:
        selected_polygon = st.session_state.last_drawn_polygon
        st.info("ℹ️ Using unsaved drawn region")
    else:
        st.warning("⚠️ Draw and save a region first")

    st.divider()
    col1, col2, col3 = st.columns(3)
    start_new = col1.button("🚀 Start New", type="primary",
        disabled=st.session_state.processing_in_progress or selected_polygon is None)
    has_cache = bool(st.session_state.downloaded_images or st.session_state.month_analysis_results)
    resume_btn = col2.button("🔄 Resume", disabled=not has_cache or st.session_state.processing_in_progress)
    has_failed = bool(st.session_state.failed_downloads)
    retry_btn = col3.button("🔁 Retry Failed", disabled=not has_failed or st.session_state.processing_in_progress)

    should_process = False; resume_mode = False
    if start_new:
        should_process = True; resume_mode = False
        for k in ['month_analysis_results','processed_months','classification_thumbnails',
                   'valid_patches_mask','valid_months','pdf_report','failed_downloads',
                   'probability_maps','change_detection_result']:
            if isinstance(st.session_state.get(k), dict): st.session_state[k] = {}
            elif isinstance(st.session_state.get(k), list): st.session_state[k] = []
            else: st.session_state[k] = None
        st.session_state.analysis_complete = False; st.session_state.download_complete = False
        st.session_state.processing_complete = False
        st.session_state.processing_config = {
            'polygon_coords': list(selected_polygon.exterior.coords),
            'start_date': start.strftime('%Y-%m-%d'), 'end_date': end.strftime('%Y-%m-%d'),
            'cloudy_pct': cloudy_pct, 'nodata_pct': nodata_pct}
        st.session_state.processing_in_progress = True
    elif resume_btn:
        should_process = True; resume_mode = True
        st.session_state.processing_in_progress = True
        if st.session_state.processing_config is None:
            st.session_state.processing_config = {
                'polygon_coords': list(selected_polygon.exterior.coords),
                'start_date': start.strftime('%Y-%m-%d'), 'end_date': end.strftime('%Y-%m-%d'),
                'cloudy_pct': cloudy_pct, 'nodata_pct': nodata_pct}
    elif retry_btn:
        should_process = True; resume_mode = True
        st.session_state.failed_downloads = []
        st.session_state.processing_in_progress = True
        if st.session_state.processing_config is None:
            st.session_state.processing_config = {
                'polygon_coords': list(selected_polygon.exterior.coords),
                'start_date': start.strftime('%Y-%m-%d'), 'end_date': end.strftime('%Y-%m-%d'),
                'cloudy_pct': cloudy_pct, 'nodata_pct': nodata_pct}
    elif st.session_state.processing_in_progress and st.session_state.processing_config is not None:
        should_process = True; resume_mode = True
        st.info("🔄 Auto-continuing...")

    if should_process:
        config = st.session_state.processing_config
        if config is None:
            st.error("❌ No config!"); st.session_state.processing_in_progress = False; st.stop()
        aoi = ee.Geometry.Polygon([config['polygon_coords']])
        try:
            thumbs = process_timeseries(aoi, config['start_date'], config['end_date'],
                st.session_state.model, st.session_state.device,
                config['cloudy_pct'], 10, config['nodata_pct'], resume=resume_mode)
            if thumbs:
                st.session_state.classification_thumbnails = thumbs
                st.session_state.processing_complete = True
            st.session_state.processing_in_progress = False
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback; st.code(traceback.format_exc())
            st.session_state.processing_in_progress = False

    if st.session_state.processing_complete and st.session_state.classification_thumbnails:
        st.divider(); st.header("📊 Results")
        display_thumbnails(st.session_state.classification_thumbnails, valid_months=st.session_state.valid_months)
        if st.session_state.probability_maps:
            st.success(f"✅ {len(st.session_state.probability_maps)} probability maps ready. Go to **Change Detection** tab.")


def change_detection_tab():
    st.title("🔍 Change Detection Analysis")
    if not st.session_state.probability_maps:
        st.warning("⚠️ No probability maps. Complete classification first.")
        return
    n_months = len(st.session_state.probability_maps)
    sorted_months = sorted(st.session_state.probability_maps.keys())
    st.success(f"✅ **{n_months} months**: {sorted_months[0]} → {sorted_months[-1]}")

    st.header("⚙️ Algorithm Parameters")
    with st.expander("ℹ️ Algorithm Explanation", expanded=False):
        st.markdown("""
        A pixel is a "change" (non-building → building) if:
        1. Starts as non-building (prob < threshold)
        2. Ends as building (prob > threshold)
        3. Only ONE transition (no back-and-forth)
        4. Stays non-building for min duration before
        5. Stays building for min duration after
        """)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📊 Thresholds")
        nb_thr = st.text_input("Non-building threshold", value="0.2", key="cd_nb")
        b_thr = st.text_input("Building threshold", value="0.8", key="cd_b")
    with c2:
        st.subheader("⏱️ Duration")
        min_nb = st.text_input("Min non-building (months)", value="2", key="cd_mnb")
        min_b = st.text_input("Min building (months)", value="2", key="cd_mb")

    try:
        nb_v = float(nb_thr); b_v = float(b_thr); mnb_v = int(min_nb); mb_v = int(min_b)
        errors = []
        if not (0.0 <= nb_v <= 0.5): errors.append("Non-building threshold: 0.0-0.5")
        if not (0.5 <= b_v <= 1.0): errors.append("Building threshold: 0.5-1.0")
        if mnb_v < 1: errors.append("Min NB >= 1")
        if mb_v < 1: errors.append("Min B >= 1")
        if mnb_v + mb_v > n_months: errors.append(f"Total duration > months ({n_months})")
        for e in errors: st.error(f"❌ {e}")
        params_valid = len(errors) == 0
    except ValueError:
        st.error("❌ Invalid values"); params_valid = False

    st.divider()
    if params_valid:
        if st.button("🔍 Run Change Detection", type="primary", use_container_width=True):
            with st.spinner("Analyzing..."):
                cm, tm, stats = analyze_building_transition(
                    st.session_state.probability_maps, nb_v, b_v, mnb_v, mb_v)
                if cm is not None:
                    st.session_state.change_detection_result = {
                        'mask': cm, 'timing_map': tm, 'stats': stats,
                        'params': {'non_building_thr': nb_v, 'building_thr': b_v,
                                   'min_non_building': mnb_v, 'min_building': mb_v}}
                    st.success("✅ Done!"); st.rerun()
                else:
                    st.error(f"❌ Failed: {stats.get('error','?') if stats else '?'}")

    if st.session_state.change_detection_result:
        result = st.session_state.change_detection_result
        cm = result['mask']; stats = result['stats']; params = result['params']
        st.divider(); st.header("📈 Results")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Change Pixels", f"{stats['change_pixels']:,}")
        c2.metric("Total Pixels", f"{stats['total_pixels']:,}")
        c3.metric("Change Rate", f"{stats['change_percentage']:.2f}%")
        c4.metric("Months", stats['n_months'])
        st.info(f"📅 {stats['first_month']} → {stats['last_month']}")

        tbm = stats.get('transition_by_month', {})
        if tbm:
            with st.expander("📊 Transitions by Month"):
                for mo, cnt in tbm.items():
                    if cnt > 0: st.write(f"  • **{mo}**: {cnt:,}")

        # Log PDF
        st.subheader("📋 Log")
        lc1, lc2 = st.columns([1, 3])
        with lc1:
            if st.button("📄 Generate Log PDF", type="primary", key="cd_log"):
                with st.spinner("Generating..."):
                    pb = generate_comprehensive_log_pdf()
                    if pb: st.session_state.log_pdf_bytes = pb; st.success("✅")
        with lc2:
            if st.session_state.get('log_pdf_bytes'):
                st.download_button("⬇️ Download Log", data=st.session_state.log_pdf_bytes,
                    file_name=f"log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf", key="cd_dl")

        # Visualization
        st.subheader("🖼️ Visualization")
        fm = stats['first_month']; lm = stats['last_month']
        fip = st.session_state.valid_months.get(fm)
        lip = st.session_state.valid_months.get(lm)
        vm = st.session_state.valid_patches_mask
        orig_size = cm.shape
        cb = get_valid_patch_bounds(vm, PATCH_SIZE, orig_size)
        frgb = generate_rgb_from_sentinel(fip) if fip else None
        lrgb = generate_rgb_from_sentinel(lip) if lip else None
        fp = st.session_state.probability_maps.get(fm)
        lp = st.session_state.probability_maps.get(lm)
        fc = (fp > 0.5).astype(np.uint8) * 255 if fp is not None else None
        lc_mask = (lp > 0.5).astype(np.uint8) * 255 if lp is not None else None

        if frgb is not None and lrgb is not None and cb is not None:
            r0, r1, c0, c1_val = cb
            frgb_c = frgb[r0:r1, c0:c1_val, :]
            lrgb_c = lrgb[r0:r1, c0:c1_val, :]
            cm_c = cm[r0:r1, c0:c1_val]
            fc_c = fc[r0:r1, c0:c1_val] if fc is not None else None
            lc_c = lc_mask[r0:r1, c0:c1_val] if lc_mask is not None else None

            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes[0,0].imshow(frgb_c); axes[0,0].set_title(f"First RGB: {fm}"); axes[0,0].axis('off')
            axes[0,1].imshow(lrgb_c); axes[0,1].set_title(f"Last RGB: {lm}"); axes[0,1].axis('off')
            chcol = np.zeros((*cm_c.shape, 3), dtype=np.uint8); chcol[cm_c==1] = [255,0,0]
            axes[0,2].imshow(lrgb_c); axes[0,2].imshow(chcol, alpha=0.6)
            axes[0,2].set_title(f"Change ({stats['change_pixels']:,} px)"); axes[0,2].axis('off')
            if fc_c is not None:
                axes[1,0].imshow(fc_c, cmap='Greens'); axes[1,0].set_title(f"First Class: {fm}"); axes[1,0].axis('off')
            if lc_c is not None:
                axes[1,1].imshow(lc_c, cmap='Reds'); axes[1,1].set_title(f"Last Class: {lm}"); axes[1,1].axis('off')
            axes[1,2].imshow(cm_c, cmap='hot'); axes[1,2].set_title(f"Change Mask ({stats['change_percentage']:.2f}%)"); axes[1,2].axis('off')
            plt.tight_layout(); st.pyplot(fig); plt.close()

            # Downloads
            st.subheader("⬇️ Downloads")
            dc1, dc2, dc3 = st.columns(3)
            with dc1:
                if fip and os.path.exists(fip):
                    try:
                        with rasterio.open(fip) as src:
                            om = src.meta.copy()
                            om.update({'count': 1, 'dtype': 'uint8', 'height': cm.shape[0], 'width': cm.shape[1]})
                            buf = BytesIO()
                            with rasterio.open(buf, 'w', **om) as dst: dst.write(cm.astype(np.uint8), 1)
                            buf.seek(0)
                            st.download_button("📥 Change Mask (GeoTIFF)", data=buf.getvalue(),
                                file_name=f"change_{fm}_to_{lm}.tif", mime="image/tiff")
                    except Exception as e:
                        ci = Image.fromarray((cm_c*255).astype(np.uint8))
                        buf = BytesIO(); ci.save(buf, format='PNG'); buf.seek(0)
                        st.download_button("📥 Change Mask (PNG)", data=buf.getvalue(),
                            file_name=f"change_{fm}_to_{lm}.png", mime="image/png")
            with dc2:
                ci = Image.fromarray((cm_c*255).astype(np.uint8))
                buf = BytesIO(); ci.save(buf, format='PNG'); buf.seek(0)
                st.download_button("📥 Preview (PNG)", data=buf.getvalue(),
                    file_name=f"preview_{fm}_to_{lm}.png", mime="image/png")
            with dc3:
                report = f"""Change Detection Report
================================
Range: {fm} to {lm} ({stats['n_months']} months)
Params: NB<{params['non_building_thr']}, B>{params['building_thr']}, minNB={params['min_non_building']}, minB={params['min_building']}
Results: {stats['change_pixels']:,} / {stats['total_pixels']:,} pixels = {stats['change_percentage']:.4f}%
"""
                st.download_button("📄 Report (TXT)", data=report,
                    file_name=f"report_{fm}_to_{lm}.txt", mime="text/plain")

            # Interactive Map
            st.divider(); st.subheader("🗺️ Interactive Map")
            display_interactive_map(fip, lip, fc, lc_mask, cm, fm, lm, cb, orig_size)
        else:
            st.error("❌ Cannot generate visualization")


def display_interactive_map(first_image_path, last_image_path, first_class, last_class,
                            change_mask, first_month, last_month, crop_bounds, original_size):
    import streamlit.components.v1 as components
    from rasterio.warp import calculate_default_transform, reproject, Resampling

    first_year = first_month.split('-')[0] if first_month else "N/A"
    last_year = last_month.split('-')[0] if last_month else "N/A"

    st.info("**Controls**: Layer control (top-right), fullscreen (top-left). Base layers need internet; overlays are embedded.")

    try:
        if not first_image_path or not os.path.exists(first_image_path):
            st.warning("Cannot create map: source not found"); return

        with rasterio.open(first_image_path) as src:
            utm_crs = src.crs; utm_transform = src.transform; utm_bounds = src.bounds
        cx = (utm_bounds.left + utm_bounds.right) / 2; cy = (utm_bounds.bottom + utm_bounds.top) / 2
        from rasterio.warp import transform as rio_transform
        clon, clat = rio_transform(utm_crs, 'EPSG:4326', [cx], [cy])
        center = [clat[0], clon[0]]

        temp_dir = tempfile.mkdtemp(); dst_crs = 'EPSG:4326'
        target_transform = target_width = target_height = target_bounds = None

        def reproject_single(data, name, dtype='uint8'):
            nonlocal target_transform, target_width, target_height, target_bounds
            up = os.path.join(temp_dir, f"{name}_utm.tif"); wp = os.path.join(temp_dir, f"{name}_wgs.tif")
            with rasterio.open(up, 'w', driver='GTiff', height=data.shape[0], width=data.shape[1],
                              count=1, dtype=dtype, crs=utm_crs, transform=utm_transform) as d:
                d.write(data.astype(dtype), 1)
            with rasterio.open(up) as s:
                if target_transform is None:
                    target_transform, target_width, target_height = calculate_default_transform(
                        s.crs, dst_crs, s.width, s.height, *s.bounds)
                kw = s.meta.copy(); kw.update({'crs': dst_crs, 'transform': target_transform,
                    'width': target_width, 'height': target_height})
                with rasterio.open(wp, 'w', **kw) as d:
                    reproject(source=rasterio.band(s, 1), destination=rasterio.band(d, 1),
                             src_transform=s.transform, src_crs=s.crs,
                             dst_transform=target_transform, dst_crs=dst_crs, resampling=Resampling.nearest)
                    if target_bounds is None: target_bounds = d.bounds
            return wp

        def reproject_rgb(rgb_data, name):
            nonlocal target_transform, target_width, target_height, target_bounds
            up = os.path.join(temp_dir, f"{name}_utm.tif"); wp = os.path.join(temp_dir, f"{name}_wgs.tif")
            with rasterio.open(up, 'w', driver='GTiff', height=rgb_data.shape[0], width=rgb_data.shape[1],
                              count=3, dtype='uint8', crs=utm_crs, transform=utm_transform) as d:
                for i in range(3): d.write(rgb_data[:,:,i], i+1)
            with rasterio.open(up) as s:
                if target_transform is None:
                    target_transform, target_width, target_height = calculate_default_transform(
                        s.crs, dst_crs, s.width, s.height, *s.bounds)
                kw = s.meta.copy(); kw.update({'crs': dst_crs, 'transform': target_transform,
                    'width': target_width, 'height': target_height})
                with rasterio.open(wp, 'w', **kw) as d:
                    for i in range(1, 4):
                        reproject(source=rasterio.band(s, i), destination=rasterio.band(d, i),
                                 src_transform=s.transform, src_crs=s.crs,
                                 dst_transform=target_transform, dst_crs=dst_crs, resampling=Resampling.bilinear)
                    if target_bounds is None: target_bounds = d.bounds
            return wp

        def to_overlay(rpath, is_binary=False, is_change=False, is_rgb=False, cmap_name='Greens'):
            with rasterio.open(rpath) as s:
                b = s.bounds; bl = [[b.bottom, b.left], [b.top, b.right]]
                if is_rgb and s.count >= 3:
                    d = np.transpose(s.read([1,2,3]), (1,2,0))
                    pi = Image.fromarray(d.astype(np.uint8))
                elif is_binary:
                    d = s.read(1); rgba = np.zeros((d.shape[0], d.shape[1], 4), dtype=np.uint8)
                    mv = d > 0
                    if cmap_name == 'Greens': rgba[mv, 0:3] = [0,255,0]
                    elif cmap_name == 'Reds': rgba[mv, 0:3] = [255,0,0]
                    rgba[mv, 3] = 180; pi = Image.fromarray(rgba, 'RGBA')
                elif is_change:
                    d = s.read(1); rgba = np.zeros((d.shape[0], d.shape[1], 4), dtype=np.uint8)
                    mv = d > 0; rgba[mv, 0:3] = [0,0,180]; rgba[mv, 3] = 220
                    pi = Image.fromarray(rgba, 'RGBA')
                else:
                    d = s.read(1)
                    dn = d.max() - d.min()
                    if dn > 0: d = (d - d.min()) / dn
                    else: d = np.zeros_like(d)
                    import matplotlib.cm as cm
                    arr = (cm.get_cmap('viridis')(d)[:,:,:3] * 255).astype(np.uint8)
                    pi = Image.fromarray(arr)
                buf = BytesIO(); pi.save(buf, format='PNG')
                return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}", bl

        # Reproject layers
        bc_path = ac_path = chm_path = br_path = ar_path = None
        if first_class is not None:
            bc_path = reproject_single((first_class > 0).astype(np.uint8), f"bc_{first_month}")
        if last_class is not None:
            ac_path = reproject_single((last_class > 0).astype(np.uint8), f"ac_{last_month}")
        if change_mask is not None:
            chm_path = reproject_single(change_mask.astype(np.uint8), "chm")
        frgb = generate_rgb_from_sentinel(first_image_path)
        lrgb = generate_rgb_from_sentinel(last_image_path) if last_image_path else None
        if frgb is not None: br_path = reproject_rgb(frgb, f"br_{first_month}")
        if lrgb is not None: ar_path = reproject_rgb(lrgb, f"ar_{last_month}")

        # Zoom level
        if target_bounds:
            ld = max(target_bounds.top - target_bounds.bottom, target_bounds.right - target_bounds.left)
            zl = 8 if ld > 1 else 10 if ld > 0.5 else 12 if ld > 0.1 else 14 if ld > 0.05 else 15
        else: zl = 15

        mp = folium.Map(location=center, zoom_start=zl, tiles=None, prefer_canvas=True)
        plugins.Fullscreen(position='topleft', force_separate_button=True).add_to(mp)

        folium.TileLayer(tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
            attr='Google', name=f'Satellite ({last_year})', overlay=False).add_to(mp)
        folium.TileLayer(tiles='https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
            attr='Google', name='Maps', overlay=False).add_to(mp)
        folium.TileLayer(tiles='OpenStreetMap', name='OSM', overlay=False).add_to(mp)

        for path, name, kwargs in [
            (br_path, f"First RGB ({first_month})", {'is_rgb': True}),
            (ar_path, f"Last RGB ({last_month})", {'is_rgb': True}),
            (bc_path, f"First Class ({first_month}) Green", {'is_binary': True, 'cmap_name': 'Greens'}),
            (ac_path, f"Last Class ({last_month}) Red", {'is_binary': True, 'cmap_name': 'Reds'}),
            (chm_path, f"Change ({first_month}→{last_month}) Blue", {'is_change': True}),
        ]:
            if path:
                try:
                    img_data, bounds = to_overlay(path, **kwargs)
                    show = 'Class' in name or 'Change' in name
                    folium.raster_layers.ImageOverlay(image=img_data, bounds=bounds,
                        opacity=0.7 if 'Class' in name else 0.8,
                        name=name, overlay=True, control=True, show=show).add_to(mp)
                except Exception as e:
                    st.warning(f"Could not add {name}: {e}")

        if target_bounds:
            mp.fit_bounds([[target_bounds.bottom, target_bounds.left], [target_bounds.top, target_bounds.right]])
        folium.LayerControl(position='topright', collapsed=False).add_to(mp)
        components.html(mp.get_root().render(), height=600)

        st.caption(f"""
        🟢 Green: First classification ({first_month}) | 🔴 Red: Last classification ({last_month}) | 🔵 Blue: Change detection
        """)
    except Exception as e:
        st.error(f"Map error: {e}")
        import traceback; st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
