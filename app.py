"""
Sentinel-2 Time Series Building Classification
VERSION 06.1 - GEE Cloud Masking + Gap-Filling + Python Patch Validation
             + POST-DOWNLOAD NaN/Nodata Completeness Validation

CHANGES FROM v06:
═══════════════════════════════════════════════════════════════════════════════
NEW: validate_geotiff_completeness() - pixel-level validation after download
  - Checks EVERY band for NaN pixels
  - Checks EVERY band for nodata-tagged pixels
  - Checks EVERY band for all-zero (empty band)
  - Checks for large contiguous zero regions (partial orbit coverage)
  - If ANY band fails → entire image is REJECTED and deleted

MODIFIED: download_composite()
  - After merging bands, calls validate_geotiff_completeness()
  - If validation fails → deletes merged file, returns None

MODIFIED: download_monthly_image_v06()
  - Cache check now also validates completeness
  - Incomplete cached files are deleted and month is REJECTED
  - Status messages indicate validation state

MODIFIED: process_timeseries() resume logic
  - Re-validates cached files on resume
  - Removes incomplete cached files from session state
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
    page_title="Building Classification Time Series v06.1",
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
_session_defaults = {
    'drawn_polygons': [],
    'last_drawn_polygon': None,
    'ee_initialized': False,
    'model_loaded': False,
    'model': None,
    'device': None,
    'classification_thumbnails': [],
    'processing_complete': False,
    'processed_months': {},
    'current_temp_dir': None,
    'downloaded_images': {},
    'valid_patches_mask': None,
    'valid_months': {},
    'pdf_report': None,
    'month_analysis_results': {},
    'failed_downloads': [],
    'analysis_complete': False,
    'download_complete': False,
    'cloud_free_collection': None,
    'processing_params': None,
    'selected_region_index': 0,
    'processing_in_progress': False,
    'processing_config': None,
    'probability_maps': {},
    'change_detection_result': None,
    'processing_log': [],
    'monthly_component_images': {},
    'monthly_image_counts': {},
    'change_timing_map': None,
    'log_pdf_bytes': None,
}

for key, default in _session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default


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
# File Validation Functions (structural only)
# =============================================================================
def validate_geotiff_file(file_path, expected_bands=1):
    """Structural validation: file exists, size, band count."""
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
# NEW v06.1: Post-Download Pixel Completeness Validation
# =============================================================================
def validate_geotiff_completeness(file_path, month_name="unknown"):
    """
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║  POST-DOWNLOAD PIXEL COMPLETENESS VALIDATION (v06.1)                ║
    ║                                                                     ║
    ║  This function is the critical safety net that catches multi-orbit  ║
    ║  partial coverage issues. When an AOI spans multiple Sentinel-2     ║
    ║  orbital swaths, one swath may have all images filtered out by the  ║
    ║  <10% cloud metadata filter, leaving a region with no data.        ║
    ║                                                                     ║
    ║  GEE-side checks (frequency map, masked_percent) should catch      ║
    ║  this, but edge cases exist. This validation ensures NO incomplete  ║
    ║  image ever reaches classification.                                 ║
    ╚═══════════════════════════════════════════════════════════════════════╝

    Checks for EVERY band:
    1. NaN pixels → REJECT
    2. Nodata-tagged pixels → REJECT
    3. Entire band is zero → REJECT
    4. >20% zero pixels overall → REJECT
    5. Any quadrant >50% zeros → REJECT (partial orbit pattern)

    If ANY band fails ANY check → the entire image is REJECTED.

    Returns:
        (is_valid: bool, message: str)
    """
    try:
        if not os.path.exists(file_path):
            return False, "File does not exist"

        with rasterio.open(file_path) as src:
            nodata_val = src.nodata
            n_bands = src.count
            height = src.height
            width = src.width
            total_pixels = height * width

            add_log_entry(
                f"{month_name}: Completeness check — {n_bands} bands, "
                f"{height}x{width} px, nodata_tag={nodata_val}",
                "INFO"
            )

            for band_idx in range(1, n_bands + 1):
                band_name = (
                    SPECTRAL_BANDS[band_idx - 1]
                    if band_idx <= len(SPECTRAL_BANDS)
                    else f"Band{band_idx}"
                )
                data = src.read(band_idx)

                # ── CHECK 1: NaN pixels ──────────────────────────────
                nan_count = int(np.sum(np.isnan(data)))
                if nan_count > 0:
                    nan_pct = (nan_count / total_pixels) * 100
                    msg = (
                        f"{band_name}: {nan_count} NaN pixels "
                        f"({nan_pct:.2f}%) — likely multi-orbit gap"
                    )
                    add_log_entry(f"{month_name}: REJECT — {msg}", "ERROR")
                    return False, msg

                # ── CHECK 2: Nodata-tagged pixels ────────────────────
                if nodata_val is not None:
                    nodata_count = int(np.sum(data == nodata_val))
                    if nodata_count > 0:
                        nodata_pct = (nodata_count / total_pixels) * 100
                        msg = (
                            f"{band_name}: {nodata_count} nodata pixels "
                            f"({nodata_pct:.2f}%), tag={nodata_val}"
                        )
                        add_log_entry(f"{month_name}: REJECT — {msg}", "ERROR")
                        return False, msg

                # ── CHECK 3: Entire band is zero ─────────────────────
                if np.all(data == 0):
                    msg = f"{band_name}: entirely zero — empty band"
                    add_log_entry(f"{month_name}: REJECT — {msg}", "ERROR")
                    return False, msg

                # ── CHECK 4: Large zero-pixel regions ────────────────
                zero_count = int(np.sum(data == 0))
                zero_pct = (zero_count / total_pixels) * 100

                if zero_pct > 5.0:
                    # 4a. Quadrant analysis — partial orbit pattern
                    mid_h, mid_w = height // 2, width // 2
                    quadrants = {
                        'top-left':     data[:mid_h, :mid_w],
                        'top-right':    data[:mid_h, mid_w:],
                        'bottom-left':  data[mid_h:, :mid_w],
                        'bottom-right': data[mid_h:, mid_w:],
                    }
                    for qname, qdata in quadrants.items():
                        qzero_pct = (int(np.sum(qdata == 0)) / qdata.size) * 100
                        if qzero_pct > 50.0:
                            msg = (
                                f"{band_name}: {zero_pct:.1f}% zeros overall, "
                                f"{qname} quadrant {qzero_pct:.1f}% zeros — "
                                f"partial orbit coverage"
                            )
                            add_log_entry(
                                f"{month_name}: REJECT — {msg}", "ERROR"
                            )
                            return False, msg

                    # 4b. Overall threshold
                    if zero_pct > 20.0:
                        msg = (
                            f"{band_name}: {zero_pct:.1f}% zeros — "
                            f"suspicious, likely incomplete"
                        )
                        add_log_entry(f"{month_name}: REJECT — {msg}", "ERROR")
                        return False, msg

                    # 4c. Warning for moderate zeros (5-20%)
                    add_log_entry(
                        f"{month_name}: NOTE — {band_name} has "
                        f"{zero_pct:.1f}% zeros (within tolerance)",
                        "WARNING"
                    )

            add_log_entry(
                f"{month_name}: Completeness PASSED — "
                f"all {n_bands} bands clean",
                "INFO"
            )
            return True, "All bands complete"

    except Exception as e:
        msg = f"Completeness validation error: {e}"
        add_log_entry(f"{month_name}: {msg}", "ERROR")
        return False, msg


# =============================================================================
# Model Download Functions
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
    except Exception:
        return None


@st.cache_resource
def load_model(model_path):
    try:
        device = torch.device('cpu')
        model = smp.UnetPlusPlus(
            encoder_name='timm-efficientnet-b8',
            encoder_weights='imagenet',
            in_channels=12, classes=1,
            decoder_attention_type='scse'
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
        st.error(f"Error loading model: {e}")
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
            return False, f"Auth failed: {auth_error}"


# =============================================================================
# Helper Functions
# =============================================================================
def get_utm_zone(longitude):
    return math.floor((longitude + 180) / 6) + 1

def get_utm_epsg(longitude, latitude):
    zone_number = get_utm_zone(longitude)
    return f"EPSG:326{zone_number:02d}" if latitude >= 0 else f"EPSG:327{zone_number:02d}"


# =============================================================================
# GEE CLOUD MASKING + GAP-FILLING (Server-Side)
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
        primary=s2_sr, secondary=s2_cloud_prob, condition=join_filter
    )
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
        for f_path in [output_path, temp_path]:
            if os.path.exists(f_path):
                try:
                    os.remove(f_path)
                except:
                    pass
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAY_BASE ** (attempt + 1))
    return False, last_error


def download_composite(composite, aoi, output_path, month_name, scale=10, status_placeholder=None):
    """
    Download composite image to GeoTIFF.
    v06.1: After merging, runs validate_geotiff_completeness().
    If ANY band has NaN/nodata/partial coverage → deletes file, returns None.
    """
    try:
        if os.path.exists(output_path):
            is_valid, msg = validate_geotiff_file(output_path, expected_bands=len(SPECTRAL_BANDS))
            if is_valid:
                # Re-validate completeness on cached files
                is_complete, cmsg = validate_geotiff_completeness(output_path, month_name)
                if is_complete:
                    if status_placeholder:
                        status_placeholder.info(f"✅ {month_name} cached (validated)")
                    return output_path
                else:
                    add_log_entry(f"{month_name}: Cached file FAILED completeness: {cmsg}", "WARNING")
                    os.remove(output_path)
                    return None  # Don't re-download — the GEE composite itself is incomplete
            else:
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
            st.error(f"❌ {month_name}: Failed bands — {'; '.join(failed_bands)}")
            return None

        if len(band_files) != len(SPECTRAL_BANDS):
            return None

        if status_placeholder:
            status_placeholder.text(f"📦 {month_name}: Merging bands...")
        with rasterio.open(band_files[0]) as src:
            meta = src.meta.copy()
        meta.update(count=len(band_files))
        with rasterio.open(output_path, 'w', **meta) as dst:
            for i, band_file in enumerate(band_files):
                with rasterio.open(band_file) as src:
                    dst.write(src.read(1), i + 1)

        # Structural check
        is_valid, msg = validate_geotiff_file(output_path, expected_bands=len(SPECTRAL_BANDS))
        if not is_valid:
            st.error(f"❌ {month_name}: Structural validation failed: {msg}")
            if os.path.exists(output_path):
                os.remove(output_path)
            return None

        # ═════════════════════════════════════════════════════════════════
        # v06.1: POST-DOWNLOAD COMPLETENESS VALIDATION
        # ═════════════════════════════════════════════════════════════════
        if status_placeholder:
            status_placeholder.text(f"🔍 {month_name}: Validating pixel completeness...")

        is_complete, completeness_msg = validate_geotiff_completeness(output_path, month_name)

        if is_complete:
            add_log_entry(f"{month_name}: Download + completeness PASSED", "INFO")
            return output_path
        else:
            st.error(f"❌ {month_name}: POST-DOWNLOAD REJECTION — {completeness_msg}")
            add_log_entry(f"{month_name}: POST-DOWNLOAD REJECTION — {completeness_msg}", "ERROR")
            if os.path.exists(output_path):
                os.remove(output_path)
            return None

    except Exception as e:
        st.error(f"❌ {month_name}: Download exception — {e}")
        return None


def download_monthly_image_v06(aoi, cloud_free_collection, month_info, temp_dir,
                                scale=10, status_placeholder=None):
    """
    Download a single monthly composite with gap-filling.
    v06.1: download_composite() now includes completeness validation.
    If the downloaded image fails → REJECTED.
    """
    month_name = month_info['month_name']
    month_index = month_info['month_index']
    origin = month_info['origin']
    output_file = os.path.join(temp_dir, f"sentinel2_{month_name}.tif")

    add_log_entry(f"Processing month: {month_name}", "INFO")

    # ── Cache check with completeness validation ──
    if os.path.exists(output_file):
        is_valid, msg = validate_geotiff_file(output_file, expected_bands=len(SPECTRAL_BANDS))
        if is_valid:
            is_complete, cmsg = validate_geotiff_completeness(output_file, month_name)
            if is_complete:
                add_log_entry(f"{month_name}: Using cached file (validated)", "INFO")
                if status_placeholder:
                    status_placeholder.info(f"✅ {month_name} cached (validated)")
                return output_file, STATUS_COMPLETE, "Cached (validated)"
            else:
                add_log_entry(f"{month_name}: Cached FAILED completeness: {cmsg}", "WARNING")
                os.remove(output_file)
                return None, STATUS_REJECTED, f"Cached file incomplete: {cmsg}"
        else:
            add_log_entry(f"{month_name}: Cache invalid ({msg}), re-processing", "WARNING")
            os.remove(output_file)

    try:
        origin_date = ee.Date(origin)
        month_start = origin_date.advance(month_index, 'month')
        month_end = origin_date.advance(ee.Number(month_index).add(1), 'month')
        month_middle = month_start.advance(15, 'day')

        if status_placeholder:
            status_placeholder.text(f"📥 {month_name}: Analyzing...")

        monthly_images = cloud_free_collection.filterDate(month_start, month_end)
        image_count = monthly_images.size().getInfo()

        st.session_state.monthly_image_counts[month_name] = image_count
        add_log_entry(f"{month_name}: Found {image_count} cloud-free images", "INFO")

        if image_count == 0:
            add_log_entry(f"{month_name}: REJECTED — No images", "WARNING")
            return None, STATUS_NO_DATA, "No images available"

        if status_placeholder:
            status_placeholder.text(f"📥 {month_name}: Composite from {image_count} images...")

        # Component image previews for logging
        try:
            component_rgbs = download_component_images_for_log(monthly_images, aoi, month_name, image_count)
            if component_rgbs:
                st.session_state.monthly_component_images[month_name] = component_rgbs
        except Exception as e:
            add_log_entry(f"{month_name}: Component preview failed: {e}", "WARNING")

        # Frequency map + composite
        def create_valid_mask(img):
            return ee.Image(1).updateMask(img.select('B4').mask()).unmask(0).toInt()
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
        add_log_entry(f"{month_name}: Masked {masked_percent:.2f}% ({masked_count}/{total_count})", "INFO")

        if masked_percent > MAX_MASKED_PERCENT_FOR_GAPFILL:
            return None, STATUS_SKIPPED, f"Masked {masked_percent:.1f}% > {MAX_MASKED_PERCENT_FOR_GAPFILL}%"

        if masked_percent == 0:
            if status_placeholder:
                status_placeholder.text(f"📥 {month_name}: 0% masked, downloading...")
            path = download_composite(composite, aoi, output_file, month_name, scale, status_placeholder)
            if path:
                return path, STATUS_COMPLETE, "Complete (0% masked, validated)"
            else:
                return None, STATUS_REJECTED, "Download failed or post-download validation rejected"

        # ── Gap-fill ──
        add_log_entry(f"{month_name}: Gap-filling ({masked_percent:.1f}% masked)", "INFO")
        if status_placeholder:
            status_placeholder.text(f"📥 {month_name}: Gap-filling ({masked_percent:.1f}%)...")

        gap_mask = frequency.eq(0)
        month_middle_millis = month_middle.millis()

        m1_past_start = origin_date.advance(ee.Number(month_index).subtract(1), 'month')
        m1_past_end = month_start
        m1_future_start = month_end
        m1_future_end = origin_date.advance(ee.Number(month_index).add(2), 'month')

        all_candidates = (cloud_free_collection.filterDate(m1_past_start, m1_past_end)
                          .merge(cloud_free_collection.filterDate(m1_future_start, m1_future_end)))

        def add_time_distance(img):
            return img.set('time_distance',
                           ee.Number(img.get('system:time_start')).subtract(month_middle_millis).abs())
        sorted_images = all_candidates.map(add_time_distance).sort('time_distance', True)

        candidate_count = sorted_images.size().getInfo()
        add_log_entry(f"{month_name}: {candidate_count} gap-fill candidates", "INFO")
        if candidate_count == 0:
            return None, STATUS_REJECTED, f"No gap-fill candidates, {masked_percent:.1f}% masked"

        closest_mosaic = sorted_images.mosaic().select(SPECTRAL_BANDS)
        has_closest = closest_mosaic.select('B4').mask()
        fill_from_closest = gap_mask.And(has_closest)
        still_masked = gap_mask.And(has_closest.Not())
        filled_composite = composite.unmask(closest_mosaic.updateMask(fill_from_closest))

        fill_source = (ee.Image.constant(0).clip(aoi).toInt8()
                       .where(fill_from_closest, 1).where(still_masked, 2)
                       .rename('fill_source'))
        still_masked_result = fill_source.eq(2).reduceRegion(
            reducer=ee.Reducer.sum(), geometry=aoi, scale=10, maxPixels=1e13
        ).get('fill_source')
        still_masked_count = ee.Number(ee.Algorithms.If(
            ee.Algorithms.IsEqual(still_masked_result, None), 0, still_masked_result
        )).getInfo()

        if still_masked_count == 0:
            if status_placeholder:
                status_placeholder.text(f"📥 {month_name}: Gap-filled, downloading...")
            path = download_composite(filled_composite, aoi, output_file, month_name, scale, status_placeholder)
            if path:
                return path, STATUS_COMPLETE, f"Gap-filled (was {masked_percent:.1f}%, validated)"
            else:
                return None, STATUS_REJECTED, "Post-download validation rejected after gap-fill"
        else:
            pct = (still_masked_count / total_count) * 100
            return None, STATUS_REJECTED, f"{pct:.1f}% still masked after gap-fill"

    except Exception as e:
        add_log_entry(f"{month_name}: ERROR — {e}", "ERROR")
        return None, STATUS_NO_DATA, f"Error: {e}"


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
                    component_rgbs.append({'date': img_date, 'rgb': np.array(img_pil.convert('RGB'))})
            except Exception as e:
                add_log_entry(f"{month_name}: Component image {i+1} failed: {e}", "WARNING")
        return component_rgbs
    except Exception as e:
        add_log_entry(f"{month_name}: Component download failed: {e}", "WARNING")
        return []


# =============================================================================
# RGB Thumbnail / Patch Validity / Classification
# (Unchanged from v06 — included for completeness)
# =============================================================================
def generate_rgb_thumbnail(image_path, month_name, max_size=256):
    try:
        with rasterio.open(image_path) as src:
            red, green, blue = src.read(4), src.read(3), src.read(2)
        rgb = np.nan_to_num(np.stack([red, green, blue], axis=-1), nan=0.0)
        def pstretch(band, lo=2, hi=98):
            v = band[band > 0]
            if len(v) == 0: return np.zeros_like(band, dtype=np.uint8)
            pl, ph = np.percentile(v, lo), np.percentile(v, hi)
            if ph <= pl: ph = pl + 0.001
            return (np.clip((band - pl) / (ph - pl), 0, 1) * 255).astype(np.uint8)
        rgb8 = np.stack([pstretch(rgb[:,:,i]) for i in range(3)], axis=-1)
        pil_img = Image.fromarray(rgb8, 'RGB')
        h, w = pil_img.size[1], pil_img.size[0]
        if max(h, w) > max_size:
            s = max_size / max(h, w)
            pil_img = pil_img.resize((int(w*s), int(h*s)), Image.LANCZOS)
        return pil_img
    except:
        return None


def check_patch_validity(patch, nodata_threshold_percent=0):
    if np.any(np.isnan(patch)):
        return False
    if np.all(patch == 0):
        return False
    zero_percent = (np.sum(patch == 0) / patch.size) * 100
    if zero_percent > nodata_threshold_percent:
        return False
    if patch.ndim == 3:
        for b in range(patch.shape[-1]):
            if np.all(patch[:, :, b] == 0):
                return False
    return True


def get_patch_validity_mask(image_path, patch_size=224, nodata_threshold_percent=0):
    try:
        with rasterio.open(image_path) as src:
            img_data = src.read()
        img = np.moveaxis(img_data, 0, -1)
        h, w, c = img.shape
        nh = int(np.ceil(h / patch_size) * patch_size)
        nw = int(np.ceil(w / patch_size) * patch_size)
        if h != nh or w != nw:
            p = np.zeros((nh, nw, c), dtype=img.dtype)
            p[:h, :w, :] = img
            img = p
        patches = patchify(img, (patch_size, patch_size, c), step=patch_size)
        nph, npw = patches.shape[0], patches.shape[1]
        mask = np.zeros((nph, npw), dtype=bool)
        for i in range(nph):
            for j in range(npw):
                mask[i, j] = check_patch_validity(patches[i, j, 0], nodata_threshold_percent)
        return mask, (h, w), (nph, npw)
    except:
        return None, None, None


def find_common_valid_patches(downloaded_images, nodata_threshold_percent=0):
    st.info("🔍 Analyzing patch validity across all months...")
    month_names = sorted(downloaded_images.keys())
    if not month_names:
        st.error("❌ No downloaded images!")
        return None, None, None

    # Dimension check
    dims = {}
    for mn in month_names:
        with rasterio.open(downloaded_images[mn]) as src:
            dims[mn] = (src.height, src.width, src.count)
    ref = dims[month_names[0]]
    bad = [f"{mn}: {d[0]}x{d[1]}" for mn, d in dims.items() if d[:2] != ref[:2]]
    if bad:
        st.error(f"❌ Dimension mismatch! Ref: {ref[0]}x{ref[1]}. Bad: {', '.join(bad)}")
        return None, None, None
    st.success(f"✅ All {len(month_names)} images: {ref[0]}x{ref[1]} ({ref[2]} bands)")

    h, w = ref[0], ref[1]
    nph = int(np.ceil(h / PATCH_SIZE))
    npw = int(np.ceil(w / PATCH_SIZE))
    total = nph * npw
    st.write(f"**Patch grid**: {nph}×{npw} = **{total} patches**")

    prog = st.progress(0)
    masks = {}
    counts = {}
    for idx, mn in enumerate(month_names):
        vm, _, _ = get_patch_validity_mask(downloaded_images[mn], PATCH_SIZE, nodata_threshold_percent)
        if vm is not None and vm.shape == (nph, npw):
            masks[mn] = vm
            counts[mn] = int(np.sum(vm))
        prog.progress((idx + 1) / len(month_names))
    prog.empty()

    if not masks:
        st.error("❌ No months analyzed!")
        return None, None, None

    max_count = max(counts.values())
    ref_month = [mn for mn, c in counts.items() if c == max_count][0]
    ref_mask = masks[ref_month]
    st.info(f"📊 Max valid patches: **{max_count}/{total}** ({100*max_count/total:.1f}%)")

    valid_months = {}
    for mn, m in masks.items():
        if np.all(m[ref_mask]):
            valid_months[mn] = downloaded_images[mn]
            st.write(f"   ✅ {mn}: {counts[mn]}/{total} — INCLUDED")
        else:
            missing = int(np.sum(ref_mask & ~m))
            st.write(f"   ❌ {mn}: {counts[mn]}/{total} — EXCLUDED (missing {missing})")

    if not valid_months:
        st.error("❌ No months match reference!")
        return None, None, None
    st.success(f"✅ **{len(valid_months)}/{len(masks)}** months valid")
    return ref_mask, (h, w), valid_months


def classify_image_with_mask(image_path, model, device, month_name, valid_mask, original_size):
    try:
        with rasterio.open(image_path) as src:
            img_data = src.read()
        img = np.moveaxis(img_data, 0, -1)
        h, w, c = img.shape
        if original_size and (h, w) != original_size:
            return None, None, 0
        nh = int(np.ceil(h / PATCH_SIZE) * PATCH_SIZE)
        nw = int(np.ceil(w / PATCH_SIZE) * PATCH_SIZE)
        nph, npw = nh // PATCH_SIZE, nw // PATCH_SIZE
        if valid_mask.shape != (nph, npw):
            return None, None, 0
        if h != nh or w != nw:
            p = np.zeros((nh, nw, c), dtype=img.dtype)
            p[:h, :w, :] = img
            img = p
        patches = patchify(img, (PATCH_SIZE, PATCH_SIZE, c), step=PATCH_SIZE)
        cls_patches = np.zeros((nph, npw, PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
        prob_patches = np.zeros((nph, npw, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
        count = 0
        for i in range(nph):
            for j in range(npw):
                if not valid_mask[i, j]:
                    continue
                patch = normalized(patches[i, j, 0])
                t = torch.tensor(np.moveaxis(patch, -1, 0), dtype=torch.float32).unsqueeze(0)
                with torch.inference_mode():
                    pred = torch.sigmoid(model(t)).cpu().squeeze().numpy()
                cls_patches[i, j] = (pred > 0.5).astype(np.uint8) * 255
                prob_patches[i, j] = pred
                count += 1
        rec = unpatchify(cls_patches, (nh, nw))[:original_size[0], :original_size[1]]
        rec_p = unpatchify(prob_patches, (nh, nw))[:original_size[0], :original_size[1]]
        return rec, rec_p, count
    except:
        return None, None, 0


def get_valid_patch_bounds(valid_mask, patch_size=224, original_size=None):
    if valid_mask is None or not np.any(valid_mask):
        return None
    rows = np.where(np.any(valid_mask, axis=1))[0]
    cols = np.where(np.any(valid_mask, axis=0))[0]
    if not len(rows) or not len(cols):
        return None
    rs, re = rows[0] * patch_size, (rows[-1] + 1) * patch_size
    cs, ce = cols[0] * patch_size, (cols[-1] + 1) * patch_size
    if original_size:
        re = min(re, original_size[0])
        ce = min(ce, original_size[1])
    return (rs, re, cs, ce)


def create_pixel_mask_from_patches(valid_mask, patch_size=224, target_size=None):
    if valid_mask is None:
        return None
    pm = np.repeat(np.repeat(valid_mask, patch_size, axis=0), patch_size, axis=1)
    if target_size:
        pm = pm[:target_size[0], :target_size[1]]
    return pm


def generate_thumbnails(image_path, classification_mask, month_name,
                        valid_mask=None, original_size=None, max_size=256):
    try:
        crop_bounds = get_valid_patch_bounds(valid_mask, PATCH_SIZE, original_size) if valid_mask is not None else None
        pixel_mask = create_pixel_mask_from_patches(valid_mask, PATCH_SIZE, original_size) if valid_mask is not None else None
        with rasterio.open(image_path) as src:
            rgb = np.nan_to_num(np.stack([src.read(4), src.read(3), src.read(2)], axis=-1), nan=0.0)
        def ps(b, lo=2, hi=98):
            v = b[b > 0]
            if not len(v): return np.zeros_like(b, dtype=np.uint8)
            pl, ph = np.percentile(v, lo), np.percentile(v, hi)
            if ph <= pl: ph = pl + .001
            return (np.clip((b - pl) / (ph - pl), 0, 1) * 255).astype(np.uint8)
        rgb8 = np.stack([ps(rgb[:,:,i]) for i in range(3)], axis=-1)
        if pixel_mask is not None:
            rgb8 = np.where(np.stack([pixel_mask]*3, -1), rgb8, 0)
        if crop_bounds:
            rs, re, cs, ce = crop_bounds
            rgb8 = rgb8[rs:re, cs:ce, :]
            cm = classification_mask[rs:re, cs:ce]
        else:
            cm = classification_mask
        pr = Image.fromarray(rgb8, 'RGB')
        pc = Image.fromarray(cm.astype(np.uint8))
        h, w = pr.size[1], pr.size[0]
        if max(h, w) > max_size:
            s = max_size / max(h, w)
            pr = pr.resize((int(w*s), int(h*s)), Image.LANCZOS)
            pc = pc.resize((int(w*s), int(h*s)), Image.NEAREST)
        return {
            'rgb_image': pr, 'classification_image': pc, 'month_name': month_name,
            'original_size': classification_mask.shape, 'cropped_size': cm.shape,
            'building_pixels': int(np.sum(cm > 0)), 'total_pixels': cm.size
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
        add_log_entry(f"{'RESUME' if resume else 'START'}: Pipeline v06.1", "INFO")
        add_log_entry(f"Date: {start_date} → {end_date}, cloud<{cloudy_pixel_percentage}%, scale={scale}m", "INFO")
        add_log_entry("Post-download completeness validation: ENABLED", "INFO")

        if st.session_state.current_temp_dir is None or not os.path.exists(st.session_state.current_temp_dir):
            st.session_state.current_temp_dir = tempfile.mkdtemp()
        temp_dir = st.session_state.current_temp_dir

        start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        total_months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)
        st.info(f"📅 {total_months} months | 📁 {temp_dir}")

        extended_start = (start_dt - datetime.timedelta(days=31)).strftime('%Y-%m-%d')
        extended_end = (end_dt + datetime.timedelta(days=31)).strftime('%Y-%m-%d')

        # PHASE 1: Cloud-free collection
        st.header("Phase 1: Cloud-Free Collection")
        cloud_free_collection = create_cloud_free_collection(aoi, extended_start, extended_end, cloudy_pixel_percentage)

        # PHASE 2: Download
        st.header("Phase 2: Download & Gap-Fill")
        st.info("🔍 **v06.1**: Every download validated for NaN/nodata/partial coverage")

        downloaded_images = {}
        month_statuses = {}
        month_infos = []
        current_range = set()
        for mi in range(total_months):
            y = start_dt.year + (start_dt.month - 1 + mi) // 12
            m = (start_dt.month - 1 + mi) % 12 + 1
            mn = f"{y}-{m:02d}"
            month_infos.append({'month_name': mn, 'month_index': mi, 'origin': start_date})
            current_range.add(mn)

        # Resume: re-validate cached downloads
        if resume and st.session_state.downloaded_images:
            for mn, path in list(st.session_state.downloaded_images.items()):
                if mn not in current_range:
                    continue
                if os.path.exists(path):
                    ok, _ = validate_geotiff_file(path, len(SPECTRAL_BANDS))
                    if ok:
                        complete, cmsg = validate_geotiff_completeness(path, mn)
                        if complete:
                            downloaded_images[mn] = path
                            month_statuses[mn] = {'status': STATUS_COMPLETE, 'message': 'Cached (validated)'}
                        else:
                            del st.session_state.downloaded_images[mn]
                            os.remove(path)
                            month_statuses[mn] = {'status': STATUS_REJECTED, 'message': f'Incomplete: {cmsg}'}
                            st.session_state.month_analysis_results[mn] = month_statuses[mn]

        if resume and st.session_state.month_analysis_results:
            for mn, si in st.session_state.month_analysis_results.items():
                if mn in current_range and mn not in month_statuses:
                    month_statuses[mn] = si

        to_process = [m for m in month_infos
                      if m['month_name'] not in downloaded_images
                      and m['month_name'] not in month_statuses]

        if to_process:
            prog = st.progress(0)
            stxt = st.empty()
            for idx, mi in enumerate(to_process):
                mn = mi['month_name']
                path, status, msg = download_monthly_image_v06(
                    aoi, cloud_free_collection, mi, temp_dir, scale, stxt)
                month_statuses[mn] = {'status': status, 'message': msg}
                st.session_state.month_analysis_results[mn] = {'status': status, 'message': msg}
                icon = {"no_data": "⚫", "skipped": "🟡", "complete": "🟢", "rejected": "🔴"}.get(status, "❓")
                st.write(f"{icon} **{mn}**: {status} — {msg}")
                if path:
                    downloaded_images[mn] = path
                    st.session_state.downloaded_images[mn] = path
                prog.progress((idx + 1) / len(to_process))
            prog.empty()
            stxt.empty()

        # Summary
        st.divider()
        sc = {s: sum(1 for ms in month_statuses.values() if ms['status'] == s)
              for s in [STATUS_NO_DATA, STATUS_SKIPPED, STATUS_COMPLETE, STATUS_REJECTED]}
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("✅ Complete", sc[STATUS_COMPLETE])
        c2.metric("🔴 Rejected", sc[STATUS_REJECTED])
        c3.metric("🟡 Skipped", sc[STATUS_SKIPPED])
        c4.metric("⚫ No Data", sc[STATUS_NO_DATA])

        st.session_state.failed_downloads = [mn for mn, ms in month_statuses.items() if ms['status'] != STATUS_COMPLETE]
        st.session_state.month_analysis_results = month_statuses

        if not downloaded_images:
            st.error("❌ No images downloaded!")
            return []
        st.success(f"✅ {len(downloaded_images)}/{total_months} months (all validated)")

        # PHASE 3: Patch validity
        st.header("Phase 3: Patch Validity")
        valid_mask, original_size, valid_months = find_common_valid_patches(downloaded_images, nodata_threshold_percent)
        if valid_mask is None:
            return []
        st.session_state.valid_patches_mask = valid_mask
        st.session_state.valid_months = valid_months
        st.session_state.processed_months = {}

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.imshow(valid_mask, cmap='RdYlGn', vmin=0, vmax=1)
        ax.set_title(f"Valid Patches ({int(np.sum(valid_mask))})")
        st.pyplot(fig)
        plt.close()

        # PHASE 4: Classification
        st.header("Phase 4: Classification")
        excluded = len(downloaded_images) - len(valid_months)
        st.info(f"🧠 Classifying {len(valid_months)} months (excluded {excluded})")

        thumbnails = []
        probability_maps = {}
        prog = st.progress(0)
        stxt = st.empty()
        for idx, mn in enumerate(sorted(valid_months.keys())):
            if mn in st.session_state.processed_months:
                thumbnails.append(st.session_state.processed_months[mn])
                if mn in st.session_state.probability_maps:
                    probability_maps[mn] = st.session_state.probability_maps[mn]
                prog.progress((idx + 1) / len(valid_months))
                continue
            stxt.text(f"🧠 {mn} ({idx+1}/{len(valid_months)})...")
            mask, prob, vc = classify_image_with_mask(valid_months[mn], model, device, mn, valid_mask, original_size)
            if mask is not None:
                probability_maps[mn] = prob
                st.session_state.probability_maps[mn] = prob
                thumb = generate_thumbnails(valid_months[mn], mask, mn, valid_mask, original_size)
                if thumb:
                    thumb['valid_patches'] = vc
                    thumb['gap_filled'] = 'gap-fill' in month_statuses.get(mn, {}).get('message', '').lower()
                    thumbnails.append(thumb)
                    st.session_state.processed_months[mn] = thumb
            prog.progress((idx + 1) / len(valid_months))
        prog.empty()
        stxt.empty()
        st.success(f"✅ Classified {len(thumbnails)} months!")

        # Filter low building %
        if len(thumbnails) >= 2:
            pcts = [(t['building_pixels'] / t['total_pixels']) * 100 for t in thumbnails]
            med = np.median(pcts)
            filtered = []
            for t in thumbnails:
                pct = (t['building_pixels'] / t['total_pixels']) * 100
                if pct < (med - 8.0):
                    for d in [st.session_state.valid_months, st.session_state.probability_maps]:
                        d.pop(t['month_name'], None)
                else:
                    filtered.append(t)
            thumbnails = filtered

        return thumbnails
    except Exception as e:
        st.error(f"Error: {e}")
        import traceback
        st.error(traceback.format_exc())
        return []


# =============================================================================
# Change Detection Algorithm (unchanged from v06)
# =============================================================================
def analyze_building_transition(probability_maps, non_building_thr=0.2, building_thr=0.8,
                                non_building_duration=2, building_duration=2):
    add_log_entry("Starting change detection", "INFO")
    sorted_months = sorted(probability_maps.keys())
    if len(sorted_months) < (non_building_duration + building_duration):
        return None, None, {"error": f"Need {non_building_duration + building_duration} months, got {len(sorted_months)}"}

    first_map = probability_maps[sorted_months[0]]
    h, w = first_map.shape
    data = np.zeros((len(sorted_months), h, w), dtype=np.float32)
    for i, m in enumerate(sorted_months):
        data[i] = probability_maps[m]

    results = np.zeros((h, w), dtype=np.uint8)
    timing = np.zeros((h, w), dtype=np.int16)

    for y in range(h):
        for x in range(w):
            ps = data[:, y, x]
            if np.all(ps == 0) or ps[0] > non_building_thr or ps[-1] < building_thr:
                continue
            tp = None
            bad = False
            for t in range(1, len(ps)):
                if ps[t-1] > non_building_thr and ps[t] <= non_building_thr:
                    bad = True; break
                if ps[t-1] >= building_thr and ps[t] < building_thr:
                    bad = True; break
                if ps[t-1] <= non_building_thr and ps[t] >= building_thr and tp is None:
                    tp = t
            if tp is not None and not bad:
                if tp >= non_building_duration and (len(ps) - tp) >= building_duration:
                    results[y, x] = 1
                    timing[y, x] = tp

    cp = int(np.sum(results))
    tot = int(np.sum(data[0] > 0))
    pct = (cp / tot * 100) if tot > 0 else 0
    tbm = {}
    for i, m in enumerate(sorted_months):
        if i > 0:
            tbm[m] = int(np.sum(timing == i))
    st.session_state.change_timing_map = timing
    return results, timing, {
        'change_pixels': cp, 'total_pixels': tot, 'change_percentage': pct,
        'n_months': len(sorted_months), 'months': sorted_months,
        'first_month': sorted_months[0], 'last_month': sorted_months[-1],
        'transition_by_month': tbm
    }


def generate_rgb_from_sentinel(image_path, max_size=None):
    try:
        with rasterio.open(image_path) as src:
            rgb = np.nan_to_num(np.stack([src.read(4), src.read(3), src.read(2)], axis=-1), nan=0.0)
        def ps(b, lo=2, hi=98):
            v = b[b > 0]
            if not len(v): return np.zeros_like(b, dtype=np.uint8)
            pl, ph = np.percentile(v, lo), np.percentile(v, hi)
            if ph <= pl: ph = pl + .001
            return (np.clip((b - pl) / (ph - pl), 0, 1) * 255).astype(np.uint8)
        return np.stack([ps(rgb[:,:,i]) for i in range(3)], axis=-1)
    except:
        return None


# =============================================================================
# Display Thumbnails
# =============================================================================
def get_image_download_data(image_path, month_name):
    try:
        with open(image_path, 'rb') as f:
            return f.read()
    except:
        return None


def display_thumbnails(thumbnails, valid_months=None):
    if not thumbnails:
        return
    mode = st.radio("Display:", ["Side by Side", "Classification", "RGB"], horizontal=True)
    st.divider()
    if mode == "Side by Side":
        for i in range(0, len(thumbnails), 2):
            cols = st.columns(4)
            for j in range(2):
                idx = i + j
                if idx < len(thumbnails):
                    t = thumbnails[idx]
                    pct = (t['building_pixels'] / t['total_pixels']) * 100
                    sfx = " (filled)" if t.get('gap_filled') else ""
                    cols[j*2].image(t['rgb_image'], caption=f"{t['month_name']} RGB{sfx}")
                    if valid_months and t['month_name'] in valid_months:
                        d = get_image_download_data(valid_months[t['month_name']], t['month_name'])
                        if d:
                            cols[j*2].download_button(f"⬇️ {t['month_name']}", d,
                                f"sentinel2_{t['month_name']}_12bands.tif", "image/tiff",
                                key=f"dl_s_{t['month_name']}")
                    cols[j*2+1].image(t['classification_image'], caption=f"{t['month_name']} ({pct:.1f}%)")
    else:
        key = 'classification_image' if mode == "Classification" else 'rgb_image'
        for row in range((len(thumbnails) + 3) // 4):
            cols = st.columns(4)
            for c in range(4):
                idx = row * 4 + c
                if idx < len(thumbnails):
                    t = thumbnails[idx]
                    pct = (t['building_pixels'] / t['total_pixels']) * 100
                    cap = f"{t['month_name']} ({pct:.1f}%)" if mode == "Classification" else t['month_name']
                    if t.get(key):
                        cols[c].image(t[key], caption=cap)
                        if mode == "RGB" and valid_months and t['month_name'] in valid_months:
                            d = get_image_download_data(valid_months[t['month_name']], t['month_name'])
                            if d:
                                cols[c].download_button("⬇️", d,
                                    f"sentinel2_{t['month_name']}_12bands.tif", "image/tiff",
                                    key=f"dl_r_{t['month_name']}")


# =============================================================================
# Main Application
# =============================================================================
def main():
    tab1, tab2 = st.tabs(["🏗️ Classification", "🔍 Change Detection"])
    with tab1:
        main_classification_tab()
    with tab2:
        change_detection_tab()


def main_classification_tab():
    st.title("🏗️ Building Classification v06.1")
    st.markdown("""
    **v06.1 — Post-download pixel completeness validation**
    
    | Status | Condition | Download? |
    |--------|-----------|-----------|
    | `no_data` | No images | ❌ |
    | `skipped` | masked > 30% | ❌ |
    | `complete` | masked == 0% **AND** pixel-complete | ✅ |
    | `rejected` | masked > 0% after gap-fill **OR** NaN/nodata detected | ❌ |
    """)

    ee_ok, ee_msg = initialize_earth_engine()
    if not ee_ok:
        st.error(ee_msg); st.stop()
    st.sidebar.success(ee_msg)

    # Model
    st.sidebar.header("🧠 Model")
    model_path = "best_model_version_Unet++_v02_e7.pt"
    if not os.path.exists(model_path):
        if not download_model_from_gdrive("", model_path):
            st.stop()
    if not st.session_state.model_loaded:
        model, device = load_model(model_path)
        if model:
            st.session_state.model = model
            st.session_state.device = device
            st.session_state.model_loaded = True
            st.sidebar.success("✅ Model loaded")
        else:
            st.stop()
    else:
        st.sidebar.success("✅ Model loaded")

    # Parameters
    st.sidebar.header("⚙️ Parameters")
    cloudy_pct = st.sidebar.slider("Max Cloud %", 0, 50, 10, 5,
        disabled=st.session_state.processing_in_progress)
    nodata_pct = 0

    # Cache status
    st.sidebar.header("🗂️ Cache")
    for k, lbl in [('month_analysis_results', '📊 analyzed'), ('downloaded_images', '📥 downloaded'), ('processed_months', '🧠 classified')]:
        if st.session_state.get(k):
            st.sidebar.success(f"{lbl}: {len(st.session_state[k])}")

    if st.sidebar.button("🗑️ Clear All", disabled=st.session_state.processing_in_progress):
        for k in list(_session_defaults.keys()):
            if k in ['model', 'device', 'model_loaded', 'ee_initialized']:
                continue
            st.session_state[k] = type(_session_defaults[k])() if isinstance(_session_defaults[k], (dict, list)) else _session_defaults[k]
        st.session_state.processing_complete = False
        st.session_state.processing_in_progress = False
        clear_log()
        st.rerun()

    if st.session_state.processing_in_progress:
        if st.sidebar.button("🛑 Stop", type="primary"):
            st.session_state.processing_in_progress = False
            st.rerun()

    # Region
    st.header("1️⃣ Region")
    if not st.session_state.processing_in_progress:
        m = folium.Map(location=[35.6892, 51.3890], zoom_start=8)
        plugins.Draw(export=True, draw_options={
            'polyline': False, 'rectangle': True, 'polygon': True,
            'circle': False, 'marker': False, 'circlemarker': False}).add_to(m)
        folium.TileLayer(tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
                         attr='Google', name='Satellite').add_to(m)
        folium.LayerControl().add_to(m)
        map_data = st_folium(m, width=800, height=500)

        if map_data and map_data.get('last_active_drawing'):
            geom = map_data['last_active_drawing'].get('geometry', {})
            if geom.get('type') == 'Polygon':
                st.session_state.last_drawn_polygon = Polygon(geom['coordinates'][0])
                st.success("✅ Region selected")

        if st.button("💾 Save Region"):
            if st.session_state.last_drawn_polygon:
                if not any(p.equals(st.session_state.last_drawn_polygon) for p in st.session_state.drawn_polygons):
                    st.session_state.drawn_polygons.append(st.session_state.last_drawn_polygon)
                    st.rerun()

    if st.session_state.drawn_polygons:
        st.subheader("📍 Saved Regions")
        for i, p in enumerate(st.session_state.drawn_polygons):
            c1, c2 = st.columns([4, 1])
            c1.write(f"**Region {i+1}**: ~{p.area * 111 * 111:.2f} km²")
            if c2.button("🗑️", key=f"del_{i}", disabled=st.session_state.processing_in_progress):
                st.session_state.drawn_polygons.pop(i)
                st.rerun()

    # Date
    st.header("2️⃣ Time Period")
    c1, c2 = st.columns(2)
    start = c1.date_input("Start", value=date(2024, 1, 1), disabled=st.session_state.processing_in_progress)
    end = c2.date_input("End (exclusive)", value=date(2025, 1, 1), disabled=st.session_state.processing_in_progress)
    if start >= end:
        st.error("Invalid dates"); st.stop()
    months = (end.year - start.year) * 12 + (end.month - start.month)
    st.info(f"📅 {months} months")

    # Process
    st.header("3️⃣ Process")
    selected_polygon = None
    if st.session_state.drawn_polygons:
        opts = [f"Region {i+1}" for i in range(len(st.session_state.drawn_polygons))]
        si = st.selectbox("Region", range(len(opts)), format_func=lambda i: opts[i],
                          disabled=st.session_state.processing_in_progress)
        selected_polygon = st.session_state.drawn_polygons[si]
    elif st.session_state.last_drawn_polygon:
        selected_polygon = st.session_state.last_drawn_polygon

    st.divider()
    c1, c2, c3 = st.columns(3)
    start_new = c1.button("🚀 Start New", type="primary",
                           disabled=st.session_state.processing_in_progress or selected_polygon is None)
    resume_btn = c2.button("🔄 Resume",
                            disabled=not st.session_state.downloaded_images or st.session_state.processing_in_progress)

    should_process = False
    resume_mode = False

    if start_new:
        should_process = True
        for k in ['month_analysis_results', 'processed_months', 'downloaded_images',
                   'classification_thumbnails', 'valid_months', 'probability_maps',
                   'failed_downloads']:
            st.session_state[k] = type(_session_defaults[k])()
        st.session_state.processing_config = {
            'polygon_coords': list(selected_polygon.exterior.coords),
            'start_date': start.strftime('%Y-%m-%d'), 'end_date': end.strftime('%Y-%m-%d'),
            'cloudy_pct': cloudy_pct, 'nodata_pct': nodata_pct
        }
        st.session_state.processing_in_progress = True
    elif resume_btn:
        should_process = True
        resume_mode = True
        st.session_state.processing_in_progress = True
        if not st.session_state.processing_config:
            st.session_state.processing_config = {
                'polygon_coords': list(selected_polygon.exterior.coords),
                'start_date': start.strftime('%Y-%m-%d'), 'end_date': end.strftime('%Y-%m-%d'),
                'cloudy_pct': cloudy_pct, 'nodata_pct': nodata_pct
            }
    elif st.session_state.processing_in_progress and st.session_state.processing_config:
        should_process = True
        resume_mode = True

    if should_process:
        cfg = st.session_state.processing_config
        aoi = ee.Geometry.Polygon([cfg['polygon_coords']])
        try:
            thumbs = process_timeseries(
                aoi, cfg['start_date'], cfg['end_date'],
                st.session_state.model, st.session_state.device,
                cfg['cloudy_pct'], 10, cfg['nodata_pct'], resume=resume_mode)
            if thumbs:
                st.session_state.classification_thumbnails = thumbs
                st.session_state.processing_complete = True
            st.session_state.processing_in_progress = False
        except Exception as e:
            st.error(f"❌ {e}")
            st.session_state.processing_in_progress = False

    if st.session_state.processing_complete and st.session_state.classification_thumbnails:
        st.divider()
        st.header("📊 Results")
        display_thumbnails(st.session_state.classification_thumbnails, st.session_state.valid_months)


def change_detection_tab():
    st.title("🔍 Change Detection")
    if not st.session_state.probability_maps:
        st.warning("Complete classification first.")
        return
    n = len(st.session_state.probability_maps)
    sm = sorted(st.session_state.probability_maps.keys())
    st.success(f"✅ {n} months: {sm[0]} → {sm[-1]}")

    c1, c2 = st.columns(2)
    nbt = c1.text_input("Non-building threshold", "0.2")
    bt = c1.text_input("Building threshold", "0.8")
    mnb = c2.text_input("Min non-building months", "2")
    mb = c2.text_input("Min building months", "2")

    try:
        nbt_v, bt_v, mnb_v, mb_v = float(nbt), float(bt), int(mnb), int(mb)
        ok = 0 <= nbt_v <= 0.5 and 0.5 <= bt_v <= 1.0 and mnb_v >= 1 and mb_v >= 1
    except:
        ok = False
    if not ok:
        st.error("Invalid parameters")
        return

    if st.button("🔍 Run Change Detection", type="primary"):
        with st.spinner("Analyzing..."):
            cm, tm, stats = analyze_building_transition(
                st.session_state.probability_maps, nbt_v, bt_v, mnb_v, mb_v)
            if cm is not None:
                st.session_state.change_detection_result = {
                    'mask': cm, 'timing_map': tm, 'stats': stats,
                    'params': {'non_building_thr': nbt_v, 'building_thr': bt_v,
                               'min_non_building': mnb_v, 'min_building': mb_v}
                }
                st.rerun()

    if st.session_state.change_detection_result:
        r = st.session_state.change_detection_result
        s = r['stats']
        c1, c2, c3 = st.columns(3)
        c1.metric("Change Pixels", f"{s['change_pixels']:,}")
        c2.metric("Total Pixels", f"{s['total_pixels']:,}")
        c3.metric("Change %", f"{s['change_percentage']:.2f}%")

        # Visualization
        fm, lm = s['first_month'], s['last_month']
        fp = st.session_state.valid_months.get(fm)
        lp = st.session_state.valid_months.get(lm)
        vm = st.session_state.valid_patches_mask
        cb = get_valid_patch_bounds(vm, PATCH_SIZE, r['mask'].shape)
        fr = generate_rgb_from_sentinel(fp) if fp else None
        lr = generate_rgb_from_sentinel(lp) if lp else None

        if fr is not None and lr is not None and cb:
            rs, re, cs, ce = cb
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(fr[rs:re, cs:ce]); axes[0].set_title(f"First: {fm}"); axes[0].axis('off')
            axes[1].imshow(lr[rs:re, cs:ce]); axes[1].set_title(f"Last: {lm}"); axes[1].axis('off')
            cm_crop = r['mask'][rs:re, cs:ce]
            overlay = np.zeros((*cm_crop.shape, 3), dtype=np.uint8)
            overlay[cm_crop == 1] = [255, 0, 0]
            axes[2].imshow(lr[rs:re, cs:ce])
            axes[2].imshow(overlay, alpha=0.6)
            axes[2].set_title(f"Changes ({s['change_pixels']:,} px)"); axes[2].axis('off')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()


if __name__ == "__main__":
    main()
