"""
Sentinel-2 Time Series Building Classification
VERSION 06 - GEE Cloud Masking + Gap-Filling + Python Patch Validation

ALGORITHM (matching JS code):
═══════════════════════════════════════════════════════════════════════════════
GEE SERVER-SIDE:
1. Filter images by CLOUDY_PIXEL_PERCENTAGE metadata (<10%)
2. Join S2_SR with S2_CLOUD_PROBABILITY
3. Apply cloud mask: cloudProb > 65 AND cdi < -0.5, dilate 20m
4. For each month M:
   a. Create median composite from cloud-free collection
   b. Count masked pixels (frequency == 0)
   c. Status logic:
      - image_count == 0        → "no_data"   → SKIP
      - masked_percent > 30%    → "skipped"   → SKIP (don't try gap-fill)
      - masked_percent == 0%    → "complete"  → DOWNLOAD ✅
      - 0% < masked <= 30%      → TRY GAP-FILL from M-1, M+1:
          * Collect cloud-free images from M-1 and M+1
          * Sort by time distance to 15th of month M
          * Fill gaps with closest cloud-free pixel (mosaic)
          * If masked_after == 0% → "complete" → DOWNLOAD ✅
          * If masked_after > 0%  → "rejected" → SKIP

PYTHON CLIENT-SIDE (after download):
5. Validate all downloaded images have SAME dimensions
6. Check patch validity (NaN/zeros) for each month
7. Find the month with MAXIMUM valid patches (reference)
8. EXCLUDE months that don't have ALL reference patches valid
9. Classify only valid months using reference patch mask

GUARANTEES:
═══════════════════════════════════════════════════════════════════════════════
✅ All downloaded images have 0% masked pixels (GEE-side guarantee)
✅ All downloaded images have identical dimensions (validated in Python)
✅ All classified months have SAME valid patches (reference mask)
✅ Months with missing patches are EXCLUDED (not reduced to minimum)
✅ Maximum number of valid patches preserved
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

# Download settings
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2
DOWNLOAD_TIMEOUT = 120
CHUNK_SIZE = 8192

# File validation
MIN_BAND_FILE_SIZE = 10000
MIN_MULTIBAND_FILE_SIZE = 100000

# Cloud masking thresholds (from JS)
CLOUD_PROB_THRESHOLD = 50
CDI_THRESHOLD = -0.5

# Gap-filling threshold
MAX_MASKED_PERCENT_FOR_GAPFILL = 30

# Status constants
STATUS_NO_DATA = "no_data"
STATUS_SKIPPED = "skipped"
STATUS_COMPLETE = "complete"
STATUS_REJECTED = "rejected"

# =============================================================================
# Session State Initialization
# =============================================================================
if 'drawn_polygons' not in st.session_state:
    st.session_state.drawn_polygons = []
if 'last_drawn_polygon' not in st.session_state:
    st.session_state.last_drawn_polygon = None
if 'ee_initialized' not in st.session_state:
    st.session_state.ee_initialized = False
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'device' not in st.session_state:
    st.session_state.device = None
if 'classification_thumbnails' not in st.session_state:
    st.session_state.classification_thumbnails = []
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'processed_months' not in st.session_state:
    st.session_state.processed_months = {}
if 'current_temp_dir' not in st.session_state:
    st.session_state.current_temp_dir = None
if 'downloaded_images' not in st.session_state:
    st.session_state.downloaded_images = {}
if 'valid_patches_mask' not in st.session_state:
    st.session_state.valid_patches_mask = None
if 'valid_months' not in st.session_state:
    st.session_state.valid_months = {}
if 'pdf_report' not in st.session_state:
    st.session_state.pdf_report = None
# For resume capability
if 'month_analysis_results' not in st.session_state:
    st.session_state.month_analysis_results = {}
if 'failed_downloads' not in st.session_state:
    st.session_state.failed_downloads = []
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'download_complete' not in st.session_state:
    st.session_state.download_complete = False
if 'cloud_free_collection' not in st.session_state:
    st.session_state.cloud_free_collection = None
if 'processing_params' not in st.session_state:
    st.session_state.processing_params = None
# NEW: For robust processing and region selection
if 'selected_region_index' not in st.session_state:
    st.session_state.selected_region_index = 0
if 'processing_in_progress' not in st.session_state:
    st.session_state.processing_in_progress = False
if 'processing_config' not in st.session_state:
    st.session_state.processing_config = None
# For bulk download
if 'sentinel2_zip' not in st.session_state:
    st.session_state.sentinel2_zip = None


# =============================================================================
# Normalization function
# =============================================================================
def normalized(img):
    """Normalize image data to range [0, 1] - global normalization"""
    min_val = np.nanmin(img)
    max_val = np.nanmax(img)
    if max_val == min_val:
        return np.zeros_like(img)
    return (img - min_val) / (max_val - min_val + 1e-5)


# =============================================================================
# File Validation Functions
# =============================================================================
def validate_geotiff_file(file_path, expected_bands=1):
    """Validate that a GeoTIFF file is complete and readable."""
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
    """Download model from Google Drive"""
    try:
        correct_file_id= "1_8jOOSXnELA-xOGW0DKgRMo6RvnJYV5_"
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
    """Load the UNet++ model"""
    try:
        device = torch.device('cpu')
        model = smp.UnetPlusPlus(
            encoder_name='timm-efficientnet-b8',
            encoder_weights='imagenet',
            in_channels=12,
            classes=1,
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
        st.error(f"Error loading model: {str(e)}")
        return None, None


# =============================================================================
# Earth Engine Authentication
# =============================================================================
@st.cache_resource
def initialize_earth_engine():
    """Initialize Earth Engine"""
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
# GEE CLOUD MASKING + GAP-FILLING (Server-Side)
# =============================================================================

def create_cloud_free_collection(aoi, extended_start, extended_end, cloudy_pixel_percentage=10):
    """
    Create cloud-free collection with proper cloud masking.
    Matching JS: indexJoin + maskCloudAndShadow
    """
    # Get S2 SR collection
    s2_sr = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
             .filterBounds(aoi)
             .filterDate(extended_start, extended_end)
             .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloudy_pixel_percentage))
             .select(SPECTRAL_BANDS + ['SCL']))
    
    # Get cloud probability collection
    s2_cloud_prob = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
                     .filterBounds(aoi)
                     .filterDate(extended_start, extended_end))
    
    # Join collections (indexJoin) - matching JS exactly
    join_filter = ee.Filter.equals(leftField='system:index', rightField='system:index')
    joined = ee.Join.saveFirst('cloud_probability').apply(
        primary=s2_sr, secondary=s2_cloud_prob, condition=join_filter
    )
    
    # Convert joined FeatureCollection to ImageCollection and add cloud band
    def add_cloud_band(feature):
        # Cast feature to image
        img = ee.Image(feature)
        cloud_prob_img = ee.Image(img.get('cloud_probability'))
        return img.addBands(cloud_prob_img)
    
    s2_joined = ee.ImageCollection(joined.map(add_cloud_band))
    
    # Apply cloud masking (maskCloudAndShadow) - matching JS exactly
    def mask_cloud_and_shadow(img):
        # Get cloud probability band
        cloud_prob = img.select('probability')
        
        # Calculate CDI (Cloud Displacement Index)
        cdi = ee.Algorithms.Sentinel2.CDI(img)
        
        # Cloud mask: prob > 65 AND cdi < -0.5
        is_cloud = cloud_prob.gt(CLOUD_PROB_THRESHOLD).And(cdi.lt(CDI_THRESHOLD))
        
        # Dilate with 20m kernel, 2 iterations
        kernel = ee.Kernel.circle(radius=20, units='meters')
        cloud_dilated = is_cloud.focal_max(kernel=kernel, iterations=2)
        
        # Mask clouds and scale
        masked = img.updateMask(cloud_dilated.Not())
        scaled = masked.select(SPECTRAL_BANDS).multiply(0.0001).clip(aoi)
        
        return scaled.copyProperties(img, ['system:time_start'])
    
    return s2_joined.map(mask_cloud_and_shadow)



# =============================================================================
# Download Functions
# =============================================================================
def download_band_with_retry(image, band, aoi, output_path, scale=10):
    """Download a single band with retry mechanism."""
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
            # Get download URL
            url = image.select(band).getDownloadURL({
                'scale': scale, 'region': region, 'format': 'GEO_TIFF', 'bands': [band]
            })
            
            response = requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT)
            
            if response.status_code == 200:
                content_type = response.headers.get('content-type', '')
                if 'text/html' in content_type:
                    last_error = "GEE rate limit (HTML response)"
                    raise Exception(last_error)
                
                # Download to temp file
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
        
        # Cleanup
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
    """Download composite image to GeoTIFF with detailed error reporting."""
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


def download_monthly_image_v06(aoi, cloud_free_collection, month_info, temp_dir, 
                                scale=10, status_placeholder=None):
    """
    Download a single monthly composite with gap-filling.
    Creates composite fresh inside function (like v05) but with v06 cloud masking.
    
    This function:
    1. Creates monthly median from cloud-free collection
    2. Checks masked pixel percentage
    3. Applies gap-filling if 0% < masked <= 30%
    4. Downloads only if final masked == 0%
    
    Returns: (output_path, status, message)
    """
    month_name = month_info['month_name']
    month_index = month_info['month_index']
    origin = month_info['origin']
    
    output_file = os.path.join(temp_dir, f"sentinel2_{month_name}.tif")
    
    # Check cache first
    if os.path.exists(output_file):
        is_valid, msg = validate_geotiff_file(output_file, expected_bands=len(SPECTRAL_BANDS))
        if is_valid:
            if status_placeholder:
                status_placeholder.info(f"✅ {month_name} using cached file")
            return output_file, STATUS_COMPLETE, "Cached"
        else:
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
        
        # CHECK 1: No images
        if image_count == 0:
            return None, STATUS_NO_DATA, "No images available"
        
        if status_placeholder:
            status_placeholder.text(f"📥 {month_name}: Creating composite from {image_count} images...")
        
        # Create frequency map and median composite
        def create_valid_mask(img):
            return ee.Image(1).updateMask(img.select('B4').mask()).unmask(0).toInt()
        
        frequency = monthly_images.map(create_valid_mask).sum().toInt().rename('frequency')
        composite = monthly_images.median()
        
        # Calculate masked pixel statistics
        masked_stats = frequency.eq(0).reduceRegion(
            reducer=ee.Reducer.sum(), geometry=aoi, scale=10, maxPixels=1e13
        )
        total_stats = frequency.gte(0).reduceRegion(
            reducer=ee.Reducer.sum(), geometry=aoi, scale=10, maxPixels=1e13
        )
        
        masked_count = ee.Number(masked_stats.get('frequency')).getInfo()
        total_count = ee.Number(total_stats.get('frequency')).getInfo()
        
        if total_count == 0:
            return None, STATUS_NO_DATA, "No valid pixels"
        
        masked_percent = (masked_count / total_count) * 100
        
        # CHECK 2: Too many masked (> 30%)
        if masked_percent > MAX_MASKED_PERCENT_FOR_GAPFILL:
            return None, STATUS_SKIPPED, f"Masked {masked_percent:.1f}% > {MAX_MASKED_PERCENT_FOR_GAPFILL}%"
        
        # CHECK 3: No masked pixels - ready to download
        if masked_percent == 0:
            if status_placeholder:
                status_placeholder.text(f"📥 {month_name}: Complete (0% masked), downloading...")
            
            path = download_composite(composite, aoi, output_file, month_name, scale, status_placeholder)
            if path:
                return path, STATUS_COMPLETE, "Complete (0% masked)"
            else:
                return None, STATUS_REJECTED, "Download failed"
        
        # GAP-FILL: 0% < masked <= 30%
        if status_placeholder:
            status_placeholder.text(f"📥 {month_name}: Gap-filling ({masked_percent:.1f}% masked)...")
        
        gap_mask = frequency.eq(0)
        month_middle_millis = month_middle.millis()
        
        # M-1 and M+1 ranges
        m1_past_start = origin_date.advance(ee.Number(month_index).subtract(1), 'month')
        m1_past_end = month_start
        m1_future_start = month_end
        m1_future_end = origin_date.advance(ee.Number(month_index).add(2), 'month')
        
        # Collect and sort by time distance
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
        
        if candidate_count == 0:
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
            if status_placeholder:
                status_placeholder.text(f"📥 {month_name}: Gap-filled successfully, downloading...")
            
            path = download_composite(filled_composite, aoi, output_file, month_name, scale, status_placeholder)
            if path:
                return path, STATUS_COMPLETE, f"Complete after gap-fill (was {masked_percent:.1f}%)"
            else:
                return None, STATUS_REJECTED, "Download failed after gap-fill"
        else:
            still_masked_pct = (still_masked_count / total_count) * 100
            return None, STATUS_REJECTED, f"{still_masked_pct:.1f}% still masked after gap-fill"
        
    except Exception as e:
        return None, STATUS_NO_DATA, f"Error: {str(e)}"


# =============================================================================
# RGB Thumbnail Generation
# =============================================================================
def generate_rgb_thumbnail(image_path, month_name, max_size=256):
    """Generate RGB thumbnail from Sentinel-2 image."""
    try:
        with rasterio.open(image_path) as src:
            red = src.read(4)
            green = src.read(3)
            blue = src.read(2)
            
            rgb = np.stack([red, green, blue], axis=-1)
            rgb = np.nan_to_num(rgb, nan=0.0)
            
            def percentile_stretch(band, lower=2, upper=98):
                valid = band[band > 0]
                if len(valid) == 0:
                    return np.zeros_like(band, dtype=np.uint8)
                p_low = np.percentile(valid, lower)
                p_high = np.percentile(valid, upper)
                if p_high <= p_low:
                    p_high = p_low + 0.001
                stretched = np.clip((band - p_low) / (p_high - p_low), 0, 1)
                return (stretched * 255).astype(np.uint8)
            
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
    """Check if a patch has minimal nodata."""
    if np.any(np.isnan(patch)):
        return False
    if np.all(patch == 0):
        return False
    
    zero_percent = (np.sum(patch == 0) / patch.size) * 100
    if zero_percent > nodata_threshold_percent:
        return False
    
    if patch.ndim == 3:
        for band_idx in range(patch.shape[-1]):
            if np.all(patch[:, :, band_idx] == 0):
                return False
    
    return True


def get_patch_validity_mask(image_path, patch_size=224, nodata_threshold_percent=0):
    """Create a mask showing which patches are valid for a single image."""
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
                patch = patches[i, j, 0]
                validity_mask[i, j] = check_patch_validity(patch, nodata_threshold_percent)
        
        return validity_mask, (h, w), (n_patches_h, n_patches_w)
    except:
        return None, None, None


def find_common_valid_patches(downloaded_images, nodata_threshold_percent=0):
    """
    Find patches that are valid and EXCLUDE months that don't have all valid patches.
    
    NEW LOGIC:
    1. Calculate validity mask for each month
    2. Find the month with the MOST valid patches (reference)
    3. Use that reference mask as the "expected" valid patches
    4. Exclude any month that doesn't have ALL those patches valid
    5. Return the reference mask and list of valid months
    
    Returns: (validity_mask, original_size, valid_months_dict) or (None, None, None) if fails
    """
    st.info("🔍 Analyzing patch validity across all months...")
    
    month_names = sorted(downloaded_images.keys())
    
    if len(month_names) == 0:
        st.error("❌ No downloaded images to analyze!")
        return None, None, None
    
    # =========================================================================
    # STEP 1: Validate all images have the same dimensions
    # =========================================================================
    st.write("**Step 1: Validating image dimensions...**")
    
    dimensions = {}
    for month_name in month_names:
        image_path = downloaded_images[month_name]
        try:
            with rasterio.open(image_path) as src:
                h, w = src.height, src.width
                bands = src.count
                dimensions[month_name] = {'height': h, 'width': w, 'bands': bands}
        except Exception as e:
            st.error(f"❌ Cannot read {month_name}: {e}")
            return None, None, None
    
    # Check if all dimensions match
    first_month = month_names[0]
    reference_dim = dimensions[first_month]
    
    mismatched = []
    for month_name, dim in dimensions.items():
        if dim['height'] != reference_dim['height'] or dim['width'] != reference_dim['width']:
            mismatched.append(f"{month_name}: {dim['height']}x{dim['width']}")
    
    if mismatched:
        st.error(f"❌ **DIMENSION MISMATCH DETECTED!**")
        st.error(f"Reference ({first_month}): {reference_dim['height']}x{reference_dim['width']}")
        st.error(f"Mismatched: {', '.join(mismatched)}")
        st.warning("All months must have identical image dimensions for consistent patch analysis.")
        return None, None, None
    
    st.success(f"✅ All {len(month_names)} images have same dimensions: {reference_dim['height']}x{reference_dim['width']} ({reference_dim['bands']} bands)")
    
    # =========================================================================
    # STEP 2: Calculate patch grid and get validity mask for each month
    # =========================================================================
    h, w = reference_dim['height'], reference_dim['width']
    n_patches_h = int(np.ceil(h / PATCH_SIZE))
    n_patches_w = int(np.ceil(w / PATCH_SIZE))
    total_patches = n_patches_h * n_patches_w
    
    st.write(f"**Step 2: Patch grid**: {n_patches_h} x {n_patches_w} = **{total_patches} patches** per image")
    
    st.write("**Step 3: Calculating validity for each month...**")
    
    progress_bar = st.progress(0)
    
    # Store validity mask for each month
    month_validity_masks = {}
    month_valid_counts = {}
    
    for idx, month_name in enumerate(month_names):
        image_path = downloaded_images[month_name]
        validity_mask, orig_size, grid_size = get_patch_validity_mask(
            image_path, PATCH_SIZE, nodata_threshold_percent
        )
        
        if validity_mask is None:
            st.warning(f"⚠️ Could not analyze {month_name}")
            progress_bar.progress((idx + 1) / len(month_names))
            continue
        
        # Verify grid size matches expected
        if validity_mask.shape != (n_patches_h, n_patches_w):
            st.error(f"❌ {month_name}: Patch grid mismatch! Expected {(n_patches_h, n_patches_w)}, got {validity_mask.shape}")
            return None, None, None
        
        month_validity_masks[month_name] = validity_mask
        month_valid_counts[month_name] = np.sum(validity_mask)
        
        progress_bar.progress((idx + 1) / len(month_names))
    
    progress_bar.empty()
    
    if len(month_validity_masks) == 0:
        st.error("❌ Could not analyze any months!")
        return None, None, None
    
    # =========================================================================
    # STEP 4: Find the month with MAXIMUM valid patches (reference)
    # =========================================================================
    st.write("**Step 4: Finding reference mask (maximum valid patches)...**")
    
    # Find max valid patch count
    max_valid_count = max(month_valid_counts.values())
    
    # Find all months with max valid count (could be multiple)
    max_months = [mn for mn, count in month_valid_counts.items() if count == max_valid_count]
    
    st.info(f"📊 Maximum valid patches: **{max_valid_count}/{total_patches}** ({100*max_valid_count/total_patches:.1f}%)")
    st.write(f"   Months with max patches: {', '.join(max_months)}")
    
    # Use the first max month as reference
    reference_month = max_months[0]
    reference_mask = month_validity_masks[reference_month]
    
    # =========================================================================
    # STEP 5: Check which months have ALL reference patches valid
    # =========================================================================
    st.write("**Step 5: Filtering months that match reference mask...**")
    
    valid_months = {}
    excluded_months = {}
    
    for month_name, mask in month_validity_masks.items():
        # Check if this month has ALL patches that are valid in reference
        # A month is valid if: everywhere reference_mask is True, this mask is also True
        matches_reference = np.all(mask[reference_mask] == True)
        
        if matches_reference:
            valid_months[month_name] = downloaded_images[month_name]
            st.write(f"   ✅ {month_name}: {month_valid_counts[month_name]}/{total_patches} patches - **INCLUDED**")
        else:
            # Count how many reference patches are missing
            missing_count = np.sum(reference_mask & ~mask)
            excluded_months[month_name] = {
                'valid_count': month_valid_counts[month_name],
                'missing_count': missing_count
            }
            st.write(f"   ❌ {month_name}: {month_valid_counts[month_name]}/{total_patches} patches - **EXCLUDED** (missing {missing_count} patches)")
    
    # =========================================================================
    # STEP 6: Summary
    # =========================================================================
    st.divider()
    
    if len(valid_months) == 0:
        st.error("❌ No months match the reference mask!")
        st.warning("This shouldn't happen - at least the reference month should match.")
        return None, None, None
    
    st.success(f"✅ **{len(valid_months)}/{len(month_validity_masks)}** months have all {max_valid_count} valid patches")
    
    if excluded_months:
        with st.expander(f"🚫 Excluded Months ({len(excluded_months)})", expanded=True):
            for month_name, info in excluded_months.items():
                st.write(f"  • {month_name}: had {info['valid_count']} patches, missing {info['missing_count']} from reference")
    
    original_size = (h, w)
    
    return reference_mask, original_size, valid_months


def classify_image_with_mask(image_path, model, device, month_name, valid_mask, original_size):
    """
    Classify an image, only processing valid patches.
    
    IMPORTANT: Validates that image dimensions match expected original_size
    and that patch grid matches valid_mask shape.
    
    Returns: (binary_mask, probability_map, valid_count)
    """
    try:
        with rasterio.open(image_path) as src:
            img_data = src.read()
        
        img_for_patching = np.moveaxis(img_data, 0, -1)
        
        h, w, c = img_for_patching.shape
        
        # VALIDATION: Check image dimensions match expected size
        if original_size is not None:
            expected_h, expected_w = original_size
            if h != expected_h or w != expected_w:
                st.error(f"❌ {month_name}: Dimension mismatch! Expected {expected_h}x{expected_w}, got {h}x{w}")
                return None, None, 0
        
        new_h = int(np.ceil(h / PATCH_SIZE) * PATCH_SIZE)
        new_w = int(np.ceil(w / PATCH_SIZE) * PATCH_SIZE)
        
        # Calculate expected patch grid
        n_patches_h = new_h // PATCH_SIZE
        n_patches_w = new_w // PATCH_SIZE
        
        # VALIDATION: Check patch grid matches valid_mask
        if valid_mask.shape != (n_patches_h, n_patches_w):
            st.error(f"❌ {month_name}: Patch grid mismatch! Mask is {valid_mask.shape}, image needs {(n_patches_h, n_patches_w)}")
            return None, None, 0
        
        if h != new_h or w != new_w:
            padded_img = np.zeros((new_h, new_w, c), dtype=img_for_patching.dtype)
            padded_img[:h, :w, :] = img_for_patching
            img_for_patching = padded_img
        
        patches = patchify(img_for_patching, (PATCH_SIZE, PATCH_SIZE, c), step=PATCH_SIZE)
        n_patches_h, n_patches_w = patches.shape[0], patches.shape[1]
        
        classified_patches = np.zeros((n_patches_h, n_patches_w, PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
        probability_patches = np.zeros((n_patches_h, n_patches_w, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
        
        valid_count = 0
        for i in range(n_patches_h):
            for j in range(n_patches_w):
                if not valid_mask[i, j]:
                    continue
                
                patch = patches[i, j, 0]
                patch_normalized = normalized(patch)
                
                patch_tensor = torch.tensor(np.moveaxis(patch_normalized, -1, 0), dtype=torch.float32)
                patch_tensor = patch_tensor.unsqueeze(0)
                
                with torch.inference_mode():
                    prediction = model(patch_tensor)
                    prediction = torch.sigmoid(prediction).cpu()
                
                pred_np = prediction.squeeze().numpy()
                classified_patches[i, j] = (pred_np > 0.5).astype(np.uint8) * 255
                probability_patches[i, j] = pred_np  # Store probability values
                valid_count += 1
        
        reconstructed = unpatchify(classified_patches, (new_h, new_w))
        reconstructed = reconstructed[:original_size[0], :original_size[1]]
        
        reconstructed_prob = unpatchify(probability_patches, (new_h, new_w))
        reconstructed_prob = reconstructed_prob[:original_size[0], :original_size[1]]
        
        return reconstructed, reconstructed_prob, valid_count
    except:
        return None, None, 0


def get_valid_patch_bounds(valid_mask, patch_size=224, original_size=None):
    """
    Calculate the pixel bounds of valid patches.
    Returns: (row_start, row_end, col_start, col_end) in pixels
    """
    if valid_mask is None or not np.any(valid_mask):
        return None
    
    # Find rows and cols with valid patches
    valid_rows = np.any(valid_mask, axis=1)
    valid_cols = np.any(valid_mask, axis=0)
    
    # Get min/max patch indices
    row_indices = np.where(valid_rows)[0]
    col_indices = np.where(valid_cols)[0]
    
    if len(row_indices) == 0 or len(col_indices) == 0:
        return None
    
    min_row, max_row = row_indices[0], row_indices[-1]
    min_col, max_col = col_indices[0], col_indices[-1]
    
    # Convert to pixel coordinates
    row_start = min_row * patch_size
    row_end = (max_row + 1) * patch_size
    col_start = min_col * patch_size
    col_end = (max_col + 1) * patch_size
    
    # Clip to original size if provided
    if original_size is not None:
        row_end = min(row_end, original_size[0])
        col_end = min(col_end, original_size[1])
    
    return (row_start, row_end, col_start, col_end)


def create_pixel_mask_from_patches(valid_mask, patch_size=224, target_size=None):
    """
    Create a pixel-level mask from patch-level validity mask.
    Each patch's validity is expanded to cover all its pixels.
    
    Returns: 2D boolean array of shape target_size (or expanded patch grid size)
    """
    if valid_mask is None:
        return None
    
    n_patches_h, n_patches_w = valid_mask.shape
    
    # Create pixel mask by repeating each patch value
    pixel_mask = np.repeat(np.repeat(valid_mask, patch_size, axis=0), patch_size, axis=1)
    
    # Crop to target size if provided
    if target_size is not None:
        pixel_mask = pixel_mask[:target_size[0], :target_size[1]]
    
    return pixel_mask


def generate_thumbnails(image_path, classification_mask, month_name, valid_mask=None, original_size=None, max_size=256):
    """
    Generate both RGB and classification thumbnails.
    If valid_mask is provided:
    1. Crops both images to the valid patch bounding box
    2. Masks out invalid patches in RGB (shows black where patches are invalid)
    This ensures RGB and classification show exactly the same valid regions.
    """
    try:
        # Get crop bounds and pixel mask from valid_mask
        crop_bounds = None
        pixel_mask = None
        if valid_mask is not None:
            crop_bounds = get_valid_patch_bounds(valid_mask, PATCH_SIZE, original_size)
            pixel_mask = create_pixel_mask_from_patches(valid_mask, PATCH_SIZE, original_size)
        
        # Read and process RGB
        with rasterio.open(image_path) as src:
            red = src.read(4)
            green = src.read(3)
            blue = src.read(2)
            
            rgb = np.stack([red, green, blue], axis=-1)
            rgb = np.nan_to_num(rgb, nan=0.0)
            
            def percentile_stretch(band, lower=2, upper=98):
                valid = band[band > 0]
                if len(valid) == 0:
                    return np.zeros_like(band, dtype=np.uint8)
                p_low = np.percentile(valid, lower)
                p_high = np.percentile(valid, upper)
                if p_high <= p_low:
                    p_high = p_low + 0.001
                stretched = np.clip((band - p_low) / (p_high - p_low), 0, 1)
                return (stretched * 255).astype(np.uint8)
            
            rgb_uint8 = np.zeros_like(rgb, dtype=np.uint8)
            for i in range(3):
                rgb_uint8[:, :, i] = percentile_stretch(rgb[:, :, i])
        
        # Apply pixel mask to RGB (set invalid patches to black)
        if pixel_mask is not None:
            # Expand mask to 3 channels
            pixel_mask_3ch = np.stack([pixel_mask, pixel_mask, pixel_mask], axis=-1)
            rgb_uint8 = np.where(pixel_mask_3ch, rgb_uint8, 0)
        
        # Crop both images if bounds are available
        if crop_bounds is not None:
            row_start, row_end, col_start, col_end = crop_bounds
            rgb_cropped = rgb_uint8[row_start:row_end, col_start:col_end, :]
            class_cropped = classification_mask[row_start:row_end, col_start:col_end]
        else:
            rgb_cropped = rgb_uint8
            class_cropped = classification_mask
        
        # Create PIL images
        pil_rgb = Image.fromarray(rgb_cropped, mode='RGB')
        pil_class = Image.fromarray(class_cropped.astype(np.uint8))
        
        # Resize if needed
        h, w = pil_rgb.size[1], pil_rgb.size[0]
        if h > max_size or w > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            pil_rgb = pil_rgb.resize((new_w, new_h), Image.LANCZOS)
            pil_class = pil_class.resize((new_w, new_h), Image.NEAREST)
        
        # Calculate building stats from cropped region (only valid pixels)
        building_pixels = np.sum(class_cropped > 0)
        total_pixels = class_cropped.shape[0] * class_cropped.shape[1]
        
        return {
            'rgb_image': pil_rgb,
            'classification_image': pil_class,
            'month_name': month_name,
            'original_size': classification_mask.shape,
            'cropped_size': class_cropped.shape,
            'building_pixels': building_pixels,
            'total_pixels': total_pixels
        }
    except Exception as e:
        st.warning(f"Error generating thumbnails for {month_name}: {e}")
        return None


# =============================================================================
# MAIN PROCESSING PIPELINE WITH RESUME CAPABILITY
# =============================================================================
def process_timeseries(aoi, start_date, end_date, model, device,
                       cloudy_pixel_percentage=10, scale=10, nodata_threshold_percent=5,
                       resume=False):
    """Main processing pipeline - v05 style: combined analysis + download."""
    try:
        if st.session_state.current_temp_dir is None or not os.path.exists(st.session_state.current_temp_dir):
            st.session_state.current_temp_dir = tempfile.mkdtemp()
        temp_dir = st.session_state.current_temp_dir
        
        start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        total_months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)
        
        st.info(f"📅 Processing {total_months} months | 📁 {temp_dir}")
        
        # Extended date range (M-1 to M+1) for gap-filling
        extended_start = (start_dt - datetime.timedelta(days=31)).strftime('%Y-%m-%d')
        extended_end = (end_dt + datetime.timedelta(days=31)).strftime('%Y-%m-%d')
        
        # =====================================================================
        # PHASE 1: Create cloud-free collection
        # =====================================================================
        st.header("Phase 1: Cloud-Free Collection")
        st.info(f"☁️ Cloud mask: prob > {CLOUD_PROB_THRESHOLD}, CDI < {CDI_THRESHOLD}")
        
        cloud_free_collection = create_cloud_free_collection(
            aoi, extended_start, extended_end, cloudy_pixel_percentage
        )
        
        # =====================================================================
        # PHASE 2: Download all months (v05 style - combined analysis + download)
        # =====================================================================
        st.header("Phase 2: Download & Gap-Fill")
        
        downloaded_images = {}
        month_statuses = {}
        
        # Prepare month info list FIRST - so we know which months are in current range
        month_infos = []
        current_range_months = set()
        for month_index in range(total_months):
            # Proper month calculation: add months to start date
            year = start_dt.year + (start_dt.month - 1 + month_index) // 12
            month = (start_dt.month - 1 + month_index) % 12 + 1
            month_name = f"{year}-{month:02d}"
            month_infos.append({
                'month_name': month_name,
                'month_index': month_index,
                'origin': start_date
            })
            current_range_months.add(month_name)
        
        # Show expected months for clarity
        st.info(f"📅 Expected months: {month_infos[0]['month_name']} to {month_infos[-1]['month_name']} ({len(month_infos)} months, end date is EXCLUSIVE)")
        
        # Check existing cached downloads (ONLY for months in current range)
        if resume and st.session_state.downloaded_images:
            for month_name, path in st.session_state.downloaded_images.items():
                # Only consider months in current date range
                if month_name not in current_range_months:
                    continue
                    
                if os.path.exists(path):
                    is_valid, _ = validate_geotiff_file(path, len(SPECTRAL_BANDS))
                    if is_valid:
                        downloaded_images[month_name] = path
                        month_statuses[month_name] = {'status': STATUS_COMPLETE, 'message': 'Cached'}
            
            if downloaded_images:
                st.info(f"🔄 Found {len(downloaded_images)} cached downloads")
                for mn in sorted(downloaded_images.keys()):
                    st.write(f"🟢 **{mn}**: complete (cached)")
        
        # Restore previous month statuses (ONLY for months in current range that aren't downloaded)
        if resume and st.session_state.month_analysis_results:
            for month_name, status_info in st.session_state.month_analysis_results.items():
                # Only consider months in current date range
                if month_name not in current_range_months:
                    continue
                    
                if month_name not in month_statuses:
                    month_statuses[month_name] = status_info
                    status = status_info.get('status', 'unknown')
                    message = status_info.get('message', '')
                    icon = {"no_data": "⚫", "skipped": "🟡", "complete": "🟢", "rejected": "🔴"}.get(status, "❓")
                    st.write(f"{icon} **{month_name}**: {status} (cached) - {message}")
        
        # Process remaining months (not in downloaded_images AND not already processed with status)
        months_to_process = [m for m in month_infos 
                            if m['month_name'] not in downloaded_images 
                            and m['month_name'] not in month_statuses]
        
        if months_to_process:
            st.info(f"📥 {len(months_to_process)} months to process: {', '.join([m['month_name'] for m in months_to_process])}")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, month_info in enumerate(months_to_process):
                month_name = month_info['month_name']
                
                # Safeguard: skip if already processed (shouldn't happen, but just in case)
                if month_name in month_statuses:
                    st.warning(f"⚠️ {month_name} already processed, skipping...")
                    progress_bar.progress((idx + 1) / len(months_to_process))
                    continue
                
                # Download with v06 cloud masking + gap-filling (all in one function)
                path, status, message = download_monthly_image_v06(
                    aoi=aoi,
                    cloud_free_collection=cloud_free_collection,
                    month_info=month_info,
                    temp_dir=temp_dir,
                    scale=scale,
                    status_placeholder=status_text
                )
                
                # Update both local and session state immediately
                month_statuses[month_name] = {'status': status, 'message': message}
                st.session_state.month_analysis_results[month_name] = {'status': status, 'message': message}
                
                icon = {"no_data": "⚫", "skipped": "🟡", "complete": "🟢", "rejected": "🔴"}.get(status, "❓")
                st.write(f"{icon} **{month_name}**: {status} - {message}")
                
                if path:
                    downloaded_images[month_name] = path
                    st.session_state.downloaded_images[month_name] = path
                
                progress_bar.progress((idx + 1) / len(months_to_process))
            
            progress_bar.empty()
            status_text.empty()
        
        # Summary
        st.divider()
        status_counts = {s: sum(1 for ms in month_statuses.values() if ms['status'] == s) 
                         for s in [STATUS_NO_DATA, STATUS_SKIPPED, STATUS_COMPLETE, STATUS_REJECTED]}
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("✅ Complete", status_counts[STATUS_COMPLETE])
        col2.metric("🔴 Rejected", status_counts[STATUS_REJECTED])
        col3.metric("🟡 Skipped", status_counts[STATUS_SKIPPED])
        col4.metric("⚫ No Data", status_counts[STATUS_NO_DATA])
        
        # Update session state
        failed_downloads = [mn for mn, ms in month_statuses.items() if ms['status'] != STATUS_COMPLETE]
        st.session_state.failed_downloads = failed_downloads
        st.session_state.month_analysis_results = month_statuses
        
        if not downloaded_images:
            st.error("❌ No images downloaded!")
            return []
        
        st.success(f"✅ Downloaded {len(downloaded_images)}/{total_months} months")
        
        # =====================================================================
        # PHASE 3: Patch validity - Find MAX patches and exclude non-conforming months
        # =====================================================================
        st.header("Phase 3: Patch Validity Analysis")
        
        valid_mask, original_size, valid_months = find_common_valid_patches(downloaded_images, nodata_threshold_percent)
        
        if valid_mask is None or valid_months is None:
            return []
        
        st.session_state.valid_patches_mask = valid_mask
        st.session_state.valid_months = valid_months  # Save for PDF report generation
        
        # IMPORTANT: Clear processed_months cache when valid_mask changes
        # This ensures all thumbnails are generated with the same mask
        if st.session_state.processed_months:
            st.info("🔄 Clearing thumbnail cache to ensure consistency...")
        st.session_state.processed_months = {}
        
        # Show which months were excluded
        excluded_count = len(downloaded_images) - len(valid_months)
        if excluded_count > 0:
            st.warning(f"⚠️ {excluded_count} months excluded due to missing patches")
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.imshow(valid_mask, cmap='RdYlGn', vmin=0, vmax=1)
        ax.set_title(f"Reference Valid Patches ({np.sum(valid_mask)} patches)")
        st.pyplot(fig)
        plt.close()
        
        # =====================================================================
        # PHASE 4: Classification (only valid months, with cache)
        # =====================================================================
        st.header("Phase 4: Classification")
        
        st.info(f"🧠 Classifying **{len(valid_months)}** months (excluded {excluded_count} with missing patches)")
        
        thumbnails = []
        month_names = sorted(valid_months.keys())  # Use valid_months, not downloaded_images
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, month_name in enumerate(month_names):
            # Check cache
            if month_name in st.session_state.processed_months:
                thumbnails.append(st.session_state.processed_months[month_name])
                progress_bar.progress((idx + 1) / len(month_names))
                continue
            
            status_text.text(f"🧠 {month_name} ({idx+1}/{len(month_names)})...")
            
            mask, prob_map, valid_count = classify_image_with_mask(
                valid_months[month_name], model, device, month_name, valid_mask, original_size
            )
            
            if mask is not None:
                thumb = generate_thumbnails(
                    valid_months[month_name], mask, month_name,
                    valid_mask=valid_mask, original_size=original_size
                )
                if thumb:
                    thumb['valid_patches'] = valid_count
                    # Add gap-fill info from status
                    ms = month_statuses.get(month_name, {})
                    thumb['gap_filled'] = 'gap-fill' in ms.get('message', '').lower()
                    
                    thumbnails.append(thumb)
                    st.session_state.processed_months[month_name] = thumb
            
            progress_bar.progress((idx + 1) / len(month_names))
        
        progress_bar.empty()
        status_text.empty()
        
        st.success(f"✅ Classified {len(thumbnails)} months!")
        
        # Filter out images with abnormally low building percentage (likely haze/snow)
        if len(thumbnails) >= 2:
            building_pcts = [(t['building_pixels'] / t['total_pixels']) * 100 for t in thumbnails]
            median_pct = np.median(building_pcts)
            
            filtered_thumbnails = []
            for t in thumbnails:
                pct = (t['building_pixels'] / t['total_pixels']) * 100
                if pct < (median_pct - 8.0):
                    print(f"🚫 Excluded {t['month_name']}: {pct:.1f}% buildings (median={median_pct:.1f}%) - likely haze or snow")
                    # Also remove from valid_months
                    if t['month_name'] in st.session_state.valid_months:
                        del st.session_state.valid_months[t['month_name']]
                else:
                    filtered_thumbnails.append(t)
            
            thumbnails = filtered_thumbnails
        
        return thumbnails
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return []


# =============================================================================
# Display Thumbnails
# =============================================================================
def get_image_download_data(image_path, month_name):
    """Read GeoTIFF file and return bytes for download."""
    try:
        with open(image_path, 'rb') as f:
            return f.read()
    except Exception as e:
        st.warning(f"Error reading {month_name}: {e}")
        return None


# def create_sentinel2_images_zip(downloaded_images):
#     """
#     Create a ZIP file containing ALL downloaded Sentinel-2 composite images.
#     Includes both valid-patch and non-valid-patch months.
    
#     Args:
#         downloaded_images: Dict of {month_name: image_path} - ALL downloaded images
    
#     Returns:
#         BytesIO buffer containing the ZIP file
#     """
#     import zipfile
    
#     zip_buffer = BytesIO()
    
#     with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
#         for month_name, image_path in sorted(downloaded_images.items()):
#             if os.path.exists(image_path):
#                 # Add file to zip with a clear filename
#                 zip_file.write(image_path, f"sentinel2_{month_name}_12bands.tif")
    
#     zip_buffer.seek(0)
    # return zip_buffer
def create_sentinel2_images_zip(downloaded_images, selected_months=None):
    """
    Create a ZIP file containing selected Sentinel-2 composite images.
    
    Args:
        downloaded_images: Dict of {month_name: image_path} - ALL downloaded images
        selected_months: List of month names to include (None = all months)
    
    Returns:
        BytesIO buffer containing the ZIP file
    """
    import zipfile
    
    zip_buffer = BytesIO()
    
    # Filter months if selection provided
    months_to_zip = downloaded_images
    if selected_months is not None:
        months_to_zip = {k: v for k, v in downloaded_images.items() if k in selected_months}
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for month_name, image_path in sorted(months_to_zip.items()):
            if os.path.exists(image_path):
                # Add file to zip with a clear filename
                zip_file.write(image_path, f"sentinel2_{month_name}_12bands.tif")
    
    zip_buffer.seek(0)
    return zip_buffer


def display_thumbnails(thumbnails, valid_months=None, downloaded_images=None):
    if not thumbnails:
        return
    
    # Add "Download All Sentinel-2 Images" section at the top
    if downloaded_images and len(downloaded_images) > 0:
        st.subheader("📦 Bulk Download Sentinel-2 Images")
        
        # Categorize months
        included_months = set(valid_months.keys()) if valid_months else set()
        excluded_months = set(downloaded_images.keys()) - included_months
        
        num_included = len(included_months)
        num_excluded = len(excluded_months)
        num_total = len(downloaded_images)
        
        # Show statistics
        col1, col2, col3 = st.columns(3)
        col1.metric("✅ Included (Valid Patches)", num_included)
        col2.metric("❌ Excluded (Missing Patches)", num_excluded)
        col3.metric("📊 Total Downloaded", num_total)
        
        st.divider()
        
        # Month selection interface
        st.write("**Select months to download:**")
        
        # Selection mode
        selection_mode = st.radio(
            "Selection Mode:",
            ["All Months", "Included Only", "Excluded Only", "Custom Selection"],
            horizontal=True,
            help="Choose which months to include in the ZIP file"
        )
        
        selected_months = None
        
        if selection_mode == "All Months":
            selected_months = list(downloaded_images.keys())
            st.info(f"📦 Will download **all {num_total} months** (included + excluded)")
            
        elif selection_mode == "Included Only":
            selected_months = sorted(included_months)
            st.success(f"✅ Will download **{num_included} included months** (with valid patches)")
            with st.expander("📋 View included months"):
                st.write(", ".join(selected_months))
            
        elif selection_mode == "Excluded Only":
            selected_months = sorted(excluded_months)
            st.warning(f"❌ Will download **{num_excluded} excluded months** (with missing patches)")
            with st.expander("📋 View excluded months"):
                st.write(", ".join(selected_months))
            
        elif selection_mode == "Custom Selection":
            st.write("**Included Months (Valid Patches):**")
            selected_included = []
            if included_months:
                cols_included = st.columns(min(4, len(included_months)))
                for idx, month in enumerate(sorted(included_months)):
                    with cols_included[idx % len(cols_included)]:
                        if st.checkbox(month, value=True, key=f"inc_{month}"):
                            selected_included.append(month)
            
            st.write("**Excluded Months (Missing Patches):**")
            selected_excluded = []
            if excluded_months:
                cols_excluded = st.columns(min(4, len(excluded_months)))
                for idx, month in enumerate(sorted(excluded_months)):
                    with cols_excluded[idx % len(cols_excluded)]:
                        if st.checkbox(month, value=False, key=f"exc_{month}"):
                            selected_excluded.append(month)
            
            selected_months = selected_included + selected_excluded
            st.info(f"📦 Selected **{len(selected_months)}** months: {len(selected_included)} included + {len(selected_excluded)} excluded")
        
        # Download button
        st.divider()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            prepare_clicked = st.button(
                f"📦 Prepare ZIP ({len(selected_months)} months)", 
                type="primary",
                disabled=not selected_months
            )
            
            if prepare_clicked:
                with st.spinner(f"Creating ZIP file with {len(selected_months)} Sentinel-2 images..."):
                    zip_buffer = create_sentinel2_images_zip(downloaded_images, selected_months)
                    st.session_state.sentinel2_zip = zip_buffer.getvalue()
                    st.session_state.sentinel2_zip_info = {
                        'num_months': len(selected_months),
                        'month_list': selected_months,
                        'selection_mode': selection_mode
                    }
                    st.success("✅ ZIP file ready!")
        
        with col2:
            if 'sentinel2_zip' in st.session_state and st.session_state.sentinel2_zip:
                zip_info = st.session_state.get('sentinel2_zip_info', {})
                num_in_zip = zip_info.get('num_months', num_total)
                mode_label = zip_info.get('selection_mode', 'Custom')
                
                st.download_button(
                    label=f"⬇️ Download Sentinel-2 Images ({num_in_zip} months)",
                    data=st.session_state.sentinel2_zip,
                    file_name=f"sentinel2_{mode_label.lower().replace(' ', '_')}_{num_in_zip}months.zip",
                    mime="application/zip",
                    help=f"ZIP contains {num_in_zip} Sentinel-2 12-band composite images"
                )
                
                # Show what's in the ZIP
                if zip_info.get('month_list'):
                    with st.expander(f"📋 Contents ({num_in_zip} months)"):
                        months_display = ", ".join(sorted(zip_info['month_list']))
                        st.text(months_display)
        
        st.divider()
    
    # Rest of the thumbnail display code remains the same
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
                    suffix = " (filled)" if t.get('gap_filled') else ""
                    cols[j*2].image(t['rgb_image'], caption=f"{t['month_name']} RGB{suffix}")
                    # Add download button under RGB
                    if valid_months and t['month_name'] in valid_months:
                        image_path = valid_months[t['month_name']]
                        image_data = get_image_download_data(image_path, t['month_name'])
                        if image_data:
                            cols[j*2].download_button(
                                label=f"⬇️ Download {t['month_name']}",
                                data=image_data,
                                file_name=f"sentinel2_{t['month_name']}_12bands.tif",
                                mime="image/tiff",
                                key=f"dl_sidebyside_{t['month_name']}"
                            )
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
                        # Add download button under RGB images only
                        if mode == "RGB" and valid_months and t['month_name'] in valid_months:
                            image_path = valid_months[t['month_name']]
                            image_data = get_image_download_data(image_path, t['month_name'])
                            if image_data:
                                cols[c].download_button(
                                    label=f"⬇️ Download",
                                    data=image_data,
                                    file_name=f"sentinel2_{t['month_name']}_12bands.tif",
                                    mime="image/tiff",
                                    key=f"dl_rgb_{t['month_name']}"
                                )


def display_thumbnails(thumbnails, valid_months=None, downloaded_images=None):
    if not thumbnails:
        return
    
    # Add "Download All Sentinel-2 Images" section at the top
    if downloaded_images and len(downloaded_images) > 0:
        st.subheader("📦 Bulk Download Sentinel-2 Images")
        
        # Categorize months
        included_months = set(valid_months.keys()) if valid_months else set()
        excluded_months = set(downloaded_images.keys()) - included_months
        
        num_included = len(included_months)
        num_excluded = len(excluded_months)
        num_total = len(downloaded_images)
        
        # Show statistics
        col1, col2, col3 = st.columns(3)
        col1.metric("✅ Included (Valid Patches)", num_included)
        col2.metric("❌ Excluded (Missing Patches)", num_excluded)
        col3.metric("📊 Total Downloaded", num_total)
        
        st.divider()
        
        # Month selection interface
        st.write("**Select months to download:**")
        
        # Selection mode
        selection_mode = st.radio(
            "Selection Mode:",
            ["All Months", "Included Only", "Excluded Only", "Custom Selection"],
            horizontal=True,
            help="Choose which months to include in the ZIP file"
        )
        
        selected_months = None
        
        if selection_mode == "All Months":
            selected_months = list(downloaded_images.keys())
            st.info(f"📦 Will download **all {num_total} months** (included + excluded)")
            
        elif selection_mode == "Included Only":
            selected_months = sorted(included_months)
            st.success(f"✅ Will download **{num_included} included months** (with valid patches)")
            with st.expander("📋 View included months"):
                st.write(", ".join(selected_months))
            
        elif selection_mode == "Excluded Only":
            selected_months = sorted(excluded_months)
            st.warning(f"❌ Will download **{num_excluded} excluded months** (with missing patches)")
            with st.expander("📋 View excluded months"):
                st.write(", ".join(selected_months))
            
        elif selection_mode == "Custom Selection":
            st.write("**Included Months (Valid Patches):**")
            selected_included = []
            if included_months:
                cols_included = st.columns(min(4, len(included_months)))
                for idx, month in enumerate(sorted(included_months)):
                    with cols_included[idx % len(cols_included)]:
                        if st.checkbox(month, value=True, key=f"inc_{month}"):
                            selected_included.append(month)
            
            st.write("**Excluded Months (Missing Patches):**")
            selected_excluded = []
            if excluded_months:
                cols_excluded = st.columns(min(4, len(excluded_months)))
                for idx, month in enumerate(sorted(excluded_months)):
                    with cols_excluded[idx % len(cols_excluded)]:
                        if st.checkbox(month, value=False, key=f"exc_{month}"):
                            selected_excluded.append(month)
            
            selected_months = selected_included + selected_excluded
            st.info(f"📦 Selected **{len(selected_months)}** months: {len(selected_included)} included + {len(selected_excluded)} excluded")
        
        # Download button
        st.divider()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            prepare_clicked = st.button(
                f"📦 Prepare ZIP ({len(selected_months)} months)", 
                type="primary",
                disabled=not selected_months
            )
            
            if prepare_clicked:
                with st.spinner(f"Creating ZIP file with {len(selected_months)} Sentinel-2 images..."):
                    zip_buffer = create_sentinel2_images_zip(downloaded_images, selected_months)
                    st.session_state.sentinel2_zip = zip_buffer.getvalue()
                    st.session_state.sentinel2_zip_info = {
                        'num_months': len(selected_months),
                        'month_list': selected_months,
                        'selection_mode': selection_mode
                    }
                    st.success("✅ ZIP file ready!")
        
        with col2:
            if 'sentinel2_zip' in st.session_state and st.session_state.sentinel2_zip:
                zip_info = st.session_state.get('sentinel2_zip_info', {})
                num_in_zip = zip_info.get('num_months', num_total)
                mode_label = zip_info.get('selection_mode', 'Custom')
                
                st.download_button(
                    label=f"⬇️ Download Sentinel-2 Images ({num_in_zip} months)",
                    data=st.session_state.sentinel2_zip,
                    file_name=f"sentinel2_{mode_label.lower().replace(' ', '_')}_{num_in_zip}months.zip",
                    mime="application/zip",
                    help=f"ZIP contains {num_in_zip} Sentinel-2 12-band composite images"
                )
                
                # Show what's in the ZIP
                if zip_info.get('month_list'):
                    with st.expander(f"📋 Contents ({num_in_zip} months)"):
                        months_display = ", ".join(sorted(zip_info['month_list']))
                        st.text(months_display)
        
        st.divider()
    
    # Rest of the thumbnail display code remains the same
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
                    suffix = " (filled)" if t.get('gap_filled') else ""
                    cols[j*2].image(t['rgb_image'], caption=f"{t['month_name']} RGB{suffix}")
                    # Add download button under RGB
                    if valid_months and t['month_name'] in valid_months:
                        image_path = valid_months[t['month_name']]
                        image_data = get_image_download_data(image_path, t['month_name'])
                        if image_data:
                            cols[j*2].download_button(
                                label=f"⬇️ Download {t['month_name']}",
                                data=image_data,
                                file_name=f"sentinel2_{t['month_name']}_12bands.tif",
                                mime="image/tiff",
                                key=f"dl_sidebyside_{t['month_name']}"
                            )
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
                        # Add download button under RGB images only
                        if mode == "RGB" and valid_months and t['month_name'] in valid_months:
                            image_path = valid_months[t['month_name']]
                            image_data = get_image_download_data(image_path, t['month_name'])
                            if image_data:
                                cols[c].download_button(
                                    label=f"⬇️ Download",
                                    data=image_data,
                                    file_name=f"sentinel2_{t['month_name']}_12bands.tif",
                                    mime="image/tiff",
                                    key=f"dl_rgb_{t['month_name']}"
                                )


def display_thumbnails(thumbnails, valid_months=None, downloaded_images=None):
    if not thumbnails:
        return
    
    # Add "Download All Sentinel-2 Images" button at the top
    # Use downloaded_images (all images) instead of valid_months (only valid patches)
    if downloaded_images and len(downloaded_images) > 0:
        st.subheader("📦 Bulk Download")
        
        # Show statistics
        num_valid = len(valid_months) if valid_months else 0
        num_total = len(downloaded_images)
        num_excluded = num_total - num_valid
        
        st.info(f"📊 **Total downloaded images**: {num_total} | **Valid patches**: {num_valid} | **Excluded**: {num_excluded}")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            if st.button("📦 Prepare ZIP of All Images", type="primary"):
                with st.spinner(f"Creating ZIP file with {num_total} Sentinel-2 images..."):
                    zip_buffer = create_sentinel2_images_zip(downloaded_images)
                    st.session_state.sentinel2_zip = zip_buffer.getvalue()
                    st.success("✅ ZIP file ready!")
        
        with col2:
            if 'sentinel2_zip' in st.session_state and st.session_state.sentinel2_zip:
                st.download_button(
                    label=f"⬇️ Download ALL Sentinel-2 Images ({num_total} months)",
                    data=st.session_state.sentinel2_zip,
                    file_name=f"sentinel2_all_months_{num_total}images.zip",
                    mime="application/zip",
                    help=f"ZIP contains ALL {num_total} downloaded Sentinel-2 12-band composite images (both valid-patch and non-valid-patch months, before classification)"
                )
        
        st.divider()
    
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
                    suffix = " (filled)" if t.get('gap_filled') else ""
                    cols[j*2].image(t['rgb_image'], caption=f"{t['month_name']} RGB{suffix}")
                    # Add download button under RGB
                    if valid_months and t['month_name'] in valid_months:
                        image_path = valid_months[t['month_name']]
                        image_data = get_image_download_data(image_path, t['month_name'])
                        if image_data:
                            cols[j*2].download_button(
                                label=f"⬇️ Download {t['month_name']}",
                                data=image_data,
                                file_name=f"sentinel2_{t['month_name']}_12bands.tif",
                                mime="image/tiff",
                                key=f"dl_sidebyside_{t['month_name']}"
                            )
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
                        # Add download button under RGB images only
                        if mode == "RGB" and valid_months and t['month_name'] in valid_months:
                            image_path = valid_months[t['month_name']]
                            image_data = get_image_download_data(image_path, t['month_name'])
                            if image_data:
                                cols[c].download_button(
                                    label=f"⬇️ Download",
                                    data=image_data,
                                    file_name=f"sentinel2_{t['month_name']}_12bands.tif",
                                    mime="image/tiff",
                                    key=f"dl_rgb_{t['month_name']}"
                                )


# =============================================================================
# Main Application
# =============================================================================
def main():
    main_classification_tab()


def main_classification_tab():
    """Main classification tab content"""
    st.title("🏗️ Building Classification v06")
    st.markdown("""
    | Status | Condition | Download? |
    |--------|-----------|-----------|
    | `no_data` | No images | ❌ |
    | `skipped` | masked > 30% | ❌ |
    | `complete` | masked == 0% | ✅ |
    | `rejected` | masked > 0% after gap-fill | ❌ |
    """)
    
    ee_ok, ee_msg = initialize_earth_engine()
    if not ee_ok:
        st.error(ee_msg)
        st.stop()
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
    cloudy_pct = st.sidebar.slider(
        "Max Cloud % (metadata)", 0, 50, 10, 5,
        help="GEE: Filter images by CLOUDY_PIXEL_PERCENTAGE metadata before cloud masking",
        disabled=st.session_state.processing_in_progress
    )
    # Nodata % is now fixed at 0 (removed slider per user request)
    nodata_pct = 0
    
    # Cache Status
    st.sidebar.header("🗂️ Cache Status")
    
    cache_info = []
    if st.session_state.month_analysis_results:
        cache_info.append(f"📊 {len(st.session_state.month_analysis_results)} months analyzed")
    if st.session_state.downloaded_images:
        cache_info.append(f"📥 {len(st.session_state.downloaded_images)} images downloaded")
    if st.session_state.processed_months:
        cache_info.append(f"🧠 {len(st.session_state.processed_months)} months classified")
    
    if cache_info:
        for info in cache_info:
            st.sidebar.success(info)
    else:
        st.sidebar.info("No cached data")
    
    if st.session_state.failed_downloads:
        st.sidebar.warning(f"❌ Failed: {', '.join(st.session_state.failed_downloads)}")
    
    # Processing status indicator
    if st.session_state.processing_in_progress:
        st.sidebar.error("⏳ Processing in progress...")
    
    if st.sidebar.button("🗑️ Clear All Cache", disabled=st.session_state.processing_in_progress):
        for key in ['processed_months', 'downloaded_images', 'classification_thumbnails', 
                    'valid_patches_mask', 'valid_months', 'current_temp_dir', 'month_analysis_results',
                    'failed_downloads', 'analysis_complete', 'download_complete',
                    'processing_params', 'processing_config', 'pdf_report',
                    'sentinel2_zip']:
            if key in st.session_state:
                if isinstance(st.session_state[key], dict):
                    st.session_state[key] = {}
                elif isinstance(st.session_state[key], list):
                    st.session_state[key] = []
                else:
                    st.session_state[key] = None
        st.session_state.processing_complete = False
        st.session_state.processing_in_progress = False
        st.rerun()
    
    # Stop processing button
    if st.session_state.processing_in_progress:
        if st.sidebar.button("🛑 Stop Processing", type="primary"):
            st.session_state.processing_in_progress = False
            st.session_state.processing_config = None
            st.warning("⚠️ Processing stopped. You can resume later.")
            st.rerun()
    
    # Region Selection
    st.header("1️⃣ Region")
    
    # Show map when not actively processing (map should be visible even after processing completes)
    if not st.session_state.processing_in_progress:
        m = folium.Map(location=[35.6892, 51.3890], zoom_start=8)
        plugins.Draw(export=True, position='topleft', draw_options={
            'polyline': False, 'rectangle': True, 'polygon': True,
            'circle': False, 'marker': False, 'circlemarker': False
        }).add_to(m)
        folium.TileLayer(tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
                         attr='Google', name='Satellite').add_to(m)
        folium.LayerControl().add_to(m)
        
        # Use a unique key to force map refresh when cache is cleared
        map_key = f"region_map_{len(st.session_state.drawn_polygons)}_{st.session_state.processing_complete}"
        map_data = st_folium(m, width=800, height=500, key=map_key)
        
        if map_data and map_data.get('last_active_drawing'):
            geom = map_data['last_active_drawing'].get('geometry', {})
            if geom.get('type') == 'Polygon':
                st.session_state.last_drawn_polygon = Polygon(geom['coordinates'][0])
                st.success(f"✅ Region selected")
        
        if st.button("💾 Save Region"):
            if st.session_state.last_drawn_polygon:
                # Check if not already saved
                is_duplicate = False
                for existing in st.session_state.drawn_polygons:
                    if existing.equals(st.session_state.last_drawn_polygon):
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    st.session_state.drawn_polygons.append(st.session_state.last_drawn_polygon)
                    st.success("✅ Region saved!")
                    st.rerun()
                else:
                    st.warning("⚠️ This region is already saved")
            else:
                st.warning("⚠️ Draw a region first")
    else:
        st.info("🔒 Map is locked during processing")
    
    # Saved Regions List
    if st.session_state.drawn_polygons:
        st.subheader("📍 Saved Regions")
        for i, p in enumerate(st.session_state.drawn_polygons):
            c1, c2, c3 = st.columns([3, 1, 1])
            centroid = p.centroid
            c1.write(f"**Region {i+1}**: ~{p.area * 111 * 111:.2f} km² | Center: ({centroid.y:.4f}, {centroid.x:.4f})")
            c2.write(f"UTM Zone {get_utm_zone(centroid.x)}")
            if c3.button("🗑️", key=f"del_{i}", disabled=st.session_state.processing_in_progress):
                st.session_state.drawn_polygons.pop(i)
                # Adjust selected index if needed
                if st.session_state.selected_region_index >= len(st.session_state.drawn_polygons):
                    st.session_state.selected_region_index = max(0, len(st.session_state.drawn_polygons) - 1)
                st.rerun()
    
    # Date
    st.header("2️⃣ Time Period")
    c1, c2 = st.columns(2)
    start = c1.date_input("Start (inclusive)", value=date(2024, 1, 1), disabled=st.session_state.processing_in_progress)
    end = c2.date_input("End (exclusive)", value=date(2025, 1, 1), disabled=st.session_state.processing_in_progress,
                        help="This month is NOT included. E.g., End=2024-06-01 means last processed month is May 2024")
    
    if start >= end:
        st.error("Invalid dates")
        st.stop()
    
    months = (end.year - start.year) * 12 + (end.month - start.month)
    
    # Calculate first and last month names
    first_month = f"{start.year}-{start.month:02d}"
    last_year = start.year + (start.month - 1 + months - 1) // 12
    last_month_num = (start.month - 1 + months - 1) % 12 + 1
    last_month = f"{last_year}-{last_month_num:02d}"
    
    st.info(f"📅 **{months} months**: {first_month} → {last_month} (end date {end.strftime('%Y-%m-%d')} is excluded)")
    
    # Process
    st.header("3️⃣ Process")
    
    # Region selection dropdown
    selected_polygon = None
    
    if st.session_state.drawn_polygons:
        # Create options for dropdown
        region_options = []
        for i, p in enumerate(st.session_state.drawn_polygons):
            area = p.area * 111 * 111
            region_options.append(f"Region {i+1} (~{area:.2f} km²)")
        
        # Ensure selected index is valid
        if st.session_state.selected_region_index >= len(st.session_state.drawn_polygons):
            st.session_state.selected_region_index = 0
        
        selected_idx = st.selectbox(
            "🎯 Select Region to Process",
            range(len(region_options)),
            format_func=lambda i: region_options[i],
            index=st.session_state.selected_region_index,
            disabled=st.session_state.processing_in_progress,
            key="region_selector"
        )
        
        st.session_state.selected_region_index = selected_idx
        selected_polygon = st.session_state.drawn_polygons[selected_idx]
        
        # Show selected region info
        centroid = selected_polygon.centroid
        st.success(f"✅ Selected: Region {selected_idx + 1} | Area: ~{selected_polygon.area * 111 * 111:.2f} km² | UTM Zone {get_utm_zone(centroid.x)}")
        
    elif st.session_state.last_drawn_polygon is not None:
        selected_polygon = st.session_state.last_drawn_polygon
        st.info("ℹ️ Using unsaved drawn region (save it to keep)")
    else:
        st.warning("⚠️ Draw and save a region on the map first")
    
    # Processing buttons
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        start_new = st.button(
            "🚀 Start New", 
            type="primary", 
            disabled=st.session_state.processing_in_progress or selected_polygon is None
        )
    
    with col2:
        has_cache = bool(st.session_state.downloaded_images or st.session_state.month_analysis_results)
        resume_btn = st.button(
            "🔄 Resume", 
            disabled=not has_cache or st.session_state.processing_in_progress
        )
    
    with col3:
        has_failed = bool(st.session_state.failed_downloads)
        retry_btn = st.button(
            "🔁 Retry Failed", 
            disabled=not has_failed or st.session_state.processing_in_progress
        )
    
    # Determine if we should process
    should_process = False
    resume_mode = False
    
    if start_new:
        should_process = True
        resume_mode = False
        # Clear cache for fresh start
        st.session_state.month_analysis_results = {}
        st.session_state.analysis_complete = False
        st.session_state.download_complete = False
        st.session_state.processed_months = {}
        st.session_state.classification_thumbnails = []
        st.session_state.valid_patches_mask = None
        st.session_state.valid_months = {}
        st.session_state.pdf_report = None
        st.session_state.failed_downloads = []
        st.session_state.processing_complete = False
        st.session_state.sentinel2_zip = None
        
        # Store processing config
        st.session_state.processing_config = {
            'polygon_coords': list(selected_polygon.exterior.coords),
            'start_date': start.strftime('%Y-%m-%d'),
            'end_date': end.strftime('%Y-%m-%d'),
            'cloudy_pct': cloudy_pct,
            'nodata_pct': nodata_pct
        }
        st.session_state.processing_in_progress = True
    
    elif resume_btn:
        should_process = True
        resume_mode = True
        st.session_state.processing_in_progress = True
        
        # Use stored config or create new
        if st.session_state.processing_config is None:
            st.session_state.processing_config = {
                'polygon_coords': list(selected_polygon.exterior.coords),
                'start_date': start.strftime('%Y-%m-%d'),
                'end_date': end.strftime('%Y-%m-%d'),
                'cloudy_pct': cloudy_pct,
                'nodata_pct': nodata_pct
            }
    
    elif retry_btn:
        should_process = True
        resume_mode = True
        st.session_state.failed_downloads = []
        st.session_state.processing_in_progress = True
        
        # Use stored config or create new
        if st.session_state.processing_config is None:
            st.session_state.processing_config = {
                'polygon_coords': list(selected_polygon.exterior.coords),
                'start_date': start.strftime('%Y-%m-%d'),
                'end_date': end.strftime('%Y-%m-%d'),
                'cloudy_pct': cloudy_pct,
                'nodata_pct': nodata_pct
            }
    
    # Auto-continue if processing was in progress (handles page reruns)
    elif st.session_state.processing_in_progress and st.session_state.processing_config is not None:
        should_process = True
        resume_mode = True
        st.info("🔄 Auto-continuing processing...")
    
    # Execute processing
    if should_process:
        config = st.session_state.processing_config
        
        if config is None:
            st.error("❌ No processing configuration found!")
            st.session_state.processing_in_progress = False
            st.stop()
        
        # Create polygon from stored coords
        poly = Polygon(config['polygon_coords'])
        aoi = ee.Geometry.Polygon([config['polygon_coords']])
        
        try:
            thumbs = process_timeseries(
                aoi, 
                config['start_date'], 
                config['end_date'],
                st.session_state.model, 
                st.session_state.device,
                config['cloudy_pct'], 
                10, 
                config['nodata_pct'], 
                resume=resume_mode
            )
            
            if thumbs:
                st.session_state.classification_thumbnails = thumbs
                st.session_state.processing_complete = True
            
            # Processing finished
            st.session_state.processing_in_progress = False
            
        except Exception as e:
            st.error(f"❌ Processing error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            st.session_state.processing_in_progress = False
    
    # Display Results
    if st.session_state.processing_complete and st.session_state.classification_thumbnails:
        st.divider()
        st.header("📊 Results")
        display_thumbnails(
            st.session_state.classification_thumbnails,
            valid_months=st.session_state.valid_months,
            downloaded_images=st.session_state.downloaded_images
        )
        
        # Info about change detection tab
        if st.session_state.probability_maps:
            st.success(f"✅ Classification complete! {len(st.session_state.probability_maps)} probability maps available. Go to **Change Detection** tab to analyze building changes.")


if __name__ == "__main__":
    main()
