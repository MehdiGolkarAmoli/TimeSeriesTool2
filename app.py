"""
Sentinel-2 Time Series Building Classification
VERSION 09 - FIXED GAP-FILLING (Matching JS Implementation)

Key Fix: Create cloudFreeCollection FIRST, then use it for both
monthly composites AND gap-filling (exactly like the JS code).

Features:
- Scene filter: CLOUDY_PIXEL_PERCENTAGE < 10%
- Pixel cloud mask: probability > 65 AND CDI < -0.5 + 20m dilation
- Pre-filter: Skip months with >30% masked pixels
- Gap-fill from M-1, M+1 using MOSAIC (closest first)
- Post-filter: Only download images with 0 masked pixels
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, box
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from patchify import patchify, unpatchify
import datetime
import torch
import math
from scipy import ndimage
import ee
import tempfile
from pathlib import Path
import requests
import time
import io
import warnings
import sys
import base64
import json
import subprocess
from datetime import date
from PIL import Image
import hashlib

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Examining the path of torch.classes")

import streamlit as st

st.set_page_config(
    layout="wide", 
    page_title="Building Classification v09 - Fixed Gap-Fill",
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

# Cloud detection parameters (UNCHANGED - same as JS)
SCENE_CLOUD_THRESHOLD = 10          # Scene-level: CLOUDY_PIXEL_PERCENTAGE < 10%
CLOUD_PROBABILITY_THRESHOLD = 65    # Pixel-level: probability > 65 = cloud
CDI_THRESHOLD = -0.5                # Pixel-level: CDI < -0.5 = cloud

# Gap-filling thresholds
MAX_MASKED_PERCENT_FOR_GAPFILL = 30  # Skip month if >30% masked (don't attempt gap-fill)
MAX_MASKED_PERCENT_AFTER_GAPFILL = 0  # Reject if ANY pixel masked after gap-fill

# Download settings
MAX_RETRIES = 5
RETRY_DELAY_BASE = 2
DOWNLOAD_TIMEOUT = 180
CHUNK_SIZE = 8192

# Minimum expected file sizes
MIN_BAND_FILE_SIZE = 10000
MIN_MULTIBAND_FILE_SIZE = 100000

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
if 'failed_months' not in st.session_state:
    st.session_state.failed_months = []
if 'skipped_months' not in st.session_state:
    st.session_state.skipped_months = []
if 'rejected_months' not in st.session_state:
    st.session_state.rejected_months = []
if 'current_temp_dir' not in st.session_state:
    st.session_state.current_temp_dir = None
if 'downloaded_images' not in st.session_state:
    st.session_state.downloaded_images = {}
if 'download_phase_complete' not in st.session_state:
    st.session_state.download_phase_complete = False
if 'valid_months' not in st.session_state:
    st.session_state.valid_months = []
if 'total_requested_months' not in st.session_state:
    st.session_state.total_requested_months = 0
if 'month_reports' not in st.session_state:
    st.session_state.month_reports = []


# =============================================================================
# Normalization function
# =============================================================================
def normalized(img):
    """Normalize image data to range [0, 1]"""
    min_val = np.nanmin(img)
    max_val = np.nanmax(img)
    
    if max_val == min_val:
        return np.zeros_like(img)
    
    img_norm = (img - min_val) / (max_val - min_val)
    return img_norm


# =============================================================================
# File Validation Functions
# =============================================================================
def validate_geotiff_comprehensive(file_path, expected_bands=1):
    """Comprehensive validation that checks entire file."""
    try:
        if not os.path.exists(file_path):
            return False, "File does not exist"
        
        file_size = os.path.getsize(file_path)
        min_size = MIN_BAND_FILE_SIZE if expected_bands == 1 else MIN_MULTIBAND_FILE_SIZE
        
        if file_size < min_size:
            return False, f"File too small ({file_size} bytes)"
        
        with open(file_path, 'rb') as f:
            header = f.read(4)
            if header[:2] not in [b'II', b'MM']:
                return False, "Invalid TIFF header"
        
        with rasterio.open(file_path) as src:
            if src.count < expected_bands:
                return False, f"Wrong band count ({src.count}, expected {expected_bands})"
            
            h, w = src.height, src.width
            if h < 10 or w < 10:
                return False, f"Image too small ({w}x{h})"
        
        return True, "File is valid"
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def validate_band_file_robust(band_file_path, band_name):
    """Robust validation for a single band file."""
    try:
        if not os.path.exists(band_file_path):
            return False, "File does not exist"
        
        file_size = os.path.getsize(band_file_path)
        if file_size < MIN_BAND_FILE_SIZE:
            return False, f"File too small ({file_size} bytes)"
        
        with open(band_file_path, 'rb') as f:
            header = f.read(8)
            if header[:2] not in [b'II', b'MM']:
                return False, "Invalid TIFF header"
        
        with rasterio.open(band_file_path) as src:
            h, w = src.height, src.width
            if h < 10 or w < 10:
                return False, "Image dimensions too small"
        
        return True, "File is valid"
        
    except Exception as e:
        return False, f"Validation failed: {str(e)}"


# =============================================================================
# Model Functions
# =============================================================================
@st.cache_data
def download_model_from_gdrive(gdrive_url, local_filename):
    """Download model from Google Drive."""
    try:
        correct_file_id = "1m6EScw-mpBIvWV78h4pyjWq1OLQtn2ov"
        st.info(f"Downloading model from Google Drive...")
        
        try:
            import gdown
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
            import gdown
        
        for method in [f"https://drive.google.com/uc?id={correct_file_id}", correct_file_id]:
            try:
                gdown.download(method, local_filename, quiet=False, fuzzy=True)
                if os.path.exists(local_filename) and os.path.getsize(local_filename) > 1024:
                    return local_filename
            except:
                continue
        
        return None
    except Exception as e:
        st.error(f"Download error: {str(e)}")
        return None


@st.cache_resource
def load_model(model_path):
    """Load the UNet++ model."""
    try:
        device = torch.device('cpu')
        model = smp.UnetPlusPlus(
            encoder_name='timm-efficientnet-b7',
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
    """Initialize Earth Engine."""
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
                return True, "Earth Engine initialized (Service Account)"
            else:
                ee.Authenticate()
                ee.Initialize()
                return True, "Earth Engine initialized"
        except Exception as e:
            return False, f"Authentication failed: {str(e)}"


# =============================================================================
# Helper Functions
# =============================================================================
def get_utm_zone(longitude):
    return math.floor((longitude + 180) / 6) + 1

def get_utm_epsg(longitude, latitude):
    zone_number = get_utm_zone(longitude)
    if latitude >= 0:
        return f"EPSG:326{zone_number:02d}"
    else:
        return f"EPSG:327{zone_number:02d}"


# =============================================================================
# GEE: CLOUD MASKING FUNCTION (Exact match to JS)
# =============================================================================
def mask_cloud_and_shadow(img, aoi):
    """
    Apply cloud masking to a single image.
    EXACTLY matches the JS function maskCloudAndShadow.
    
    Cloud detection:
    - probability > 65 AND CDI < -0.5 = CLOUD
    - Dilate with 20m circle kernel, 2 iterations
    
    Returns: scaled spectral bands only, clipped to AOI
    """
    cloud_prob = img.select('probability')
    cdi = ee.Algorithms.Sentinel2.CDI(img)
    
    # Cloud detection: high probability AND low CDI
    is_cloud = cloud_prob.gt(CLOUD_PROBABILITY_THRESHOLD).And(cdi.lt(CDI_THRESHOLD))
    
    # Dilate cloud mask with 20m kernel, 2 iterations
    kernel = ee.Kernel.circle(radius=20, units='meters')
    cloud_dilated = is_cloud.focal_max(kernel=kernel, iterations=2)
    
    # Apply mask
    masked = img.updateMask(cloud_dilated.Not())
    
    # Scale spectral bands and clip to AOI
    scaled = masked.select(SPECTRAL_BANDS).multiply(0.0001).clip(aoi)
    
    return scaled.copyProperties(img, ['system:time_start'])


def get_cloud_free_collection(aoi, start_date, end_date):
    """
    Get Sentinel-2 cloud-free collection.
    EXACTLY matches the JS data preparation section.
    
    Steps:
    1. Get S2_SR_HARMONIZED with scene cloud filter
    2. Get S2_CLOUD_PROBABILITY
    3. Join them
    4. Apply cloud masking to ALL images
    
    Returns: ImageCollection with cloud-masked, scaled spectral bands
    """
    
    # S2 Surface Reflectance
    s2_sr = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
             .filterBounds(aoi)
             .filterDate(start_date, end_date)
             .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', SCENE_CLOUD_THRESHOLD))
             .select(['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12', 'SCL']))
    
    # Cloud Probability
    s2_cloud = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
                .filterBounds(aoi)
                .filterDate(start_date, end_date))
    
    # Join collections (exactly like JS indexJoin)
    joined = ee.ImageCollection(
        ee.Join.saveFirst('cloud_probability').apply(
            primary=s2_sr,
            secondary=s2_cloud,
            condition=ee.Filter.equals(
                leftField='system:index',
                rightField='system:index'
            )
        )
    )
    
    # Add cloud probability band to each image
    def add_cloud_band(img):
        return img.addBands(ee.Image(img.get('cloud_probability')))
    
    s2_joined = joined.map(add_cloud_band)
    
    # Apply cloud masking to ALL images (create cloudFreeCollection)
    cloud_free_collection = s2_joined.map(lambda img: mask_cloud_and_shadow(img, aoi))
    
    return cloud_free_collection


# =============================================================================
# GEE: MONTHLY COMPOSITE WITH GAP-FILLING (Exact match to JS)
# =============================================================================
def create_monthly_composite_with_gapfill(cloud_free_collection, aoi, origin, month_index, 
                                           num_months, status_callback=None):
    """
    Create a single monthly composite with gap-filling.
    EXACTLY matches the JS gap-filling logic.
    
    Gap-fill strategy (from JS):
    1. Get current month composite (median)
    2. Identify masked pixels (frequency == 0)
    3. Collect images from M-1, M+1 (from cloudFreeCollection!)
    4. Sort by time distance from middle of current month
    5. Create mosaic (first valid pixel wins = closest in time)
    6. Fill masked pixels from mosaic
    
    Returns: dict with image and status information
    """
    
    month_index = ee.Number(month_index)
    origin_date = ee.Date(origin)
    
    # Current month boundaries
    month_start = origin_date.advance(month_index, 'month')
    month_end = origin_date.advance(month_index.add(1), 'month')
    month_middle = month_start.advance(15, 'day')
    month_middle_millis = month_middle.millis()
    
    month_name = month_start.format('YYYY-MM').getInfo()
    
    if status_callback:
        status_callback(f"Processing {month_name}...")
    
    # Get images for current month (from cloud-free collection!)
    monthly_images = cloud_free_collection.filterDate(month_start, month_end)
    image_count = monthly_images.size().getInfo()
    
    # =========================================================================
    # CASE 1: No images available
    # =========================================================================
    if image_count == 0:
        return {
            'image': None,
            'status': 'no_data',
            'status_reason': f'No cloud-free images (CLOUDY_PIXEL_PERCENTAGE < {SCENE_CLOUD_THRESHOLD}%) available for this month',
            'masked_before': 100,
            'masked_after': 100,
            'month_name': month_name,
            'image_count': 0,
            'masked_pixels_before': 'N/A',
            'masked_pixels_after': 'N/A',
            'total_pixels': 'N/A'
        }
    
    # Create median composite (exactly like JS)
    median_composite = monthly_images.median()
    
    # Create frequency map (count of valid pixels per location)
    # This matches JS: monthlyImages.map(...).sum()
    def create_valid_mask(img):
        return ee.Image(1).updateMask(img.select('B4').mask()).unmask(0).toInt()
    
    frequency_map = monthly_images.map(create_valid_mask).sum().toInt().rename('frequency')
    
    # Calculate masked pixels BEFORE gap-filling
    gap_mask = frequency_map.eq(0)  # True where NO valid data
    
    # Count pixels
    pixel_stats = ee.Image.cat([
        gap_mask.rename('masked'),
        ee.Image(1).rename('total')
    ]).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=aoi,
        scale=10,
        maxPixels=1e13
    )
    
    masked_pixels_before = ee.Number(pixel_stats.get('masked')).round().getInfo()
    total_pixels = ee.Number(pixel_stats.get('total')).round().getInfo()
    
    if total_pixels is None or total_pixels == 0:
        total_pixels = 1
    if masked_pixels_before is None:
        masked_pixels_before = 0
    
    masked_pixels_before = int(masked_pixels_before)
    total_pixels = int(total_pixels)
    
    masked_percent_before = 100 * masked_pixels_before / total_pixels
    
    # =========================================================================
    # CASE 2: Too many masked pixels (>30%) - SKIP
    # =========================================================================
    if masked_percent_before > MAX_MASKED_PERCENT_FOR_GAPFILL:
        return {
            'image': None,
            'status': 'skipped',
            'status_reason': f'Too many masked pixels ({masked_percent_before:.1f}% > {MAX_MASKED_PERCENT_FOR_GAPFILL}% threshold) - gap-filling not attempted',
            'masked_before': masked_percent_before,
            'masked_after': masked_percent_before,
            'month_name': month_name,
            'image_count': image_count,
            'masked_pixels_before': masked_pixels_before,
            'masked_pixels_after': masked_pixels_before,
            'total_pixels': total_pixels
        }
    
    # =========================================================================
    # CASE 3: Already complete (0% masked)
    # =========================================================================
    if masked_pixels_before == 0:
        return {
            'image': median_composite,
            'status': 'complete',
            'status_reason': 'Image has 0 masked pixels - ready for download',
            'masked_before': 0,
            'masked_after': 0,
            'month_name': month_name,
            'image_count': image_count,
            'masked_pixels_before': 0,
            'masked_pixels_after': 0,
            'total_pixels': total_pixels,
            'was_gapfilled': False
        }
    
    # =========================================================================
    # CASE 4: Need gap-filling
    # =========================================================================
    if status_callback:
        status_callback(f"Gap-filling {month_name}...")
    
    # Define search ranges for M-1, M+1 (exactly like JS)
    # M-1 (previous month)
    m1_past_start = origin_date.advance(month_index.subtract(1), 'month')
    m1_past_end = month_start
    
    # M+1 (next month)
    m1_future_start = month_end
    m1_future_end = origin_date.advance(month_index.add(2), 'month')
    
    # Collect images from M-1 and M+1 (from cloudFreeCollection!)
    m1_past_images = cloud_free_collection.filterDate(m1_past_start, m1_past_end)
    m1_future_images = cloud_free_collection.filterDate(m1_future_start, m1_future_end)
    
    # Merge all candidate images
    all_candidate_images = m1_past_images.merge(m1_future_images)
    candidate_count = all_candidate_images.size().getInfo()
    
    if candidate_count == 0:
        # No candidates available
        return {
            'image': None,
            'status': 'rejected',
            'status_reason': f'Has {masked_pixels_before:,} masked pixels and no gap-fill candidates available from M-1 or M+1',
            'masked_before': masked_percent_before,
            'masked_after': masked_percent_before,
            'month_name': month_name,
            'image_count': image_count,
            'masked_pixels_before': masked_pixels_before,
            'masked_pixels_after': masked_pixels_before,
            'total_pixels': total_pixels,
            'candidate_count': 0
        }
    
    # Add time distance property to each candidate (exactly like JS)
    def add_time_distance(img):
        img_time = ee.Number(img.get('system:time_start'))
        # Absolute time distance from middle of current month
        time_diff = img_time.subtract(month_middle_millis).abs()
        return img.set('time_distance', time_diff)
    
    images_with_distance = all_candidate_images.map(add_time_distance)
    
    # Sort by time distance (closest first) - exactly like JS
    sorted_images = images_with_distance.sort('time_distance', True)
    
    # Create mosaic: first valid pixel wins (closest in time)
    # This is EXACTLY what the JS code does!
    closest_mosaic = sorted_images.mosaic().select(SPECTRAL_BANDS)
    
    # Check if mosaic has valid data at gap locations
    has_closest = closest_mosaic.select('B4').mask()
    
    # Apply gap-filling (exactly like JS)
    fill_from_closest = gap_mask.And(has_closest)
    still_masked = gap_mask.And(has_closest.Not())
    
    # Fill gaps using unmask (exactly like JS)
    filled_spectral = median_composite.unmask(closest_mosaic.updateMask(fill_from_closest))
    
    # Calculate masked percentage AFTER gap-filling
    filled_gap_mask = filled_spectral.select('B4').mask().Not()
    
    after_stats = filled_gap_mask.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=aoi,
        scale=10,
        maxPixels=1e13
    )
    
    masked_pixels_after = ee.Number(after_stats.get('B4')).round().getInfo()
    
    if masked_pixels_after is None:
        masked_pixels_after = 0
    
    masked_pixels_after = int(masked_pixels_after)
    masked_percent_after = 100 * masked_pixels_after / total_pixels
    
    # =========================================================================
    # CASE 4a: Still has masked pixels after gap-filling - REJECT
    # =========================================================================
    if masked_pixels_after > 0:
        return {
            'image': None,
            'status': 'rejected',
            'status_reason': f'Still has {masked_pixels_after:,} masked pixels after gap-filling from {candidate_count} candidates',
            'masked_before': masked_percent_before,
            'masked_after': masked_percent_after,
            'month_name': month_name,
            'image_count': image_count,
            'masked_pixels_before': masked_pixels_before,
            'masked_pixels_after': masked_pixels_after,
            'total_pixels': total_pixels,
            'candidate_count': candidate_count
        }
    
    # =========================================================================
    # CASE 4b: Gap-filling successful (0 masked pixels)
    # =========================================================================
    return {
        'image': filled_spectral,
        'status': 'complete',
        'status_reason': f'Gap-filled successfully from {candidate_count} candidates - now has 0 masked pixels',
        'masked_before': masked_percent_before,
        'masked_after': 0,
        'month_name': month_name,
        'image_count': image_count,
        'masked_pixels_before': masked_pixels_before,
        'masked_pixels_after': 0,
        'total_pixels': total_pixels,
        'candidate_count': candidate_count,
        'was_gapfilled': True
    }


# =============================================================================
# Download Functions
# =============================================================================
def download_band_robust(image, band, aoi, output_path, scale=10, status_callback=None):
    """Download a single band with retry mechanism."""
    region = aoi.bounds().getInfo()['coordinates']
    temp_path = output_path + '.tmp'
    
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    if os.path.exists(output_path):
        is_valid, _ = validate_band_file_robust(output_path, band)
        if is_valid:
            return True, "cached"
        os.remove(output_path)
    
    for attempt in range(MAX_RETRIES):
        try:
            if status_callback:
                status_callback(f"📥 {band} - attempt {attempt + 1}/{MAX_RETRIES}...")
            
            url = image.select(band).getDownloadURL({
                'scale': scale,
                'region': region,
                'format': 'GEO_TIFF',
                'bands': [band]
            })
            
            response = requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT)
            
            if response.status_code != 200:
                raise Exception(f"HTTP {response.status_code}")
            
            content_type = response.headers.get('content-type', '')
            if 'text/html' in content_type:
                raise Exception("Received HTML instead of GeoTIFF")
            
            expected_size = int(response.headers.get('content-length', 0))
            actual_size = 0
            
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        actual_size += len(chunk)
            
            if expected_size > 0 and actual_size < expected_size * 0.95:
                raise Exception(f"Incomplete download: {actual_size}/{expected_size}")
            
            is_valid, msg = validate_band_file_robust(temp_path, band)
            if not is_valid:
                raise Exception(f"Validation failed: {msg}")
            
            os.replace(temp_path, output_path)
            return True, "downloaded"
            
        except Exception as e:
            for f in [output_path, temp_path]:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except:
                        pass
            
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY_BASE ** (attempt + 1))
    
    return False, f"Failed after {MAX_RETRIES} attempts"


def download_gapfilled_image(composite_result, aoi, temp_dir, scale=10, status_placeholder=None):
    """Download a gap-filled composite image."""
    month_name = composite_result['month_name']
    image = composite_result['image']
    
    output_file = os.path.join(temp_dir, f"sentinel2_{month_name}.tif")
    bands_dir = os.path.join(temp_dir, f"bands_{month_name}")
    
    def update_status(msg):
        if status_placeholder:
            status_placeholder.text(f"📥 {month_name}: {msg}")
    
    if os.path.exists(output_file):
        is_valid, _ = validate_geotiff_comprehensive(output_file, expected_bands=len(SPECTRAL_BANDS))
        if is_valid:
            update_status("✅ Using cached file")
            return output_file
        os.remove(output_file)
    
    os.makedirs(bands_dir, exist_ok=True)
    
    band_files = []
    for i, band in enumerate(SPECTRAL_BANDS):
        band_file = os.path.join(bands_dir, f"{band}.tif")
        update_status(f"Downloading {band} ({i+1}/{len(SPECTRAL_BANDS)})...")
        
        success, result = download_band_robust(image, band, aoi, band_file, scale, update_status)
        
        if not success:
            return None
        
        band_files.append(band_file)
    
    update_status("Creating multiband GeoTIFF...")
    
    with rasterio.open(band_files[0]) as src:
        meta = src.meta.copy()
    
    meta.update(count=len(band_files))
    
    with rasterio.open(output_file, 'w', **meta) as dst:
        for i, band_file in enumerate(band_files):
            with rasterio.open(band_file) as src:
                dst.write(src.read(1), i + 1)
    
    is_valid, msg = validate_geotiff_comprehensive(output_file, expected_bands=len(SPECTRAL_BANDS))
    if not is_valid:
        if os.path.exists(output_file):
            os.remove(output_file)
        return None
    
    update_status("✅ Complete!")
    return output_file


# =============================================================================
# RGB Thumbnail Generation
# =============================================================================
def generate_rgb_thumbnail(image_path, month_name, max_size=256):
    """Generate RGB thumbnail from downloaded image."""
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
                new_h, new_w = int(h * scale), int(w * scale)
                pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
            
            return pil_img
            
    except Exception as e:
        return None


# =============================================================================
# Classification Functions
# =============================================================================
def classify_image(image_path, model, device, month_name):
    """Classify an entire image for building detection."""
    try:
        with rasterio.open(image_path) as src:
            img_data = src.read()
        
        img_for_patching = np.moveaxis(img_data, 0, -1)
        
        h, w, c = img_for_patching.shape
        new_h = int(np.ceil(h / PATCH_SIZE) * PATCH_SIZE)
        new_w = int(np.ceil(w / PATCH_SIZE) * PATCH_SIZE)
        
        if h != new_h or w != new_w:
            padded_img = np.zeros((new_h, new_w, c), dtype=img_for_patching.dtype)
            padded_img[:h, :w, :] = img_for_patching
            img_for_patching = padded_img
        
        patches = patchify(img_for_patching, (PATCH_SIZE, PATCH_SIZE, c), step=PATCH_SIZE)
        n_patches_h, n_patches_w = patches.shape[0], patches.shape[1]
        
        classified_patches = np.zeros((n_patches_h, n_patches_w, PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
        
        for i in range(n_patches_h):
            for j in range(n_patches_w):
                patch = patches[i, j, 0]
                patch_normalized = normalized(patch)
                patch_tensor = torch.tensor(np.moveaxis(patch_normalized, -1, 0), dtype=torch.float32)
                patch_tensor = patch_tensor.unsqueeze(0)
                
                with torch.inference_mode():
                    prediction = model(patch_tensor)
                    prediction = torch.sigmoid(prediction).cpu()
                
                pred_np = prediction.squeeze().numpy()
                binary_mask = (pred_np > 0.5).astype(np.uint8) * 255
                classified_patches[i, j] = binary_mask
        
        reconstructed = unpatchify(classified_patches, (new_h, new_w))
        reconstructed = reconstructed[:h, :w]
        
        return reconstructed
        
    except Exception as e:
        st.error(f"Error classifying {month_name}: {str(e)}")
        return None


# =============================================================================
# Display Detailed Report
# =============================================================================
def display_detailed_report(month_reports, total_months):
    """Display a detailed month-by-month report before downloading."""
    
    st.subheader("📋 DETAILED MONTH-BY-MONTH REPORT")
    
    # Categorize months
    complete = [r for r in month_reports if r['status'] == 'complete']
    skipped = [r for r in month_reports if r['status'] == 'skipped']
    rejected = [r for r in month_reports if r['status'] == 'rejected']
    no_data = [r for r in month_reports if r['status'] == 'no_data']
    
    # Summary box
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📅 Total Requested", total_months)
    with col2:
        st.metric("✅ Will Download", len(complete), 
                 delta=f"{100*len(complete)/total_months:.0f}%" if total_months > 0 else "0%")
    with col3:
        st.metric("⏭️ Skipped", len(skipped))
    with col4:
        st.metric("❌ Rejected", len(rejected) + len(no_data))
    
    st.markdown("---")
    
    # Detailed report for EACH month
    st.markdown("### 📊 Status of Each Month:")
    
    for idx, report in enumerate(month_reports):
        month_name = report['month_name']
        status = report['status']
        reason = report.get('status_reason', '')
        masked_before = report.get('masked_pixels_before', 'N/A')
        masked_after = report.get('masked_pixels_after', 'N/A')
        total_pixels = report.get('total_pixels', 'N/A')
        image_count = report.get('image_count', 0)
        candidate_count = report.get('candidate_count', 0)
        was_gapfilled = report.get('was_gapfilled', False)
        
        # Determine icon and action
        if status == 'complete':
            if was_gapfilled:
                icon = "✅🔄"
                action = "WILL DOWNLOAD (gap-filled)"
                box_type = "success"
            else:
                icon = "✅"
                action = "WILL DOWNLOAD"
                box_type = "success"
        elif status == 'skipped':
            icon = "⏭️"
            action = "SKIPPED (>30% masked)"
            box_type = "warning"
        elif status == 'rejected':
            icon = "❌"
            action = "REJECTED (still has masked pixels)"
            box_type = "error"
        else:  # no_data
            icon = "📭"
            action = "NO DATA"
            box_type = "info"
        
        # Create expandable section for each month
        with st.expander(f"{icon} **Month {idx+1}: {month_name}** - {action}", expanded=False):
            # Status and reason
            if box_type == "success":
                st.success(f"**Status:** {action}")
            elif box_type == "warning":
                st.warning(f"**Status:** {action}")
            elif box_type == "error":
                st.error(f"**Status:** {action}")
            else:
                st.info(f"**Status:** {action}")
            
            st.markdown(f"**Reason:** {reason}")
            
            # Details
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"**Images found:** {image_count}")
                if candidate_count > 0:
                    st.markdown(f"**Gap-fill candidates:** {candidate_count}")
            
            with col2:
                if masked_before != 'N/A':
                    if isinstance(masked_before, (int, float)):
                        st.markdown(f"**Masked pixels (before):** {int(masked_before):,}")
                    else:
                        st.markdown(f"**Masked pixels (before):** {masked_before}")
                if masked_after != 'N/A' and masked_after != masked_before:
                    if isinstance(masked_after, (int, float)):
                        st.markdown(f"**Masked pixels (after):** {int(masked_after):,}")
                    else:
                        st.markdown(f"**Masked pixels (after):** {masked_after}")
            
            with col3:
                if total_pixels != 'N/A':
                    st.markdown(f"**Total pixels:** {total_pixels:,}")
    
    st.markdown("---")
    
    # Summary lists
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ✅ Months to Download:")
        if complete:
            for r in complete:
                gf = " 🔄(gap-filled)" if r.get('was_gapfilled') else ""
                st.markdown(f"- **{r['month_name']}**{gf} - 0 masked pixels")
        else:
            st.warning("⚠️ No months available for download!")
    
    with col2:
        st.markdown("### ❌ Months NOT Downloaded:")
        
        if skipped:
            st.markdown(f"**Skipped ({len(skipped)})** - >{MAX_MASKED_PERCENT_FOR_GAPFILL}% masked:")
            for r in skipped:
                mp = r.get('masked_pixels_before', 0)
                mp = int(mp) if isinstance(mp, (int, float)) else mp
                st.markdown(f"- {r['month_name']} ({mp:,} masked pixels, {r['masked_before']:.1f}%)")
        
        if rejected:
            st.markdown(f"**Rejected ({len(rejected)})** - still has masked pixels after gap-fill:")
            for r in rejected:
                mp = r.get('masked_pixels_after', 0)
                mp = int(mp) if isinstance(mp, (int, float)) else mp
                st.markdown(f"- {r['month_name']} ({mp:,} masked pixels remaining)")
        
        if no_data:
            st.markdown(f"**No Data ({len(no_data)})** - no images with <{SCENE_CLOUD_THRESHOLD}% cloud:")
            for r in no_data:
                st.markdown(f"- {r['month_name']}")
    
    return complete


# =============================================================================
# Main Processing Pipeline
# =============================================================================
def process_timeseries_with_gapfill(aoi, start_date, end_date, model, device, scale=10, resume=False):
    """
    Main pipeline with gap-filling.
    EXACTLY matches the JS workflow.
    """
    try:
        if st.session_state.current_temp_dir is None or not os.path.exists(st.session_state.current_temp_dir):
            st.session_state.current_temp_dir = tempfile.mkdtemp()
        temp_dir = st.session_state.current_temp_dir
        
        st.info(f"📁 Working directory: {temp_dir}")
        
        # Calculate months
        start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        total_months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)
        
        st.session_state.total_requested_months = total_months
        st.info(f"📅 Requested: {total_months} months ({start_date} to {end_date})")
        
        # Extended date range for gap-filling (1 month before, 1 month after)
        extended_start = ee.Date(start_date).advance(-1, 'month')
        extended_end = ee.Date(end_date).advance(1, 'month')
        
        # =====================================================================
        # PHASE 1: Create Cloud-Free Collection (exactly like JS!)
        # =====================================================================
        st.header("Phase 1: Creating Cloud-Free Collection")
        
        with st.spinner("Loading and cloud-masking Sentinel-2 data..."):
            cloud_free_collection = get_cloud_free_collection(aoi, extended_start, extended_end)
            total_images = cloud_free_collection.size().getInfo()
            st.success(f"✅ Created cloud-free collection with {total_images} images")
            st.info(f"""
            **Cloud Detection Applied:**
            - Scene filter: `CLOUDY_PIXEL_PERCENTAGE < {SCENE_CLOUD_THRESHOLD}%`
            - Pixel mask: `probability > {CLOUD_PROBABILITY_THRESHOLD} AND CDI < {CDI_THRESHOLD}` + 20m dilation
            """)
        
        # =====================================================================
        # PHASE 2: Analyze each month
        # =====================================================================
        st.header("Phase 2: Creating Monthly Composites with Gap-Filling")
        
        st.info(f"""
        **Gap-Fill Strategy (matching JS):**
        1. Create median composite from current month
        2. If masked pixels exist and ≤{MAX_MASKED_PERCENT_FOR_GAPFILL}%:
           - Collect cloud-free images from M-1, M+1
           - Sort by time distance (closest first)
           - Use mosaic to fill gaps (first valid pixel wins)
        3. Only download if final image has 0 masked pixels
        """)
        
        origin = start_date
        month_reports = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for month_idx in range(total_months):
            result = create_monthly_composite_with_gapfill(
                cloud_free_collection, aoi, origin, month_idx, total_months,
                status_callback=lambda msg: status_text.text(msg)
            )
            
            month_reports.append(result)
            progress_bar.progress((month_idx + 1) / total_months)
        
        progress_bar.empty()
        status_text.empty()
        
        st.session_state.month_reports = month_reports
        
        # =====================================================================
        # PHASE 3: Display detailed report
        # =====================================================================
        st.header("Phase 3: Analysis Report")
        
        complete_months = display_detailed_report(month_reports, total_months)
        
        if not complete_months:
            st.error("❌ No months available for download! All months have masked pixels or no data.")
            return []
        
        # =====================================================================
        # PHASE 4: Download
        # =====================================================================
        st.header("Phase 4: Downloading Cloud-Free Months")
        
        st.success(f"""
        📥 **Download Summary:**
        - From {total_months} requested months
        - Downloading {len(complete_months)} months with **0 masked pixels**
        - Skipping {len(month_reports) - len(complete_months)} months (masked pixels or no data)
        """)
        
        downloaded_images = {}
        
        if resume and st.session_state.downloaded_images:
            downloaded_images = st.session_state.downloaded_images.copy()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, result in enumerate(complete_months):
            month_name = result['month_name']
            
            if month_name in downloaded_images and os.path.exists(downloaded_images[month_name]):
                progress_bar.progress((idx + 1) / len(complete_months))
                continue
            
            status_text.text(f"📥 Downloading {month_name} ({idx+1}/{len(complete_months)})...")
            
            output_file = download_gapfilled_image(result, aoi, temp_dir, scale, status_text)
            
            if output_file:
                downloaded_images[month_name] = output_file
                st.session_state.downloaded_images = downloaded_images.copy()
            
            progress_bar.progress((idx + 1) / len(complete_months))
        
        progress_bar.empty()
        status_text.empty()
        
        st.success(f"✅ Downloaded {len(downloaded_images)} / {len(complete_months)} months successfully!")
        st.session_state.valid_months = list(downloaded_images.keys())
        
        # =====================================================================
        # PHASE 5: Classification
        # =====================================================================
        st.header("Phase 5: Building Classification")
        
        thumbnails = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        month_names = sorted(downloaded_images.keys())
        
        for idx, month_name in enumerate(month_names):
            image_path = downloaded_images[month_name]
            status_text.text(f"🧠 Classifying {month_name} ({idx+1}/{len(month_names)})...")
            
            classification_mask = classify_image(image_path, model, device, month_name)
            
            if classification_mask is not None:
                rgb_thumbnail = generate_rgb_thumbnail(image_path, month_name)
                
                h, w = classification_mask.shape
                pil_class = Image.fromarray(classification_mask.astype(np.uint8))
                if h > 256 or w > 256:
                    scale_factor = 256 / max(h, w)
                    pil_class = pil_class.resize((int(w * scale_factor), int(h * scale_factor)), Image.NEAREST)
                
                was_gapfilled = any(r['month_name'] == month_name and r.get('was_gapfilled', False) 
                                   for r in complete_months)
                
                thumbnails.append({
                    'rgb_image': rgb_thumbnail,
                    'classification_image': pil_class,
                    'month_name': month_name,
                    'building_pixels': np.sum(classification_mask > 0),
                    'total_pixels': h * w,
                    'was_gapfilled': was_gapfilled
                })
            
            progress_bar.progress((idx + 1) / len(month_names))
        
        progress_bar.empty()
        status_text.empty()
        
        st.success(f"✅ Classified {len(thumbnails)} months!")
        
        return thumbnails
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return []


# =============================================================================
# Display Functions
# =============================================================================
def display_classification_thumbnails(thumbnails):
    """Display thumbnails."""
    if not thumbnails:
        st.info("No classifications to display.")
        return
    
    display_mode = st.radio(
        "Display Mode:",
        ["Side by Side (RGB + Classification)", "Classification Only", "RGB Only"],
        horizontal=True
    )
    
    st.caption("🔄 = Gap-filled from adjacent months")
    st.divider()
    
    if display_mode == "Side by Side (RGB + Classification)":
        for i in range(0, len(thumbnails), 2):
            cols = st.columns(4)
            for j in range(2):
                idx = i + j
                if idx < len(thumbnails):
                    thumb = thumbnails[idx]
                    building_pct = (thumb['building_pixels'] / thumb['total_pixels']) * 100
                    gf = " 🔄" if thumb.get('was_gapfilled') else ""
                    
                    with cols[j * 2]:
                        if thumb.get('rgb_image'):
                            st.image(thumb['rgb_image'], caption=f"{thumb['month_name']}{gf} (RGB)")
                    
                    with cols[j * 2 + 1]:
                        if thumb.get('classification_image'):
                            st.image(thumb['classification_image'], caption=f"{thumb['month_name']} ({building_pct:.1f}%)")
    
    elif display_mode == "Classification Only":
        for row in range((len(thumbnails) + 3) // 4):
            cols = st.columns(4)
            for col_idx in range(4):
                idx = row * 4 + col_idx
                if idx < len(thumbnails):
                    thumb = thumbnails[idx]
                    building_pct = (thumb['building_pixels'] / thumb['total_pixels']) * 100
                    gf = " 🔄" if thumb.get('was_gapfilled') else ""
                    with cols[col_idx]:
                        if thumb.get('classification_image'):
                            st.image(thumb['classification_image'], caption=f"{thumb['month_name']}{gf} ({building_pct:.1f}%)")
    
    else:
        for row in range((len(thumbnails) + 3) // 4):
            cols = st.columns(4)
            for col_idx in range(4):
                idx = row * 4 + col_idx
                if idx < len(thumbnails):
                    thumb = thumbnails[idx]
                    gf = " 🔄" if thumb.get('was_gapfilled') else ""
                    with cols[col_idx]:
                        if thumb.get('rgb_image'):
                            st.image(thumb['rgb_image'], caption=f"{thumb['month_name']}{gf}")


# =============================================================================
# Main Application
# =============================================================================
def main():
    st.title("🏗️ Building Classification v09 - Fixed Gap-Fill")
    st.markdown(f"""
    **Cloud Detection Pipeline (Exact JS Match):**
    - ☁️ Scene filter: `CLOUDY_PIXEL_PERCENTAGE < {SCENE_CLOUD_THRESHOLD}%`
    - 🎯 Pixel mask: `probability > {CLOUD_PROBABILITY_THRESHOLD} AND CDI < {CDI_THRESHOLD}` + 20m dilation
    - 📦 **Cloud-free collection created FIRST** (like JS `cloudFreeCollection`)
    - ⏭️ Pre-filter: Skip if >{MAX_MASKED_PERCENT_FOR_GAPFILL}% masked
    - 🔄 Gap-fill: M-1, M+1 mosaic (closest first) from cloud-free collection
    - ✅ Download only: 0 masked pixels
    """)
    
    # Initialize Earth Engine
    ee_initialized, ee_message = initialize_earth_engine()
    if not ee_initialized:
        st.error(ee_message)
        st.stop()
    else:
        st.sidebar.success(ee_message)
    
    # Model Loading
    st.sidebar.header("🧠 Model")
    model_path = "best_model_version_Unet++_v02_e7.pt"
    
    if not os.path.exists(model_path):
        download_model_from_gdrive("", model_path)
    
    if not st.session_state.model_loaded:
        with st.spinner("Loading model..."):
            model, device = load_model(model_path)
            if model:
                st.session_state.model = model
                st.session_state.device = device
                st.session_state.model_loaded = True
                st.sidebar.success("✅ Model loaded")
            else:
                st.sidebar.error("❌ Model failed")
                st.stop()
    else:
        st.sidebar.success("✅ Model loaded")
    
    # Cache info
    st.sidebar.header("🗂️ Cache")
    if st.session_state.downloaded_images:
        st.sidebar.success(f"📥 {len(st.session_state.downloaded_images)} downloaded")
    if st.session_state.valid_months:
        st.sidebar.info(f"✅ {len(st.session_state.valid_months)} complete")
    
    if st.sidebar.button("🗑️ Clear Cache"):
        st.session_state.downloaded_images = {}
        st.session_state.classification_thumbnails = []
        st.session_state.processing_complete = False
        st.session_state.month_reports = []
        st.session_state.valid_months = []
        st.rerun()
    
    # Map
    st.header("1️⃣ Select Region")
    
    m = folium.Map(location=[35.6892, 51.3890], zoom_start=8)
    draw = plugins.Draw(export=True, position='topleft',
        draw_options={'polyline': False, 'rectangle': True, 'polygon': True,
                     'circle': False, 'marker': False, 'circlemarker': False})
    m.add_child(draw)
    folium.TileLayer(tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
                    attr='Google', name='Satellite').add_to(m)
    folium.LayerControl().add_to(m)
    
    map_data = st_folium(m, width=800, height=500)
    
    if map_data and 'last_active_drawing' in map_data and map_data['last_active_drawing']:
        drawn = map_data['last_active_drawing']
        if 'geometry' in drawn and drawn['geometry']['type'] == 'Polygon':
            coords = drawn['geometry']['coordinates'][0]
            polygon = Polygon(coords)
            st.session_state.last_drawn_polygon = polygon
            st.success(f"✅ Region: ~{polygon.area * 111 * 111:.2f} km²")
    
    if st.button("💾 Save Region"):
        if st.session_state.last_drawn_polygon:
            if not any(p.equals(st.session_state.last_drawn_polygon) for p in st.session_state.drawn_polygons):
                st.session_state.drawn_polygons.append(st.session_state.last_drawn_polygon)
                st.success("✅ Saved!")
    
    if st.session_state.drawn_polygons:
        st.subheader("📍 Saved Regions")
        for i, poly in enumerate(st.session_state.drawn_polygons):
            col1, col2, col3 = st.columns([4, 3, 1])
            with col1:
                st.write(f"**Region {i+1}**")
            with col2:
                st.write(f"~{poly.area * 111 * 111:.2f} km²")
            with col3:
                if st.button("🗑️", key=f"del_{i}"):
                    st.session_state.drawn_polygons.pop(i)
                    st.rerun()
    
    # Dates
    st.header("2️⃣ Select Dates")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start", value=date(2023, 6, 1), min_value=date(2017, 1, 1))
    with col2:
        end_date = st.date_input("End", value=date(2024, 2, 1), min_value=date(2017, 1, 1))
    
    if start_date >= end_date:
        st.error("❌ End must be after start!")
        st.stop()
    
    num_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
    st.info(f"📅 {num_months} months selected")
    
    # Process
    st.header("3️⃣ Generate")
    
    selected_polygon = None
    if st.session_state.drawn_polygons:
        idx = st.selectbox("Select region", range(len(st.session_state.drawn_polygons)),
                          format_func=lambda i: f"Region {i+1}")
        selected_polygon = st.session_state.drawn_polygons[idx]
    elif st.session_state.last_drawn_polygon:
        selected_polygon = st.session_state.last_drawn_polygon
    
    col1, col2 = st.columns(2)
    with col1:
        start_new = st.button("🚀 Start New", type="primary")
    with col2:
        resume = st.button("🔄 Resume", disabled=not st.session_state.downloaded_images)
    
    if start_new or resume:
        if not selected_polygon:
            st.error("❌ Select region first!")
            st.stop()
        
        if start_new:
            st.session_state.downloaded_images = {}
            st.session_state.classification_thumbnails = []
            st.session_state.processing_complete = False
        
        geojson = {"type": "Polygon", "coordinates": [list(selected_polygon.exterior.coords)]}
        aoi = ee.Geometry.Polygon(geojson['coordinates'])
        
        thumbnails = process_timeseries_with_gapfill(
            aoi, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'),
            st.session_state.model, st.session_state.device, scale=10, resume=resume
        )
        
        if thumbnails:
            st.session_state.classification_thumbnails = thumbnails
            st.session_state.processing_complete = True
    
    # Results
    if st.session_state.processing_complete and st.session_state.classification_thumbnails:
        st.divider()
        st.header("📊 Results")
        display_classification_thumbnails(st.session_state.classification_thumbnails)


if __name__ == "__main__":
    main()
