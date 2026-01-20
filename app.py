"""
Sentinel-2 Time Series Building Classification
VERSION 05 - NODATA HANDLING + FULL CACHE SUPPORT

Features:
- Two-pass approach: Download all → Find valid patches → Classify
- Patches with ANY nodata in ANY month are excluded from ALL months
- Full cache support with .tmp file approach (from v04)
- Resume processing if interrupted
- Failed months tracking and retry
- Raw DN values (no 0.0001 scaling)
- Per-patch global normalization
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

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Examining the path of torch.classes")

import streamlit as st

st.set_page_config(
    layout="wide", 
    page_title="Building Classification Time Series v05",
    page_icon="🏗️"
)

import folium
from folium import plugins
from streamlit_folium import st_folium
import segmentation_models_pytorch as smp

SPECTRAL_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
PATCH_SIZE = 224

# RGB bands for visualization (True Color: B4=Red, B3=Green, B2=Blue)
RGB_BAND_INDICES = {'red': 3, 'green': 2, 'blue': 1}

# Download settings - KEPT FROM V04
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2
DOWNLOAD_TIMEOUT = 120
CHUNK_SIZE = 8192

# Minimum expected file sizes for validation - KEPT FROM V04
MIN_BAND_FILE_SIZE = 10000
MIN_MULTIBAND_FILE_SIZE = 100000

# Session state initialization - EXPANDED
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
if 'data_summary' not in st.session_state:
    st.session_state.data_summary = None
if 'processed_months' not in st.session_state:
    st.session_state.processed_months = {}
if 'failed_months' not in st.session_state:
    st.session_state.failed_months = []
if 'current_temp_dir' not in st.session_state:
    st.session_state.current_temp_dir = None
if 'processing_in_progress' not in st.session_state:
    st.session_state.processing_in_progress = False
if 'last_processed_index' not in st.session_state:
    st.session_state.last_processed_index = 0
# NEW for v05: Valid patch mask and downloaded images tracking
if 'valid_patches_mask' not in st.session_state:
    st.session_state.valid_patches_mask = None
if 'downloaded_images' not in st.session_state:
    st.session_state.downloaded_images = {}
if 'download_phase_complete' not in st.session_state:
    st.session_state.download_phase_complete = False
if 'validity_analysis_complete' not in st.session_state:
    st.session_state.validity_analysis_complete = False


# =============================================================================
# Normalization function - matches working app exactly
# =============================================================================
def normalized(img):
    """
    Normalize image data to range [0, 1]
    This normalizes the ENTIRE array globally (all bands together)
    """
    min_val = np.nanmin(img)
    max_val = np.nanmax(img)
    
    if max_val == min_val:
        return np.zeros_like(img)
    
    img_norm = (img - min_val) / (max_val - min_val)
    return img_norm


# =============================================================================
# File Validation Functions - KEPT FROM V04
# =============================================================================
def validate_geotiff_file(file_path, expected_bands=1):
    """Validate that a GeoTIFF file is complete and readable."""
    try:
        if not os.path.exists(file_path):
            return False, "File does not exist"
        
        file_size = os.path.getsize(file_path)
        min_size = MIN_BAND_FILE_SIZE if expected_bands == 1 else MIN_MULTIBAND_FILE_SIZE
        
        if file_size < min_size:
            return False, f"File too small ({file_size} bytes, expected > {min_size})"
        
        with rasterio.open(file_path) as src:
            if src.count < expected_bands:
                return False, f"Wrong band count ({src.count}, expected {expected_bands})"
            
            for band_idx in range(1, min(src.count + 1, expected_bands + 1)):
                window = rasterio.windows.Window(0, 0, min(10, src.width), min(10, src.height))
                data = src.read(band_idx, window=window)
                if np.all(np.isnan(data)):
                    return False, f"Band {band_idx} contains only NaN values"
        
        return True, "File is valid"
        
    except rasterio.errors.RasterioIOError as e:
        return False, f"Rasterio cannot read file: {str(e)}"
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def validate_band_file(band_file_path, band_name):
    """Validate a single band GeoTIFF file."""
    return validate_geotiff_file(band_file_path, expected_bands=1)


# =============================================================================
# Model Download Functions - KEPT FROM V04
# =============================================================================
@st.cache_data
def download_model_from_gdrive(gdrive_url, local_filename):
    """Download a file from Google Drive with improved error handling"""
    try:
        correct_file_id = "1m6EScw-mpBIvWV78h4pyjWq1OLQtn2ov"
        st.info(f"Downloading model from Google Drive (File ID: {correct_file_id})...")
        
        try:
            import gdown
        except ImportError:
            st.info("Installing gdown library...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
            import gdown
        
        download_methods = [
            f"https://drive.google.com/uc?id={correct_file_id}",
            f"https://drive.google.com/file/d/{correct_file_id}/view",
            correct_file_id
        ]
        
        for i, method in enumerate(download_methods):
            try:
                st.info(f"Trying download method {i+1}/3...")
                gdown.download(method, local_filename, quiet=False, fuzzy=True)
                
                if os.path.exists(local_filename) and os.path.getsize(local_filename) > 1024:
                    file_size = os.path.getsize(local_filename)
                    with open(local_filename, 'rb') as f:
                        header = f.read(10)
                        if header.startswith(b'\x80\x02') or header.startswith(b'\x80\x03') or header.startswith(b'PK'):
                            st.success(f"Model downloaded successfully! Size: {file_size / (1024*1024):.1f} MB")
                            return local_filename
                        else:
                            if os.path.exists(local_filename):
                                os.remove(local_filename)
                else:
                    if os.path.exists(local_filename):
                        os.remove(local_filename)
                        
            except Exception as e:
                st.warning(f"Download method {i+1} failed: {str(e)}")
                if os.path.exists(local_filename):
                    os.remove(local_filename)
                continue
        
        return manual_download_fallback(correct_file_id, local_filename)
            
    except Exception as e:
        st.error(f"Error in download function: {str(e)}")
        return None


def manual_download_fallback(file_id, local_filename):
    """Fallback manual download method using requests"""
    try:
        urls_to_try = [
            f"https://drive.google.com/uc?export=download&id={file_id}",
            f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t",
            f"https://drive.usercontent.google.com/download?id={file_id}&export=download",
        ]
        
        session = requests.Session()
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        
        for i, url in enumerate(urls_to_try):
            try:
                st.info(f"Trying manual method {i+1}/3...")
                response = session.get(url, headers=headers, stream=True, timeout=30)
                
                if response.status_code == 200:
                    content_type = response.headers.get('content-type', '').lower()
                    
                    if 'text/html' not in content_type:
                        total_size = int(response.headers.get('content-length', 0))
                        downloaded_size = 0
                        
                        if total_size > 0:
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                        
                        with open(local_filename, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                                    downloaded_size += len(chunk)
                                    if total_size > 0:
                                        progress = downloaded_size / total_size
                                        progress_bar.progress(progress)
                                        status_text.text(f"Downloaded: {downloaded_size / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB")
                        
                        if total_size > 0:
                            progress_bar.empty()
                            status_text.empty()
                        
                        if os.path.exists(local_filename) and os.path.getsize(local_filename) > 1024:
                            file_size = os.path.getsize(local_filename)
                            st.success(f"Manual download successful! Size: {file_size / (1024*1024):.1f} MB")
                            return local_filename
                        
            except Exception as e:
                st.warning(f"Manual method {i+1} failed: {e}")
                continue
        
        st.error("All automatic download methods failed. Please download manually:")
        st.markdown(f"""
        1. **Open this link:** https://drive.google.com/file/d/{file_id}/view
        2. **Click the Download button**
        3. **Save the file as:** `{local_filename}`
        4. **Upload it using the file uploader below**
        """)
        
        uploaded_file = st.file_uploader(
            f"Upload the model file ({local_filename}) after manual download:",
            type=['pt', 'pth']
        )
        
        if uploaded_file is not None:
            with open(local_filename, 'wb') as f:
                f.write(uploaded_file.read())
            file_size = os.path.getsize(local_filename)
            st.success(f"Model uploaded successfully! Size: {file_size / (1024*1024):.1f} MB")
            return local_filename
        
        return None
        
    except Exception as e:
        st.error(f"Manual download fallback failed: {e}")
        return None


# =============================================================================
# Model Loading Function - KEPT FROM V04
# =============================================================================
@st.cache_resource
def load_model(model_path):
    """Load the UNet++ model for building detection"""
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
            st.info("Model loaded from checkpoint dictionary.")
        elif isinstance(loaded_object, dict):
            model.load_state_dict(loaded_object)
            st.info("Model loaded directly from state_dict.")
        else:
            st.error("Loaded model file is not a recognized state_dict or checkpoint format.")
            return None, None

        model.eval()
        st.success("Model loaded successfully!")
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None


# =============================================================================
# Earth Engine Authentication - KEPT FROM V04
# =============================================================================
@st.cache_resource
def initialize_earth_engine():
    """Initialize Earth Engine with authentication."""
    try:
        ee.Initialize()
        return True, "Earth Engine already initialized"
    except Exception:
        try:
            base64_key = os.environ.get('GOOGLE_EARTH_ENGINE_KEY_BASE64')
            
            if base64_key:
                key_json = base64.b64decode(base64_key).decode()
                key_data = json.loads(key_json)
                
                key_file = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
                with open(key_file.name, 'w') as f:
                    json.dump(key_data, f)
                
                credentials = ee.ServiceAccountCredentials(
                    key_data['client_email'],
                    key_file.name
                )
                ee.Initialize(credentials)
                os.unlink(key_file.name)
                return True, "Successfully authenticated with Earth Engine (Service Account)!"
            else:
                ee.Authenticate()
                ee.Initialize()
                return True, "Successfully authenticated with Earth Engine!"
        except Exception as auth_error:
            return False, f"Authentication failed: {str(auth_error)}"


# =============================================================================
# Helper Functions
# =============================================================================
def get_utm_zone(longitude):
    """Determine the UTM zone for a given longitude."""
    return math.floor((longitude + 180) / 6) + 1

def get_utm_epsg(longitude, latitude):
    """Determine the EPSG code for UTM zone based on longitude and latitude."""
    zone_number = get_utm_zone(longitude)
    if latitude >= 0:
        return f"EPSG:326{zone_number:02d}"
    else:
        return f"EPSG:327{zone_number:02d}"


# =============================================================================
# RGB Image Generation Functions - KEPT FROM V04
# =============================================================================
def generate_rgb_thumbnail(image_path, month_name, max_size=256):
    """
    Generate an RGB thumbnail from a Sentinel-2 multiband image.
    Uses B4 (Red), B3 (Green), B2 (Blue) for true color visualization.
    """
    try:
        with rasterio.open(image_path) as src:
            red = src.read(4)    # B4 - Red
            green = src.read(3)  # B3 - Green
            blue = src.read(2)   # B2 - Blue
            
            rgb = np.stack([red, green, blue], axis=-1)
            rgb = np.nan_to_num(rgb, nan=0.0)
            
            def percentile_stretch(band, lower=2, upper=98):
                """Apply percentile stretching to enhance contrast"""
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
        st.warning(f"Error generating RGB thumbnail for {month_name}: {str(e)}")
        return None


# =============================================================================
# GEE Functions - Simple median composite (matching working app)
# =============================================================================
def create_monthly_composites_list(aoi, start_date, end_date, cloudy_pixel_percentage=10):
    """
    Create a list of month info for downloading.
    Uses simple median composites - matching working app approach.
    """
    start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    total_months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)
    
    composites = []
    
    for month_offset in range(total_months):
        current_date = start_dt + datetime.timedelta(days=month_offset * 30)
        year = current_date.year
        month = current_date.month
        
        month_start = f"{year}-{month:02d}-01"
        if month == 12:
            month_end = f"{year + 1}-01-01"
        else:
            month_end = f"{year}-{month + 1:02d}-01"
        
        month_name = f"{year}-{month:02d}"
        
        composites.append({
            'month_name': month_name,
            'start_date': month_start,
            'end_date': month_end,
            'year': year,
            'month': month
        })
    
    return composites, total_months


# =============================================================================
# Download Functions with .tmp file approach - KEPT FROM V04
# =============================================================================
def download_band_with_retry(image, band, aoi, output_path, scale=10):
    """Download a single band with retry mechanism using .tmp file approach."""
    region = aoi.bounds().getInfo()['coordinates']
    
    # If .tmp file exists from previous crash, delete it
    temp_path = output_path + '.tmp'
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    # If final file exists and is valid, skip
    if os.path.exists(output_path):
        is_valid, msg = validate_band_file(output_path, band)
        if is_valid:
            return True
        else:
            os.remove(output_path)
    
    for attempt in range(MAX_RETRIES):
        try:
            url = image.select(band).getDownloadURL({
                'scale': scale,
                'region': region,
                'format': 'GEO_TIFF',
                'bands': [band]
            })
            
            response = requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT)
            
            if response.status_code == 200:
                content_type = response.headers.get('content-type', '')
                if 'text/html' in content_type:
                    raise Exception("Received HTML instead of GeoTIFF - possible rate limit")
                
                # Download to temp file first
                with open(temp_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                
                # Validate the downloaded file
                is_valid, msg = validate_band_file(temp_path, band)
                
                if is_valid:
                    # Move temp file to final location (atomic operation)
                    os.replace(temp_path, output_path)
                    return True
                else:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    raise Exception(f"Downloaded file invalid: {msg}")
            else:
                raise Exception(f"HTTP {response.status_code}")
                
        except requests.exceptions.Timeout:
            st.warning(f"⏱️ Timeout downloading {band} (attempt {attempt + 1}/{MAX_RETRIES})")
        except requests.exceptions.ConnectionError:
            st.warning(f"🔌 Connection error downloading {band} (attempt {attempt + 1}/{MAX_RETRIES})")
        except Exception as e:
            st.warning(f"⚠️ Error downloading {band}: {str(e)} (attempt {attempt + 1}/{MAX_RETRIES})")
        
        # Clean up any partial files
        for f in [output_path, temp_path]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    pass
        
        if attempt < MAX_RETRIES - 1:
            wait_time = RETRY_DELAY_BASE ** (attempt + 1)
            st.info(f"⏳ Waiting {wait_time}s before retry...")
            time.sleep(wait_time)
    
    return False


def download_monthly_image(aoi, month_info, temp_dir, cloudy_pixel_percentage=10, scale=10, status_placeholder=None):
    """
    Download a single monthly Sentinel-2 composite from GEE.
    Uses simple median composite - matching working app.
    Includes cache checking and .tmp file approach.
    """
    try:
        month_name = month_info['month_name']
        start_date = month_info['start_date']
        end_date = month_info['end_date']
        
        output_file = os.path.join(temp_dir, f"sentinel2_{month_name}.tif")
        
        # Check if already downloaded AND VALID (cache hit)
        if os.path.exists(output_file):
            is_valid, msg = validate_geotiff_file(output_file, expected_bands=len(SPECTRAL_BANDS))
            if is_valid:
                if status_placeholder:
                    status_placeholder.info(f"✅ {month_name} using cached file")
                return output_file
            else:
                if status_placeholder:
                    status_placeholder.warning(f"⚠️ {month_name} cached file corrupted ({msg}), re-downloading...")
                os.remove(output_file)
        
        if status_placeholder:
            status_placeholder.text(f"📥 {month_name}: Searching for images...")
        
        # Simple collection filter - matches working app
        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                     .filterBounds(aoi)
                     .filterDate(start_date, end_date)
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloudy_pixel_percentage))
                     .select(SPECTRAL_BANDS))
        
        count = collection.size().getInfo()
        
        if count == 0:
            # Try higher cloud cover
            if status_placeholder:
                status_placeholder.text(f"📥 {month_name}: No images found, trying higher cloud cover...")
            
            collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                         .filterBounds(aoi)
                         .filterDate(start_date, end_date)
                         .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', min(cloudy_pixel_percentage * 3, 50)))
                         .select(SPECTRAL_BANDS))
            count = collection.size().getInfo()
            
            if count == 0:
                if status_placeholder:
                    status_placeholder.warning(f"⚠️ {month_name}: No images found")
                return None
        
        if status_placeholder:
            status_placeholder.text(f"📥 {month_name}: Creating median from {count} images...")
        
        # Simple median composite - NO scaling, raw DN values
        median_image = collection.median()
        
        # Download bands
        bands_dir = os.path.join(temp_dir, f"bands_{month_name}")
        os.makedirs(bands_dir, exist_ok=True)
        
        band_files = []
        failed_bands = []
        
        for i, band in enumerate(SPECTRAL_BANDS):
            band_file = os.path.join(bands_dir, f"{band}.tif")
            
            if status_placeholder:
                status_placeholder.text(f"📥 {month_name}: Downloading {band} ({i+1}/{len(SPECTRAL_BANDS)})...")
            
            success = download_band_with_retry(median_image, band, aoi, band_file, scale)
            
            if success:
                band_files.append(band_file)
            else:
                failed_bands.append(band)
        
        if failed_bands:
            st.error(f"❌ {month_name}: Failed to download bands: {', '.join(failed_bands)}")
            return None
        
        if len(band_files) == len(SPECTRAL_BANDS):
            if status_placeholder:
                status_placeholder.text(f"📦 {month_name}: Creating multiband GeoTIFF...")
            
            with rasterio.open(band_files[0]) as src:
                meta = src.meta.copy()
            
            meta.update(count=len(band_files))
            
            with rasterio.open(output_file, 'w', **meta) as dst:
                for i, band_file in enumerate(band_files):
                    with rasterio.open(band_file) as src:
                        dst.write(src.read(1), i+1)
            
            is_valid, msg = validate_geotiff_file(output_file, expected_bands=len(SPECTRAL_BANDS))
            if not is_valid:
                st.error(f"❌ {month_name}: Final multiband file validation failed: {msg}")
                if os.path.exists(output_file):
                    os.remove(output_file)
                return None
            
            return output_file
        else:
            return None
        
    except Exception as e:
        st.error(f"❌ Error downloading {month_name}: {str(e)}")
        return None


# =============================================================================
# NODATA HANDLING - NEW FOR V05
# =============================================================================
def check_patch_validity(patch, nodata_threshold_percent=5):
    """
    Check if a patch has any nodata (NaN, 0, or invalid values).
    Returns True if patch is valid (minimal nodata), False otherwise.
    
    Args:
        patch: numpy array of shape (H, W, C) or (H, W)
        nodata_threshold_percent: Maximum percentage of zeros allowed
    """
    # Check for NaN
    if np.any(np.isnan(patch)):
        return False
    
    # Check for all-zero patch (common nodata indicator in Sentinel-2)
    if np.all(patch == 0):
        return False
    
    # Check percentage of zeros
    total_elements = patch.size
    zero_count = np.sum(patch == 0)
    zero_percent = (zero_count / total_elements) * 100
    
    if zero_percent > nodata_threshold_percent:
        return False
    
    # Check if any band is all zeros (for 3D arrays)
    if patch.ndim == 3:
        for band_idx in range(patch.shape[-1]):
            if np.all(patch[:, :, band_idx] == 0):
                return False
    
    return True


def get_patch_validity_mask(image_path, patch_size=224, nodata_threshold_percent=5):
    """
    Create a mask showing which patches are valid (no nodata) for a single image.
    Returns: 2D boolean array where True = valid patch
    """
    try:
        with rasterio.open(image_path) as src:
            img_data = src.read()  # (C, H, W)
        
        # Convert to (H, W, C)
        img_for_patching = np.moveaxis(img_data, 0, -1)
        
        h, w, c = img_for_patching.shape
        new_h = int(np.ceil(h / patch_size) * patch_size)
        new_w = int(np.ceil(w / patch_size) * patch_size)
        
        # Pad if needed
        if h != new_h or w != new_w:
            padded_img = np.zeros((new_h, new_w, c), dtype=img_for_patching.dtype)
            padded_img[:h, :w, :] = img_for_patching
            img_for_patching = padded_img
        
        # Create patches
        patches = patchify(img_for_patching, (patch_size, patch_size, c), step=patch_size)
        
        n_patches_h, n_patches_w = patches.shape[0], patches.shape[1]
        
        # Check each patch
        validity_mask = np.zeros((n_patches_h, n_patches_w), dtype=bool)
        nodata_info = []
        
        for i in range(n_patches_h):
            for j in range(n_patches_w):
                patch = patches[i, j, 0]
                is_valid = check_patch_validity(patch, nodata_threshold_percent)
                validity_mask[i, j] = is_valid
                
                if not is_valid:
                    # Calculate nodata percentage for reporting
                    zero_percent = (np.sum(patch == 0) / patch.size) * 100
                    nodata_info.append((i, j, zero_percent))
        
        return validity_mask, (h, w), (n_patches_h, n_patches_w), nodata_info
        
    except Exception as e:
        st.error(f"Error checking validity: {str(e)}")
        return None, None, None, None


def find_common_valid_patches(downloaded_images, nodata_threshold_percent=5):
    """
    Find patches that are valid across ALL months.
    Returns: Combined validity mask (True = valid in all months)
    """
    st.info("🔍 Analyzing patch validity across all months...")
    
    combined_mask = None
    original_size = None
    patch_grid_size = None
    
    month_names = sorted(downloaded_images.keys())
    
    progress_bar = st.progress(0)
    validity_report = []
    
    for idx, month_name in enumerate(month_names):
        image_path = downloaded_images[month_name]
        
        validity_mask, orig_size, grid_size, nodata_info = get_patch_validity_mask(
            image_path, PATCH_SIZE, nodata_threshold_percent
        )
        
        if validity_mask is None:
            st.warning(f"Could not check {month_name}")
            continue
        
        if combined_mask is None:
            combined_mask = validity_mask.copy()
            original_size = orig_size
            patch_grid_size = grid_size
        else:
            # Combine: only keep patches valid in ALL months
            combined_mask = combined_mask & validity_mask
        
        # Report for this month
        valid_count = np.sum(validity_mask)
        total_count = validity_mask.size
        validity_report.append({
            'month': month_name,
            'valid': valid_count,
            'total': total_count,
            'percent': 100 * valid_count / total_count,
            'nodata_patches': len(nodata_info)
        })
        
        progress_bar.progress((idx + 1) / len(month_names))
    
    progress_bar.empty()
    
    # Display validity report
    st.subheader("📊 Patch Validity Report")
    
    report_cols = st.columns(4)
    for idx, report in enumerate(validity_report):
        col_idx = idx % 4
        with report_cols[col_idx]:
            st.metric(
                report['month'],
                f"{report['valid']}/{report['total']}",
                f"{report['percent']:.1f}% valid"
            )
    
    if combined_mask is not None:
        final_valid = np.sum(combined_mask)
        total = combined_mask.size
        
        st.divider()
        st.success(f"✅ Found **{final_valid}/{total}** patches valid across ALL months ({100*final_valid/total:.1f}%)")
        
        if final_valid == 0:
            st.error("❌ No patches are valid across all months! Try:")
            st.write("- Selecting a different region")
            st.write("- Choosing a different time period")
            st.write("- Increasing the nodata threshold")
            return None, None, None
    
    return combined_mask, original_size, patch_grid_size


def classify_image_with_mask(image_path, model, device, month_name, valid_mask, original_size):
    """
    Classify an image, but only process patches marked as valid.
    Invalid patches are set to 0 (no buildings).
    """
    try:
        with rasterio.open(image_path) as src:
            img_data = src.read()  # (C, H, W)
        
        # Convert to (H, W, C)
        img_for_patching = np.moveaxis(img_data, 0, -1)
        
        h, w, c = img_for_patching.shape
        new_h = int(np.ceil(h / PATCH_SIZE) * PATCH_SIZE)
        new_w = int(np.ceil(w / PATCH_SIZE) * PATCH_SIZE)
        
        # Pad if needed
        if h != new_h or w != new_w:
            padded_img = np.zeros((new_h, new_w, c), dtype=img_for_patching.dtype)
            padded_img[:h, :w, :] = img_for_patching
            img_for_patching = padded_img
        
        # Create patches
        patches = patchify(img_for_patching, (PATCH_SIZE, PATCH_SIZE, c), step=PATCH_SIZE)
        
        n_patches_h, n_patches_w = patches.shape[0], patches.shape[1]
        
        # Initialize output
        classified_patches = np.zeros((n_patches_h, n_patches_w, PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
        
        valid_count = 0
        skipped_count = 0
        
        for i in range(n_patches_h):
            for j in range(n_patches_w):
                # Check if this patch is valid
                if not valid_mask[i, j]:
                    skipped_count += 1
                    continue  # Leave as zeros
                
                patch = patches[i, j, 0]  # (H, W, C)
                
                # Normalize entire patch (all bands together) - matches working app!
                patch_normalized = normalized(patch)
                
                # Convert to (C, H, W) tensor
                patch_tensor = torch.tensor(np.moveaxis(patch_normalized, -1, 0), dtype=torch.float32)
                patch_tensor = patch_tensor.unsqueeze(0)
                
                # Inference
                with torch.inference_mode():
                    prediction = model(patch_tensor)
                    prediction = torch.sigmoid(prediction).cpu()
                
                pred_np = prediction.squeeze().numpy()
                binary_mask = (pred_np > 0.5).astype(np.uint8) * 255
                
                classified_patches[i, j] = binary_mask
                valid_count += 1
        
        # Reconstruct
        reconstructed = unpatchify(classified_patches, (new_h, new_w))
        reconstructed = reconstructed[:original_size[0], :original_size[1]]
        
        return reconstructed, valid_count, skipped_count
        
    except Exception as e:
        st.error(f"Error classifying {month_name}: {str(e)}")
        return None, 0, 0


# =============================================================================
# Generate Thumbnails - KEPT FROM V04
# =============================================================================
def generate_thumbnails(image_path, classification_mask, month_name, max_size=256):
    """Generate both RGB and classification thumbnails."""
    try:
        rgb_thumbnail = generate_rgb_thumbnail(image_path, month_name, max_size)
        
        h, w = classification_mask.shape
        
        if h > max_size or w > max_size:
            scale = max_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            pil_class = Image.fromarray(classification_mask.astype(np.uint8))
            pil_class = pil_class.resize((new_w, new_h), Image.NEAREST)
        else:
            pil_class = Image.fromarray(classification_mask.astype(np.uint8))
        
        return {
            'rgb_image': rgb_thumbnail,
            'classification_image': pil_class,
            'month_name': month_name,
            'original_size': (h, w),
            'building_pixels': np.sum(classification_mask > 0),
            'total_pixels': h * w
        }
        
    except Exception as e:
        st.warning(f"Error creating thumbnails for {month_name}: {str(e)}")
        return None


# =============================================================================
# Main Processing Pipeline - TWO PHASES with CACHE SUPPORT
# =============================================================================
def download_all_images(composites, aoi, temp_dir, cloudy_pixel_percentage=10, scale=10, resume=False):
    """
    PHASE 1: Download all monthly images.
    Supports resuming from previous downloads (cache).
    """
    downloaded_images = {}
    failed_months = []
    
    if resume and st.session_state.downloaded_images:
        downloaded_images = st.session_state.downloaded_images.copy()
        st.info(f"🔄 Resuming... {len(downloaded_images)} months already downloaded")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, month_info in enumerate(composites):
        month_name = month_info['month_name']
        
        # Skip if already downloaded
        if month_name in downloaded_images:
            # Verify the file still exists and is valid
            cached_path = downloaded_images[month_name]
            if os.path.exists(cached_path):
                is_valid, _ = validate_geotiff_file(cached_path, expected_bands=len(SPECTRAL_BANDS))
                if is_valid:
                    progress_bar.progress((idx + 1) / len(composites))
                    continue
        
        status_text.text(f"📥 Downloading {month_name} ({idx+1}/{len(composites)})...")
        
        image_path = download_monthly_image(
            aoi, month_info, temp_dir,
            cloudy_pixel_percentage=cloudy_pixel_percentage,
            scale=scale,
            status_placeholder=status_text
        )
        
        if image_path:
            downloaded_images[month_name] = image_path
        else:
            failed_months.append(month_name)
            st.warning(f"⚠️ Failed to download {month_name}")
        
        progress_bar.progress((idx + 1) / len(composites))
        
        # Save progress to session state
        st.session_state.downloaded_images = downloaded_images
    
    progress_bar.empty()
    status_text.empty()
    
    return downloaded_images, failed_months


def classify_all_images(downloaded_images, model, device, valid_mask, original_size, resume=False):
    """
    PHASE 3: Classify all downloaded images using the valid patch mask.
    Supports resuming from previous classifications (cache).
    """
    thumbnails = []
    
    if resume and st.session_state.processed_months:
        already_processed = set(st.session_state.processed_months.keys())
        st.info(f"🔄 Resuming... {len(already_processed)} months already classified")
    else:
        already_processed = set()
        st.session_state.processed_months = {}
    
    month_names = sorted(downloaded_images.keys())
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, month_name in enumerate(month_names):
        # Skip if already processed
        if month_name in already_processed:
            # Add to thumbnails from cache
            if month_name in st.session_state.processed_months:
                thumbnails.append(st.session_state.processed_months[month_name])
            progress_bar.progress((idx + 1) / len(month_names))
            continue
        
        image_path = downloaded_images[month_name]
        status_text.text(f"🧠 Classifying {month_name} ({idx+1}/{len(month_names)})...")
        
        classification_mask, valid_count, skipped_count = classify_image_with_mask(
            image_path, model, device, month_name, valid_mask, original_size
        )
        
        if classification_mask is not None:
            thumbnail_data = generate_thumbnails(image_path, classification_mask, month_name)
            
            if thumbnail_data:
                thumbnail_data['valid_patches'] = valid_count
                thumbnail_data['skipped_patches'] = skipped_count
                thumbnails.append(thumbnail_data)
                st.session_state.processed_months[month_name] = thumbnail_data
        
        progress_bar.progress((idx + 1) / len(month_names))
    
    progress_bar.empty()
    status_text.empty()
    
    return thumbnails


def process_timeseries_with_nodata_handling(aoi, start_date, end_date, model, device,
                                             cloudy_pixel_percentage=10, scale=10,
                                             nodata_threshold_percent=5, resume=False):
    """
    Main processing pipeline with THREE PHASES:
    1. Download all images (with cache)
    2. Analyze patch validity across all months
    3. Classify only valid patches (with cache)
    """
    try:
        # Setup temp directory
        if st.session_state.current_temp_dir is None or not os.path.exists(st.session_state.current_temp_dir):
            st.session_state.current_temp_dir = tempfile.mkdtemp()
        temp_dir = st.session_state.current_temp_dir
        
        st.info(f"📁 Working directory: {temp_dir}")
        
        # Get month list
        composites, total_months = create_monthly_composites_list(aoi, start_date, end_date, cloudy_pixel_percentage)
        st.info(f"📅 Processing {total_months} months...")
        
        # =====================================================================
        # PHASE 1: Download all images (with cache support)
        # =====================================================================
        st.header("Phase 1: Downloading Images")
        
        downloaded_images, failed_downloads = download_all_images(
            composites, aoi, temp_dir,
            cloudy_pixel_percentage=cloudy_pixel_percentage,
            scale=scale,
            resume=resume
        )
        
        if len(downloaded_images) == 0:
            st.error("❌ No images could be downloaded!")
            return []
        
        st.success(f"✅ Downloaded {len(downloaded_images)}/{total_months} months")
        
        if failed_downloads:
            st.warning(f"⚠️ Failed to download: {', '.join(failed_downloads)}")
            st.session_state.failed_months = failed_downloads
        
        st.session_state.download_phase_complete = True
        
        # =====================================================================
        # PHASE 2: Analyze patch validity
        # =====================================================================
        st.header("Phase 2: Finding Valid Patches")
        
        # Only re-analyze if not already done or if downloads changed
        if (not st.session_state.validity_analysis_complete or 
            st.session_state.valid_patches_mask is None):
            
            valid_mask, original_size, patch_grid_size = find_common_valid_patches(
                downloaded_images, nodata_threshold_percent
            )
            
            if valid_mask is None:
                return []
            
            st.session_state.valid_patches_mask = valid_mask
            st.session_state.original_size = original_size
            st.session_state.patch_grid_size = patch_grid_size
            st.session_state.validity_analysis_complete = True
            
            # Visualize valid patch mask
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(valid_mask, cmap='RdYlGn', vmin=0, vmax=1)
            ax.set_title(f"Valid Patches Mask\nGreen = Valid in ALL months, Red = Has nodata")
            ax.set_xlabel("Patch Column")
            ax.set_ylabel("Patch Row")
            st.pyplot(fig)
        else:
            valid_mask = st.session_state.valid_patches_mask
            original_size = st.session_state.original_size
            st.info("✅ Using cached validity analysis")
        
        # =====================================================================
        # PHASE 3: Classify all images (with cache support)
        # =====================================================================
        st.header("Phase 3: Classifying Images")
        
        thumbnails = classify_all_images(
            downloaded_images, model, device, 
            valid_mask, original_size,
            resume=resume
        )
        
        if thumbnails:
            st.success(f"✅ Successfully classified {len(thumbnails)} months!")
        
        return thumbnails
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return []


# =============================================================================
# Display Thumbnails - KEPT FROM V04
# =============================================================================
def display_classification_thumbnails(thumbnails):
    """Display RGB and classification thumbnails side by side."""
    
    if not thumbnails:
        st.info("No classifications to display.")
        return
    
    display_mode = st.radio(
        "Display Mode:",
        ["Side by Side (RGB + Classification)", "Classification Only", "RGB Only"],
        horizontal=True
    )
    
    st.divider()
    
    if display_mode == "Side by Side (RGB + Classification)":
        num_cols = 4
        
        for i in range(0, len(thumbnails), 2):
            cols = st.columns(num_cols)
            
            for j in range(2):
                idx = i + j
                if idx < len(thumbnails):
                    thumb = thumbnails[idx]
                    building_pct = (thumb['building_pixels'] / thumb['total_pixels']) * 100
                    
                    with cols[j * 2]:
                        if thumb.get('rgb_image') is not None:
                            st.image(
                                thumb['rgb_image'],
                                caption=f"{thumb['month_name']} (RGB)",
                                use_column_width=True
                            )
                        else:
                            st.warning(f"No RGB for {thumb['month_name']}")
                    
                    with cols[j * 2 + 1]:
                        class_img = thumb.get('classification_image') or thumb.get('image')
                        if class_img is not None:
                            st.image(
                                class_img,
                                caption=f"{thumb['month_name']} ({building_pct:.1f}% buildings)",
                                use_column_width=True
                            )
                        else:
                            st.warning(f"No classification for {thumb['month_name']}")
    
    elif display_mode == "Classification Only":
        num_cols = 4
        num_rows = (len(thumbnails) + num_cols - 1) // num_cols
        
        for row in range(num_rows):
            cols = st.columns(num_cols)
            for col_idx in range(num_cols):
                img_idx = row * num_cols + col_idx
                if img_idx < len(thumbnails):
                    with cols[col_idx]:
                        thumb = thumbnails[img_idx]
                        building_pct = (thumb['building_pixels'] / thumb['total_pixels']) * 100
                        
                        class_img = thumb.get('classification_image') or thumb.get('image')
                        if class_img is not None:
                            st.image(
                                class_img,
                                caption=f"{thumb['month_name']} ({building_pct:.1f}%)",
                                use_column_width=True
                            )
    
    else:  # RGB Only
        num_cols = 4
        num_rows = (len(thumbnails) + num_cols - 1) // num_cols
        
        for row in range(num_rows):
            cols = st.columns(num_cols)
            for col_idx in range(num_cols):
                img_idx = row * num_cols + col_idx
                if img_idx < len(thumbnails):
                    with cols[col_idx]:
                        thumb = thumbnails[img_idx]
                        
                        if thumb.get('rgb_image') is not None:
                            st.image(
                                thumb['rgb_image'],
                                caption=f"{thumb['month_name']} (RGB)",
                                use_column_width=True
                            )


# =============================================================================
# Main Application
# =============================================================================
def main():
    st.title("🏗️ Building Classification Time Series v05")
    st.markdown("""
    **Features:**
    - ✅ Smart nodata handling: Only processes patches valid in ALL months
    - ✅ Full cache support: Resume interrupted downloads
    - ✅ Raw DN values (no 0.0001 scaling)
    - ✅ Per-patch global normalization (matches working app)
    - ✅ RGB thumbnails alongside classifications
    """)
    
    # Initialize Earth Engine
    ee_initialized, ee_message = initialize_earth_engine()
    
    if not ee_initialized:
        st.error(ee_message)
        st.stop()
    else:
        st.sidebar.success(ee_message)
    
    # Model Loading
    st.sidebar.header("🧠 Model Status")
    
    model_path = "best_model_version_Unet++_v02_e7.pt"
    gdrive_model_url = "https://drive.google.com/file/d/1m6EScw-mpBIvWV78h4pyjWq1OLQtn2ov/view?usp=drive_link"
    
    if not os.path.exists(model_path):
        st.sidebar.info("Model not found locally. Downloading...")
        downloaded_path = download_model_from_gdrive(gdrive_model_url, model_path)
        if downloaded_path is None:
            st.sidebar.error("Model download failed. Please upload manually.")
            st.stop()
    
    if not st.session_state.model_loaded:
        with st.spinner("Loading UNet++ model..."):
            model, device = load_model(model_path)
            if model is not None:
                st.session_state.model = model
                st.session_state.device = device
                st.session_state.model_loaded = True
                st.sidebar.success("✅ Model loaded successfully!")
            else:
                st.sidebar.error("❌ Failed to load model")
                st.stop()
    else:
        st.sidebar.success("✅ Model already loaded")
    
    # Sidebar - Parameters
    st.sidebar.header("⚙️ Parameters")
    
    cloudy_pixel_percentage = st.sidebar.slider(
        "Max Cloud Cover %", 0, 100, 20, 5
    )
    
    nodata_threshold_percent = st.sidebar.slider(
        "Nodata Threshold %",
        min_value=1, max_value=50, value=5, step=1,
        help="Patches with more than this % of zero values will be considered invalid"
    )
    
    # Sidebar - Cache Management (KEPT FROM V04)
    st.sidebar.header("🗂️ Cache Management")
    
    cache_info = []
    if st.session_state.downloaded_images:
        cache_info.append(f"📥 {len(st.session_state.downloaded_images)} images downloaded")
    if st.session_state.processed_months:
        cache_info.append(f"🧠 {len(st.session_state.processed_months)} months classified")
    if st.session_state.valid_patches_mask is not None:
        valid_count = np.sum(st.session_state.valid_patches_mask)
        total_count = st.session_state.valid_patches_mask.size
        cache_info.append(f"✅ {valid_count}/{total_count} valid patches")
    
    if cache_info:
        for info in cache_info:
            st.sidebar.success(info)
    else:
        st.sidebar.info("No cached data")
    
    if st.session_state.failed_months:
        st.sidebar.warning(f"❌ {len(st.session_state.failed_months)} failed: {', '.join(st.session_state.failed_months)}")
    
    if st.sidebar.button("🗑️ Clear All Cache"):
        st.session_state.processed_months = {}
        st.session_state.downloaded_images = {}
        st.session_state.failed_months = []
        st.session_state.classification_thumbnails = []
        st.session_state.processing_complete = False
        st.session_state.current_temp_dir = None
        st.session_state.valid_patches_mask = None
        st.session_state.download_phase_complete = False
        st.session_state.validity_analysis_complete = False
        st.sidebar.success("Cache cleared!")
        st.rerun()
    
    # Region Selection
    st.header("1️⃣ Select Region of Interest")
    
    m = folium.Map(location=[35.6892, 51.3890], zoom_start=8)
    
    draw = plugins.Draw(
        export=True,
        position='topleft',
        draw_options={
            'polyline': False,
            'rectangle': True,
            'polygon': True,
            'circle': False,
            'marker': False,
            'circlemarker': False
        }
    )
    m.add_child(draw)
    
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google Satellite',
        name='Google Satellite'
    ).add_to(m)
    
    folium.LayerControl().add_to(m)
    
    map_data = st_folium(m, width=800, height=500)
    
    if map_data is not None and 'last_active_drawing' in map_data and map_data['last_active_drawing'] is not None:
        drawn_shape = map_data['last_active_drawing']
        if 'geometry' in drawn_shape:
            geometry = drawn_shape['geometry']
            if geometry['type'] == 'Polygon':
                coords = geometry['coordinates'][0]
                polygon = Polygon(coords)
                st.session_state.last_drawn_polygon = polygon
                
                centroid = polygon.centroid
                utm_zone = get_utm_zone(centroid.x)
                utm_epsg = get_utm_epsg(centroid.x, centroid.y)
                area_sq_km = polygon.area * 111 * 111
                
                st.success(f"✅ Region captured! UTM Zone {utm_zone} ({utm_epsg}), Area: ~{area_sq_km:.2f} km²")
    
    if st.button("💾 Save Selected Region"):
        if st.session_state.last_drawn_polygon is not None:
            if not any(p.equals(st.session_state.last_drawn_polygon) for p in st.session_state.drawn_polygons):
                st.session_state.drawn_polygons.append(st.session_state.last_drawn_polygon)
                st.success(f"✅ Region saved!")
    
    # Saved Regions
    if st.session_state.drawn_polygons:
        st.subheader("📍 Saved Regions")
        
        for i, poly in enumerate(st.session_state.drawn_polygons):
            col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
            
            with col1:
                st.write(f"**Region {i+1}**")
            
            with col2:
                centroid = poly.centroid
                st.write(f"UTM: {get_utm_zone(centroid.x)}")
            
            with col3:
                area_sq_km = poly.area * 111 * 111
                st.write(f"Area: ~{area_sq_km:.2f} km²")
            
            with col4:
                if st.button("🗑️", key=f"delete_region_{i}"):
                    st.session_state.drawn_polygons.pop(i)
                    st.rerun()
    
    # Date Selection
    st.header("2️⃣ Select Time Period")
    
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=date(2023, 6, 1),
            min_value=date(2017, 1, 1),
            max_value=date.today()
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=date(2024, 2, 1),
            min_value=date(2017, 1, 1),
            max_value=date.today()
        )
    
    if start_date >= end_date:
        st.error("❌ End date must be after start date!")
        st.stop()
    
    num_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
    st.info(f"📅 Time period: {start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')} ({num_months} months)")
    
    # Process Buttons
    st.header("3️⃣ Generate Building Classifications")
    
    selected_polygon = None
    if len(st.session_state.drawn_polygons) > 0:
        polygon_index = st.selectbox(
            "Select region to process",
            range(len(st.session_state.drawn_polygons)),
            format_func=lambda i: f"Region {i+1} (~{st.session_state.drawn_polygons[i].area * 111 * 111:.2f} km²)",
            key="polygon_selector"
        )
        selected_polygon = st.session_state.drawn_polygons[polygon_index]
    elif st.session_state.last_drawn_polygon is not None:
        selected_polygon = st.session_state.last_drawn_polygon
        st.info("Using the last drawn polygon (not saved)")
    
    # Show current progress
    if st.session_state.downloaded_images:
        st.info(f"📊 Current progress: {len(st.session_state.downloaded_images)} images downloaded, {len(st.session_state.processed_months)} classified")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        start_new = st.button("🚀 Start New Processing", type="primary")
    
    with col2:
        resume_processing = st.button(
            "🔄 Resume Processing", 
            disabled=not (st.session_state.downloaded_images or st.session_state.processed_months)
        )
    
    with col3:
        retry_failed = st.button(
            "🔁 Retry Failed", 
            disabled=not st.session_state.failed_months
        )
    
    should_process = False
    resume_mode = False
    
    if start_new:
        should_process = True
        resume_mode = False
        # Clear cache
        st.session_state.processed_months = {}
        st.session_state.downloaded_images = {}
        st.session_state.failed_months = []
        st.session_state.classification_thumbnails = []
        st.session_state.processing_complete = False
        st.session_state.valid_patches_mask = None
        st.session_state.download_phase_complete = False
        st.session_state.validity_analysis_complete = False
        st.session_state.current_temp_dir = None
        
    elif resume_processing:
        should_process = True
        resume_mode = True
        
    elif retry_failed:
        should_process = True
        resume_mode = True
        # Only clear failed months, keep successful ones
        st.session_state.failed_months = []
    
    if should_process:
        if selected_polygon is None:
            st.error("❌ Please select a region of interest first!")
            st.stop()
        
        if not st.session_state.model_loaded:
            st.error("❌ Model not loaded!")
            st.stop()
        
        geojson = {"type": "Polygon", "coordinates": [list(selected_polygon.exterior.coords)]}
        aoi = ee.Geometry.Polygon(geojson['coordinates'])
        
        thumbnails = process_timeseries_with_nodata_handling(
            aoi=aoi,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            model=st.session_state.model,
            device=st.session_state.device,
            cloudy_pixel_percentage=cloudy_pixel_percentage,
            nodata_threshold_percent=nodata_threshold_percent,
            resume=resume_mode
        )
        
        if thumbnails:
            st.session_state.classification_thumbnails = thumbnails
            st.session_state.processing_complete = True
    
    # Display Results
    if st.session_state.processing_complete and st.session_state.classification_thumbnails:
        st.divider()
        st.header("📊 Results")
        
        if st.session_state.valid_patches_mask is not None:
            valid_count = np.sum(st.session_state.valid_patches_mask)
            total_count = st.session_state.valid_patches_mask.size
            st.info(f"Showing {len(st.session_state.classification_thumbnails)} months using {valid_count}/{total_count} patches ({100*valid_count/total_count:.1f}%) valid across all months")
        
        display_classification_thumbnails(st.session_state.classification_thumbnails)


if __name__ == "__main__":
    main()
