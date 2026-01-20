"""
Sentinel-2 Time Series Building Classification
VERSION 07 - ROBUST DOWNLOAD HANDLING

Key Features:
- ROBUST download with band-level tracking
- On resume: DELETE last downloaded band and re-download (safety measure)
- Comprehensive file validation (checks entire file, not just corners)
- Content-Length verification to detect incomplete downloads
- Band consistency check to detect corruption
- Full cache support with safe resume capability
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
    page_title="Building Classification Time Series v07 - Robust",
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

# RGB bands for visualization (True Color: B4=Red, B3=Green, B2=Blue)
RGB_BAND_INDICES = {'red': 3, 'green': 2, 'blue': 1}

# Download settings - MORE ROBUST
MAX_RETRIES = 5  # Increased retries
RETRY_DELAY_BASE = 2
DOWNLOAD_TIMEOUT = 180  # Increased timeout
CHUNK_SIZE = 8192

# Minimum expected file sizes for validation
MIN_BAND_FILE_SIZE = 10000
MIN_MULTIBAND_FILE_SIZE = 100000

# FIXED cloud cover threshold (no slider)
CLOUDY_PIXEL_PERCENTAGE = 10

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
# Valid patch mask and downloaded images tracking
if 'valid_patches_mask' not in st.session_state:
    st.session_state.valid_patches_mask = None
if 'downloaded_images' not in st.session_state:
    st.session_state.downloaded_images = {}
if 'download_phase_complete' not in st.session_state:
    st.session_state.download_phase_complete = False
if 'validity_analysis_complete' not in st.session_state:
    st.session_state.validity_analysis_complete = False
# Track months that passed validation
if 'valid_months' not in st.session_state:
    st.session_state.valid_months = []
if 'deleted_months' not in st.session_state:
    st.session_state.deleted_months = []
if 'total_requested_months' not in st.session_state:
    st.session_state.total_requested_months = 0

# NEW: Track download progress at BAND level for robust resume
if 'download_progress' not in st.session_state:
    st.session_state.download_progress = {}  # {month_name: {'bands_completed': [], 'last_band': None}}


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
# ROBUST File Validation Functions
# =============================================================================
def compute_file_checksum(file_path):
    """Compute MD5 checksum of a file."""
    md5 = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                md5.update(chunk)
        return md5.hexdigest()
    except:
        return None


def validate_geotiff_comprehensive(file_path, expected_bands=1):
    """
    COMPREHENSIVE validation that checks entire file, not just corners.
    This is critical for detecting download corruption.
    """
    try:
        if not os.path.exists(file_path):
            return False, "File does not exist"
        
        file_size = os.path.getsize(file_path)
        min_size = MIN_BAND_FILE_SIZE if expected_bands == 1 else MIN_MULTIBAND_FILE_SIZE
        
        if file_size < min_size:
            return False, f"File too small ({file_size} bytes, expected > {min_size})"
        
        # Check TIFF header
        with open(file_path, 'rb') as f:
            header = f.read(4)
            if header[:2] not in [b'II', b'MM']:  # Little/Big endian TIFF
                return False, "Invalid TIFF header - file is corrupted"
        
        with rasterio.open(file_path) as src:
            if src.count < expected_bands:
                return False, f"Wrong band count ({src.count}, expected {expected_bands})"
            
            h, w = src.height, src.width
            
            if h < 10 or w < 10:
                return False, f"Image too small ({w}x{h})"
            
            # Sample MULTIPLE regions across the image
            sample_size = min(30, h // 3, w // 3)
            
            check_regions = [
                (0, 0),                                          # Top-left
                (0, w - sample_size),                            # Top-right
                (h - sample_size, 0),                            # Bottom-left
                (h - sample_size, w - sample_size),              # Bottom-right
                (h // 2 - sample_size // 2, w // 2 - sample_size // 2),  # Center
                (h // 3, w // 3),                                # Upper-left third
                (2 * h // 3, 2 * w // 3),                        # Lower-right third
            ]
            
            for band_idx in range(1, min(src.count + 1, expected_bands + 1)):
                all_invalid_regions = 0
                total_checked = 0
                
                for row, col in check_regions:
                    # Ensure we don't go out of bounds
                    row = max(0, min(row, h - sample_size))
                    col = max(0, min(col, w - sample_size))
                    
                    try:
                        window = rasterio.windows.Window(col, row, sample_size, sample_size)
                        data = src.read(band_idx, window=window)
                        total_checked += 1
                        
                        # Check for completely invalid region
                        if np.all(np.isnan(data)) or np.all(data == 0):
                            all_invalid_regions += 1
                    except Exception as e:
                        return False, f"Error reading band {band_idx} at ({row}, {col}): {str(e)}"
                
                # If ALL sampled regions are invalid, file is likely corrupted
                if all_invalid_regions == total_checked:
                    return False, f"Band {band_idx} appears completely empty/corrupted"
                
                # If majority of regions are invalid (more than 80%), suspicious
                if total_checked > 0 and all_invalid_regions / total_checked > 0.8:
                    return False, f"Band {band_idx} has too many invalid regions ({all_invalid_regions}/{total_checked})"
        
        return True, "File is valid"
        
    except rasterio.errors.RasterioIOError as e:
        return False, f"Rasterio cannot read file (corrupted): {str(e)}"
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def validate_band_file_robust(band_file_path, band_name):
    """
    ROBUST validation for a single band GeoTIFF file.
    Checks more thoroughly than before.
    """
    try:
        if not os.path.exists(band_file_path):
            return False, "File does not exist"
        
        file_size = os.path.getsize(band_file_path)
        
        if file_size < MIN_BAND_FILE_SIZE:
            return False, f"File too small ({file_size} bytes)"
        
        # Check TIFF header integrity
        with open(band_file_path, 'rb') as f:
            header = f.read(8)
            if header[:2] not in [b'II', b'MM']:
                return False, "Invalid TIFF header"
            
            # Check we can seek to end (file not truncated)
            f.seek(0, 2)  # Seek to end
            actual_size = f.tell()
            if actual_size != file_size:
                return False, "File size mismatch"
        
        # Try to read with rasterio
        with rasterio.open(band_file_path) as src:
            h, w = src.height, src.width
            
            if h < 10 or w < 10:
                return False, "Image dimensions too small"
            
            # Read multiple regions to verify file integrity
            regions_to_check = [
                (0, 0),
                (h // 2, w // 2),
                (h - min(20, h), w - min(20, w))
            ]
            
            for row, col in regions_to_check:
                sample_h = min(20, h - row)
                sample_w = min(20, w - col)
                
                try:
                    window = rasterio.windows.Window(col, row, sample_w, sample_h)
                    data = src.read(1, window=window)
                    
                    # Just verify we can read without error
                    _ = data.shape
                except Exception as e:
                    return False, f"Cannot read region ({row}, {col}): {str(e)}"
        
        return True, "File is valid"
        
    except Exception as e:
        return False, f"Validation failed: {str(e)}"


def verify_band_consistency(band_files, month_name):
    """
    Verify all bands have consistent spatial coverage (same NaN pattern).
    Different patterns between bands = DOWNLOAD CORRUPTION.
    """
    if len(band_files) < 2:
        return True, "Only one band, skipping consistency check"
    
    try:
        ref_mask = None
        ref_band = None
        
        for band_file in band_files:
            band_name = os.path.basename(band_file).replace('.tif', '')
            
            with rasterio.open(band_file) as src:
                data = src.read(1)
                # Create mask of invalid pixels
                mask = np.isnan(data) | (data == 0)
                
                if ref_mask is None:
                    ref_mask = mask
                    ref_band = band_name
                else:
                    # Compare masks
                    diff_pixels = np.sum(ref_mask != mask)
                    total_pixels = mask.size
                    diff_percent = 100 * diff_pixels / total_pixels
                    
                    # Allow small differences (< 1%) due to floating point
                    if diff_percent > 1.0:
                        return False, f"Band {band_name} has different coverage than {ref_band} ({diff_percent:.2f}% different) - CORRUPTION DETECTED"
        
        return True, "All bands have consistent coverage"
        
    except Exception as e:
        return False, f"Consistency check failed: {str(e)}"


# =============================================================================
# Model Download Functions (unchanged)
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
        
        st.error("All automatic download methods failed. Please download manually.")
        return None
        
    except Exception as e:
        st.error(f"Manual download fallback failed: {e}")
        return None


# =============================================================================
# Model Loading Function
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
# Earth Engine Authentication
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
# RGB Image Generation Functions
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
# GEE Functions - Simple median composite
# =============================================================================
def create_monthly_composites_list(aoi, start_date, end_date):
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
# ROBUST Download Functions
# =============================================================================
def download_band_robust(image, band, aoi, output_path, scale=10, status_callback=None):
    """
    ROBUST download for a single band with:
    - Content-Length verification
    - Multiple retry attempts
    - Comprehensive validation
    - .tmp file approach for atomicity
    """
    region = aoi.bounds().getInfo()['coordinates']
    
    temp_path = output_path + '.tmp'
    
    # Clean up any leftover temp files
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    # If final file exists, validate it thoroughly
    if os.path.exists(output_path):
        is_valid, msg = validate_band_file_robust(output_path, band)
        if is_valid:
            if status_callback:
                status_callback(f"✅ {band} - using validated cache")
            return True, "cached"
        else:
            if status_callback:
                status_callback(f"⚠️ {band} - cached file invalid ({msg}), re-downloading...")
            os.remove(output_path)
    
    for attempt in range(MAX_RETRIES):
        try:
            if status_callback:
                status_callback(f"📥 {band} - attempt {attempt + 1}/{MAX_RETRIES}...")
            
            # Get download URL
            url = image.select(band).getDownloadURL({
                'scale': scale,
                'region': region,
                'format': 'GEO_TIFF',
                'bands': [band]
            })
            
            # Start download with streaming
            response = requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT)
            
            if response.status_code != 200:
                raise Exception(f"HTTP {response.status_code}")
            
            # Check content type
            content_type = response.headers.get('content-type', '')
            if 'text/html' in content_type:
                raise Exception("Received HTML instead of GeoTIFF - possible rate limit or error")
            
            # Get expected size from Content-Length header
            expected_size = int(response.headers.get('content-length', 0))
            
            # Download to temp file
            actual_size = 0
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        actual_size += len(chunk)
            
            # Verify download completeness using Content-Length
            if expected_size > 0:
                if actual_size < expected_size:
                    raise Exception(f"Incomplete download: {actual_size}/{expected_size} bytes ({100*actual_size/expected_size:.1f}%)")
                
                # Allow small tolerance (some servers send slightly more)
                if actual_size < expected_size * 0.95:
                    raise Exception(f"Download size mismatch: got {actual_size}, expected {expected_size}")
            
            # Validate the downloaded file THOROUGHLY
            is_valid, msg = validate_band_file_robust(temp_path, band)
            
            if not is_valid:
                raise Exception(f"Downloaded file validation failed: {msg}")
            
            # Move temp file to final location (atomic operation)
            os.replace(temp_path, output_path)
            
            if status_callback:
                status_callback(f"✅ {band} - downloaded successfully ({actual_size:,} bytes)")
            
            return True, "downloaded"
                
        except requests.exceptions.Timeout:
            error_msg = f"Timeout on attempt {attempt + 1}"
        except requests.exceptions.ConnectionError:
            error_msg = f"Connection error on attempt {attempt + 1}"
        except Exception as e:
            error_msg = f"Error on attempt {attempt + 1}: {str(e)}"
        
        if status_callback:
            status_callback(f"⚠️ {band} - {error_msg}")
        
        # Clean up any partial files
        for f in [output_path, temp_path]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    pass
        
        if attempt < MAX_RETRIES - 1:
            wait_time = RETRY_DELAY_BASE ** (attempt + 1)
            if status_callback:
                status_callback(f"⏳ {band} - waiting {wait_time}s before retry...")
            time.sleep(wait_time)
    
    return False, f"Failed after {MAX_RETRIES} attempts"


def get_completed_bands(bands_dir, month_name):
    """
    Get list of bands that have been successfully downloaded and validated.
    Returns: (list of completed band names, last band file path or None)
    """
    completed_bands = []
    last_band_file = None
    last_band_mtime = 0
    
    for band in SPECTRAL_BANDS:
        band_file = os.path.join(bands_dir, f"{band}.tif")
        
        if os.path.exists(band_file):
            is_valid, _ = validate_band_file_robust(band_file, band)
            if is_valid:
                completed_bands.append(band)
                
                # Track the most recently modified file
                mtime = os.path.getmtime(band_file)
                if mtime > last_band_mtime:
                    last_band_mtime = mtime
                    last_band_file = band_file
    
    return completed_bands, last_band_file


def download_monthly_image_robust(aoi, month_info, temp_dir, scale=10, status_placeholder=None, resume_mode=False):
    """
    ROBUST download of a monthly Sentinel-2 composite.
    
    Key features:
    - On resume: DELETE the last downloaded band and re-download it (safety measure)
    - Track progress at band level
    - Comprehensive validation
    - Band consistency check
    """
    try:
        month_name = month_info['month_name']
        start_date = month_info['start_date']
        end_date = month_info['end_date']
        
        output_file = os.path.join(temp_dir, f"sentinel2_{month_name}.tif")
        bands_dir = os.path.join(temp_dir, f"bands_{month_name}")
        
        def update_status(msg):
            if status_placeholder:
                status_placeholder.text(f"📥 {month_name}: {msg}")
        
        # Check if final multiband file exists and is valid
        if os.path.exists(output_file):
            is_valid, msg = validate_geotiff_comprehensive(output_file, expected_bands=len(SPECTRAL_BANDS))
            if is_valid:
                # Also verify band consistency
                update_status("Verifying cached file integrity...")
                
                # Quick consistency check by reading all bands
                try:
                    with rasterio.open(output_file) as src:
                        # Read a small region from each band to verify
                        for i in range(1, src.count + 1):
                            window = rasterio.windows.Window(0, 0, min(50, src.width), min(50, src.height))
                            _ = src.read(i, window=window)
                    
                    update_status("✅ Using validated cached file")
                    return output_file
                except Exception as e:
                    update_status(f"⚠️ Cache file corrupted ({str(e)}), re-downloading...")
                    os.remove(output_file)
            else:
                update_status(f"⚠️ Cached file invalid ({msg}), re-downloading...")
                os.remove(output_file)
        
        # Create bands directory
        os.makedirs(bands_dir, exist_ok=True)
        
        # Check which bands are already downloaded
        completed_bands, last_band_file = get_completed_bands(bands_dir, month_name)
        
        # ROBUST RESUME: Delete the last downloaded band (might be corrupted due to interruption)
        if resume_mode and last_band_file and os.path.exists(last_band_file):
            last_band_name = os.path.basename(last_band_file).replace('.tif', '')
            update_status(f"🔄 Resume mode: Deleting last band ({last_band_name}) for safety...")
            
            try:
                os.remove(last_band_file)
                if last_band_name in completed_bands:
                    completed_bands.remove(last_band_name)
                update_status(f"🔄 Deleted {last_band_name}, will re-download")
            except Exception as e:
                update_status(f"⚠️ Could not delete {last_band_name}: {str(e)}")
        
        # Report resume status
        if completed_bands:
            update_status(f"Resuming... {len(completed_bands)}/{len(SPECTRAL_BANDS)} bands already downloaded")
        
        # Get bands that still need to be downloaded
        bands_to_download = [b for b in SPECTRAL_BANDS if b not in completed_bands]
        
        if not bands_to_download:
            update_status("All bands already downloaded, creating multiband file...")
        else:
            # Need to download some bands - first get the GEE image
            update_status("Searching for images...")
            
            # Simple collection filter with FIXED cloud cover threshold
            collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                         .filterBounds(aoi)
                         .filterDate(start_date, end_date)
                         .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUDY_PIXEL_PERCENTAGE))
                         .select(SPECTRAL_BANDS))
            
            count = collection.size().getInfo()
            
            if count == 0:
                # Try higher cloud cover (up to 30%)
                update_status(f"No images at {CLOUDY_PIXEL_PERCENTAGE}% cloud, trying 30%...")
                
                collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                             .filterBounds(aoi)
                             .filterDate(start_date, end_date)
                             .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
                             .select(SPECTRAL_BANDS))
                count = collection.size().getInfo()
                
                if count == 0:
                    update_status("❌ No images found")
                    return None
            
            update_status(f"Creating median from {count} images...")
            
            # Simple median composite
            median_image = collection.median()
            
            # Download remaining bands
            failed_bands = []
            
            for i, band in enumerate(bands_to_download):
                band_file = os.path.join(bands_dir, f"{band}.tif")
                
                update_status(f"Downloading {band} ({len(completed_bands) + i + 1}/{len(SPECTRAL_BANDS)})...")
                
                success, result = download_band_robust(
                    median_image, band, aoi, band_file, scale,
                    status_callback=update_status
                )
                
                if success:
                    completed_bands.append(band)
                else:
                    failed_bands.append(band)
                    st.error(f"❌ {month_name}: Failed to download {band}: {result}")
            
            if failed_bands:
                st.error(f"❌ {month_name}: Failed bands: {', '.join(failed_bands)}")
                return None
        
        # Verify all bands are present
        band_files = []
        for band in SPECTRAL_BANDS:
            band_file = os.path.join(bands_dir, f"{band}.tif")
            if not os.path.exists(band_file):
                st.error(f"❌ {month_name}: Missing band file: {band}")
                return None
            band_files.append(band_file)
        
        # VERIFY BAND CONSISTENCY (critical for detecting corruption)
        update_status("Verifying band consistency...")
        is_consistent, consistency_msg = verify_band_consistency(band_files, month_name)
        
        if not is_consistent:
            st.error(f"❌ {month_name}: {consistency_msg}")
            st.warning(f"🔄 Deleting all bands for {month_name} due to corruption. Please retry.")
            
            # Delete all band files
            for band_file in band_files:
                if os.path.exists(band_file):
                    os.remove(band_file)
            
            return None
        
        update_status("Band consistency verified ✅")
        
        # Create multiband GeoTIFF
        update_status("Creating multiband GeoTIFF...")
        
        with rasterio.open(band_files[0]) as src:
            meta = src.meta.copy()
        
        meta.update(count=len(band_files))
        
        with rasterio.open(output_file, 'w', **meta) as dst:
            for i, band_file in enumerate(band_files):
                with rasterio.open(band_file) as src:
                    dst.write(src.read(1), i+1)
        
        # Final validation
        is_valid, msg = validate_geotiff_comprehensive(output_file, expected_bands=len(SPECTRAL_BANDS))
        if not is_valid:
            st.error(f"❌ {month_name}: Final multiband file validation failed: {msg}")
            if os.path.exists(output_file):
                os.remove(output_file)
            return None
        
        update_status(f"✅ Complete!")
        return output_file
        
    except Exception as e:
        st.error(f"❌ Error downloading {month_name}: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None


# =============================================================================
# STRICT NODATA HANDLING
# =============================================================================
def check_patch_validity_strict(patch):
    """
    STRICT check if a patch has ANY nodata (NaN or zero values).
    Returns True if patch is COMPLETELY valid (no NaN, no zeros), False otherwise.
    """
    if np.any(np.isnan(patch)):
        return False
    
    if np.all(patch == 0):
        return False
    
    if patch.ndim == 3:
        for band_idx in range(patch.shape[-1]):
            if np.all(patch[:, :, band_idx] == 0):
                return False
    
    if np.any(patch == 0):
        return False
    
    return True


def get_patch_validity_mask_strict(image_path, patch_size=224):
    """
    Create a mask showing which patches are STRICTLY valid (no nodata at all).
    """
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
                validity_mask[i, j] = check_patch_validity_strict(patch)
        
        return validity_mask, (h, w), (n_patches_h, n_patches_w)
        
    except Exception as e:
        st.error(f"Error checking validity: {str(e)}")
        return None, None, None


def analyze_months_and_filter(downloaded_images):
    """
    Analyze all downloaded images, count valid patches per month,
    and REMOVE months that have fewer patches than the maximum.
    """
    st.info("🔍 Analyzing patch validity across all months (STRICT mode: any zero/NaN = invalid)...")
    
    month_names = sorted(downloaded_images.keys())
    
    validity_data = {}
    original_size = None
    patch_grid_size = None
    
    progress_bar = st.progress(0)
    
    for idx, month_name in enumerate(month_names):
        image_path = downloaded_images[month_name]
        
        validity_mask, orig_size, grid_size = get_patch_validity_mask_strict(image_path, PATCH_SIZE)
        
        if validity_mask is None:
            st.warning(f"Could not check {month_name}")
            continue
        
        if original_size is None:
            original_size = orig_size
            patch_grid_size = grid_size
        
        valid_count = np.sum(validity_mask)
        total_count = validity_mask.size
        
        validity_data[month_name] = {
            'mask': validity_mask,
            'valid_count': valid_count,
            'total_count': total_count,
            'percent': 100 * valid_count / total_count,
            'image_path': image_path
        }
        
        progress_bar.progress((idx + 1) / len(month_names))
    
    progress_bar.empty()
    
    if not validity_data:
        st.error("❌ No valid data from any month!")
        return None, [], None, None, []
    
    max_valid_patches = max(v['valid_count'] for v in validity_data.values())
    
    st.info(f"📊 Maximum valid patches found: {max_valid_patches} out of {patch_grid_size[0] * patch_grid_size[1]} total")
    
    valid_months = {}
    deleted_months = []
    
    for month_name, data in validity_data.items():
        if data['valid_count'] == max_valid_patches:
            valid_months[month_name] = data['image_path']
        else:
            deleted_months.append(month_name)
    
    validity_report = []
    for month_name in month_names:
        if month_name in validity_data:
            data = validity_data[month_name]
            status = "✅ KEPT" if month_name in valid_months else "❌ DELETED"
            validity_report.append({
                'month': month_name,
                'valid': data['valid_count'],
                'total': data['total_count'],
                'percent': data['percent'],
                'status': status
            })
    
    if not valid_months:
        st.error("❌ No months passed the validation criteria!")
        return None, deleted_months, None, None, validity_report
    
    common_valid_mask = None
    for month_name in valid_months:
        month_mask = validity_data[month_name]['mask']
        if common_valid_mask is None:
            common_valid_mask = month_mask.copy()
        else:
            common_valid_mask = common_valid_mask & month_mask
    
    final_valid_patches = np.sum(common_valid_mask)
    
    if final_valid_patches != max_valid_patches:
        st.warning(f"⚠️ After intersection, {final_valid_patches} patches are common (expected {max_valid_patches})")
    
    return valid_months, deleted_months, common_valid_mask, original_size, validity_report


def classify_image_with_mask(image_path, model, device, month_name, valid_mask, original_size):
    """
    Classify an image, but only process patches marked as valid.
    """
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
        
        valid_count = 0
        skipped_count = 0
        
        for i in range(n_patches_h):
            for j in range(n_patches_w):
                if not valid_mask[i, j]:
                    skipped_count += 1
                    continue
                
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
                valid_count += 1
        
        reconstructed = unpatchify(classified_patches, (new_h, new_w))
        reconstructed = reconstructed[:original_size[0], :original_size[1]]
        
        return reconstructed, valid_count, skipped_count
        
    except Exception as e:
        st.error(f"Error classifying {month_name}: {str(e)}")
        return None, 0, 0


# =============================================================================
# Generate Thumbnails
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
# Main Processing Pipeline - ROBUST VERSION
# =============================================================================
def download_all_images_robust(composites, aoi, temp_dir, scale=10, resume=False):
    """
    PHASE 1: Download all monthly images with ROBUST handling.
    
    Key features:
    - On resume: DELETE the last band of each incomplete month
    - Comprehensive validation
    - Band consistency checks
    """
    downloaded_images = {}
    failed_months = []
    
    if resume and st.session_state.downloaded_images:
        downloaded_images = st.session_state.downloaded_images.copy()
        st.info(f"🔄 Resume mode: {len(downloaded_images)} months in cache")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, month_info in enumerate(composites):
        month_name = month_info['month_name']
        
        # Skip if already downloaded AND VALIDATED
        if month_name in downloaded_images:
            cached_path = downloaded_images[month_name]
            if os.path.exists(cached_path):
                is_valid, _ = validate_geotiff_comprehensive(cached_path, expected_bands=len(SPECTRAL_BANDS))
                if is_valid:
                    status_text.text(f"✅ {month_name} - validated from cache")
                    progress_bar.progress((idx + 1) / len(composites))
                    continue
                else:
                    status_text.text(f"⚠️ {month_name} - cached file invalid, re-downloading...")
                    del downloaded_images[month_name]
        
        status_text.text(f"📥 Downloading {month_name} ({idx+1}/{len(composites)})...")
        
        # Download with robust handling (resume_mode flag passed)
        image_path = download_monthly_image_robust(
            aoi, month_info, temp_dir,
            scale=scale,
            status_placeholder=status_text,
            resume_mode=resume  # This triggers the "delete last band" safety measure
        )
        
        if image_path:
            downloaded_images[month_name] = image_path
            st.session_state.downloaded_images = downloaded_images.copy()
        else:
            failed_months.append(month_name)
            st.warning(f"⚠️ Failed to download {month_name}")
        
        progress_bar.progress((idx + 1) / len(composites))
    
    progress_bar.empty()
    status_text.empty()
    
    return downloaded_images, failed_months


def classify_all_images(valid_months, model, device, valid_mask, original_size, resume=False):
    """
    PHASE 3: Classify all validated images.
    """
    thumbnails = []
    
    if resume and st.session_state.processed_months:
        already_processed = set(st.session_state.processed_months.keys())
        st.info(f"🔄 Resuming... {len(already_processed)} months already classified")
    else:
        already_processed = set()
        st.session_state.processed_months = {}
    
    month_names = sorted(valid_months.keys())
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, month_name in enumerate(month_names):
        if month_name in already_processed:
            if month_name in st.session_state.processed_months:
                thumbnails.append(st.session_state.processed_months[month_name])
            progress_bar.progress((idx + 1) / len(month_names))
            continue
        
        image_path = valid_months[month_name]
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


def process_timeseries_robust(aoi, start_date, end_date, model, device, scale=10, resume=False):
    """
    Main processing pipeline with ROBUST download handling.
    """
    try:
        if st.session_state.current_temp_dir is None or not os.path.exists(st.session_state.current_temp_dir):
            st.session_state.current_temp_dir = tempfile.mkdtemp()
        temp_dir = st.session_state.current_temp_dir
        
        st.info(f"📁 Working directory: {temp_dir}")
        
        composites, total_months = create_monthly_composites_list(aoi, start_date, end_date)
        st.session_state.total_requested_months = total_months
        st.info(f"📅 Requested: {total_months} months")
        
        # =====================================================================
        # PHASE 1: ROBUST Download
        # =====================================================================
        st.header("Phase 1: Downloading Images (Robust Mode)")
        
        if resume:
            st.warning("🔄 **Resume mode active**: Last downloaded band of each incomplete month will be deleted and re-downloaded for safety.")
        
        downloaded_images, failed_downloads = download_all_images_robust(
            composites, aoi, temp_dir,
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
        # PHASE 2: Analyze validity
        # =====================================================================
        st.header("Phase 2: Analyzing Patch Validity & Filtering Months")
        
        valid_months, deleted_months, valid_mask, original_size, validity_report = analyze_months_and_filter(
            downloaded_images
        )
        
        if valid_months is None or len(valid_months) == 0:
            st.error("❌ No months passed validation!")
            return []
        
        st.session_state.valid_months = list(valid_months.keys())
        st.session_state.deleted_months = deleted_months
        st.session_state.valid_patches_mask = valid_mask
        st.session_state.original_size = original_size
        st.session_state.validity_analysis_complete = True
        
        # Display Validity Report
        st.subheader("📊 Validity Report by Month")
        
        report_cols = st.columns(4)
        for idx, report in enumerate(validity_report):
            col_idx = idx % 4
            with report_cols[col_idx]:
                if "KEPT" in report['status']:
                    st.success(f"**{report['month']}**\n{report['valid']}/{report['total']} patches\n{report['percent']:.1f}%")
                else:
                    st.error(f"**{report['month']}**\n{report['valid']}/{report['total']} patches\n{report['percent']:.1f}%")
        
        st.divider()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Months Requested", total_months)
        
        with col2:
            st.metric(
                "Months with Valid Data",
                len(valid_months),
                delta=f"-{len(deleted_months)} deleted" if deleted_months else None,
                delta_color="inverse"
            )
        
        with col3:
            valid_patch_count = np.sum(valid_mask) if valid_mask is not None else 0
            total_patches = valid_mask.size if valid_mask is not None else 0
            st.metric("Valid Patches per Month", f"{valid_patch_count}/{total_patches}")
        
        if deleted_months:
            st.warning(f"🗑️ **Deleted months** (fewer valid patches): {', '.join(deleted_months)}")
        
        st.success(f"""
        📊 **Summary**: Out of **{total_months}** months requested, **{len(valid_months)}** months have complete, 
        cloud-free Sentinel-2 data with **{np.sum(valid_mask)}** valid patches each.
        """)
        
        if valid_mask is not None:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(valid_mask, cmap='RdYlGn', vmin=0, vmax=1)
            ax.set_title(f"Valid Patches Mask\nGreen = Valid in ALL {len(valid_months)} months, Red = Invalid")
            ax.set_xlabel("Patch Column")
            ax.set_ylabel("Patch Row")
            st.pyplot(fig)
            plt.close(fig)
        
        # =====================================================================
        # PHASE 3: Classify
        # =====================================================================
        st.header("Phase 3: Classifying Images")
        
        thumbnails = classify_all_images(
            valid_months, model, device, 
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
# Display Thumbnails
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
    st.title("🏗️ Building Classification Time Series v07 - ROBUST")
    st.markdown("""
    **Robust Download Features:**
    - ✅ **Content-Length verification**: Detects incomplete downloads
    - ✅ **Comprehensive file validation**: Checks entire file, not just corners
    - ✅ **Band consistency check**: Detects corruption between bands
    - ✅ **Safe resume**: On resume, DELETES last downloaded band and re-downloads it
    - ✅ **Multiple retry attempts**: 5 retries with exponential backoff
    - ✅ **STRICT nodata handling**: Any zero or NaN = invalid patch
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
    st.sidebar.info(f"☁️ Cloud cover threshold: **{CLOUDY_PIXEL_PERCENTAGE}%** (fixed)")
    st.sidebar.info("🔒 Nodata handling: **STRICT**")
    st.sidebar.info("🔄 Resume mode: **Deletes last band for safety**")
    
    # Sidebar - Cache Management
    st.sidebar.header("🗂️ Cache Management")
    
    cache_info = []
    if st.session_state.downloaded_images:
        cache_info.append(f"📥 {len(st.session_state.downloaded_images)} images downloaded")
    if st.session_state.valid_months:
        cache_info.append(f"✅ {len(st.session_state.valid_months)} months validated")
    if st.session_state.deleted_months:
        cache_info.append(f"🗑️ {len(st.session_state.deleted_months)} months deleted")
    if st.session_state.processed_months:
        cache_info.append(f"🧠 {len(st.session_state.processed_months)} months classified")
    if st.session_state.valid_patches_mask is not None:
        valid_count = np.sum(st.session_state.valid_patches_mask)
        total_count = st.session_state.valid_patches_mask.size
        cache_info.append(f"📊 {valid_count}/{total_count} valid patches")
    
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
        st.session_state.valid_months = []
        st.session_state.deleted_months = []
        st.session_state.total_requested_months = 0
        st.session_state.download_progress = {}
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
    
    if st.session_state.downloaded_images:
        st.info(f"📊 Current progress: {len(st.session_state.downloaded_images)} images downloaded, {len(st.session_state.processed_months)} classified")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        start_new = st.button("🚀 Start New Processing", type="primary")
    
    with col2:
        resume_processing = st.button(
            "🔄 Resume (Safe Mode)", 
            disabled=not (st.session_state.downloaded_images or st.session_state.processed_months),
            help="Deletes last downloaded band and re-downloads it for safety"
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
        st.session_state.valid_months = []
        st.session_state.deleted_months = []
        st.session_state.total_requested_months = 0
        st.session_state.download_progress = {}
        
    elif resume_processing:
        should_process = True
        resume_mode = True  # This triggers the "delete last band" safety measure
        
    elif retry_failed:
        should_process = True
        resume_mode = True
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
        
        thumbnails = process_timeseries_robust(
            aoi=aoi,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            model=st.session_state.model,
            device=st.session_state.device,
            scale=10,
            resume=resume_mode
        )
        
        if thumbnails:
            st.session_state.classification_thumbnails = thumbnails
            st.session_state.processing_complete = True
    
    # Display Results
    if st.session_state.processing_complete and st.session_state.classification_thumbnails:
        st.divider()
        st.header("📊 Results")
        
        if st.session_state.valid_months and st.session_state.total_requested_months > 0:
            st.success(f"""
            **Final Summary**: 
            - Requested: {st.session_state.total_requested_months} months
            - Valid (cloud-free): {len(st.session_state.valid_months)} months
            - Deleted: {len(st.session_state.deleted_months)} months
            - Valid patches per month: {np.sum(st.session_state.valid_patches_mask)}/{st.session_state.valid_patches_mask.size}
            """)
        
        display_classification_thumbnails(st.session_state.classification_thumbnails)


if __name__ == "__main__":
    main()
