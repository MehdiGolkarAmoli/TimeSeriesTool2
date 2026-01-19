"""
Sentinel-2 Time Series Building Classification
A Streamlit application for viewing building classification results 
from cloud-free Sentinel-2 monthly composites using a UNet++ deep learning model.

VERSION 04 FIXES:
- FIXED: Removed 0.0001 scaling in GEE (now matches working app)
- FIXED: Normalization now happens per-patch globally (all bands together)
- FIXED: Cache validation with .tmp file approach
- Added RGB thumbnail display alongside classification masks

Workflow:
1. Generate cloud-free monthly composites on GEE
2. Download each monthly image locally (12 bands) - RAW DN VALUES
3. Apply patching → Normalize per-patch → Model inference → Reconstruction
4. Display RGB + classification thumbnails (binary masks)
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

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Examining the path of torch.classes")

import streamlit as st

st.set_page_config(
    layout="wide", 
    page_title="Building Classification Time Series",
    page_icon="🏗️"
)

import folium
from folium import plugins
from streamlit_folium import st_folium
import segmentation_models_pytorch as smp
from tqdm import tqdm

SPECTRAL_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
PATCH_SIZE = 224

# RGB bands for visualization (True Color: B4=Red, B3=Green, B2=Blue)
RGB_BAND_INDICES = {'red': 3, 'green': 2, 'blue': 1}  # 0-indexed positions in SPECTRAL_BANDS

# Download settings
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2
DOWNLOAD_TIMEOUT = 120
CHUNK_SIZE = 8192

# Minimum expected file sizes for validation
MIN_BAND_FILE_SIZE = 10000
MIN_MULTIBAND_FILE_SIZE = 100000

# Session state initialization
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


# =============================================================================
# FIXED: Normalization function - matches working app exactly
# =============================================================================
def normalized(img):
    """
    Normalize image data to range [0, 1]
    This normalizes the ENTIRE array globally (all bands together)
    
    Parameters:
    img (numpy.ndarray): Input image (can be 2D or 3D)
    
    Returns:
    numpy.ndarray: Normalized image with values in [0, 1]
    """
    min_val = np.nanmin(img)
    max_val = np.nanmax(img)
    
    if max_val == min_val:
        return np.zeros_like(img)
    
    img_norm = (img - min_val) / (max_val - min_val)
    return img_norm


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
# Model Download Functions
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
                p_low = np.percentile(band[band > 0], lower) if np.any(band > 0) else 0
                p_high = np.percentile(band[band > 0], upper) if np.any(band > 0) else 1
                
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
# FIXED: GEE Processing Functions - NO 0.0001 SCALING
# =============================================================================
def create_gapfilled_timeseries(aoi, start_date, end_date, 
                                 cloudy_pixel_percentage=10,
                                 cloud_probability_threshold=65,
                                 cdi_threshold=-0.5):
    """
    Create gap-filled monthly Sentinel-2 composites.
    
    FIXED: No longer multiplies by 0.0001 - keeps raw DN values to match working app.
    """
    
    start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    total_months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)
    
    start_date_ee = ee.Date(start_date)
    end_date_ee = ee.Date(end_date)
    num_months = ee.Number(total_months)
    
    extended_start = start_date_ee.advance(-2, 'month')
    extended_end = end_date_ee.advance(1, 'month')
    
    s2_sr = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
             .filterBounds(aoi)
             .filterDate(extended_start, extended_end)
             .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloudy_pixel_percentage))
             .select(['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12','SCL']))
    
    s2_cloud = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
                .filterBounds(aoi)
                .filterDate(extended_start, extended_end))
    
    s2_joined = ee.ImageCollection(ee.Join.saveFirst('cloud_prob').apply(
        primary=s2_sr, 
        secondary=s2_cloud,
        condition=ee.Filter.equals(leftField='system:index', rightField='system:index')
    )).map(lambda img: img.addBands(ee.Image(img.get('cloud_prob'))))
    
    # FIXED: Cloud masking WITHOUT the 0.0001 scaling
    # This matches the working app which uses raw DN values
    def mask_clouds(img):
        cloud_prob = img.select('probability')
        cdi = ee.Algorithms.Sentinel2.CDI(img)
        is_cloud = cloud_prob.gt(cloud_probability_threshold).And(cdi.lt(cdi_threshold))
        kernel = ee.Kernel.circle(radius=20, units='meters')
        cloud_dilated = is_cloud.focal_max(kernel=kernel, iterations=2)
        masked = img.updateMask(cloud_dilated.Not())
        
        # FIXED: NO scaling - keep raw DN values (typically 0-10000 range)
        # The working app does: median_image = collection.median().select(S2_BANDS)
        # without any scaling, so we match that behavior
        selected = masked.select(SPECTRAL_BANDS).clip(aoi)
        
        return selected.copyProperties(img, ['system:time_start'])
    
    cloud_free_collection = s2_joined.map(mask_clouds)
    
    origin = ee.Date(start_date)
    
    empty_image = (ee.Image.constant(ee.List.repeat(0, len(SPECTRAL_BANDS)))
                   .rename(SPECTRAL_BANDS)
                   .toFloat()
                   .updateMask(ee.Image.constant(0)))
    
    def create_monthly_composite(i):
        i = ee.Number(i)
        month_start = origin.advance(i, 'month')
        month_end = origin.advance(i.add(1), 'month')
        month_middle = month_start.advance(15, 'day')
        
        monthly_images = cloud_free_collection.filterDate(month_start, month_end)
        image_count = monthly_images.size()
        
        frequency_map = ee.Image(ee.Algorithms.If(
            image_count.gt(0),
            monthly_images.map(lambda img: 
                ee.Image(1).updateMask(img.select('B4').mask()).unmask(0).toInt()
            ).sum().toInt(),
            ee.Image.constant(0).toInt().clip(aoi)
        )).rename('frequency')
        
        monthly_composite = ee.Image(ee.Algorithms.If(
            image_count.gt(0),
            monthly_images.median(),
            empty_image.clip(aoi)
        ))
        
        validity_mask = frequency_map.gt(0).rename('validity_mask')
        
        masked_pixel_count = ee.Algorithms.If(
            image_count.gt(0),
            frequency_map.eq(0).reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=aoi,
                scale=10,
                maxPixels=1e13
            ).get('frequency'),
            0
        )
        
        return (monthly_composite
                .addBands(frequency_map)
                .addBands(validity_mask)
                .set('system:time_start', month_start.millis())
                .set('system:time_end', month_end.millis())
                .set('month_middle', month_middle.millis())
                .set('month_index', i)
                .set('month_name', month_start.format('YYYY-MM'))
                .set('image_count', image_count)
                .set('has_data', image_count.gt(0))
                .set('masked_pixel_count', masked_pixel_count))
    
    monthly_composites = ee.ImageCollection(
        ee.List.sequence(0, num_months.subtract(1)).map(create_monthly_composite)
    )
    monthly_list = monthly_composites.toList(num_months)
    month_indices = ee.List.sequence(0, num_months.subtract(1))
    
    months_with_data = month_indices.map(lambda i: 
        ee.Algorithms.If(
            ee.Image(monthly_list.get(i)).get('has_data'),
            i,
            None
        )
    ).removeAll([None])
    
    # Gap-filling function
    def gap_fill_month_closest(month_index):
        month_index = ee.Number(month_index)
        
        current_img = ee.Image(monthly_list.get(month_index))
        original_spectral = current_img.select(SPECTRAL_BANDS)
        frequency = current_img.select('frequency')
        validity_mask = current_img.select('validity_mask')
        
        gap_mask = frequency.eq(0)
        
        current_month_start = origin.advance(month_index, 'month')
        current_month_end = origin.advance(month_index.add(1), 'month')
        month_middle_millis = current_month_start.advance(15, 'day').millis()
        
        m1_past_start = origin.advance(month_index.subtract(1), 'month')
        m1_past_end = current_month_start
        m1_future_start = current_month_end
        m1_future_end = origin.advance(month_index.add(2), 'month')
        m2_past_start = origin.advance(month_index.subtract(2), 'month')
        m2_past_end = m1_past_start
        
        empty_spectral = (ee.Image.constant(ee.List.repeat(0, len(SPECTRAL_BANDS)))
                         .rename(SPECTRAL_BANDS)
                         .toFloat()
                         .updateMask(ee.Image.constant(0))
                         .clip(aoi))
        
        m1_past_images = cloud_free_collection.filterDate(m1_past_start, m1_past_end)
        m1_future_images = cloud_free_collection.filterDate(m1_future_start, m1_future_end)
        m2_past_images = cloud_free_collection.filterDate(m2_past_start, m2_past_end)
        
        all_candidate_images = m1_past_images.merge(m1_future_images).merge(m2_past_images)
        
        def add_time_distance(img):
            img_time = ee.Number(img.get('system:time_start'))
            time_diff = img_time.subtract(month_middle_millis).abs()
            return img.set('time_distance', time_diff)
        
        images_with_distance = all_candidate_images.map(add_time_distance)
        sorted_images = images_with_distance.sort('time_distance', True)
        
        closest_mosaic = ee.Image(ee.Algorithms.If(
            sorted_images.size().gt(0),
            sorted_images.mosaic().select(SPECTRAL_BANDS),
            empty_spectral
        ))
        
        has_closest = closest_mosaic.select('B4').mask()
        fill_from_closest = gap_mask.And(has_closest)
        still_masked = gap_mask.And(has_closest.Not())
        
        filled_spectral = original_spectral.unmask(closest_mosaic.updateMask(fill_from_closest))
        
        fill_source = (ee.Image.constant(0).clip(aoi).toInt8()
                       .where(fill_from_closest, 1)
                       .where(still_masked, 2)
                       .rename('fill_source'))
        
        new_validity_mask = filled_spectral.select('B4').mask().rename('filled_validity_mask')
        
        return (filled_spectral
                .addBands(frequency)
                .addBands(validity_mask)
                .addBands(new_validity_mask)
                .addBands(fill_source)
                .set('gap_filled', True)
                .set('processed', True)
                .copyProperties(current_img, current_img.propertyNames()))
    
    def prepare_complete_month(month_index):
        month_index = ee.Number(month_index)
        
        current_img = ee.Image(monthly_list.get(month_index))
        frequency = current_img.select('frequency')
        validity_mask = current_img.select('validity_mask')
        
        fill_source = ee.Image.constant(0).clip(aoi).toInt8().rename('fill_source')
        
        return (current_img.select(SPECTRAL_BANDS)
                .addBands(frequency)
                .addBands(validity_mask)
                .addBands(validity_mask.rename('filled_validity_mask'))
                .addBands(fill_source)
                .set('gap_filled', False)
                .set('processed', True)
                .copyProperties(current_img, current_img.propertyNames()))
    
    def process_month(i):
        img = ee.Image(monthly_list.get(i))
        has_data = ee.Number(img.get('has_data'))
        masked_count = ee.Number(img.get('masked_pixel_count'))
        
        return ee.Algorithms.If(
            has_data.Not(),
            None,
            ee.Algorithms.If(
                masked_count.gt(0),
                gap_fill_month_closest(i),
                prepare_complete_month(i)
            )
        )
    
    processed_months_list = month_indices.map(process_month)
    
    def check_if_complete(i):
        img = ee.Image(ee.List(processed_months_list).get(i))
        
        return ee.Algorithms.If(
            ee.Algorithms.IsEqual(img, None),
            None,
            ee.Algorithms.If(
                img.bandNames().contains('fill_source'),
                ee.Algorithms.If(
                    ee.Number(
                        img.select('fill_source').eq(2).reduceRegion(
                            reducer=ee.Reducer.sum(),
                            geometry=aoi,
                            scale=10,
                            maxPixels=1e13
                        ).get('fill_source')
                    ).eq(0),
                    i,
                    None
                ),
                None
            )
        )
    
    complete_month_indices = months_with_data.map(check_if_complete).removeAll([None])
    
    def get_complete_image(i):
        img = ee.Image(ee.List(processed_months_list).get(i))
        # FIXED: Use toDouble() to preserve raw DN values properly
        return (img.select(SPECTRAL_BANDS).toDouble()
                .set('system:index', img.get('month_name'))
                .set('month_name', img.get('month_name'))
                .set('was_gapfilled', img.get('gap_filled')))
    
    final_collection = ee.ImageCollection.fromImages(
        complete_month_indices.map(get_complete_image)
    )
    
    return final_collection, total_months


# =============================================================================
# Download Functions with .tmp file approach
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


def download_monthly_image(image, aoi, month_name, temp_dir, scale=10, status_placeholder=None):
    """Download a single monthly Sentinel-2 composite from GEE."""
    try:
        output_file = os.path.join(temp_dir, f"sentinel2_{month_name}.tif")
        
        # Check if already downloaded AND VALID
        if os.path.exists(output_file):
            is_valid, msg = validate_geotiff_file(output_file, expected_bands=len(SPECTRAL_BANDS))
            if is_valid:
                if status_placeholder:
                    status_placeholder.info(f"✅ {month_name} already downloaded and validated")
                return output_file
            else:
                if status_placeholder:
                    status_placeholder.warning(f"⚠️ {month_name} cached file corrupted ({msg}), re-downloading...")
                os.remove(output_file)
        
        bands_dir = os.path.join(temp_dir, f"bands_{month_name}")
        os.makedirs(bands_dir, exist_ok=True)
        
        band_files = []
        failed_bands = []
        
        for i, band in enumerate(SPECTRAL_BANDS):
            band_file = os.path.join(bands_dir, f"{band}.tif")
            
            if status_placeholder:
                status_placeholder.text(f"📥 {month_name}: Downloading {band} ({i+1}/{len(SPECTRAL_BANDS)})...")
            
            success = download_band_with_retry(image, band, aoi, band_file, scale)
            
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
# FIXED: Classification Functions - Normalization matches working app
# =============================================================================
def verify_image_complete(image_path, month_name):
    """Verify that a downloaded image has NO NaN or masked pixels."""
    try:
        with rasterio.open(image_path) as src:
            nodata = src.nodata
            
            for band_idx in range(1, src.count + 1):
                band_data = src.read(band_idx)
                
                nan_count = np.sum(np.isnan(band_data))
                if nan_count > 0:
                    return False, f"Band {band_idx} has {nan_count} NaN pixels"
                
                if nodata is not None:
                    nodata_count = np.sum(band_data == nodata)
                    if nodata_count > 0:
                        return False, f"Band {band_idx} has {nodata_count} nodata pixels"
                
                zero_count = np.sum(band_data == 0)
                total_pixels = band_data.size
                zero_percentage = (zero_count / total_pixels) * 100
                
                if zero_percentage > 50:
                    return False, f"Band {band_idx} has {zero_percentage:.1f}% zero pixels (likely masked)"
            
            return True, "Image is complete with no masked pixels"
            
    except Exception as e:
        return False, f"Error verifying image: {str(e)}"


def classify_monthly_image(image_path, model, device, month_name):
    """
    Apply the building detection model to a monthly Sentinel-2 image.
    
    FIXED: Normalization now matches working app exactly:
    - Normalize each PATCH globally (all bands together)
    - NOT per-band normalization
    """
    try:
        is_complete, verify_msg = verify_image_complete(image_path, month_name)
        if not is_complete:
            st.warning(f"⚠️ {month_name}: {verify_msg}")
            return None
        
        with rasterio.open(image_path) as src:
            img_data = src.read()  # Shape: (bands, height, width)
            meta = src.meta.copy()
        
        if img_data.shape[1] < PATCH_SIZE or img_data.shape[2] < PATCH_SIZE:
            st.warning(f"{month_name}: Image too small for patching ({img_data.shape[1]}x{img_data.shape[2]})")
            return None
        
        # Convert to H x W x C format for patchify (NO normalization yet!)
        img_for_patching = np.moveaxis(img_data, 0, -1)
        
        h, w, c = img_for_patching.shape
        new_h = int(np.ceil(h / PATCH_SIZE) * PATCH_SIZE)
        new_w = int(np.ceil(w / PATCH_SIZE) * PATCH_SIZE)
        
        if h != new_h or w != new_w:
            padded_img = np.zeros((new_h, new_w, c), dtype=img_for_patching.dtype)
            padded_img[:h, :w, :] = img_for_patching
            img_for_patching = padded_img
        
        # Create patches
        patches = patchify(img_for_patching, (PATCH_SIZE, PATCH_SIZE, c), step=PATCH_SIZE)
        
        n_patches_h, n_patches_w = patches.shape[0], patches.shape[1]
        
        classified_patches = np.zeros((n_patches_h, n_patches_w, PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
        
        for i in range(n_patches_h):
            for j in range(n_patches_w):
                patch = patches[i, j, 0]  # Shape: (H, W, C)
                
                # FIXED: Normalize the ENTIRE PATCH globally (all bands together)
                # This matches the working app exactly:
                # patch_normalized = normalized(patch)
                patch_normalized = normalized(patch)
                
                # Convert to torch tensor (C, H, W)
                patch_tensor = torch.tensor(np.moveaxis(patch_normalized, -1, 0), dtype=torch.float32)
                patch_tensor = patch_tensor.unsqueeze(0)  # Add batch dimension
                
                # Run inference
                with torch.inference_mode():
                    prediction = model(patch_tensor)
                    prediction = torch.sigmoid(prediction).cpu()
                
                # Convert to binary mask
                pred_np = prediction.squeeze().numpy()
                binary_mask = (pred_np > 0.5).astype(np.uint8) * 255
                
                classified_patches[i, j] = binary_mask
        
        # Reconstruct the full classification image
        reconstructed = unpatchify(classified_patches, (new_h, new_w))
        reconstructed = reconstructed[:h, :w]
        
        return reconstructed
        
    except Exception as e:
        st.warning(f"Error classifying {month_name}: {str(e)}")
        import traceback
        st.warning(traceback.format_exc())
        return None


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
# Process Time Series
# =============================================================================
def process_timeseries_classification(final_collection, aoi, model, device, scale=10, resume=False):
    """Process the entire time series: download, classify, and create thumbnails."""
    
    try:
        st.info("📥 Fetching collection metadata...")
        collection_info = final_collection.getInfo()
        features = collection_info.get('features', [])
        num_images = len(features)
        
        if num_images == 0:
            st.warning("No images found for the selected parameters.")
            return []
        
        st.success(f"✅ Found {num_images} monthly composites to process")
        
        if st.session_state.current_temp_dir is None or not os.path.exists(st.session_state.current_temp_dir):
            st.session_state.current_temp_dir = tempfile.mkdtemp()
        temp_dir = st.session_state.current_temp_dir
        
        st.info(f"📁 Working directory: {temp_dir}")
        
        if resume and st.session_state.processed_months:
            already_processed = set(st.session_state.processed_months.keys())
            st.info(f"🔄 Resuming... {len(already_processed)} months already processed")
        else:
            already_processed = set()
            st.session_state.processed_months = {}
            st.session_state.failed_months = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        detail_text = st.empty()
        
        stats_container = st.container()
        
        image_list = final_collection.toList(num_images)
        
        processed_count = len(already_processed)
        failed_count = 0
        skipped_count = 0
        
        st.session_state.processing_in_progress = True
        
        for idx in range(num_images):
            try:
                month_name = features[idx]['properties'].get('month_name', f'Month {idx+1}')
                
                if month_name in already_processed:
                    skipped_count += 1
                    progress_bar.progress((idx + 1) / num_images)
                    continue
                
                status_text.text(f"🔄 Processing {month_name} ({idx+1}/{num_images})...")
                
                img = ee.Image(image_list.get(idx))
                
                detail_text.text(f"📥 Downloading {month_name} at {scale}m resolution...")
                image_path = download_monthly_image(
                    img, aoi, month_name, temp_dir, 
                    scale=scale, 
                    status_placeholder=detail_text
                )
                
                if image_path is None:
                    st.warning(f"⚠️ Failed to download {month_name} after {MAX_RETRIES} retries")
                    st.session_state.failed_months.append(month_name)
                    failed_count += 1
                    progress_bar.progress((idx + 1) / num_images)
                    continue
                
                detail_text.text(f"🧠 Classifying {month_name}...")
                classification_mask = classify_monthly_image(image_path, model, device, month_name)
                
                if classification_mask is None:
                    st.warning(f"⚠️ Failed to classify {month_name}")
                    st.session_state.failed_months.append(month_name)
                    failed_count += 1
                    progress_bar.progress((idx + 1) / num_images)
                    continue
                
                thumbnail_data = generate_thumbnails(image_path, classification_mask, month_name)
                
                if thumbnail_data:
                    st.session_state.processed_months[month_name] = thumbnail_data
                    processed_count += 1
                
                progress_bar.progress((idx + 1) / num_images)
                
                with stats_container:
                    st.write(f"✅ Processed: {processed_count} | ❌ Failed: {failed_count} | ⏭️ Skipped: {skipped_count}")
                
                st.session_state.last_processed_index = idx
                
                time.sleep(0.3)
                
            except Exception as e:
                st.warning(f"⚠️ Error processing {month_name}: {str(e)}")
                st.session_state.failed_months.append(month_name)
                failed_count += 1
                continue
        
        st.session_state.processing_in_progress = False
        
        status_text.empty()
        detail_text.empty()
        progress_bar.empty()
        
        thumbnails = []
        for month_name in sorted(st.session_state.processed_months.keys()):
            thumbnails.append(st.session_state.processed_months[month_name])
        
        st.success(f"""
        **Processing Complete!**
        - ✅ Successfully processed: {processed_count} months
        - ❌ Failed: {failed_count} months
        - ⏭️ Skipped (already done): {skipped_count} months
        """)
        
        if st.session_state.failed_months:
            st.warning(f"Failed months: {', '.join(st.session_state.failed_months)}")
            st.info("💡 You can click 'Resume Processing' to retry failed months")
        
        return thumbnails
        
    except Exception as e:
        st.session_state.processing_in_progress = False
        st.error(f"Error processing time series: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return []


# =============================================================================
# Display Thumbnails
# =============================================================================
def display_classification_thumbnails(thumbnails):
    """Display RGB and classification thumbnails side by side."""
    
    if not thumbnails:
        st.info("No classifications to display. Click 'Generate Classifications' to process.")
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
                                caption=f"{thumb['month_name']} ({building_pct:.1f}% buildings)",
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
                        else:
                            st.warning(f"No RGB for {thumb['month_name']}")


# =============================================================================
# Main Application
# =============================================================================
def main():
    st.title("🏗️ Building Classification Time Series")
    st.markdown("""
    View building classification results from cloud-free Sentinel-2 monthly composites 
    using a UNet++ deep learning model.
    
    **Version 04 - FIXED preprocessing to match working app:**
    - ✅ Raw DN values (no 0.0001 scaling in GEE)
    - ✅ Per-patch global normalization (all bands together)
    - ✅ Cache validation with .tmp file approach
    - ✅ RGB thumbnails alongside classifications
    """)
    
    # Initialize Earth Engine
    ee_initialized, ee_message = initialize_earth_engine()
    
    if not ee_initialized:
        st.error(ee_message)
        st.info("""
        **To authenticate with Earth Engine:**
        
        **Option 1: Interactive Authentication (Colab/Local)**
        - Run `ee.Authenticate()` in a Python console first
        
        **Option 2: Service Account (Production)**
        - Set the `GOOGLE_EARTH_ENGINE_KEY_BASE64` environment variable
        """)
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
    
    # Sidebar - Cloud Parameters
    st.sidebar.header("⚙️ Cloud Filtering Parameters")
    
    cloudy_pixel_percentage = st.sidebar.slider(
        "Max Cloudy Pixel Percentage",
        min_value=0, max_value=100, value=10, step=5
    )
    
    cloud_probability_threshold = st.sidebar.slider(
        "Cloud Probability Threshold",
        min_value=0, max_value=100, value=65, step=5
    )
    
    cdi_threshold = st.sidebar.slider(
        "CDI Threshold",
        min_value=-1.0, max_value=0.0, value=-0.5, step=0.1
    )
    
    # Sidebar - Download Settings
    st.sidebar.header("📥 Download Settings")
    download_resolution = 10
    st.sidebar.info("📏 Resolution: 10m (full resolution)")
    
    # Sidebar - Cache Management
    st.sidebar.header("🗂️ Cache Management")
    
    if st.session_state.processed_months:
        st.sidebar.success(f"✅ {len(st.session_state.processed_months)} months cached")
        
        if st.sidebar.button("🗑️ Clear Cache"):
            st.session_state.processed_months = {}
            st.session_state.failed_months = []
            st.session_state.classification_thumbnails = []
            st.session_state.processing_complete = False
            st.session_state.current_temp_dir = None
            st.session_state.last_processed_index = 0
            st.sidebar.success("Cache cleared!")
            st.rerun()
    else:
        st.sidebar.info("No cached data")
    
    if st.session_state.failed_months:
        st.sidebar.warning(f"❌ {len(st.session_state.failed_months)} failed months")
    
    # Region Selection
    st.header("1️⃣ Select Region of Interest")
    st.info("Draw a rectangle or polygon on the map to define your area of interest.")
    
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
                
                if area_sq_km > 50:
                    st.warning("⚠️ Large area selected. Processing may take a long time.")
    
    if st.button("💾 Save Selected Region"):
        if st.session_state.last_drawn_polygon is not None:
            if not any(p.equals(st.session_state.last_drawn_polygon) for p in st.session_state.drawn_polygons):
                st.session_state.drawn_polygons.append(st.session_state.last_drawn_polygon)
                st.success(f"✅ Region saved! Total regions: {len(st.session_state.drawn_polygons)}")
            else:
                st.info("This polygon is already saved.")
        else:
            st.warning("Please draw a polygon on the map first")
    
    # Saved Regions
    if st.session_state.drawn_polygons:
        st.subheader("📍 Saved Regions")
        
        for i, poly in enumerate(st.session_state.drawn_polygons):
            col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
            
            with col1:
                st.write(f"**Region {i+1}**")
            
            with col2:
                centroid = poly.centroid
                utm_zone = get_utm_zone(centroid.x)
                utm_epsg = get_utm_epsg(centroid.x, centroid.y)
                st.write(f"UTM: {utm_zone} ({utm_epsg})")
            
            with col3:
                area_sq_km = poly.area * 111 * 111
                st.write(f"Area: ~{area_sq_km:.2f} km²")
            
            with col4:
                if st.button("🗑️", key=f"delete_region_{i}"):
                    st.session_state.drawn_polygons.pop(i)
                    st.session_state.classification_thumbnails = []
                    st.session_state.processing_complete = False
                    st.rerun()
        
        st.divider()
    
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
    
    if num_months > 12:
        st.warning(f"⚡ Processing {num_months} months - this may take several minutes.")
    
    # Generate Classifications
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
    
    if st.session_state.processed_months:
        st.info(f"📊 Current progress: {len(st.session_state.processed_months)} months processed")
        if st.session_state.failed_months:
            st.warning(f"⚠️ {len(st.session_state.failed_months)} months failed: {', '.join(st.session_state.failed_months)}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        start_new = st.button("🚀 Start New Processing", type="primary")
    
    with col2:
        resume_processing = st.button("🔄 Resume Processing", disabled=not st.session_state.processed_months)
    
    with col3:
        retry_failed = st.button("🔁 Retry Failed", disabled=not st.session_state.failed_months)
    
    should_process = False
    resume_mode = False
    
    if start_new:
        should_process = True
        resume_mode = False
        st.session_state.processed_months = {}
        st.session_state.failed_months = []
        st.session_state.classification_thumbnails = []
        st.session_state.processing_complete = False
        st.session_state.current_temp_dir = None
        
    elif resume_processing:
        should_process = True
        resume_mode = True
        
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
        
        with st.spinner("⚙️ Creating cloud-free monthly composites..."):
            try:
                start_date_str = start_date.strftime('%Y-%m-%d')
                end_date_str = end_date.strftime('%Y-%m-%d')
                
                final_collection, total_months = create_gapfilled_timeseries(
                    aoi=aoi,
                    start_date=start_date_str,
                    end_date=end_date_str,
                    cloudy_pixel_percentage=cloudy_pixel_percentage,
                    cloud_probability_threshold=cloud_probability_threshold,
                    cdi_threshold=cdi_threshold
                )
                
                complete_count = final_collection.size().getInfo()
                excluded_count = total_months - complete_count
                
                st.success(f"✅ Cloud-free composites created!")
                st.info(f"""
                📊 **Image Quality Summary:**
                - Total months requested: {total_months}
                - Complete images: {complete_count}
                - Excluded images: {excluded_count}
                """)
                
                if complete_count == 0:
                    st.error("❌ No complete images available!")
                    st.warning("Try: Increasing cloud cover %, selecting a different time period, or choosing a smaller region.")
                    st.stop()
                
                st.session_state.data_summary = {
                    'total_months': total_months,
                    'complete_months': complete_count,
                    'excluded_months': excluded_count
                }
                
            except Exception as e:
                st.error(f"❌ Error creating composites: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
                st.stop()
        
        st.info(f"📥 Download resolution: {download_resolution}m")
        
        thumbnails = process_timeseries_classification(
            final_collection, 
            aoi, 
            st.session_state.model, 
            st.session_state.device,
            scale=download_resolution,
            resume=resume_mode
        )
        
        if thumbnails:
            st.session_state.classification_thumbnails = thumbnails
            st.session_state.processing_complete = True
            if st.session_state.data_summary:
                st.session_state.data_summary['months_processed'] = len(thumbnails)
            st.success(f"✅ Successfully classified {len(thumbnails)} monthly images!")
        else:
            st.warning("No classifications were generated.")
    
    # Display Results
    if st.session_state.processing_complete and st.session_state.classification_thumbnails:
        st.divider()
        
        if st.session_state.data_summary:
            total = st.session_state.data_summary.get('total_months', 0)
            complete = st.session_state.data_summary.get('complete_months', 0)
            excluded = st.session_state.data_summary.get('excluded_months', 0)
            processed = st.session_state.data_summary.get('months_processed', len(st.session_state.classification_thumbnails))
            
            st.success(f"""
            **📊 Processing Summary:**
            - Total months: {total}
            - Complete images: {complete}
            - Excluded: {excluded}
            - Successfully classified: {processed}
            """)
        
        st.subheader("📅 Monthly Building Classifications")
        st.caption("White = Buildings | Black = No buildings | Percentage shows building coverage")
        display_classification_thumbnails(st.session_state.classification_thumbnails)


if __name__ == "__main__":
    main()
