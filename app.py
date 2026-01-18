"""
Sentinel-2 Time Series Building Classification
A Streamlit application for viewing building classification results 
from cloud-free Sentinel-2 monthly composites using a UNet++ deep learning model.

Workflow:
1. Generate cloud-free monthly composites on GEE
2. Download each monthly image locally (12 bands)
3. Apply patching → Model inference → Reconstruction
4. Display classification thumbnails (binary masks)
"""

# =============================================================================
# Import non-streamlit packages first
# =============================================================================
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

# =============================================================================
# Now import streamlit as the first streamlit-related import
# =============================================================================
import streamlit as st

# Set page configuration - MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    layout="wide", 
    page_title="Building Classification Time Series",
    page_icon="🏗️"
)

# =============================================================================
# Now import other streamlit-related packages
# =============================================================================
import folium
from folium import plugins
from streamlit_folium import st_folium
import segmentation_models_pytorch as smp
from tqdm import tqdm

# =============================================================================
# Constants
# =============================================================================
SPECTRAL_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
PATCH_SIZE = 224

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

# =============================================================================
# Model Download Functions (from Document 1)
# =============================================================================
@st.cache_data
def download_model_from_gdrive(gdrive_url, local_filename):
    """
    Download a file from Google Drive using the sharing URL with improved error handling
    """
    try:
        # File ID from the URL
        correct_file_id = "1m6EScw-mpBIvWV78h4pyjWq1OLQtn2ov"
        
        st.info(f"Downloading model from Google Drive (File ID: {correct_file_id})...")
        
        # Use gdown library
        try:
            import gdown
        except ImportError:
            st.info("Installing gdown library for reliable Google Drive downloads...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
            import gdown
        
        # Try multiple download approaches
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
        
        # Fallback to manual download
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
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
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
        
        # All methods failed - provide manual instructions
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
# Model Loading Function (from Document 1)
# =============================================================================
@st.cache_resource
def load_model(model_path):
    """Load the UNet++ model for building detection"""
    try:
        device = torch.device('cpu')
        model = smp.UnetPlusPlus(
            encoder_name='timm-efficientnet-b7',
            encoder_weights='imagenet',
            in_channels=12,  # 12 bands for Sentinel-2
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
# Earth Engine Authentication (from Document 2)
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
# Helper Functions (from Document 1)
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

def normalized(img):
    """Normalize image data to range [0, 1]"""
    min_val = np.nanmin(img)
    max_val = np.nanmax(img)
    
    if max_val == min_val:
        return np.zeros_like(img)
    
    img_norm = (img - min_val) / (max_val - min_val)
    return img_norm

# =============================================================================
# GEE Processing Functions (from Document 2, modified)
# =============================================================================
def create_gapfilled_timeseries(aoi, start_date, end_date, 
                                 cloudy_pixel_percentage=10,
                                 cloud_probability_threshold=65,
                                 cdi_threshold=-0.5):
    """
    Create gap-filled monthly Sentinel-2 composites.
    Uses only M-1 and M+1 for gap-filling.
    
    Returns:
        tuple: (final_collection, total_months)
    """
    
    # Date calculations
    start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    total_months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)
    
    start_date_ee = ee.Date(start_date)
    end_date_ee = ee.Date(end_date)
    num_months = ee.Number(total_months)
    
    extended_start = start_date_ee.advance(-1, 'month')
    extended_end = end_date_ee.advance(1, 'month')
    
    # Load collections
    s2_sr = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
             .filterBounds(aoi)
             .filterDate(extended_start, extended_end)
             .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloudy_pixel_percentage))
             .select(['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12','SCL']))
    
    s2_cloud = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
                .filterBounds(aoi)
                .filterDate(extended_start, extended_end))
    
    # Join collections
    s2_joined = ee.ImageCollection(ee.Join.saveFirst('cloud_prob').apply(
        primary=s2_sr, 
        secondary=s2_cloud,
        condition=ee.Filter.equals(leftField='system:index', rightField='system:index')
    )).map(lambda img: img.addBands(ee.Image(img.get('cloud_prob'))))
    
    # Cloud masking function
    def mask_clouds(img):
        is_cloud = img.select('probability').gt(cloud_probability_threshold).And(
            ee.Algorithms.Sentinel2.CDI(img).lt(cdi_threshold)
        )
        cloud_dilated = is_cloud.focal_max(kernel=ee.Kernel.circle(20, 'meters'), iterations=2)
        return (img.updateMask(cloud_dilated.Not())
                .select(SPECTRAL_BANDS)
                .multiply(0.0001)
                .clip(aoi)
                .copyProperties(img, ['system:time_start']))
    
    cloud_free = s2_joined.map(mask_clouds)
    
    # Create monthly composites
    origin = ee.Date(start_date)
    empty_img = (ee.Image.constant(ee.List.repeat(0, len(SPECTRAL_BANDS)))
                 .rename(SPECTRAL_BANDS)
                 .toFloat()
                 .updateMask(ee.Image.constant(0)))
    
    def create_monthly(i):
        i = ee.Number(i)
        m_start = origin.advance(i, 'month')
        m_end = origin.advance(i.add(1), 'month')
        monthly = cloud_free.filterDate(m_start, m_end)
        count = monthly.size()
        
        freq = ee.Image(ee.Algorithms.If(
            count.gt(0),
            monthly.map(lambda img: ee.Image(1).updateMask(img.select('B4').mask()).unmask(0).toInt()).sum().toInt(),
            ee.Image.constant(0).toInt().clip(aoi)
        )).rename('frequency')
        
        composite = ee.Image(ee.Algorithms.If(count.gt(0), monthly.median(), empty_img.clip(aoi)))
        
        return (composite.addBands(freq)
                .addBands(freq.gt(0).rename('validity_mask'))
                .set('system:time_start', m_start.millis())
                .set('month_index', i)
                .set('month_name', m_start.format('YYYY-MM'))
                .set('image_count', count)
                .set('has_data', count.gt(0)))
    
    monthly_composites = ee.ImageCollection(ee.List.sequence(0, num_months.subtract(1)).map(create_monthly))
    monthly_list = monthly_composites.toList(num_months)
    month_indices = ee.List.sequence(0, num_months.subtract(1))
    
    # Check which months have masked pixels
    def check_has_gaps(img):
        freq = img.select('frequency')
        min_freq = freq.reduceRegion(
            reducer=ee.Reducer.min(),
            geometry=aoi,
            scale=10,
            maxPixels=1e9
        ).get('frequency')
        has_gaps = ee.Number(min_freq).eq(0)
        return img.set('has_masked_pixels', has_gaps)
    
    monthly_with_gap_info = monthly_composites.map(check_has_gaps)
    monthly_list_updated = monthly_with_gap_info.toList(num_months)
    
    # Gap-filling function (M-1 and M+1 only)
    def gap_fill(month_idx):
        month_idx = ee.Number(month_idx)
        curr = ee.Image(monthly_list_updated.get(month_idx))
        freq = curr.select('frequency')
        gap_mask = freq.eq(0)
        
        m_start = origin.advance(month_idx, 'month')
        m_end = origin.advance(month_idx.add(1), 'month')
        m_mid_millis = m_start.advance(15, 'day').millis()
        
        candidates = (cloud_free.filterDate(origin.advance(month_idx.subtract(1), 'month'), m_start)
            .merge(cloud_free.filterDate(m_end, origin.advance(month_idx.add(2), 'month'))))
        
        sorted_candidates = candidates.map(
            lambda img: img.set('time_dist', ee.Number(img.get('system:time_start')).subtract(m_mid_millis).abs())
        ).sort('time_dist', True)
        
        empty = (ee.Image.constant(ee.List.repeat(0, len(SPECTRAL_BANDS)))
                 .rename(SPECTRAL_BANDS)
                 .toFloat()
                 .updateMask(ee.Image.constant(0))
                 .clip(aoi))
        
        mosaic = ee.Image(ee.Algorithms.If(
            sorted_candidates.size().gt(0),
            sorted_candidates.mosaic().select(SPECTRAL_BANDS),
            empty
        ))
        
        has_fill = mosaic.select('B4').mask()
        fill_mask = gap_mask.And(has_fill)
        still_masked = gap_mask.And(has_fill.Not())
        
        filled = curr.select(SPECTRAL_BANDS).unmask(mosaic.updateMask(fill_mask))
        fill_source = (ee.Image.constant(0).clip(aoi).toInt8()
                       .where(fill_mask, 1)
                       .where(still_masked, 2)
                       .rename('fill_source'))
        
        return (filled.addBands(freq)
                .addBands(fill_source)
                .set('month_name', curr.get('month_name'))
                .copyProperties(curr, ['system:time_start', 'month_index', 'has_data']))
    
    def prepare_complete(month_idx):
        curr = ee.Image(monthly_list_updated.get(month_idx))
        return (curr.select(SPECTRAL_BANDS)
                .addBands(curr.select('frequency'))
                .addBands(ee.Image.constant(0).clip(aoi).toInt8().rename('fill_source'))
                .set('month_name', curr.get('month_name'))
                .copyProperties(curr, ['system:time_start', 'month_index', 'has_data']))
    
    # Process all months
    def process_month(i):
        img = ee.Image(monthly_list_updated.get(i))
        has_data = ee.Number(img.get('has_data'))
        has_gaps = img.get('has_masked_pixels')
        
        return ee.Algorithms.If(
            has_data.And(has_gaps),
            gap_fill(i),
            ee.Algorithms.If(has_data, prepare_complete(i), None)
        )
    
    processed_list = ee.List(month_indices.map(process_month)).removeAll([None])
    
    # Create final collection
    final_collection = ee.ImageCollection.fromImages(processed_list.map(
        lambda img: ee.Image(img).select(SPECTRAL_BANDS).toDouble()
            .set('system:index', ee.Image(img).get('month_name'))
            .set('month_name', ee.Image(img).get('month_name'))
            .set('was_gapfilled', ee.Image(img).propertyNames().contains('fill_source'))
    ))
    
    return final_collection, total_months

# =============================================================================
# Download Monthly Image from GEE
# =============================================================================
def download_monthly_image(image, aoi, month_name, temp_dir):
    """
    Download a single monthly Sentinel-2 composite from GEE.
    
    Returns:
        str: Path to downloaded GeoTIFF or None if failed
    """
    try:
        output_file = os.path.join(temp_dir, f"sentinel2_{month_name}.tif")
        bands_dir = os.path.join(temp_dir, f"bands_{month_name}")
        os.makedirs(bands_dir, exist_ok=True)
        
        # Get the region bounds for download
        region = aoi.bounds().getInfo()['coordinates']
        
        band_files = []
        
        for band in SPECTRAL_BANDS:
            band_file = os.path.join(bands_dir, f"{band}.tif")
            
            url = image.select(band).getDownloadURL({
                'scale': 10,
                'region': region,
                'format': 'GEO_TIFF',
                'bands': [band]
            })
            
            response = requests.get(url, stream=True, timeout=60)
            if response.status_code == 200:
                with open(band_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                band_files.append(band_file)
            else:
                return None
        
        # Create multiband GeoTIFF
        if len(band_files) == len(SPECTRAL_BANDS):
            with rasterio.open(band_files[0]) as src:
                meta = src.meta.copy()
            
            meta.update(count=len(band_files))
            
            with rasterio.open(output_file, 'w', **meta) as dst:
                for i, band_file in enumerate(band_files):
                    with rasterio.open(band_file) as src:
                        dst.write(src.read(1), i+1)
            
            return output_file
        else:
            return None
        
    except Exception as e:
        st.warning(f"Error downloading {month_name}: {str(e)}")
        return None

# =============================================================================
# Classification Functions (from Document 1)
# =============================================================================
def classify_monthly_image(image_path, model, device, month_name):
    """
    Apply the building detection model to a monthly Sentinel-2 image.
    Uses the patching approach from Document 1.
    
    Returns:
        numpy.ndarray: Binary classification mask or None if failed
    """
    try:
        with rasterio.open(image_path) as src:
            img_data = src.read()  # Shape: (bands, height, width)
            meta = src.meta.copy()
        
        # Check if image is large enough for patching
        if img_data.shape[1] < PATCH_SIZE or img_data.shape[2] < PATCH_SIZE:
            st.warning(f"{month_name}: Image too small for patching ({img_data.shape[1]}x{img_data.shape[2]})")
            return None
        
        # Normalize the image
        img_normalized = np.zeros_like(img_data, dtype=np.float32)
        for i in range(img_data.shape[0]):
            img_normalized[i] = normalized(img_data[i])
        
        # Convert to H x W x C format for patchify
        img_for_patching = np.moveaxis(img_normalized, 0, -1)
        
        # Calculate the padded size to make it divisible by patch size
        h, w, c = img_for_patching.shape
        new_h = int(np.ceil(h / PATCH_SIZE) * PATCH_SIZE)
        new_w = int(np.ceil(w / PATCH_SIZE) * PATCH_SIZE)
        
        # Pad the image if necessary
        if h != new_h or w != new_w:
            padded_img = np.zeros((new_h, new_w, c), dtype=np.float32)
            padded_img[:h, :w, :] = img_for_patching
            img_for_patching = padded_img
        
        # Create patches
        patches = patchify(img_for_patching, (PATCH_SIZE, PATCH_SIZE, c), step=PATCH_SIZE)
        
        # Get patch grid dimensions
        n_patches_h, n_patches_w = patches.shape[0], patches.shape[1]
        
        # Classify each patch
        classified_patches = np.zeros((n_patches_h, n_patches_w, PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
        
        for i in range(n_patches_h):
            for j in range(n_patches_w):
                patch = patches[i, j, 0]  # Shape: (H, W, C)
                
                # Convert to torch tensor (C, H, W)
                patch_tensor = torch.tensor(np.moveaxis(patch, -1, 0), dtype=torch.float32)
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
        
        # Crop back to original size
        reconstructed = reconstructed[:h, :w]
        
        return reconstructed
        
    except Exception as e:
        st.warning(f"Error classifying {month_name}: {str(e)}")
        import traceback
        st.warning(traceback.format_exc())
        return None

# =============================================================================
# Generate Classification Thumbnails
# =============================================================================
def generate_classification_thumbnail(classification_mask, month_name):
    """
    Generate a thumbnail image from a classification mask.
    
    Returns:
        dict: Thumbnail data including PIL image and metadata
    """
    try:
        # Create thumbnail (resize if too large)
        max_size = 256
        h, w = classification_mask.shape
        
        if h > max_size or w > max_size:
            scale = max_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            # Use PIL for resizing
            pil_img = Image.fromarray(classification_mask.astype(np.uint8))
            pil_img = pil_img.resize((new_w, new_h), Image.NEAREST)
        else:
            pil_img = Image.fromarray(classification_mask.astype(np.uint8))
        
        return {
            'image': pil_img,
            'month_name': month_name,
            'original_size': (h, w),
            'building_pixels': np.sum(classification_mask > 0),
            'total_pixels': h * w
        }
        
    except Exception as e:
        st.warning(f"Error creating thumbnail for {month_name}: {str(e)}")
        return None

# =============================================================================
# Process Time Series - Main Function
# =============================================================================
def process_timeseries_classification(final_collection, aoi, model, device):
    """
    Process the entire time series: download, classify, and create thumbnails.
    
    Returns:
        list: List of classification thumbnail data
    """
    thumbnails = []
    
    try:
        # Get collection info
        st.info("📥 Fetching collection metadata...")
        collection_info = final_collection.getInfo()
        features = collection_info.get('features', [])
        num_images = len(features)
        
        if num_images == 0:
            st.warning("No images found for the selected parameters.")
            return []
        
        st.success(f"✅ Found {num_images} monthly composites to process")
        
        # Create temp directory
        temp_dir = tempfile.mkdtemp()
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        image_list = final_collection.toList(num_images)
        
        for idx in range(num_images):
            try:
                month_name = features[idx]['properties'].get('month_name', f'Month {idx+1}')
                status_text.text(f"🔄 Processing {month_name} ({idx+1}/{num_images})...")
                
                # Get the image
                img = ee.Image(image_list.get(idx))
                
                # Download the image
                status_text.text(f"📥 Downloading {month_name}...")
                image_path = download_monthly_image(img, aoi, month_name, temp_dir)
                
                if image_path is None:
                    st.warning(f"⚠️ Failed to download {month_name}, skipping...")
                    continue
                
                # Classify the image
                status_text.text(f"🧠 Classifying {month_name}...")
                classification_mask = classify_monthly_image(image_path, model, device, month_name)
                
                if classification_mask is None:
                    st.warning(f"⚠️ Failed to classify {month_name}, skipping...")
                    continue
                
                # Create thumbnail
                thumbnail_data = generate_classification_thumbnail(classification_mask, month_name)
                
                if thumbnail_data:
                    thumbnails.append(thumbnail_data)
                
                # Update progress
                progress_bar.progress((idx + 1) / num_images)
                
                # Small delay to avoid overwhelming the API
                time.sleep(0.2)
                
            except Exception as e:
                st.warning(f"⚠️ Error processing image {idx+1}: {str(e)}")
                continue
        
        # Clear progress indicators
        status_text.empty()
        progress_bar.empty()
        
        return thumbnails
        
    except Exception as e:
        st.error(f"Error processing time series: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return []

# =============================================================================
# Display Classification Thumbnails
# =============================================================================
def display_classification_thumbnails(thumbnails):
    """
    Display classification thumbnails in a grid layout.
    Fixed 4 columns.
    """
    
    if not thumbnails:
        st.info("No classifications to display. Click 'Generate Classifications' to process.")
        return
    
    num_cols = 4
    num_rows = (len(thumbnails) + num_cols - 1) // num_cols
    
    for row in range(num_rows):
        cols = st.columns(num_cols)
        for col_idx in range(num_cols):
            img_idx = row * num_cols + col_idx
            if img_idx < len(thumbnails):
                with cols[col_idx]:
                    thumb = thumbnails[img_idx]
                    
                    # Calculate building percentage
                    building_pct = (thumb['building_pixels'] / thumb['total_pixels']) * 100
                    
                    # Display the image
                    st.image(
                        thumb['image'],
                        caption=f"{thumb['month_name']} ({building_pct:.1f}% buildings)",
                        use_column_width=True
                    )

# =============================================================================
# Main Application
# =============================================================================
def main():
    # Title and description
    st.title("🏗️ Building Classification Time Series")
    st.markdown("""
    View building classification results from cloud-free Sentinel-2 monthly composites 
    using a UNet++ deep learning model.
    
    **Workflow:**
    1. Select region of interest and time period
    2. Generate cloud-free monthly composites (GEE)
    3. Apply UNet++ building detection model
    4. Display classification masks for each month
    """)
    
    # =========================================================================
    # Initialize Earth Engine
    # =========================================================================
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
    
    # =========================================================================
    # Model Loading
    # =========================================================================
    st.sidebar.header("🧠 Model Status")
    
    model_path = "best_model_version_Unet++_v02_e7.pt"
    gdrive_model_url = "https://drive.google.com/file/d/1m6EScw-mpBIvWV78h4pyjWq1OLQtn2ov/view?usp=drive_link"
    
    # Download model if not exists
    if not os.path.exists(model_path):
        st.sidebar.info("Model not found locally. Downloading...")
        downloaded_path = download_model_from_gdrive(gdrive_model_url, model_path)
        if downloaded_path is None:
            st.sidebar.error("Model download failed. Please upload manually.")
            st.stop()
    
    # Load model if not already loaded
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
    
    # =========================================================================
    # Sidebar - Cloud Parameters
    # =========================================================================
    st.sidebar.header("⚙️ Cloud Filtering Parameters")
    
    cloudy_pixel_percentage = st.sidebar.slider(
        "Max Cloudy Pixel Percentage",
        min_value=0,
        max_value=100,
        value=10,
        step=5,
        help="Filter out images with cloud cover above this percentage"
    )
    
    cloud_probability_threshold = st.sidebar.slider(
        "Cloud Probability Threshold",
        min_value=0,
        max_value=100,
        value=65,
        step=5,
        help="Pixels with cloud probability above this threshold are masked"
    )
    
    cdi_threshold = st.sidebar.slider(
        "CDI Threshold",
        min_value=-1.0,
        max_value=0.0,
        value=-0.5,
        step=0.1,
        help="Cloud Displacement Index threshold for cloud detection"
    )
    
    # =========================================================================
    # Region Selection
    # =========================================================================
    st.header("1️⃣ Select Region of Interest")
    st.info("Draw a rectangle or polygon on the map to define your area of interest.")
    
    # Create folium map
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
    
    # Display map
    map_data = st_folium(m, width=800, height=500)
    
    # Process drawn shape
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
    
    # Save region button
    if st.button("💾 Save Selected Region"):
        if st.session_state.last_drawn_polygon is not None:
            if not any(p.equals(st.session_state.last_drawn_polygon) for p in st.session_state.drawn_polygons):
                st.session_state.drawn_polygons.append(st.session_state.last_drawn_polygon)
                st.success(f"✅ Region saved! Total regions: {len(st.session_state.drawn_polygons)}")
            else:
                st.info("This polygon is already saved.")
        else:
            st.warning("Please draw a polygon on the map first")
    
    # =========================================================================
    # Saved Regions
    # =========================================================================
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
    
    # =========================================================================
    # Date Selection
    # =========================================================================
    st.header("2️⃣ Select Time Period")
    
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=date(2023, 6, 1),
            min_value=date(2017, 1, 1),
            max_value=date.today(),
            help="Select the start date for the time series"
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=date(2024, 2, 1),
            min_value=date(2017, 1, 1),
            max_value=date.today(),
            help="Select the end date for the time series"
        )
    
    if start_date >= end_date:
        st.error("❌ End date must be after start date!")
        st.stop()
    
    num_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
    st.info(f"📅 Time period: {start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')} ({num_months} months)")
    
    if num_months > 12:
        st.warning(f"⚡ Processing {num_months} months with model inference - this may take several minutes.")
    
    # =========================================================================
    # Generate Classifications
    # =========================================================================
    st.header("3️⃣ Generate Building Classifications")
    
    # Region selector
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
    
    # Process button
    if st.button("🚀 Generate Building Classifications", type="primary"):
        
        if selected_polygon is None:
            st.error("❌ Please select a region of interest first!")
            st.stop()
        
        if not st.session_state.model_loaded:
            st.error("❌ Model not loaded!")
            st.stop()
        
        # Clear previous results
        st.session_state.classification_thumbnails = []
        st.session_state.processing_complete = False
        st.session_state.data_summary = None
        
        # Convert polygon to GEE geometry
        geojson = {"type": "Polygon", "coordinates": [list(selected_polygon.exterior.coords)]}
        aoi = ee.Geometry.Polygon(geojson['coordinates'])
        
        with st.spinner("⚙️ Creating cloud-free monthly composites..."):
            try:
                start_date_str = start_date.strftime('%Y-%m-%d')
                end_date_str = end_date.strftime('%Y-%m-%d')
                
                # Create time series
                final_collection, total_months = create_gapfilled_timeseries(
                    aoi=aoi,
                    start_date=start_date_str,
                    end_date=end_date_str,
                    cloudy_pixel_percentage=cloudy_pixel_percentage,
                    cloud_probability_threshold=cloud_probability_threshold,
                    cdi_threshold=cdi_threshold
                )
                
                st.success("✅ Cloud-free composites created!")
                
            except Exception as e:
                st.error(f"❌ Error creating composites: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
                st.stop()
        
        with st.spinner("🧠 Processing images with building detection model..."):
            try:
                # Process and classify
                thumbnails = process_timeseries_classification(
                    final_collection, 
                    aoi, 
                    st.session_state.model, 
                    st.session_state.device
                )
                
                if thumbnails:
                    st.session_state.classification_thumbnails = thumbnails
                    st.session_state.processing_complete = True
                    st.session_state.data_summary = {
                        'total_months': total_months,
                        'months_processed': len(thumbnails)
                    }
                    st.success(f"✅ Successfully classified {len(thumbnails)} monthly images!")
                else:
                    st.warning("No classifications were generated.")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
    
    # =========================================================================
    # Display Results
    # =========================================================================
    if st.session_state.processing_complete and st.session_state.classification_thumbnails:
        st.divider()
        
        # Display data summary
        if st.session_state.data_summary:
            total = st.session_state.data_summary['total_months']
            processed = st.session_state.data_summary['months_processed']
            st.success(f"✅ {processed} months classified from {total} months in the selected period.")
        
        # Display thumbnails
        st.subheader("📅 Monthly Building Classifications")
        st.caption("White = Buildings detected | Black = No buildings | Percentage shows building coverage")
        display_classification_thumbnails(st.session_state.classification_thumbnails)

# =============================================================================
# Run Application
# =============================================================================
if __name__ == "__main__":
    main()
