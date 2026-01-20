"""
Sentinel-2 Time Series Building Classification
VERSION 10 - EXACT JS ALGORITHM

Strategy:
1. Skip month entirely if masked_pixels > 30% (don't attempt gap-fill)
2. Gap-fill from M-1, M+1 if 0 < masked_pixels <= 30%
3. Only download months with 0 masked pixels after gap-fill
4. Reject months that still have masked pixels after gap-fill

Cloud Detection (from JS):
- Scene filter: CLOUDY_PIXEL_PERCENTAGE < 10%
- Pixel mask: probability > 65 AND CDI < -0.5
- Dilate: 20m kernel, 2 iterations
- Scale: multiply by 0.0001
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
    page_title="Building Classification v10 - JS Algorithm",
    page_icon="🏗️"
)

import folium
from folium import plugins
from streamlit_folium import st_folium
import segmentation_models_pytorch as smp

# =============================================================================
# CONSTANTS - FROM JS CODE EXACTLY
# =============================================================================
SPECTRAL_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
PATCH_SIZE = 224

# Cloud detection parameters (FROM JS - EXACT)
# JS: .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
SCENE_CLOUD_THRESHOLD = 10

# JS: var iscloud = cloudProb.gt(65).and(cdi.lt(-0.5));
CLOUD_PROBABILITY_THRESHOLD = 65
CDI_THRESHOLD = -0.5

# Pre-filter threshold: Skip month if > 30% masked (don't attempt gap-fill)
MAX_MASKED_PERCENT_FOR_GAPFILL = 30

# Download settings
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2
DOWNLOAD_TIMEOUT = 120
CHUNK_SIZE = 8192

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
if 'current_temp_dir' not in st.session_state:
    st.session_state.current_temp_dir = None
if 'downloaded_images' not in st.session_state:
    st.session_state.downloaded_images = {}
if 'month_reports' not in st.session_state:
    st.session_state.month_reports = []
if 'processed_composites' not in st.session_state:
    st.session_state.processed_composites = []
if 'cached_aoi_hash' not in st.session_state:
    st.session_state.cached_aoi_hash = None
if 'cached_date_range' not in st.session_state:
    st.session_state.cached_date_range = None


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
    """Validate a single band GeoTIFF file."""
    return validate_geotiff_file(band_file_path, expected_bands=1)


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
# GEE SECTION 2 & 3: DATA PREPARATION & CLOUD MASKING (EXACT JS)
# =============================================================================
def create_cloud_free_collection(aoi, extended_start_date, extended_end_date):
    """
    EXACT translation of JS Sections 2 & 3:
    - Join S2_SR_HARMONIZED with S2_CLOUD_PROBABILITY
    - Apply cloud masking: probability > 65 AND CDI < -0.5
    - Dilate with 20m kernel, 2 iterations
    - Scale by 0.0001
    """
    
    # JS: var s2SR = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    #       .filterBounds(aoi)
    #       .filterDate(extendedStartDate, extendedEndDate)
    #       .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
    #       .select(['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12','SCL']);
    s2_sr = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
             .filterBounds(aoi)
             .filterDate(extended_start_date, extended_end_date)
             .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', SCENE_CLOUD_THRESHOLD))
             .select(['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12', 'SCL']))
    
    # JS: var s2CloudProb = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
    #       .filterBounds(aoi)
    #       .filterDate(extendedStartDate, extendedEndDate);
    s2_cloud_prob = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
                    .filterBounds(aoi)
                    .filterDate(extended_start_date, extended_end_date))
    
    # JS: function indexJoin(collectionA, collectionB, propertyName)
    def index_join(collection_a, collection_b, property_name):
        joined = ee.ImageCollection(
            ee.Join.saveFirst(property_name).apply(
                primary=collection_a,
                secondary=collection_b,
                condition=ee.Filter.equals(
                    leftField='system:index',
                    rightField='system:index'
                )
            )
        )
        return joined.map(lambda img: img.addBands(ee.Image(img.get(property_name))))
    
    # JS: var s2Joined = indexJoin(s2SR, s2CloudProb, 'cloud_probability');
    s2_joined = index_join(s2_sr, s2_cloud_prob, 'cloud_probability')
    
    # JS: function maskCloudAndShadow(img)
    def mask_cloud_and_shadow(img):
        # JS: var cloudProb = img.select('probability');
        cloud_prob = img.select('probability')
        
        # JS: var cdi = ee.Algorithms.Sentinel2.CDI(img);
        cdi = ee.Algorithms.Sentinel2.CDI(img)
        
        # JS: var iscloud = cloudProb.gt(65).and(cdi.lt(-0.5));
        is_cloud = cloud_prob.gt(CLOUD_PROBABILITY_THRESHOLD).And(cdi.lt(CDI_THRESHOLD))
        
        # JS: var kernel = ee.Kernel.circle({radius: 20, units: 'meters'});
        kernel = ee.Kernel.circle(radius=20, units='meters')
        
        # JS: var cloudDilated = iscloud.focal_max({kernel: kernel, iterations: 2});
        cloud_dilated = is_cloud.focal_max(kernel=kernel, iterations=2)
        
        # JS: var masked = img.updateMask(cloudDilated.not());
        masked = img.updateMask(cloud_dilated.Not())
        
        # JS: var scaled = masked.select(spectralBands).multiply(0.0001).clip(aoi);
        scaled = masked.select(SPECTRAL_BANDS).multiply(0.0001).clip(aoi)
        
        # JS: return scaled.copyProperties(img, ['system:time_start']);
        return scaled.copyProperties(img, ['system:time_start'])
    
    # JS: var cloudFreeCollection = s2Joined.map(maskCloudAndShadow);
    cloud_free_collection = s2_joined.map(mask_cloud_and_shadow)
    
    return cloud_free_collection


# =============================================================================
# GEE SECTION 4: CREATE MONTHLY COMPOSITES (EXACT JS)
# =============================================================================
def create_monthly_composite(cloud_free_collection, aoi, origin, month_index):
    """
    Create a single monthly composite with frequency map.
    EXACT translation of JS Section 4.
    """
    month_index = ee.Number(month_index)
    origin_date = ee.Date(origin)
    
    # JS: var monthStart = origin.advance(i, 'month');
    month_start = origin_date.advance(month_index, 'month')
    month_end = origin_date.advance(month_index.add(1), 'month')
    
    # JS: var monthlyImages = collection.filterDate(monthStart, monthEnd);
    monthly_images = cloud_free_collection.filterDate(month_start, month_end)
    
    # JS: var imageCount = monthlyImages.size();
    image_count = monthly_images.size()
    
    # Empty image template
    empty_image = (ee.Image.constant(ee.List.repeat(0, len(SPECTRAL_BANDS)))
                   .rename(SPECTRAL_BANDS)
                   .toFloat()
                   .updateMask(ee.Image.constant(0)))
    
    # JS: var frequencyMap = ee.Image(ee.Algorithms.If(...))
    frequency_map = ee.Image(ee.Algorithms.If(
        image_count.gt(0),
        monthly_images.map(lambda img: 
            ee.Image(1)
            .updateMask(img.select('B4').mask())
            .unmask(0)
            .toInt()
        ).sum().toInt(),
        ee.Image.constant(0).toInt().clip(aoi)
    )).rename('frequency')
    
    # JS: var monthlyComposite = ee.Image(ee.Algorithms.If(...))
    monthly_composite = ee.Image(ee.Algorithms.If(
        image_count.gt(0),
        monthly_images.median(),
        empty_image.clip(aoi)
    ))
    
    # JS: var validityMask = frequencyMap.gt(0).rename('validity_mask');
    validity_mask = frequency_map.gt(0).rename('validity_mask')
    
    # JS: var monthMiddle = monthStart.advance(15, 'day');
    month_middle = month_start.advance(15, 'day')
    
    # JS: var monthName = monthStart.format('YYYY-MM');
    month_name = month_start.format('YYYY-MM')
    
    # Build result image
    result_image = (monthly_composite
                   .addBands(frequency_map)
                   .addBands(validity_mask)
                   .set('system:time_start', month_start.millis())
                   .set('system:time_end', month_end.millis())
                   .set('month_middle', month_middle.millis())
                   .set('month_index', month_index)
                   .set('month_name', month_name)
                   .set('image_count', image_count)
                   .set('has_data', image_count.gt(0)))
    
    return {
        'image': result_image,
        'month_index': month_index,
        'month_start': month_start,
        'month_end': month_end,
        'month_middle': month_middle
    }


# =============================================================================
# GEE SECTION 7: GAP-FILLING (EXACT JS - M-1, M+1 only)
# =============================================================================
def gap_fill_month(monthly_composite_info, cloud_free_collection, aoi, origin):
    """
    EXACT translation of JS Section 7: gapFillMonthClosest
    
    Strategy (from JS):
    1. Collect cloud-free images from M-1, M+1
    2. Sort by time distance from middle of current month
    3. Mosaic: first valid pixel wins (closest in time)
    4. Fill using unmask()
    """
    
    month_index = monthly_composite_info['month_index']
    current_img = monthly_composite_info['image']
    origin_date = ee.Date(origin)
    
    # JS: var originalSpectral = currentImg.select(spectralBands);
    original_spectral = current_img.select(SPECTRAL_BANDS)
    
    # JS: var frequency = currentImg.select('frequency');
    frequency = current_img.select('frequency')
    
    # JS: var validityMask = currentImg.select('validity_mask');
    validity_mask = current_img.select('validity_mask')
    
    # JS: var gapMask = frequency.eq(0);
    gap_mask = frequency.eq(0)
    
    # Current month boundaries
    month_start = monthly_composite_info['month_start']
    month_end = monthly_composite_info['month_end']
    month_middle = monthly_composite_info['month_middle']
    month_middle_millis = month_middle.millis()
    
    # JS: Define search ranges for M-1, M+1
    # var m1PastStart = origin.advance(monthIndex.subtract(1), 'month');
    # var m1PastEnd = currentMonthStart;
    m1_past_start = origin_date.advance(ee.Number(month_index).subtract(1), 'month')
    m1_past_end = month_start
    
    # var m1FutureStart = currentMonthEnd;
    # var m1FutureEnd = origin.advance(monthIndex.add(2), 'month');
    m1_future_start = month_end
    m1_future_end = origin_date.advance(ee.Number(month_index).add(2), 'month')
    
    # Empty image template
    empty_spectral = (ee.Image.constant(ee.List.repeat(0, len(SPECTRAL_BANDS)))
                     .rename(SPECTRAL_BANDS)
                     .toFloat()
                     .updateMask(ee.Image.constant(0))
                     .clip(aoi))
    
    # JS: var m1PastImages = cloudFreeCollection.filterDate(m1PastStart, m1PastEnd);
    m1_past_images = cloud_free_collection.filterDate(m1_past_start, m1_past_end)
    
    # JS: var m1FutureImages = cloudFreeCollection.filterDate(m1FutureStart, m1FutureEnd);
    m1_future_images = cloud_free_collection.filterDate(m1_future_start, m1_future_end)
    
    # JS: var allCandidateImages = m1PastImages.merge(m1FutureImages);
    # NOTE: Skipping M-2 as requested - only M-1 and M+1
    all_candidate_images = m1_past_images.merge(m1_future_images)
    
    # JS: Add time distance property to each image
    def add_time_distance(img):
        img_time = ee.Number(img.get('system:time_start'))
        time_diff = img_time.subtract(month_middle_millis).abs()
        return img.set('time_distance', time_diff)
    
    images_with_distance = all_candidate_images.map(add_time_distance)
    
    # JS: var sortedImages = imagesWithDistance.sort('time_distance', true);
    sorted_images = images_with_distance.sort('time_distance', True)
    
    # JS: var closestMosaic = ee.Image(ee.Algorithms.If(
    #       sortedImages.size().gt(0),
    #       sortedImages.mosaic().select(spectralBands),
    #       emptySpectral
    #     ));
    closest_mosaic = ee.Image(ee.Algorithms.If(
        sorted_images.size().gt(0),
        sorted_images.mosaic().select(SPECTRAL_BANDS),
        empty_spectral
    ))
    
    # JS: var hasClosest = closestMosaic.select('B4').mask();
    has_closest = closest_mosaic.select('B4').mask()
    
    # JS: var fillFromClosest = gapMask.and(hasClosest);
    fill_from_closest = gap_mask.And(has_closest)
    
    # JS: var stillMasked = gapMask.and(hasClosest.not());
    still_masked = gap_mask.And(has_closest.Not())
    
    # JS: var filledSpectral = originalSpectral.unmask(closestMosaic.updateMask(fillFromClosest));
    filled_spectral = original_spectral.unmask(closest_mosaic.updateMask(fill_from_closest))
    
    # JS: var fillSource = ee.Image.constant(0).clip(aoi).toInt8()
    #       .where(fillFromClosest, 1)
    #       .where(stillMasked, 2)
    #       .rename('fill_source');
    fill_source = (ee.Image.constant(0).clip(aoi).toInt8()
                   .where(fill_from_closest, 1)
                   .where(still_masked, 2)
                   .rename('fill_source'))
    
    # JS: var newValidityMask = filledSpectral.select('B4').mask().rename('filled_validity_mask');
    new_validity_mask = filled_spectral.select('B4').mask().rename('filled_validity_mask')
    
    # Build result
    result = (filled_spectral
             .addBands(frequency)
             .addBands(validity_mask)
             .addBands(new_validity_mask)
             .addBands(fill_source)
             .set('gap_filled', True)
             .set('candidate_images', all_candidate_images.size())
             .copyProperties(current_img, current_img.propertyNames()))
    
    return result


def prepare_complete_month(monthly_composite_info, aoi):
    """
    EXACT translation of JS: prepareCompleteMonth
    For months that already have 0 masked pixels.
    """
    current_img = monthly_composite_info['image']
    frequency = current_img.select('frequency')
    validity_mask = current_img.select('validity_mask')
    
    # JS: var fillSource = ee.Image.constant(0).clip(aoi).toInt8().rename('fill_source');
    fill_source = ee.Image.constant(0).clip(aoi).toInt8().rename('fill_source')
    
    result = (current_img.select(SPECTRAL_BANDS)
             .addBands(frequency)
             .addBands(validity_mask)
             .addBands(validity_mask.rename('filled_validity_mask'))
             .addBands(fill_source)
             .set('gap_filled', False)
             .set('candidate_images', 0)
             .copyProperties(current_img, current_img.propertyNames()))
    
    return result


# =============================================================================
# MAIN GEE PROCESSING FUNCTION
# =============================================================================
def process_gee_composites(aoi, start_date, end_date, status_callback=None):
    """
    Main GEE processing following exact JS workflow:
    
    1. Create cloud-free collection
    2. Create monthly composites
    3. For each month:
       - If no_data → SKIP
       - If masked_percent > 30% → SKIP (don't gap-fill)
       - If masked_percent == 0 → COMPLETE (ready for download)
       - If 0 < masked_percent <= 30% → GAP-FILL
         - If 0 masked after → COMPLETE
         - If still masked → REJECTED
    4. Return only COMPLETE months for download
    """
    
    # Calculate number of months
    start_date_ee = ee.Date(start_date)
    end_date_ee = ee.Date(end_date)
    
    num_months = (end_date_ee.get('year').subtract(start_date_ee.get('year')).multiply(12)
                  .add(end_date_ee.get('month').subtract(start_date_ee.get('month'))))
    num_months_val = num_months.getInfo()
    
    if status_callback:
        status_callback(f"Processing {num_months_val} months...")
    
    # Extended date range for gap-filling (1 month before, 1 month after)
    # JS: var extendedStartDate = startDateEE.advance(-1, 'month');
    extended_start = start_date_ee.advance(-1, 'month')
    extended_end = end_date_ee.advance(1, 'month')
    
    # =================================================================
    # SECTION 2 & 3: Create cloud-free collection
    # =================================================================
    if status_callback:
        status_callback("Creating cloud-free collection...")
    
    cloud_free_collection = create_cloud_free_collection(aoi, extended_start, extended_end)
    total_images = cloud_free_collection.size().getInfo()
    
    if status_callback:
        status_callback(f"Found {total_images} cloud-free images")
    
    # =================================================================
    # SECTION 4 & 5: Create monthly composites and analyze
    # =================================================================
    if status_callback:
        status_callback("Creating monthly composites...")
    
    month_reports = []
    monthly_composites = []
    
    for i in range(num_months_val):
        if status_callback:
            status_callback(f"Analyzing month {i+1}/{num_months_val}...")
        
        # Create composite
        comp_info = create_monthly_composite(cloud_free_collection, aoi, start_date, i)
        monthly_composites.append(comp_info)
        
        img = comp_info['image']
        
        # Get properties
        month_name = img.get('month_name').getInfo()
        has_data = img.get('has_data').getInfo()
        image_count = img.get('image_count').getInfo()
        
        if not has_data or image_count == 0:
            month_reports.append({
                'month_index': i,
                'month_name': month_name,
                'has_data': False,
                'image_count': 0,
                'valid_pixels': 0,
                'masked_pixels': 0,
                'total_pixels': 0,
                'masked_percent': 100,
                'status': 'no_data',
                'status_reason': 'No images available'
            })
            continue
        
        # Calculate pixel statistics
        frequency = img.select('frequency')
        pixel_stats = ee.Image.cat([
            frequency.gt(0).rename('valid'),
            frequency.eq(0).rename('masked')
        ]).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi,
            scale=10,
            maxPixels=int(1e13)
        )
        
        valid_pixels = ee.Number(pixel_stats.get('valid')).round().getInfo()
        masked_pixels = ee.Number(pixel_stats.get('masked')).round().getInfo()
        
        if valid_pixels is None:
            valid_pixels = 0
        if masked_pixels is None:
            masked_pixels = 0
            
        total_pixels = valid_pixels + masked_pixels
        
        if total_pixels > 0:
            masked_percent = 100 * masked_pixels / total_pixels
        else:
            masked_percent = 100
        
        # Determine status
        if masked_pixels == 0:
            status = 'complete'
            status_reason = '0 masked pixels - ready for download'
        elif masked_percent > MAX_MASKED_PERCENT_FOR_GAPFILL:
            status = 'skipped'
            status_reason = f'>{MAX_MASKED_PERCENT_FOR_GAPFILL}% masked ({masked_percent:.1f}%) - skipped'
        else:
            status = 'needs_gapfill'
            status_reason = f'{masked_percent:.1f}% masked - will attempt gap-fill'
        
        month_reports.append({
            'month_index': i,
            'month_name': month_name,
            'has_data': True,
            'image_count': image_count,
            'valid_pixels': valid_pixels,
            'masked_pixels': masked_pixels,
            'total_pixels': total_pixels,
            'masked_percent': masked_percent,
            'status': status,
            'status_reason': status_reason
        })
    
    # =================================================================
    # SECTION 7: Gap-filling
    # =================================================================
    processed_composites = []
    
    for i, comp_info in enumerate(monthly_composites):
        report = month_reports[i]
        month_name = report['month_name']
        
        # NO_DATA: Skip entirely
        if report['status'] == 'no_data':
            processed_composites.append({
                'image': None,
                'month_name': month_name,
                'status': 'no_data',
                'report': report
            })
            continue
        
        # SKIPPED: > 30% masked, don't attempt gap-fill
        if report['status'] == 'skipped':
            processed_composites.append({
                'image': None,
                'month_name': month_name,
                'status': 'skipped',
                'report': report
            })
            continue
        
        # COMPLETE: 0 masked pixels, ready for download
        if report['status'] == 'complete':
            if status_callback:
                status_callback(f"{month_name}: Already complete (0 masked)")
            
            processed_img = prepare_complete_month(comp_info, aoi)
            processed_composites.append({
                'image': processed_img,
                'month_name': month_name,
                'status': 'complete',
                'report': report
            })
            continue
        
        # NEEDS_GAPFILL: Try gap-filling
        if report['status'] == 'needs_gapfill':
            if status_callback:
                status_callback(f"Gap-filling {month_name}...")
            
            processed_img = gap_fill_month(comp_info, cloud_free_collection, aoi, start_date)
            
            # Check if gap-filling was successful (0 masked pixels remaining)
            fill_source = processed_img.select('fill_source')
            still_masked_count = fill_source.eq(2).reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=aoi,
                scale=10,
                maxPixels=int(1e13)
            ).get('fill_source')
            
            still_masked = ee.Number(ee.Algorithms.If(
                ee.Algorithms.IsEqual(still_masked_count, None),
                0,
                still_masked_count
            )).round().getInfo()
            
            report['masked_after_gapfill'] = still_masked
            
            if still_masked == 0:
                report['status'] = 'complete'
                report['status_reason'] = 'Gap-filled successfully - 0 masked pixels'
                processed_composites.append({
                    'image': processed_img,
                    'month_name': month_name,
                    'status': 'complete',
                    'report': report
                })
            else:
                report['status'] = 'rejected'
                report['status_reason'] = f'Still has {still_masked:,} masked pixels after gap-fill'
                processed_composites.append({
                    'image': None,
                    'month_name': month_name,
                    'status': 'rejected',
                    'report': report
                })
    
    return processed_composites, month_reports, cloud_free_collection


# =============================================================================
# Download Functions
# =============================================================================
def download_band_with_retry(image, band, aoi, output_path, scale=10):
    """Download a single band with retry mechanism."""
    region = aoi.bounds().getInfo()['coordinates']
    temp_path = output_path + '.tmp'
    
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    if os.path.exists(output_path):
        is_valid, msg = validate_band_file(output_path, band)
        if is_valid:
            return True
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
                    raise Exception("Received HTML instead of GeoTIFF")
                
                with open(temp_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                
                is_valid, msg = validate_band_file(temp_path, band)
                
                if is_valid:
                    os.replace(temp_path, output_path)
                    return True
                else:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    raise Exception(f"Invalid file: {msg}")
            else:
                raise Exception(f"HTTP {response.status_code}")
                
        except Exception as e:
            for f in [output_path, temp_path]:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except:
                        pass
            
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY_BASE ** (attempt + 1))
    
    return False


def download_processed_image(processed_composite, aoi, temp_dir, scale=10, status_placeholder=None):
    """Download a processed (gap-filled) composite image."""
    month_name = processed_composite['month_name']
    image = processed_composite['image']
    
    if image is None:
        return None
    
    output_file = os.path.join(temp_dir, f"sentinel2_{month_name}.tif")
    bands_dir = os.path.join(temp_dir, f"bands_{month_name}")
    
    # Check cache
    if os.path.exists(output_file):
        is_valid, msg = validate_geotiff_file(output_file, expected_bands=len(SPECTRAL_BANDS))
        if is_valid:
            if status_placeholder:
                status_placeholder.text(f"✅ {month_name}: Using cached file")
            return output_file
        os.remove(output_file)
    
    os.makedirs(bands_dir, exist_ok=True)
    
    band_files = []
    for i, band in enumerate(SPECTRAL_BANDS):
        band_file = os.path.join(bands_dir, f"{band}.tif")
        
        if status_placeholder:
            status_placeholder.text(f"📥 {month_name}: Downloading {band} ({i+1}/{len(SPECTRAL_BANDS)})...")
        
        success = download_band_with_retry(image, band, aoi, band_file, scale)
        
        if success:
            band_files.append(band_file)
        else:
            return None
    
    # Create multiband GeoTIFF
    if status_placeholder:
        status_placeholder.text(f"📦 {month_name}: Creating multiband GeoTIFF...")
    
    with rasterio.open(band_files[0]) as src:
        meta = src.meta.copy()
    
    meta.update(count=len(band_files))
    
    with rasterio.open(output_file, 'w', **meta) as dst:
        for i, band_file in enumerate(band_files):
            with rasterio.open(band_file) as src:
                dst.write(src.read(1), i + 1)
    
    is_valid, msg = validate_geotiff_file(output_file, expected_bands=len(SPECTRAL_BANDS))
    if not is_valid:
        if os.path.exists(output_file):
            os.remove(output_file)
        return None
    
    return output_file


# =============================================================================
# RGB Thumbnail Generation
# =============================================================================
def generate_rgb_thumbnail(image_path, month_name, max_size=256):
    """Generate RGB thumbnail from downloaded image."""
    try:
        with rasterio.open(image_path) as src:
            red = src.read(4)    # B4
            green = src.read(3)  # B3
            blue = src.read(2)   # B2
            
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
    """Classify an image for building detection."""
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
                
                # Check for valid data (skip empty patches)
                if np.all(patch == 0) or np.any(np.isnan(patch)):
                    continue
                
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
# Display Report
# =============================================================================
def display_month_reports(month_reports):
    """Display detailed report for each month."""
    
    st.subheader("📊 Month-by-Month Analysis")
    
    # Categorize
    complete = [r for r in month_reports if r['status'] == 'complete']
    skipped = [r for r in month_reports if r['status'] == 'skipped']
    rejected = [r for r in month_reports if r['status'] == 'rejected']
    no_data = [r for r in month_reports if r['status'] == 'no_data']
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("✅ Complete", len(complete))
    with col2:
        st.metric("⏭️ Skipped (>30%)", len(skipped))
    with col3:
        st.metric("❌ Rejected", len(rejected))
    with col4:
        st.metric("📭 No Data", len(no_data))
    
    st.divider()
    
    # Detailed report for each month
    for report in month_reports:
        status = report['status']
        
        if status == 'complete':
            icon = "✅"
            color = "success"
        elif status == 'skipped':
            icon = "⏭️"
            color = "warning"
        elif status == 'rejected':
            icon = "❌"
            color = "error"
        else:
            icon = "📭"
            color = "info"
        
        with st.expander(f"{icon} {report['month_name']} - {report['status_reason']}"):
            if report['has_data']:
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Images:** {report['image_count']}")
                    st.write(f"**Valid pixels:** {report['valid_pixels']:,}")
                    st.write(f"**Masked pixels:** {report['masked_pixels']:,}")
                with col2:
                    st.write(f"**Total pixels:** {report['total_pixels']:,}")
                    st.write(f"**Masked %:** {report['masked_percent']:.2f}%")
                    if 'masked_after_gapfill' in report:
                        st.write(f"**After gap-fill:** {report['masked_after_gapfill']:,} masked")
            else:
                st.info("No images available for this month")
    
    st.divider()
    
    # Summary lists
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ✅ Months to Download:")
        if complete:
            for r in complete:
                gf = " (gap-filled)" if r.get('masked_after_gapfill') == 0 and r.get('masked_pixels', 0) > 0 else ""
                st.write(f"- **{r['month_name']}**{gf}")
        else:
            st.warning("No months available for download!")
    
    with col2:
        st.markdown("### ❌ Months NOT Downloaded:")
        if skipped:
            st.write(f"**Skipped ({len(skipped)})** - >{MAX_MASKED_PERCENT_FOR_GAPFILL}% masked:")
            for r in skipped:
                st.write(f"- {r['month_name']} ({r['masked_percent']:.1f}%)")
        if rejected:
            st.write(f"**Rejected ({len(rejected)})** - still masked after gap-fill:")
            for r in rejected:
                st.write(f"- {r['month_name']} ({r.get('masked_after_gapfill', '?'):,} remaining)")
        if no_data:
            st.write(f"**No Data ({len(no_data)}):**")
            for r in no_data:
                st.write(f"- {r['month_name']}")
    
    return complete


# =============================================================================
# Main Processing Pipeline
# =============================================================================
def process_timeseries(aoi, start_date, end_date, model, device, scale=10, resume=False):
    """Main processing pipeline."""
    try:
        # Setup temp directory
        if st.session_state.current_temp_dir is None or not os.path.exists(st.session_state.current_temp_dir):
            st.session_state.current_temp_dir = tempfile.mkdtemp()
        temp_dir = st.session_state.current_temp_dir
        
        st.info(f"📁 Working directory: {temp_dir}")
        
        # Check if we can resume
        aoi_hash = hash(str(aoi.coordinates().getInfo()))
        date_range = (start_date, end_date)
        
        can_resume = (resume and 
                      st.session_state.cached_aoi_hash == aoi_hash and 
                      st.session_state.cached_date_range == date_range and
                      st.session_state.month_reports)
        
        # =================================================================
        # Phase 1: GEE Processing (skip if resuming with cache)
        # =================================================================
        if can_resume and st.session_state.month_reports:
            st.header("Phase 1: ⏭️ Using Cached Analysis")
            st.success(f"✅ Found cached analysis for {len(st.session_state.month_reports)} months")
            month_reports = st.session_state.month_reports
            processed_composites = st.session_state.processed_composites
        else:
            st.header("Phase 1: Cloud Masking & Gap-Filling")
            
            progress_status = st.empty()
            
            processed_composites, month_reports, cloud_free_collection = process_gee_composites(
                aoi, start_date, end_date,
                status_callback=lambda msg: progress_status.text(msg)
            )
            
            progress_status.empty()
            
            # Cache results
            st.session_state.month_reports = month_reports
            st.session_state.processed_composites = processed_composites
            st.session_state.cached_aoi_hash = aoi_hash
            st.session_state.cached_date_range = date_range
        
        # =================================================================
        # Phase 2: Display Report
        # =================================================================
        st.header("Phase 2: Analysis Report")
        complete_months = display_month_reports(month_reports)
        
        if not complete_months:
            st.error("❌ No complete months available for download!")
            return []
        
        # Get complete composites for download
        complete_composites = [pc for pc in processed_composites if pc['status'] == 'complete']
        
        st.success(f"✅ {len(complete_composites)} months ready for download (0 masked pixels)")
        
        # =================================================================
        # Phase 3: Download
        # =================================================================
        st.header("Phase 3: Downloading Images")
        
        downloaded_images = {}
        if resume and st.session_state.downloaded_images:
            for month_name, path in st.session_state.downloaded_images.items():
                if os.path.exists(path):
                    is_valid, _ = validate_geotiff_file(path, expected_bands=len(SPECTRAL_BANDS))
                    if is_valid:
                        downloaded_images[month_name] = path
            if downloaded_images:
                st.success(f"✅ Found {len(downloaded_images)} cached downloads")
        
        # Download remaining
        to_download = [pc for pc in complete_composites if pc['month_name'] not in downloaded_images]
        
        if to_download:
            st.info(f"📥 Downloading {len(to_download)} months...")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, pc in enumerate(to_download):
                output_file = download_processed_image(pc, aoi, temp_dir, scale, status_text)
                
                if output_file:
                    downloaded_images[pc['month_name']] = output_file
                    st.session_state.downloaded_images = downloaded_images.copy()
                
                progress_bar.progress((idx + 1) / len(to_download))
            
            progress_bar.empty()
            status_text.empty()
        else:
            st.success("All months already downloaded!")
        
        st.success(f"✅ Downloaded {len(downloaded_images)} months")
        
        # =================================================================
        # Phase 4: Classification
        # =================================================================
        st.header("Phase 4: Building Classification")
        
        thumbnails = []
        existing = {t['month_name']: t for t in st.session_state.classification_thumbnails} if resume else {}
        
        to_classify = [m for m in sorted(downloaded_images.keys()) if m not in existing]
        
        if existing:
            thumbnails = list(existing.values())
            st.success(f"✅ Found {len(existing)} cached classifications")
        
        if to_classify:
            st.info(f"🧠 Classifying {len(to_classify)} months...")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, month_name in enumerate(to_classify):
                image_path = downloaded_images[month_name]
                status_text.text(f"🧠 Classifying {month_name}...")
                
                classification_mask = classify_image(image_path, model, device, month_name)
                
                if classification_mask is not None:
                    rgb_thumbnail = generate_rgb_thumbnail(image_path, month_name)
                    
                    h, w = classification_mask.shape
                    pil_class = Image.fromarray(classification_mask.astype(np.uint8))
                    if h > 256 or w > 256:
                        scale_factor = 256 / max(h, w)
                        pil_class = pil_class.resize((int(w * scale_factor), int(h * scale_factor)), Image.NEAREST)
                    
                    thumb = {
                        'rgb_image': rgb_thumbnail,
                        'classification_image': pil_class,
                        'month_name': month_name,
                        'building_pixels': np.sum(classification_mask > 0),
                        'total_pixels': h * w
                    }
                    thumbnails.append(thumb)
                    st.session_state.classification_thumbnails = thumbnails.copy()
                
                progress_bar.progress((idx + 1) / len(to_classify))
            
            progress_bar.empty()
            status_text.empty()
        else:
            st.success("All months already classified!")
        
        thumbnails = sorted(thumbnails, key=lambda x: x['month_name'])
        st.session_state.classification_thumbnails = thumbnails
        
        st.success(f"✅ Classified {len(thumbnails)} months!")
        
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
    """Display thumbnails."""
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
        for i in range(0, len(thumbnails), 2):
            cols = st.columns(4)
            for j in range(2):
                idx = i + j
                if idx < len(thumbnails):
                    thumb = thumbnails[idx]
                    building_pct = (thumb['building_pixels'] / thumb['total_pixels']) * 100
                    
                    with cols[j * 2]:
                        if thumb.get('rgb_image'):
                            st.image(thumb['rgb_image'], caption=f"{thumb['month_name']} (RGB)")
                    
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
                    with cols[col_idx]:
                        if thumb.get('classification_image'):
                            st.image(thumb['classification_image'], caption=f"{thumb['month_name']} ({building_pct:.1f}%)")
    
    else:
        for row in range((len(thumbnails) + 3) // 4):
            cols = st.columns(4)
            for col_idx in range(4):
                idx = row * 4 + col_idx
                if idx < len(thumbnails):
                    thumb = thumbnails[idx]
                    with cols[col_idx]:
                        if thumb.get('rgb_image'):
                            st.image(thumb['rgb_image'], caption=f"{thumb['month_name']}")


# =============================================================================
# Main Application
# =============================================================================
def main():
    st.title("🏗️ Building Classification v10")
    st.markdown(f"""
    **Strategy:**
    1. ⏭️ **Skip** month if masked > {MAX_MASKED_PERCENT_FOR_GAPFILL}% (don't gap-fill)
    2. 🔄 **Gap-fill** from M-1, M+1 if 0 < masked ≤ {MAX_MASKED_PERCENT_FOR_GAPFILL}%
    3. ✅ **Download** only months with **0 masked pixels**
    4. ❌ **Reject** months still masked after gap-fill
    
    **Cloud Detection (from JS):**
    - Scene: `CLOUDY_PIXEL_PERCENTAGE < {SCENE_CLOUD_THRESHOLD}%`
    - Pixel: `probability > {CLOUD_PROBABILITY_THRESHOLD} AND CDI < {CDI_THRESHOLD}`
    - Dilate: 20m kernel, 2 iterations
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
    if st.session_state.classification_thumbnails:
        st.sidebar.info(f"🧠 {len(st.session_state.classification_thumbnails)} classified")
    
    if st.sidebar.button("🗑️ Clear Cache"):
        st.session_state.downloaded_images = {}
        st.session_state.classification_thumbnails = []
        st.session_state.processing_complete = False
        st.session_state.month_reports = []
        st.session_state.processed_composites = []
        st.session_state.cached_aoi_hash = None
        st.session_state.cached_date_range = None
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
        resume = st.button("🔄 Resume", disabled=not (st.session_state.downloaded_images or st.session_state.month_reports))
    
    if start_new or resume:
        if not selected_polygon:
            st.error("❌ Select region first!")
            st.stop()
        
        if start_new:
            st.session_state.downloaded_images = {}
            st.session_state.classification_thumbnails = []
            st.session_state.processing_complete = False
            st.session_state.month_reports = []
            st.session_state.processed_composites = []
            st.session_state.cached_aoi_hash = None
            st.session_state.cached_date_range = None
        
        geojson = {"type": "Polygon", "coordinates": [list(selected_polygon.exterior.coords)]}
        aoi = ee.Geometry.Polygon(geojson['coordinates'])
        
        thumbnails = process_timeseries(
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
