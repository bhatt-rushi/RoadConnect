
import glob
import io
import itertools
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
import warnings
import zipfile
from datetime import datetime
from logging.handlers import RotatingFileHandler
from math import ceil, sqrt
from pathlib import Path
from multiprocessing import Pool, cpu_count

import contextily as ctx
import fiona
import geopandas as gpd
import geoutils as gu
import graphviz
import joblib
import laspy
import matplotlib.pyplot as plt

# import ogr, gdal, osr, os
import numpy as np
import osmnx as ox
import pandas as pd
import pdal
import planetary_computer
import pystac_client
import rasterio
import rasterstats
import requests
import richdem as rd
import momepy

# import rioxarray
# import rioxarray as rxr
import seaborn as sns
import shapely
from dateutil.relativedelta import relativedelta
from ipyleaflet import (
    DrawControl,
    GeoJSON,
    Map,
    SearchControl,
    basemap_to_tiles,
    basemaps,
)
from ipywidgets import HTML, FileUpload, Layout, Output, VBox
from joblib import Parallel, delayed
from osgeo import gdal, ogr
from pyproj import CRS, Transformer
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.io import MemoryFile
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.plot import show
from rasterio.warp import Resampling, calculate_default_transform, reproject
from shapely.geometry import LineString, MultiLineString, Point, Polygon, box, mapping
from shapely.ops import linemerge, unary_union
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import export_graphviz
from tqdm import tqdm

# |%%--%%| <Z9F7ShkxVA|AMOFKjPpQ8>
r"""°°°
# Welcome
°°°"""
# |%%--%%| <AMOFKjPpQ8|9eMorIunFj>
r"""°°°
[!NOTE]
I've spent most of my programming time in Go where we have explicit error handeling. Now, I've grown very fond of this and so you'll find that I try to replicate that experience for myself here through the use of try/except blocks. You may not like it, and that's okay. It's designed to avoid digging through trace stacks for simple issues created by the user. And I'm very prone to creating such issues.

[!NOTE]
I'll use LLMs to quickly generate test code but I have found it's resulting code to be unnecessarily complicated and buggy. For all the hours I've let myself be fooled into trusting LLMs, there's an equal number of hours spent troubleshooting it's code and then eventually writing it myself. Now, it still works well in some cases like writing raster transformation functions and such but now I'm trusting that it is accurate. Out of laziness, I still do this and sometimes my re-writes still involve copying/pasting LLM-generated code. I do my best to ensure all LLM-code is accurate through a re-write process. This is all a disclaimer so that you know what to expect in this codebase. When writing boilerplate (transformations, plotting functions, etc.) you should basically assume that it has been LLM generated. You can expect all other logic to be hand-written.

Code Author: Rushi Bhatt
E-Mail: bhattrushi@utexas.edu

Advisor: Carlos E Ramos-Scharrón
E-Mail: cramos@austin.utexas.edu

Institution(s): University of Texas at Austin, Protectores de Cuencas Inc.
°°°"""
# |%%--%%| <9eMorIunFj|TnPCKHtiMi>

# User-Defined Parameters
PATH_SHAPEFILE_TO_EXTRACT_EXTENT = "/home/rbhatt/Projects/RoadConnect/CACHE/__EPSG_6566__EXTENT_-65_303_18_277_-65_281_18_302/EXTENT/EXTENT.shp"
LOCAL_EPSG_CODE = "EPSG:6566" # The program will reproject everything to this but please make sure this is set to the epsg code of your DEM/LiDAR Data as those are most sensitive to reprojects.
LIDAR_REPROJECTION_CODE = "EPSG:32161"


# Only change code here if you know what you're doing! If not, please e-mail and ask me for help.
# I try to include visuals and logging where I can so that you know what's happening.

os.environ["GDAL_CACHEMAX"] = "32768"

# |%%--%%| <TnPCKHtiMi|Mn1Mtc3Zev>
r"""°°°
# Let's Prepare To Do Magic
°°°"""
# |%%--%%| <Mn1Mtc3Zev|ZPJW9q1dAU>
r"""°°°
## Setup Logger
°°°"""
# |%%--%%| <ZPJW9q1dAU|xYjmswyIQm>

# Get the root logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Clear all existing handlers to prevent duplication
if logger.hasHandlers():
    logger.handlers.clear()

# File handler with rotation at 500MB and no backups
file_handler = RotatingFileHandler(
    filename="Logfile.log",
    maxBytes=500 * 1024 * 1024,  # 500 MB
    backupCount=0  # No backups
)
file_formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
file_handler.setFormatter(file_formatter)
file_handler.setLevel(logging.DEBUG)

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(file_formatter)

# Attach handlers
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Suppress excessive logging from certain libraries
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logging.getLogger('rasterio').setLevel(logging.WARNING)

logger.info("Logger has been initialized.")

# No magic happened here but information is a pre-requisite to doing magic so keep scrolling

# |%%--%%| <xYjmswyIQm|NBz0VcfCCf>
r"""°°°
## Get Extent
°°°"""
# |%%--%%| <NBz0VcfCCf|02eGDVs5z2>

# I wrote some of this and then I handed it off to the LLMs without checking any of the code except for functionality.

# Output widget
output = Output()

# Final extent GeoDataFrame
EXTENT_GDF = None

# Store reference to the last drawn/uploaded layer
current_extent_layer = None

# Map setup
m = Map(
    center=(18.2208, -66.5901),
    zoom=10,
    basemap=basemaps.OpenStreetMap.Mapnik,
    layout=Layout(width='100%', height='600px')
)

# Draw control for polygons and rectangles
draw_control = DrawControl(
    rectangle={'shapeOptions': {'color': '#0000FF', 'fillOpacity': 0.1, 'weight': 2}},
    polygon={'shapeOptions': {'color': '#0000FF', 'fillOpacity': 0.1, 'weight': 2}},
    circle={}, polyline={}, marker={}, circlemarker={},
    edit=False, remove=True
)
m.add_control(draw_control)

# File upload widget
file_upload = FileUpload(
    accept='.zip', multiple=False, description='Upload Shapefile (.zip)',
    layout=Layout(width='auto'), style={'button_color': 'lightgreen'}
)

# Helper to validate or generate convex hull GeoDataFrame
def prepare_extent_gdf(gdf):
    if gdf.empty:
        raise ValueError("Input GeoDataFrame is empty.")
    if not gdf.crs:
        raise ValueError("Input GeoDataFrame has no CRS defined.")
    if gdf.crs.to_epsg() != LOCAL_EPSG_CODE:
        gdf = gdf.to_crs(LOCAL_EPSG_CODE)

    # Use original geometry if all are valid polygons
    if all(gdf.geometry.type.isin(["Polygon", "MultiPolygon"])):
        return gdf
    else:
        convex_hull = gdf.union_all().convex_hull
        return gpd.GeoDataFrame(geometry=[convex_hull], crs=f"EPSG:{LOCAL_EPSG_CODE}")

# Remove existing extent layer if present
def remove_current_extent_layer():
    global current_extent_layer
    if current_extent_layer and current_extent_layer in m.layers:
        m.remove_layer(current_extent_layer)
        current_extent_layer = None

# Draw event handler
@draw_control.on_draw
def handle_draw(_, action, geo_json):
    global EXTENT_GDF, current_extent_layer

    if action == 'created' and geo_json['geometry']['type'] == 'Polygon':
        try:
            remove_current_extent_layer()
            shape = Polygon(geo_json['geometry']['coordinates'][0])
            gdf_4326 = gpd.GeoDataFrame(geometry=[shape], crs='EPSG:4326')
            EXTENT_GDF = gdf_4326.to_crs(LOCAL_EPSG_CODE)
            current_extent_layer = GeoJSON(data=json.loads(gdf_4326.to_json()), style={
                'color': '#008000', 'opacity': 0.7, 'weight': 2, 'fillOpacity': 0.2
            })
            m.add_layer(current_extent_layer)
            with output:
                output.clear_output()
                print("Drawn extent loaded as GeoDataFrame.")
        except Exception as e:
            with output:
                output.clear_output()
                print(f"Error processing drawn shape: {e}")

    elif action == 'deleted':
        EXTENT_GDF = None
        remove_current_extent_layer()
        with output:
            output.clear_output()
            print("Extent cleared.")

# File upload handler
def on_upload_change(change):
    global EXTENT_GDF, current_extent_layer

    if not change['new']:
        return

    uploaded_file_info = change['new'][0]
    file_name = uploaded_file_info['name']
    file_content = uploaded_file_info['content']

    with output:
        output.clear_output()
        print(f"Processing uploaded file: {file_name}...")

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                zip_file_bytes = io.BytesIO(file_content)
                with zipfile.ZipFile(zip_file_bytes, 'r') as zip_ref:
                    zip_ref.extractall(tmpdir)

                shp_file = next((
                    os.path.join(root, f)
                    for root, _, files in os.walk(tmpdir)
                    for f in files if f.lower().endswith('.shp')
                ), None)

                if not shp_file:
                    raise FileNotFoundError("No .shp file found in archive.")

                gdf = gpd.read_file(shp_file)
                EXTENT_GDF = prepare_extent_gdf(gdf)

                gdf_4326 = EXTENT_GDF.to_crs(epsg=4326)
                remove_current_extent_layer()
                current_extent_layer = GeoJSON(data=json.loads(gdf_4326.to_json()), style={
                    'color': '#008000', 'opacity': 0.7, 'weight': 2, 'fillOpacity': 0.2
                })
                m.add_layer(current_extent_layer)
                m.fit_bounds(gdf_4326.total_bounds)

                print("Extent created from uploaded shapefile.")

        except Exception as e:
            with output:
                print(f"Error: {e}")
                traceback.print_exc()
        finally:
            file_upload.value = ()

file_upload.observe(on_upload_change, names='value')

# Instructions
instructions_html = HTML(
    value=f"""<div style='padding: 10px; border: 1px solid #ccc; border-radius: 5px; background-color: #f9f9f9;'>
    <p><b>Instructions:</b></p>
    <ul>
        <li>Draw a polygon/rectangle or upload a <b>.zip</b> shapefile to define the extent.</li>
        <li>If the uploaded file does not contain valid polygons, a convex hull will be generated.</li>
        <li>The GeoDataFrame <code>EXTENT_GDF</code> will contain the final extent in the local CRS (EPSG:{LOCAL_EPSG_CODE}).</li>
    </ul></div>""")

# Final UI layout
ui_layout = VBox([instructions_html, file_upload, m, output])

# Display UI
display(ui_layout)


# |%%--%%| <02eGDVs5z2|lqCFCkGLvA>
r"""°°°
## Cache and Data Sources Setup
°°°"""
# |%%--%%| <lqCFCkGLvA|5kF7m18Goa>

try:
    CACHE_BASE_DIR = 'CACHE'
    os.makedirs(CACHE_BASE_DIR, exist_ok=True)
except Exception as e:
    logger.fatal(f"Unable to create base cache dir. Error: {e}")
    sys.exit(1)

# GC_ is for 'Global [Unprojected] Cache' (basically, it will sit between the program and the internet).
# These vars only have relevancy for downloading files for the first time.
# This is important to save time on downloading large files repeatedly but also to save you and the government some money on bandwith

# LC_ is for 'Local Extent Cache'
# All files here should match the EPSG and Extent on the Parent Dir Name

# Setup Local Cache (Unique Extent and CRS)
try:

    EXTENT_GDF = gpd.read_file(PATH_SHAPEFILE_TO_EXTRACT_EXTENT).to_crs(LOCAL_EPSG_CODE)
    
    bounds = EXTENT_GDF.total_bounds  # [minx, miny, maxx, maxy]
    MINX_LOCALEPSG, MINY_LOCALEPSG, MAXX_LOCALEPSG, MAXY_LOCALEPSG = bounds

    bounds = EXTENT_GDF.to_crs("EPSG:4326").total_bounds
    MINX_WGS84, MINY_WGS84, MAXX_WGS84, MAXY_WGS84 = bounds


    LC_CACHE_PATH = os.path.join(
        CACHE_BASE_DIR,
        f"__EPSG_{LOCAL_EPSG_CODE[5:]}__EXTENT_"
        f"{format(MINX_WGS84, '.3f').replace('.', '_')}_"
        f"{format(MINY_WGS84, '.3f').replace('.', '_')}_"
        f"{format(MAXX_WGS84, '.3f').replace('.', '_')}_"
        f"{format(MAXY_WGS84, '.3f').replace('.', '_')}"
    )
    os.makedirs(LC_CACHE_PATH, exist_ok=True)

    LC_EXTENT_LAYER_PATH = (os.path.join(LC_CACHE_PATH, "EXTENT"))
    os.makedirs(LC_EXTENT_LAYER_PATH, exist_ok=True)

    EXTENT_GDF.to_file(os.path.join(LC_EXTENT_LAYER_PATH, 'EXTENT.shp'))

    logger.info(f"\n--- WKT: ----\n    {EXTENT_GDF.to_crs('EPSG:4326').union_all().wkt}")
    logger.info(f"\n--- Program will henceforth use ----\n    {LOCAL_EPSG_CODE} (ensure this matches your LiDAR Data) \n    EXTENT: {EXTENT_GDF.total_bounds}\n")
except Exception as e:
    logger.fatal(f"Unable to create local cache dir: {e} %tb")
    traceback.print_exc()
    sys.exit(1)

# DEM
try:
    GC_DEM_CACHE_SUBDIR = os.path.join(CACHE_BASE_DIR, 'DEM')

    GC_DEM_TILE_INDEX_DIR = os.path.join(GC_DEM_CACHE_SUBDIR, 'TileIndex_2018_DEM')
    os.makedirs(GC_DEM_TILE_INDEX_DIR, exist_ok=True)

    GC_DEM_TILE_INDEX_URL = 'https://noaa-nos-coastal-lidar-pds.s3.amazonaws.com/dem/USGS_PostMaria_PuertoRico_DEM_2018_9101/tileindex_USGS_PostMaria_PuertoRico_DEM_2018.zip'
    GC_DEM_TILE_INDEX_SHAPEFILE_NAME_IN_ZIP = 'tileindex_USGS_PostMaria_PuertoRico_DEM_2018.shp'
    GC_DEM_TILE_INDEX_URL_COLUMN_NAME = 'url'
    GC_DEM_TILE_INDEX_PATH = os.path.join(GC_DEM_TILE_INDEX_DIR, GC_DEM_TILE_INDEX_SHAPEFILE_NAME_IN_ZIP)

    GC_DEM_ORIGINAL_CACHE_DIR = os.path.join(GC_DEM_CACHE_SUBDIR, "Rasters")
    os.makedirs(GC_DEM_ORIGINAL_CACHE_DIR, exist_ok=True)

    LC_DEM_DIR = os.path.join(LC_CACHE_PATH, "DEM")
    os.makedirs(LC_DEM_DIR, exist_ok=True)

    LC_DEM_PATH = os.path.join(LC_DEM_DIR, "dem.tif")
except Exception as e:
    logger.fatal(f"Unable to create cache directory for DEM related files. Error: {e}")
    sys.exit(1)

# LiDAR
try:
    GC_LIDAR_CACHE_SUBDIR = os.path.join(CACHE_BASE_DIR, 'LIDAR')

    GC_LIDAR_TILE_INDEX_DIR = os.path.join(GC_LIDAR_CACHE_SUBDIR, 'TileIndex_2018_LiDAR')
    os.makedirs(GC_LIDAR_TILE_INDEX_DIR, exist_ok=True)

    GC_LIDAR_TILE_INDEX_URL = 'https://noaa-nos-coastal-lidar-pds.s3.amazonaws.com/laz/geoid18/9102/tileindex.zip'
    GC_LIDAR_TILE_INDEX_SHAPEFILE_NAME_IN_ZIP = '2018_USGS_PostMaria_PR_lidar_index.shp'
    GC_LIDAR_TILE_INDEX_URL_COLUMN_NAME = 'URL'
    GC_LIDAR_TILE_INDEX_PATH = os.path.join(GC_LIDAR_TILE_INDEX_DIR, GC_LIDAR_TILE_INDEX_SHAPEFILE_NAME_IN_ZIP)

    GC_LIDAR_ORIGINAL_CACHE_DIR = os.path.join(GC_LIDAR_CACHE_SUBDIR, "LAZ")
    os.makedirs(GC_LIDAR_ORIGINAL_CACHE_DIR, exist_ok=True)

    LC_LIDAR_DIR = os.path.join(LC_CACHE_PATH, "LIDAR")
    os.makedirs(LC_LIDAR_DIR, exist_ok=True)

    LC_LIDAR_PATH = os.path.join(LC_LIDAR_DIR, "lidar.las")
    LC_LIDAR_PROJECTED_PATH = os.path.join(LC_LIDAR_DIR, f"lidar_projected_{LIDAR_REPROJECTION_CODE[5:]}.las")
    LC_LIDAR_ROAD_ONLY_PATH = os.path.join(LC_LIDAR_DIR, f"lidar_road.las")
    LC_LIDAR_ROAD_ONLY_PROJECTED_PATH = os.path.join(LC_LIDAR_DIR, f"lidar_road_projected_{LIDAR_REPROJECTION_CODE[5:]}.las")
    LC_LIDAR_GROUND_ONLY_PROJECTED_PATH = os.path.join(LC_LIDAR_DIR, f"lidar_ground_projected_{LIDAR_REPROJECTION_CODE[5:]}.las")

    LC_LIDAR_EXPERIMENTAL_DIR = os.path.join(LC_LIDAR_DIR, "LIDAR_EXPERIMENTAL")
    os.makedirs(LC_LIDAR_EXPERIMENTAL_DIR, exist_ok=True)
except Exception as e:
    logger.fatal(f"Unable to create cache directory for LiDAR Tile Index. Error: {e}")
    sys.exit(1)

# Roads
try:
    LC_ROAD_CACHE_SUBDIR = os.path.join(LC_CACHE_PATH, 'ROAD')
    os.makedirs(LC_ROAD_CACHE_SUBDIR, exist_ok=True)

    LC_ROAD_EDGE_LAYER_PATH = os.path.join(LC_ROAD_CACHE_SUBDIR, 'road_edges.shp')
    LC_ROAD_EDGE_MODIFIED_LAYER_PATH = os.path.join(LC_ROAD_CACHE_SUBDIR, 'road_edges_modified.shp')
    LC_ROAD_EDGE_SPLIT_PATH = os.path.join(LC_ROAD_CACHE_SUBDIR, 'road_edges_split.shp')
    LC_ROAD_NODE_LAYER_PATH = os.path.join(LC_ROAD_CACHE_SUBDIR, 'road_nodes.shp')
    LC_ROAD_EDGE_BUFFERED_PATH = os.path.join(LC_ROAD_CACHE_SUBDIR, 'road_edges_voronoi.shp')
except Exception as e:
    logger.fatal(f"Unable to create cache directory for LiDAR Tile Index. Error: {e}")
    sys.exit(1)

# |%%--%%| <5kF7m18Goa|csKX9epmwg>
r"""°°°
# Automatic Data Retrevial and Processing
°°°"""
# |%%--%%| <csKX9epmwg|lA5lBBwswT>
r"""°°°
 ## DEM
°°°"""
# |%%--%%| <lA5lBBwswT|yPWItMAfdZ>

def download_dem_tile_index():
    """
    Download and extract DEM tile index if not already present.
    
    Returns:
        gpd.GeoDataFrame: DEM tile index GeoDataFrame
    """
    if not Path(GC_DEM_TILE_INDEX_PATH).is_file():
        logger.info(f"Downloading DEM tile index from {GC_DEM_TILE_INDEX_URL}")
        dem_tile_index_zip = requests.get(GC_DEM_TILE_INDEX_URL)
        with zipfile.ZipFile(io.BytesIO(dem_tile_index_zip.content)) as z:
            z.extractall(GC_DEM_TILE_INDEX_DIR)
        del dem_tile_index_zip
    
    return gpd.read_file(GC_DEM_TILE_INDEX_PATH).to_crs(LOCAL_EPSG_CODE)

def find_intersecting_dem_tiles(extent_gdf, dem_tile_index):
    """
    Find DEM tiles that intersect with the study area extent.
    
    Args:
        extent_gdf (gpd.GeoDataFrame): Study area extent
        dem_tile_index (gpd.GeoDataFrame): DEM tile index
    
    Returns:
        gpd.GeoDataFrame: Intersecting DEM tiles
    """
    return gpd.overlay(
        extent_gdf, 
        dem_tile_index, 
        how="intersection"
    ).to_crs(LOCAL_EPSG_CODE).drop_duplicates(
        subset=GC_DEM_TILE_INDEX_URL_COLUMN_NAME
    )

def download_dem_tiles(intersections):
    """
    Download DEM tiles that intersect with the study area.
    
    Args:
        intersections (gpd.GeoDataFrame): Intersecting DEM tiles
    
    Returns:
        list: Paths to downloaded DEM tiles
    """
    downloaded_tiles = []
    
    for _, tile in intersections.iterrows():
        dem_url = tile[GC_DEM_TILE_INDEX_URL_COLUMN_NAME]
        tile_filename = os.path.basename(dem_url)
        tile_path = os.path.join(GC_DEM_ORIGINAL_CACHE_DIR, tile_filename)
        
        if not Path(tile_path).is_file():
            logger.info(f"Downloading DEM tile: {dem_url}")
            response = requests.get(dem_url)
            response.raise_for_status()
            
            with open(tile_path, 'wb') as f:
                f.write(response.content)
        else:
            logger.info(f"Cache hit for DEM tile: {tile_filename}")
        
        # Unzip if necessary (assuming tiles are zipped)
        if tile_path.endswith('.zip'):
            unzipped_path = tile_path.replace('.zip', '.tif')
            if not Path(unzipped_path).is_file():
                with zipfile.ZipFile(tile_path, 'r') as zip_ref:
                    zip_ref.extractall(GC_DEM_ORIGINAL_CACHE_DIR)
            tile_path = unzipped_path
        
        downloaded_tiles.append(tile_path)
    
    return downloaded_tiles

def merge_and_process_dem_rasters(extent_gdf, raster_paths):
    """
    Merge DEM raster tiles, reproject, and clip to study area extent.
    
    Args:
        extent_gdf (gpd.GeoDataFrame): GeoDataFrame defining the clipping extent
        raster_paths (list): Paths to DEM raster tiles
    
    Returns:
        None: Saves processed DEM raster to specified path
    """
    # Check if final raster already exists
    if Path(LC_DEM_PATH).is_file():
        logger.info(f"CACHE HIT! DEM raster already processed at {LC_DEM_PATH}")
        return

    # Temporary files for intermediate processing
    merged_vrt = tempfile.mktemp(suffix='.vrt')
    merged_raster = tempfile.mktemp(suffix='.tif')
    reprojected_raster = tempfile.mktemp(suffix='.tif')

    try:
        # Create a virtual raster (VRT) from input rasters
        logger.info(f"Merging {len(raster_paths)} DEM raster tiles...")
        gdal.BuildVRT(
            merged_vrt, 
            raster_paths
        )

        # Translate VRT to a single merged raster
        gdal.Translate(
            merged_raster, 
            merged_vrt, 
            options=gdal.TranslateOptions(
                format='GTiff',
                noData=-9999  # Set consistent nodata value
            )
        )

        # Reproject merged raster to local coordinate system
        logger.info(f"Reprojecting merged raster to {LOCAL_EPSG_CODE}...")
        gdal.Warp(
            reprojected_raster, 
            merged_raster, 
            dstSRS=LOCAL_EPSG_CODE,
            # Use nearest neighbor for elevation data
            resampleAlg=gdal.GRIORA_NearestNeighbour  
        )

        # Clip raster to extent of study area
        logger.info(f"Clipping raster to study area extent...")
        gdal.Warp(
            LC_DEM_PATH, 
            reprojected_raster, 
            cutlineDSName=extent_gdf.to_json(), 
            cutlineLayer=extent_gdf.index.name,
            dstNodata=-9999  # Consistent nodata value
        )

        logger.info(f"Successfully processed DEM raster: {LC_DEM_PATH}")

    except Exception as e:
        logger.fatal(f"Error processing DEM rasters: {e}")
        logger.error(traceback.format_exc())
        raise

    finally:
        # Clean up temporary files
        for temp_file in [merged_vrt, merged_raster, reprojected_raster]:
            if os.path.exists(temp_file):
                os.remove(temp_file)

try:
    # Download or load DEM tile index
    DEM_TILE_INDEX = download_dem_tile_index()

    # Find intersecting DEM tiles
    INTERSECTING_DEM_TILES = find_intersecting_dem_tiles(
        extent_gdf=EXTENT_GDF, 
        dem_tile_index=DEM_TILE_INDEX
    )

    # Download intersecting DEM tiles
    DEM_TILE_PATHS = download_dem_tiles(INTERSECTING_DEM_TILES)

except Exception as e:
    logger.fatal(f"Something went wrong finding or retrieving intersecting DEM Tiles. {e}")
    logger.error(traceback.format_exc())
    sys.exit(1)

try:
    # Merge and process DEM rasters
    merge_and_process_dem_rasters(
        extent_gdf=EXTENT_GDF, 
        raster_paths=DEM_TILE_PATHS
    )

except Exception as e:
    logger.fatal(f"Something went wrong processing DEM rasters. Error: {e}")
    logger.error(traceback.format_exc())
    sys.exit(1)


# |%%--%%| <yPWItMAfdZ|8IzjMW6otb>
r"""°°°
## Road
°°°"""
# |%%--%%| <8IzjMW6otb|vDdWRsPzhh>
r"""°°°
### Download Road Network From OSM
°°°"""
# |%%--%%| <vDdWRsPzhh|3GDq7v6b4B>

try:
    if Path(LC_ROAD_EDGE_MODIFIED_LAYER_PATH).is_file():
        logger.info(f"MODIFIED ROADLAYER BEING USED!!! CACHE HIT! CACHE HIT!")
        ROAD_EDGES_GDF = gpd.read_file(LC_ROAD_EDGE_MODIFIED_LAYER_PATH).to_crs(LOCAL_EPSG_CODE)
        ROAD_EDGES_GDF.plot()
        
    elif Path(LC_ROAD_EDGE_LAYER_PATH).is_file():
        logger.info(f"CACHE HIT! CACHE HIT!")
        ROAD_EDGES_GDF = gpd.read_file(LC_ROAD_EDGE_LAYER_PATH).to_crs(LOCAL_EPSG_CODE)
        ROAD_EDGES_GDF.plot()

        # May 23, 2025: I'm adding this as a temporary step
        #ROAD_EDGES_GDF = gpd.read_file(os.path.join('RoadConnect_LosPozos','INPUT','Roads.shp'))
        #ROAD_EDGES_GDF = ROAD_EDGES_GDF.to_crs(LOCAL_EPSG_CODE)
        #ROAD_EDGES_GDF.plot()
    else:
        logger.info(f"CACHE MISS. Downloading OpenStreetMaps Data.")

        extent = EXTENT_GDF.to_crs(4326).total_bounds
        min_x, min_y, max_x, max_y = extent
        graphs = ox.graph_from_bbox(bbox=(min_x, min_y, max_x, max_y), network_type='all', simplify=False, retain_all=True)
        ROAD_NODES_GDF, ROAD_EDGES_GDF = ox.graph_to_gdfs(graphs)

        ROAD_NODES_GDF = ROAD_NODES_GDF.to_crs(LOCAL_EPSG_CODE)
        ROAD_EDGES_GDF = ROAD_EDGES_GDF.to_crs(LOCAL_EPSG_CODE)

        ROAD_NODES_GDF = ROAD_NODES_GDF.clip(EXTENT_GDF)
        ROAD_EDGES_GDF = ROAD_EDGES_GDF.clip(EXTENT_GDF)

        ROAD_EDGES_GDF.plot()

        ROAD_NODES_GDF.to_file(LC_ROAD_NODE_LAYER_PATH)
        ROAD_EDGES_GDF.to_file(LC_ROAD_EDGE_LAYER_PATH)
    # May 19, 2025: I'm adding this as a temporary step
        #ROAD_EDGES_GDF = gpd.read_file(os.path.join('RoadConnect_LosPozos','INPUT','Roads.shp'))
        #ROAD_EDGES_GDF = ROAD_EDGES_GDF.to_crs(LOCAL_EPSG_CODE)
        #ROAD_EDGES_GDF.plot()
except Exception as e:
    logger.info(f"Oops, something went wrong getting or loading road layer data: {e}.")
    sys.exit(1)

# |%%--%%| <3GDq7v6b4B|dntPeviJx2>
r"""°°°
### Segmentize, Explode, and add Elevation Values
°°°"""
# |%%--%%| <dntPeviJx2|VThZIBTOnl>
r"""°°°
I really didn't think that line_merge and union_all would be important and only added because I was waiting for another model to finish running and to deal with potential edge cases of bad data. However, it turns out, it drastically changes model outputs. Without this step, the standard deviation of segment length is greater. This has a large impact on the algorithim searching for local minima. Ideally, all edges are of equal length. However, this is not possible to do and so we should strive to keep the distribution as tight as possible.
°°°"""
# |%%--%%| <VThZIBTOnl|zoXdY21x4y>
r"""°°°
Road data quality matters a lot here. I tried running this with the mapped data provided in the prev RoadConnect dir and road segments not meeting at single nodes causes breaking issues (simplifying the road network did not fix this because it creates noise in the data i.e., extra road segments that do not exist). 
°°°"""
# |%%--%%| <zoXdY21x4y|Hgz1fOZvC7>
r"""°°°
I experimented with two libraries for Voronoi tessellation to create the buffer without overlap: (1) Geoutils and (2) momepy (which uses libpysal in the background). I stuck with geoutils due to how road intersections end up being treated.
°°°"""
# |%%--%%| <Hgz1fOZvC7|sYHf2ejYnP>

def line_segments(line: shapely.LineString) -> np.ndarray:
    """
    Explode a linestring into constituent pairwise coordinates.
    
    Args:
        line (shapely.LineString): Input linestring
    
    Returns:
        np.ndarray: Array of line segments
    """
    xys = shapely.get_coordinates(line)
    return shapely.linestrings(
        np.column_stack((xys[:-1], xys[1:])).reshape(xys.shape[0] - 1, 2, 2)
    )

def simplify_and_segmentize_roads(road_edges_gdf, simplify_tolerance=2, segmentize_distance=2):
    """
    Simplify and segmentize road geometries.
    
    Args:
        road_edges_gdf (gpd.GeoDataFrame): Input road edges GeoDataFrame
        simplify_tolerance (float): Tolerance for geometry simplification
        segmentize_distance (float): Distance for segmentizing lines
    
    Returns:
        gpd.GeoSeries: Simplified and segmentized road geometries
    """
    # Merge, simplify, and segmentize road geometries
    merged_simplified_roads = linemerge(unary_union(road_edges_gdf.geometry))
    simplified_roads = merged_simplified_roads.simplify(tolerance=simplify_tolerance)
    
    segmentized_roads = gpd.GeoSeries([
        line.segmentize(segmentize_distance) 
        for line in simplified_roads.geoms
    ])
    
    return segmentized_roads

def explode_lines_to_segments(road_edges_gdf):
    """
    Convert road geometries into individual line segments.
    
    Args:
        road_edges_gdf (gpd.GeoDataFrame): Input road edges GeoDataFrame
    
    Returns:
        gpd.GeoDataFrame: GeoDataFrame of road line segments
    """
    segment_geometries = []
    for geom in road_edges_gdf.geometry:
        segment_geometries.extend(line_segments(geom))
    
    split_gdf = gpd.GeoDataFrame(
        geometry=segment_geometries, 
        crs=LOCAL_EPSG_CODE
    )

    # Calculate segment lengths
    split_gdf['length'] = split_gdf.geometry.length
    
    return split_gdf

def process_road_network(road_edges_gdf, simplify_tolerance=2, segmentize_distance=2):
    """
    Process road network by simplifying, segmentizing, and splitting geometries.
    
    Args:
        road_edges_gdf (gpd.GeoDataFrame): Input road edges GeoDataFrame
        simplify_tolerance (float): Tolerance for geometry simplification
        segmentize_distance (float): Distance for segmentizing lines
    
    Returns:
        gpd.GeoDataFrame: Processed road segments GeoDataFrame
    """
    # Create a copy to avoid modifying the original
    processed_gdf = road_edges_gdf.copy()
    
    # Simplify and segmentize road geometries
    segmentized_roads = simplify_and_segmentize_roads(
        processed_gdf, 
        simplify_tolerance, 
        segmentize_distance
    )

    # Create and return segmented road network
    return explode_lines_to_segments(segmentized_roads)

def identify_lowest_elevation_segments(gdf, elevation_field='ele_fmean'):
    """
    Identify road segments with the lowest elevation among their neighboring segments.
    
    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame of buffered road segments
        elevation_field (str, optional): Field to use for elevation comparison. 
                                         Defaults to 'ele_fmean'.
    
    Returns:
        gpd.GeoDataFrame: Updated GeoDataFrame with lowest elevation flag
    """
    # Create a copy to avoid modifying the original
    result_gdf = gdf.copy()
    
    # Add a column to track lowest elevation neighbors
    result_gdf['local_low'] = False
    
    # Spatial join to find touching polygons
    spatial_index = result_gdf.sindex
    
    # Iterate through each polygon
    for idx, row in result_gdf.iterrows():
        # Find intersecting polygons
        possible_matches_index = list(spatial_index.intersection(row.geometry.bounds))
        
        # Remove the current polygon's index from possible matches
        possible_matches_index = [i for i in possible_matches_index if i != idx]
        
        # Check if any neighbors exist
        if not possible_matches_index:
            continue
        
        # Get neighboring polygons that actually touch
        neighbors = result_gdf.iloc[possible_matches_index]
        touching_neighbors = neighbors[
            neighbors.geometry.touches(row.geometry)
        ]
        
        # If no touching neighbors, continue
        if touching_neighbors.empty:
            continue
        
        # Check if current polygon has the lowest elevation among touching neighbors
        current_elevation = row[elevation_field]
        is_lowest = all(current_elevation < neighbor[elevation_field] for _, neighbor in touching_neighbors.iterrows())
        
        # Set the flag if it's the lowest
        if is_lowest:
            result_gdf.at[idx, 'local_low'] = True
    
    return result_gdf

def calculate_elevation_stats(buffered_gdf, dem_path):
    """
    Calculate elevation statistics for buffered road segments using zonal stats.
    
    Args:
        buffered_gdf (gpd.GeoDataFrame): Buffered road segments
        dem_path (str): Path to the Digital Elevation Model raster
    
    Returns:
        gpd.GeoDataFrame: GeoDataFrame with added elevation statistics
    """
    # Calculate standard deviation of elevation to define filtering range
    def get_filtered_mean(x):
        """
        Calculate mean after removing outliers using Interquartile Range (IQR) method.
        
        Args:
            x (numpy.ma.masked_array): Input elevation values
        
        Returns:
            float: Mean of filtered values
        """
        if len(x) == 0:
            return -9999
        
        # Calculate Q1, Q3, and IQR
        q1 = np.percentile(x.compressed(), 25)
        q3 = np.percentile(x.compressed(), 75)
        iqr = q3 - q1
        
        # Define outlier boundaries
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Filter values within IQR boundaries
        filtered_values = x[(x >= lower_bound) & (x <= upper_bound)]
        
        # Return mean of filtered values
        return np.ma.mean(filtered_values) if len(filtered_values) > 0 else -9999


    # Perform zonal statistics using gen_zonal_stats with geojson_out=True
    elevation_stats = list(rasterstats.gen_zonal_stats(
        buffered_gdf, 
        dem_path, 
        stats=['count', 'min', 'max','std', 'mean'],
        add_stats={'fmean': get_filtered_mean},
        band_num=1,
        nodata=-9999,
        prefix='ele_',
        geojson_out=True
    ))
    
    # Create a new GeoDataFrame from the geojson output
    stats_gdf = gpd.GeoDataFrame.from_features(elevation_stats, crs=LOCAL_EPSG_CODE)

    stats_gdf = identify_lowest_elevation_segments(stats_gdf)

    return stats_gdf

def create_buffered_road_segments(split_road_segments, buffer_distance=2):
    """
    Create a buffered GeoDataFrame using geoutils.Vector.buffer_without_overlap.
    
    Args:
        split_road_segments (gpd.GeoDataFrame): Input road segments GeoDataFrame
        buffer_distance (float): Buffer distance in meters
    
    Returns:
        gpd.GeoDataFrame: Buffered road segments GeoDataFrame without overlapping buffers
    """
    
    # Apply buffer without overlap using geoutils
    #buffered_gdf = gu.Vector(split_road_segments).buffer_without_overlap(buffer_distance).ds
    buffered_gdf = momepy.morphological_tessellation(split_road_segments, clip=split_road_segments.union_all().buffer(6), simplify=False)
    
    # Calculate and store the area of each buffered segment
    buffered_gdf['area'] = buffered_gdf.geometry.area
    
    buffered_gdf = calculate_elevation_stats(buffered_gdf, LC_DEM_PATH)
    
    return buffered_gdf

# |%%--%%| <sYHf2ejYnP|mYlo0brg3e>

try:
    if not Path(LC_ROAD_EDGE_SPLIT_PATH).is_file():
        logger.info(f"CACHE HIT! Split road network already exists.")
        ROAD_EDGES_SPLIT_GDF = gpd.read_file(LC_ROAD_EDGE_SPLIT_PATH).to_crs(LOCAL_EPSG_CODE)
        
        # Calculate standard deviation of segment lengths
        length_std = ROAD_EDGES_SPLIT_GDF['length'].std()
        print(f"Standard Deviation of Road Segment Lengths: {length_std:.2f} meters")
        
        # Plot histogram of segment lengths
        plt.figure(figsize=(10, 6))
        ROAD_EDGES_SPLIT_GDF['length'].hist(bins=50, edgecolor='black')
        plt.title('Distribution of Road Segment Lengths')
        plt.xlabel('Segment Length (meters)')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()
    else:
        logger.info(f"CACHE MISS. Processing road network...")
        ROAD_EDGES_SPLIT_GDF = process_road_network(
            ROAD_EDGES_GDF, 
            simplify_tolerance=2, 
            segmentize_distance=2
        )

        # Calculate standard deviation of segment lengths
        length_std = ROAD_EDGES_SPLIT_GDF['length'].std()
        print(f"Standard Deviation of Road Segment Lengths: {length_std:.2f} meters")
        
        # Plot histogram of segment lengths
        plt.figure(figsize=(10, 6))
        ROAD_EDGES_SPLIT_GDF['length'].hist(bins=50, edgecolor='black')
        plt.title('Distribution of Road Segment Lengths')
        plt.xlabel('Segment Length (meters)')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()

        ROAD_EDGES_SPLIT_GDF.to_file(LC_ROAD_EDGE_SPLIT_PATH)

except Exception as e:
    logger.info(f"Oops, something went wrong processing road network: {e}.")
    sys.exit(1)

# |%%--%%| <mYlo0brg3e|13Hxs2N36U>

try:
    if not Path(LC_ROAD_EDGE_BUFFERED_PATH).is_file():
        logger.info(f"CACHE HIT! Road edge buffer already exists.")
        ROAD_EDGES_BUFFERED_GDF = gpd.read_file(LC_ROAD_EDGE_BUFFERED_PATH).to_crs(LOCAL_EPSG_CODE)
        
        # Calculate standard deviation of buffer areas
        area_std = ROAD_EDGES_BUFFERED_GDF['area'].std()
        print(f"Standard Deviation of Road Segment Buffer Areas: {area_std:.2f} sq meters")
        
        # Print how many local minima
        local_minima_count = ROAD_EDGES_BUFFERED_GDF['local_low'].sum()
        print(f"Number of Local Minima: {local_minima_count}")

        # Plot histogram of buffer areas
        plt.figure(figsize=(10, 6))
        ROAD_EDGES_BUFFERED_GDF['area'].hist(bins=50, edgecolor='black')
        plt.title('Distribution of Road Segment Buffer Areas')
        plt.xlabel('Buffer Area (sq meters)')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()

        # Plot histogram of elevation statistics
        plt.figure(figsize=(20, 5))
        plt.subplot(1, 4, 1)
        ROAD_EDGES_BUFFERED_GDF['ele_min'].hist(bins=50, edgecolor='black')
        plt.title('Distribution of Minimum Elevation')
        plt.xlabel('Minimum Elevation')
        plt.ylabel('Frequency')

        plt.subplot(1, 4, 2)
        ROAD_EDGES_BUFFERED_GDF['ele_max'].hist(bins=50, edgecolor='black')
        plt.title('Distribution of Maximum Elevation')
        plt.xlabel('Maximum Elevation')
        plt.ylabel('Frequency')

        plt.subplot(1, 4, 3)
        ROAD_EDGES_BUFFERED_GDF['ele_fmean'].hist(bins=50, edgecolor='black')
        plt.title('Distribution of Filtered Mean Elevation')
        plt.xlabel('Filtered Mean Elevation')
        plt.ylabel('Frequency')

        plt.subplot(1, 4, 4)
        ROAD_EDGES_BUFFERED_GDF['ele_std'].hist(bins=50, edgecolor='black')
        plt.title('Distribution of Elevation Standard Deviation')
        plt.xlabel('Elevation Standard Deviation')
        plt.ylabel('Frequency')

        plt.tight_layout()
        plt.show()
    else:
        logger.info(f"CACHE MISS. Creating road segment buffers...")
        # Create buffered road segments
        ROAD_EDGES_BUFFERED_GDF = create_buffered_road_segments(ROAD_EDGES_SPLIT_GDF, buffer_distance=6)
        
        # Calculate standard deviation of buffer areas
        area_std = ROAD_EDGES_BUFFERED_GDF['area'].std()
        print(f"Standard Deviation of Road Segment Buffer Areas: {area_std:.2f} sq meters")
        
        # Print how many local minima
        local_minima_count = ROAD_EDGES_BUFFERED_GDF['local_low'].sum()
        print(f"Number of Local Minima: {local_minima_count}")

        # Plot histogram of buffer areas
        plt.figure(figsize=(10, 6))
        ROAD_EDGES_BUFFERED_GDF['area'].hist(bins=50, edgecolor='black')
        plt.title('Distribution of Road Segment Buffer Areas')
        plt.xlabel('Buffer Area (sq meters)')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()

        # Plot histogram of elevation statistics
        plt.figure(figsize=(20, 5))
        plt.subplot(1, 4, 1)
        ROAD_EDGES_BUFFERED_GDF['ele_min'].hist(bins=50, edgecolor='black')
        plt.title('Distribution of Minimum Elevation')
        plt.xlabel('Minimum Elevation')
        plt.ylabel('Frequency')

        plt.subplot(1, 4, 2)
        ROAD_EDGES_BUFFERED_GDF['ele_max'].hist(bins=50, edgecolor='black')
        plt.title('Distribution of Maximum Elevation')
        plt.xlabel('Maximum Elevation')
        plt.ylabel('Frequency')

        plt.subplot(1, 4, 3)
        ROAD_EDGES_BUFFERED_GDF['ele_fmean'].hist(bins=50, edgecolor='black')
        plt.title('Distribution of Filtered Mean Elevation')
        plt.xlabel('Filtered Mean Elevation')
        plt.ylabel('Frequency')

        plt.subplot(1, 4, 4)
        ROAD_EDGES_BUFFERED_GDF['ele_std'].hist(bins=50, edgecolor='black')
        plt.title('Distribution of Elevation Standard Deviation')
        plt.xlabel('Elevation Standard Deviation')
        plt.ylabel('Frequency')

        plt.tight_layout()
        plt.show()

        # Save the buffered GeoDataFrame
        ROAD_EDGES_BUFFERED_GDF.to_file(LC_ROAD_EDGE_BUFFERED_PATH)

except Exception as e:
    logger.info(f"Oops, something went wrong creating road segment buffers: {e}.\n{traceback.format_exc()}")
    sys.exit(1)

# |%%--%%| <13Hxs2N36U|Zzf9MBMFbJ>

ROAD_EDGES_BUFFERED_GDF = identify_lowest_elevation_segments(ROAD_EDGES_BUFFERED_GDF, elevation_field='ele_mean')
local_minima_count = ROAD_EDGES_BUFFERED_GDF['local_low'].sum()
print(f"Number of Local Minima: {local_minima_count}")

# |%%--%%| <Zzf9MBMFbJ|fKIeorFz7X>

def prepare_raster_mask(input_raster_path, temp_dir):
    """
    Rounds a raster to the nearest 1 decimal place and creates an empty mask raster.
    
    Parameters:
    -----------
    input_raster_path : str
        Path to the input raster file
    temp_dir : str
        Temporary directory to store output files
    
    Returns:
    --------
    tuple
        Paths to the rounded raster and the empty mask raster
    """
    # Open the input raster
    with rasterio.open(input_raster_path) as src:
        # Read the raster data
        raster_data = src.read(1)  # Assumes single-band raster
        
        # Round the raster to 1 decimal place
        rounded_data = np.round(raster_data, decimals=1)
        
        # Prepare rounded raster metadata
        rounded_profile = src.profile.copy()
        
        # Path for rounded raster
        rounded_raster_path = os.path.join(temp_dir, 'rounded_raster.tif')
        
        # Write rounded raster
        with rasterio.open(rounded_raster_path, 'w', **rounded_profile) as dst:
            dst.write(rounded_data.astype(rounded_profile['dtype']), 1)
        
        # Prepare empty mask raster metadata
        mask_profile = src.profile.copy()
        mask_profile.update({
            'nodata': 0,  # Set no data value
            'dtype': 'uint8'  # Binary raster
        })
        
        # Path for mask raster
        mask_raster_path = os.path.join(temp_dir, 'mask_raster.tif')
        
        # Create empty mask raster with same extent and CRS
        with rasterio.open(mask_raster_path, 'w', **mask_profile) as dst:
            empty_mask = np.zeros_like(raster_data, dtype=np.uint8)
            dst.write(empty_mask, 1)
    
    return rounded_raster_path, mask_raster_path

def create_majority_mask(rounded_raster_path, shapefile_path, mask_raster_path):
    """
    Create a mask raster based on majority values in each polygon.
    """
    # Read the shapefile
    gdf = gpd.read_file(shapefile_path)
    
    # Open the rounded raster and mask raster with GDAL
    with rasterio.open(rounded_raster_path) as src_rounded, \
         rasterio.open(mask_raster_path, 'r+') as src_mask:
        
        # Read raster data
        rounded_data = src_rounded.read(1)
        mask_data = src_mask.read(1)
        
        # Perform zonal statistics for each polygon
        zstats = rasterstats.zonal_stats(
            gdf, 
            rounded_raster_path, 
            stats=['majority'],
            all_touched=True
        )
        
        # Iterate through polygons with progress bar
        for idx, (polygon, stat) in tqdm(enumerate(zip(gdf.geometry, zstats)), 
                                         total=len(gdf), 
                                         desc="Processing Polygons"):
            # Skip if no majority value
            if stat['majority'] is None:
                continue
            
            # Rasterize the polygon
            poly_raster_mask = gdal.Rasterize(
                '', 
                shapefile_path,
                format='MEM',
                outputType=gdal.GDT_Byte,
                width=mask_data.shape[1], 
                height=mask_data.shape[0],
                outputBounds=[
                    src_mask.bounds.left, 
                    src_mask.bounds.bottom, 
                    src_mask.bounds.right, 
                    src_mask.bounds.top
                ],
                where=f"FID = {idx}",
                allTouched=True
            )
            
            # Read the rasterized polygon
            poly_array = poly_raster_mask.ReadAsArray()
            
            # Find cells within majority ± 0.3
            matching_cells = (
                (rounded_data >= stat['majority'] - 0.3) & 
                (rounded_data <= stat['majority'] + 0.3) & 
                (poly_array > 0)
            )
            
            # Set matching cells in mask to 1
            mask_data[matching_cells] = 1
        
        # Write updated mask data
        src_mask.write(mask_data, 1)
    
    return mask_raster_path

def process_raster_by_shapefile(input_raster_path, shapefile_path):
    """
    Comprehensive raster processing workflow with error handling and temp file management.
    
    Parameters:
    -----------
    input_raster_path : str
        Path to the input raster file
    shapefile_path : str
        Path to the shapefile with polygons
    
    Returns:
    --------
    str
        Path to the final mask raster
    """
    # Create a single temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Step 1: Prepare raster and create initial mask
        rounded_raster_path, mask_raster_path = prepare_raster_mask(
            input_raster_path, 
            temp_dir
        )
        
        # Step 2: Create majority mask
        final_mask_path = create_majority_mask(
            rounded_raster_path, 
            shapefile_path, 
            mask_raster_path
        )
        
        # Optional: Copy final mask to a more permanent location if needed
        final_output_path = os.path.join(LC_CACHE_PATH,'TESTING', 'final_mask.tif')
        shutil.copy(final_mask_path, final_output_path)
        
        return final_output_path
    
    except Exception as e:
        print(f"An error occurred during processing: {e}")
        return None
    
    finally:
        # Always attempt to clean up temporary files
        try:
            shutil.rmtree(temp_dir)
        except Exception as cleanup_error:
            print(f"Error during cleanup: {cleanup_error}")

# Jupyter Notebook usage example
try:
    # Specify your input paths
    input_raster = LC_DEM_PATH
    input_shapefile = LC_ROAD_EDGE_BUFFERED_PATH
    
    # Process the raster
    result_mask = process_raster_by_shapefile(input_raster, input_shapefile)
    
    if result_mask:
        print(f"Mask created successfully at: {result_mask}")
    else:
        print("Mask creation failed")

except Exception as notebook_error:
    print(f"Notebook-level error: {notebook_error}")

# |%%--%%| <fKIeorFz7X|r9w2QdQ5tR>

import os
import tempfile
import rasterio
import numpy as np
import geopandas as gpd
from multiprocessing import Pool, cpu_count
import shutil

def split_polygons_for_workers(gdf, num_workers=None):
    """
    Split polygons into groups for parallel processing.
    
    Parameters:
    -----------
    gdf : GeoDataFrame
        Input geodataframe of polygons
    num_workers : int, optional
        Number of workers. Defaults to CPU count if not specified
    
    Returns:
    --------
    list of lists
        List of polygon groups, one for each worker
    """
    
    # Ensure num_workers doesn't exceed polygon count
    num_workers = min(num_workers, len(gdf))
    
    # Calculate group sizes
    group_size = len(gdf) // num_workers
    remainder = len(gdf) % num_workers
    
    # Split polygons into groups
    polygon_groups = []
    start = 0
    
    for i in range(num_workers):
        # Distribute remainder groups across first few workers
        current_group_size = group_size + (1 if i < remainder else 0)
        
        # Select group of polygons
        group = gdf.iloc[start:start+current_group_size]
        polygon_groups.append(group)
        
        start += current_group_size
    
    return polygon_groups

def worker_prepare_raster_mask(input_raster_path, temp_dir, worker_id):
    """
    Prepare a worker-specific copy of the raster and mask.
    
    Parameters:
    -----------
    input_raster_path : str
        Path to the input raster file
    temp_dir : str
        Temporary directory to store output files
    worker_id : int
        Unique identifier for the worker
    
    Returns:
    --------
    tuple
        Paths to the worker-specific rounded raster and mask raster
    """
    # Open the input raster
    with rasterio.open(input_raster_path) as src:
        # Read the raster data
        raster_data = src.read(1)  # Assumes single-band raster
        
        # Round the raster to 1 decimal place
        rounded_data = np.round(raster_data, decimals=1)
        
        # Prepare rounded raster metadata
        rounded_profile = src.profile.copy()
        
        # Path for worker-specific rounded raster
        rounded_raster_path = os.path.join(temp_dir, f'rounded_raster_worker_{worker_id}.tif')
        
        # Write rounded raster
        with rasterio.open(rounded_raster_path, 'w', **rounded_profile) as dst:
            dst.write(rounded_data.astype(rounded_profile['dtype']), 1)
        
        # Prepare empty mask raster metadata
        mask_profile = src.profile.copy()
        mask_profile.update({
            'nodata': 0,  # Set no data value
            'dtype': 'uint8'  # Binary raster
        })
        
        # Path for worker-specific mask raster
        mask_raster_path = os.path.join(temp_dir, f'mask_raster_worker_{worker_id}.tif')
        
        # Create empty mask raster with same extent and CRS
        with rasterio.open(mask_raster_path, 'w', **mask_profile) as dst:
            empty_mask = np.zeros_like(raster_data, dtype=np.uint8)
            dst.write(empty_mask, 1)
    
    return rounded_raster_path, mask_raster_path

def worker_create_majority_mask(worker_args):
    """
    Worker function to process a subset of polygons and create a mask.
    
    Parameters:
    -----------
    worker_args : tuple
        Contains:
        - rounded_raster_path: Path to the worker-specific rounded raster
        - mask_raster_path: Path to the worker-specific mask raster
        - polygon_group: GeoDataFrame of polygons for this worker
    
    Returns:
    --------
    numpy.ndarray
        Mask data created by this worker
    """
    rounded_raster_path, mask_raster_path, polygon_group = worker_args
    
    # Open the rounded raster and mask raster
    with rasterio.open(rounded_raster_path) as src_rounded, \
         rasterio.open(mask_raster_path, 'r+') as src_mask:
        
        # Read raster data
        rounded_data = src_rounded.read(1)
        mask_data = src_mask.read(1)
        
        # Perform zonal statistics for each polygon in this group
        zstats = rasterstats.zonal_stats(
            polygon_group, 
            rounded_raster_path, 
            stats=['majority'],
            all_touched=True
        )
        
        # Create a temporary directory for the shapefile
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create full path for temporary shapefile
            temp_shapefile = os.path.join(temp_dir, 'temp_polygons.shp')
            
            # Ensure the polygon group is saved with all necessary files
            polygon_group.to_file(temp_shapefile)
            
            try:
                # Open the shapefile using fiona to verify it exists
                import fiona
                with fiona.open(temp_shapefile) as src:
                    # Verify the shapefile can be read
                    pass
                
                # Open with GDAL
                shapefile_ds = gdal.OpenEx(temp_shapefile, gdal.OF_VECTOR)
                if shapefile_ds is None:
                    raise RuntimeError(f"Could not open {temp_shapefile}")
                
                # Iterate through polygons in this group
                for idx, (polygon, stat) in enumerate(zip(polygon_group.geometry, zstats)):
                    # Skip if no majority value
                    if stat['majority'] is None:
                        continue
                    
                    # Rasterize options
                    rasterize_options = gdal.RasterizeOptions(
                        format='MEM',
                        outputType=gdal.GDT_Byte,
                        width=mask_data.shape[1], 
                        height=mask_data.shape[0],
                        outputBounds=[
                            src_mask.bounds.left, 
                            src_mask.bounds.bottom, 
                            src_mask.bounds.right, 
                            src_mask.bounds.top
                        ],
                        where=f"FID = {idx}",
                        allTouched=True
                    )
                    
                    # Rasterize the polygon
                    poly_raster_mask = gdal.Rasterize(
                        '', 
                        shapefile_ds,
                        options=rasterize_options
                    )
                    
                    # Read the rasterized polygon
                    poly_array = poly_raster_mask.ReadAsArray()
                    
                    # Find cells within majority ± 0.3
                    matching_cells = (
                        (rounded_data >= stat['majority'] - 0.3) & 
                        (rounded_data <= stat['majority'] + 0.3) & 
                        (poly_array > 0)
                    )
                    
                    # Set matching cells in mask to 1
                    mask_data[matching_cells] = 1
                
                # Clean up GDAL dataset
                shapefile_ds = None
            except Exception as e:
                print(f"Error processing shapefile: {e}")
                # Optionally log the full traceback
                import traceback
                traceback.print_exc()
                return None  # or handle the error appropriately
        
        # Return the worker's mask data
        return mask_data

def process_raster_by_shapefile_parallel(input_raster_path, input_shapefile_path, num_workers=None):
    """
    Process raster by shapefile in parallel
    
    Parameters:
    -----------
    input_raster_path : str
        Path to the input raster file
    input_shapefile_path : str
        Path to the input shapefile
    num_workers : int, optional
        Number of workers to use. Defaults to CPU count.
    
    Returns:
    --------
    numpy.ndarray
        Final mask raster
    """
    # Read input shapefile
    gdf = gpd.read_file(input_shapefile_path)
    
    # Determine number of workers
    if num_workers is None:
        num_workers = cpu_count()-2
    
    # Split polygons into groups for workers
    polygon_groups = split_polygons_for_workers(gdf, num_workers)
    
    # Create a temporary directory for worker files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Prepare worker-specific rasters
        worker_raster_paths = []
        for worker_id, polygon_group in enumerate(polygon_groups):
            # Prepare raster and mask for each worker
            rounded_raster_path, mask_raster_path = worker_prepare_raster_mask(
                input_raster_path, 
                temp_dir, 
                worker_id
            )
            
            # Store worker paths along with polygon group
            worker_raster_paths.append((
                rounded_raster_path, 
                mask_raster_path, 
                polygon_group
            ))
        
        # Use multiprocessing to create masks
        try:
            with Pool(processes=num_workers) as pool:
                # Map worker function to worker paths
                worker_mask_results = pool.map(worker_create_majority_mask, worker_raster_paths)
            
            # Check if all workers completed successfully
            if any(result is None for result in worker_mask_results):
                raise RuntimeError("One or more workers failed to process")
            
            # Combine masks from all workers
            final_mask = np.zeros_like(worker_mask_results[0], dtype=np.uint8)
            for worker_mask in worker_mask_results:
                final_mask |= worker_mask
            
            return final_mask
        
        except Exception as e:
            print(f"Parallel processing error: {e}")
            import traceback
            traceback.print_exc()
            return None

# Specify your input paths
input_raster = LC_DEM_PATH
input_shapefile = LC_ROAD_EDGE_BUFFERED_PATH

#input_raster = '/home/rbhatt/Projects/RoadConnect/CACHE/__EPSG_6566__EXTENT_-65_303_18_277_-65_281_18_302/DEM/Temp/dem_clipped.tif'
#input_shapefile = '/home/rbhatt/Projects/RoadConnect/CACHE/__EPSG_6566__EXTENT_-65_303_18_277_-65_281_18_302/TESTING/geopandas_voronoi_clipped.shp'

# Process the raster in parallel
result_mask = process_raster_by_shapefile_parallel(
    input_raster, 
    input_shapefile,
    num_workers=10  # Use default (CPU count)
)

# Correct way to check if the mask was created successfully
if result_mask is not None and result_mask.size > 0:
    print(f"Parallel mask created successfully")
    
    # Optional: Save the mask to a file
    with rasterio.open(input_raster) as src:
        # Create a clean output profile
        output_profile = {
            'driver': 'GTiff',
            'dtype': 'uint8',
            'nodata': 0,
            'width': src.width,
            'height': src.height,
            'count': 1,
            'crs': src.crs,
            'transform': src.transform,
        }
        
        output_path = os.path.join(os.path.dirname(input_raster), 'output_mask.tif')
        
        with rasterio.open(output_path, 'w', **output_profile) as dst:
            dst.write(result_mask.astype(np.uint8), 1)
        
        print(f"Mask saved to: {output_path}")
else:
    print("Failed to create mask")

# |%%--%%| <r9w2QdQ5tR|OIraz6C0Le>

import os
import tempfile
import rasterio
import numpy as np
import geopandas as gpd
from multiprocessing import Pool, cpu_count
import shutil

def split_polygons_for_workers(gdf, num_workers=None):
    """
    Split polygons into groups for parallel processing.
    
    Parameters:
    -----------
    gdf : GeoDataFrame
        Input geodataframe of polygons
    num_workers : int, optional
        Number of workers. Defaults to CPU count if not specified
    
    Returns:
    --------
    list of lists
        List of polygon groups, one for each worker
    """
    
    # Ensure num_workers doesn't exceed polygon count
    num_workers = min(num_workers, len(gdf))
    
    # Calculate group sizes
    group_size = len(gdf) // num_workers
    remainder = len(gdf) % num_workers
    
    # Split polygons into groups
    polygon_groups = []
    start = 0
    
    for i in range(num_workers):
        # Distribute remainder groups across first few workers
        current_group_size = group_size + (1 if i < remainder else 0)
        
        # Select group of polygons
        group = gdf.iloc[start:start+current_group_size]
        polygon_groups.append(group)
        
        start += current_group_size
    
    return polygon_groups

def worker_prepare_raster_mask(input_raster_path, polygon_group, temp_dir, worker_id):
    """
    Prepare a worker-specific copy of the raster clipped to polygon group extent.
    
    Parameters:
    -----------
    input_raster_path : str
        Path to the input raster file
    polygon_group : GeoDataFrame
        Group of polygons for this worker
    temp_dir : str
        Temporary directory to store output files
    worker_id : int
        Unique identifier for the worker
    
    Returns:
    --------
    tuple
        Paths to the worker-specific rounded raster and mask raster
    """
    # Open the input raster
    with rasterio.open(input_raster_path) as src:
        # Compute the combined bounds of the polygon group
        worker_bounds = polygon_group.total_bounds
        
        # Compute the window for reading
        window = rasterio.windows.from_bounds(
            left=worker_bounds[0], 
            bottom=worker_bounds[1], 
            right=worker_bounds[2], 
            top=worker_bounds[3], 
            transform=src.transform
        )
        
        # Read the raster data for this window
        raster_data = src.read(1, window=window)
        
        # Round the raster to 1 decimal place
        rounded_data = np.round(raster_data, decimals=1)
        
        # Prepare rounded raster metadata
        rounded_profile = src.profile.copy()
        
        # Update the profile with the new window's transform and dimensions
        rounded_profile.update({
            'height': window.height,
            'width': window.width,
            'transform': rasterio.windows.transform(window, src.transform)
        })
        
        # Path for worker-specific rounded raster
        rounded_raster_path = os.path.join(temp_dir, f'rounded_raster_worker_{worker_id}.tif')
        
        # Write rounded raster
        with rasterio.open(rounded_raster_path, 'w', **rounded_profile) as dst:
            dst.write(rounded_data.astype(rounded_profile['dtype']), 1)
        
        # Prepare empty mask raster metadata
        mask_profile = rounded_profile.copy()
        mask_profile.update({
            'nodata': 0,  # Set no data value
            'dtype': 'uint8'  # Binary raster
        })
        
        # Path for worker-specific mask raster
        mask_raster_path = os.path.join(temp_dir, f'mask_raster_worker_{worker_id}.tif')
        
        # Create empty mask raster with same extent and CRS
        with rasterio.open(mask_raster_path, 'w', **mask_profile) as dst:
            empty_mask = np.zeros_like(rounded_data, dtype=np.uint8)
            dst.write(empty_mask, 1)
    
    return rounded_raster_path, mask_raster_path

def worker_create_majority_mask(worker_args):
    """
    Worker function to process a subset of polygons and create a mask.
    
    Parameters:
    -----------
    worker_args : tuple
        Contains:
        - rounded_raster_path: Path to the worker-specific rounded raster
        - mask_raster_path: Path to the worker-specific mask raster
        - polygon_group: GeoDataFrame of polygons for this worker
    
    Returns:
    --------
    numpy.ndarray
        Mask data created by this worker
    """
    rounded_raster_path, mask_raster_path, polygon_group = worker_args
    
    # Open the rounded raster and mask raster
    with rasterio.open(rounded_raster_path) as src_rounded, \
         rasterio.open(mask_raster_path, 'r+') as src_mask:
        
        # Read raster data
        rounded_data = src_rounded.read(1)
        mask_data = src_mask.read(1)
        
        # Perform zonal statistics for each polygon in this group
        zstats = rasterstats.zonal_stats(
            polygon_group, 
            rounded_raster_path, 
            stats=['majority'],
            all_touched=True
        )
        
        # Create a temporary directory for the shapefile
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create full path for temporary shapefile
            temp_shapefile = os.path.join(temp_dir, 'temp_polygons.shp')
            
            # Ensure the polygon group is saved with all necessary files
            polygon_group.to_file(temp_shapefile)
            
            try:
                # Open the shapefile using fiona to verify it exists
                import fiona
                with fiona.open(temp_shapefile) as src:
                    # Verify the shapefile can be read
                    pass
                
                # Open with GDAL
                shapefile_ds = gdal.OpenEx(temp_shapefile, gdal.OF_VECTOR)
                if shapefile_ds is None:
                    raise RuntimeError(f"Could not open {temp_shapefile}")
                
                # Iterate through polygons in this group
                for idx, (polygon, stat) in enumerate(zip(polygon_group.geometry, zstats)):
                    # Skip if no majority value
                    if stat['majority'] is None:
                        continue
                    
                    # Rasterize options
                    rasterize_options = gdal.RasterizeOptions(
                        format='MEM',
                        outputType=gdal.GDT_Byte,
                        width=mask_data.shape[1], 
                        height=mask_data.shape[0],
                        outputBounds=[
                            src_mask.bounds.left, 
                            src_mask.bounds.bottom, 
                            src_mask.bounds.right, 
                            src_mask.bounds.top
                        ],
                        where=f"FID = {idx}",
                        allTouched=True
                    )
                    
                    # Rasterize the polygon
                    poly_raster_mask = gdal.Rasterize(
                        '', 
                        shapefile_ds,
                        options=rasterize_options
                    )
                    
                    # Read the rasterized polygon
                    poly_array = poly_raster_mask.ReadAsArray()
                    
                    # Find cells within majority ± 0.3
                    matching_cells = (
                        (rounded_data >= stat['majority'] - 0.3) & 
                        (rounded_data <= stat['majority'] + 0.3) & 
                        (poly_array > 0)
                    )
                    
                    # Set matching cells in mask to 1
                    mask_data[matching_cells] = 1
                
                # Clean up GDAL dataset
                shapefile_ds = None
            except Exception as e:
                print(f"Error processing shapefile: {e}")
                # Optionally log the full traceback
                import traceback
                traceback.print_exc()
                return None  # or handle the error appropriately
        
        # Return the worker's mask data
        return mask_data

def process_raster_by_shapefile_parallel(input_raster_path, input_shapefile_path, num_workers=None):
    """
    Process raster by shapefile in parallel
    
    Parameters:
    -----------
    input_raster_path : str
        Path to the input raster file
    input_shapefile_path : str
        Path to the input shapefile
    num_workers : int, optional
        Number of workers to use. Defaults to CPU count.
    
    Returns:
    --------
    numpy.ndarray
        Final mask raster
    """
    # Read input shapefile
    gdf = gpd.read_file(input_shapefile_path)
    
    # Determine number of workers
    if num_workers is None:
        num_workers = cpu_count()-2
    
    # Split polygons into groups for workers
    polygon_groups = split_polygons_for_workers(gdf, num_workers)
    
    # Create a temporary directory for worker files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Open the original raster to get full dimensions
        with rasterio.open(input_raster_path) as src:
            full_raster_profile = src.profile.copy()
            full_raster_shape = (src.height, src.width)
        
        # Prepare worker-specific rasters
        worker_raster_paths = []
        for worker_id, polygon_group in enumerate(polygon_groups):
            # Prepare raster and mask for each worker
            rounded_raster_path, mask_raster_path = worker_prepare_raster_mask(
                input_raster_path, 
                polygon_group,
                temp_dir, 
                worker_id
            )
            
            # Store worker paths along with polygon group
            worker_raster_paths.append((
                rounded_raster_path, 
                mask_raster_path, 
                polygon_group
            ))
        
        # Use multiprocessing to create masks
        try:
            with Pool(processes=num_workers) as pool:
                # Map worker function to worker paths
                worker_mask_results = pool.map(worker_create_majority_mask, worker_raster_paths)
            
            # Check if all workers completed successfully
            if any(result is None for result in worker_mask_results):
                raise RuntimeError("One or more workers failed to process")
            
            # Create a full-sized final mask initialized with zeros
            final_mask = np.zeros(full_raster_shape, dtype=np.uint8)
            
            # Combine masks from all workers
            for worker_mask, worker_raster_path in zip(worker_mask_results, worker_raster_paths):
                # Open the corresponding rounded raster to get its transform
                with rasterio.open(worker_raster_path[0]) as src:
                    # Calculate the window in the full raster
                    window = rasterio.windows.from_bounds(
                        left=src.bounds.left,
                        bottom=src.bounds.bottom,
                        right=src.bounds.right,
                        top=src.bounds.top,
                        transform=full_raster_profile['transform']
                    )
                    
                    # Update the final mask with the worker's mask in its correct location
                    final_mask[
                        int(window.row_off):int(window.row_off + window.height),
                        int(window.col_off):int(window.col_off + window.width)
                    ] |= worker_mask
            
            return final_mask
        
        except Exception as e:
            print(f"Parallel processing error: {e}")
            import traceback
            traceback.print_exc()
            return None

# Specify your input paths
input_raster = LC_DEM_PATH
input_shapefile = LC_ROAD_EDGE_BUFFERED_PATH

#input_raster = '/home/rbhatt/Projects/RoadConnect/CACHE/__EPSG_6566__EXTENT_-65_303_18_277_-65_281_18_302/DEM/Temp/dem_clipped.tif'
#input_shapefile = '/home/rbhatt/Projects/RoadConnect/CACHE/__EPSG_6566__EXTENT_-65_303_18_277_-65_281_18_302/TESTING/geopandas_voronoi_clipped.shp'

# Process the raster in parallel
result_mask = process_raster_by_shapefile_parallel(
    input_raster, 
    input_shapefile,
    num_workers=None
)

if result_mask is not None and result_mask.size > 0:
    print(f"Parallel mask created successfully")
    
    # Optional: Save the mask to a file
    with rasterio.open(input_raster) as src:
        # Create a clean output profile
        output_profile = {
            'driver': 'GTiff',
            'dtype': 'uint8',
            'nodata': 0,
            'width': src.width,
            'height': src.height,
            'count': 1,
            'crs': src.crs,
            'transform': src.transform,
        }
        
        output_path = os.path.join(os.path.dirname(input_raster), 'output_mask.tif')
        
        with rasterio.open(output_path, 'w', **output_profile) as dst:
            dst.write(result_mask.astype(np.uint8), 1)
        
        print(f"Mask saved to: {output_path}")
else:
    print("Failed to create mask")

# |%%--%%| <OIraz6C0Le|pUXPdTm3y2>

import contextily as ctx
import geopandas as gpd
import os

def download_esri_imagery(gdf, output_path):
    """
    Download ESRI World Imagery for a geodataframe
    
    Args:
    - gdf: Input geodataframe
    - output_path: Path to save the output raster
    """
    # Ensure the geodataframe is in Web Mercator
    gdf = gdf.to_crs(epsg=3857)
    
    # Get bounds
    w, s, e, n = gdf.total_bounds
    
    # Download and save raster
    ctx.bounds2raster(
        w, s, e, n, 
        path=output_path, 
        zoom=19,  # Let contextily determine optimal zoom
        source=ctx.providers.Esri.WorldImagery,
        ll=False,  # Coordinates are already in Web Mercator
        use_cache=True,
        wait=30,
        n_connections=3
    )
    
    return output_path

# Usage example
output_raster = download_esri_imagery(EXTENT_GDF, os.path.join(LC_CACHE_PATH, 'TEST_SAT.tif'))

# |%%--%%| <pUXPdTm3y2|OhQnbBpXXU>
r"""°°°
### Road Flowpath Simulation
°°°"""
# |%%--%%| <OhQnbBpXXU|z4hWMu1fNn>
r"""°°°
For each minima, we're trying to find all of the contributing road segments. This is done using a reverse search. Currently, several minima may share overlapping road segments (i.e., divergent flow).

This needs to be reworked so that only the steepest flow in chosen. To verify, the sum of all individual flow paths should equal the total length of the road network.
°°°"""
# |%%--%%| <z4hWMu1fNn|09JrU2ohgA>

def find_upstream_segments(split_gdf, minima_gdf, output_dir):
    split_gdf = split_gdf.reset_index(drop=True)
    minima_gdf = minima_gdf.reset_index(drop=True)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create spatial index
    sindex = split_gdf.sindex

    for i, minima_point in tqdm(minima_gdf.iterrows(), total=len(minima_gdf)):
        # Find the segment that this minima point belongs to
        possible_matches_index = list(sindex.intersection(minima_point.geometry.bounds))
        possible_matches = split_gdf.iloc[possible_matches_index]
        containing_segment = possible_matches[possible_matches.geometry.distance(minima_point.geometry) < 1e-6]

        if containing_segment.empty:
            print(f"No matching segment for minima point {i}")
            continue

        seed_idx = containing_segment.index[0]
        seed_elevation = split_gdf.loc[seed_idx, 'elevation']

        # Perform reverse search
        visited = set()
        to_visit = [seed_idx]

        while to_visit:
            current_idx = to_visit.pop()
            if current_idx in visited:
                continue
            visited.add(current_idx)

            current_geom = split_gdf.loc[current_idx, 'geometry']
            current_elev = split_gdf.loc[current_idx, 'elevation']

            # Find connected segments
            possible = list(sindex.intersection(current_geom.bounds))
            connected = split_gdf.iloc[possible]
            neighbors = connected[connected.geometry.touches(current_geom)]

            # Filter only those with higher elevation (reverse traversal)
            lower_neighbors = neighbors[neighbors.elevation > current_elev]

            for n_idx in lower_neighbors.index:
                if n_idx not in visited:
                    to_visit.append(n_idx)


        # Get all geometries and perform union
        merged_geom = unary_union(split_gdf.loc[list(visited)].geometry)

        # Wrap in GeoDataFrame
        union_gdf = gpd.GeoDataFrame(geometry=[merged_geom], crs=split_gdf.crs)

        # Save to shapefile
        out_path = os.path.join(output_dir, f"minima_{i}.shp")
        union_gdf.to_file(out_path)

minima_gdf = find_minima_points(split_gdf, LC_CACHE_PATH)
find_upstream_segments(split_gdf, minima_gdf, os.path.join(LC_CACHE_PATH, "TEST"))

# |%%--%%| <09JrU2ohgA|BBXc0LSx8x>
r"""°°°
 ## LiDAR
°°°"""
# |%%--%%| <BBXc0LSx8x|MPzJHZcAAY>
r"""°°°
### Retrieve LiDAR Tile Index and get intersecting tiles
°°°"""
# |%%--%%| <MPzJHZcAAY|RCDQbmUIHk>

# Ignore the ridiculousness in the code here, it's pretty late on a friday or early on a saturday depending on who you ask
try:
    if Path(GC_LIDAR_TILE_INDEX_PATH).is_file():
        logger.info(f"CACHE HIT! CACHE HIT!")
    else:
        logger.info(f"CACHE MISS. DOWNLOADING {GC_LIDAR_TILE_INDEX_URL}")
        lidar_tile_index_zip = requests.get(GC_LIDAR_TILE_INDEX_URL)
        with zipfile.ZipFile(io.BytesIO(lidar_tile_index_zip.content)) as z:
            z.extractall(GC_LIDAR_TILE_INDEX_DIR)
        del lidar_tile_index_zip

    LIDAR_TILE_INDEX = gpd.read_file(GC_LIDAR_TILE_INDEX_PATH).to_crs(LOCAL_EPSG_CODE)
    intersections = gpd.overlay(EXTENT_GDF, LIDAR_TILE_INDEX, how="intersection").to_crs(LOCAL_EPSG_CODE).drop_duplicates(subset=GC_LIDAR_TILE_INDEX_URL_COLUMN_NAME)
except Exception as e:
    logger.fatal(f"Something went wrong finding or retrieving intersecting DEM Tiles. {e}\n{traceback.format_exc()}")
    sys.exit(1)

# |%%--%%| <RCDQbmUIHk|ZC58gb5Z4F>
r"""°°°
### Download LiDAR Tiles
°°°"""
# |%%--%%| <ZC58gb5Z4F|ayXaetFGQv>

try:
    for _, tile in intersections.iterrows():
        lidar_url = tile[GC_LIDAR_TILE_INDEX_URL_COLUMN_NAME].strip()
        tile_filename = os.path.basename(lidar_url)
        tile_path = os.path.join(GC_LIDAR_ORIGINAL_CACHE_DIR, tile_filename)

        if Path(tile_path).is_file():
            logger.info(f"CACHE HIT! CACHE HIT!")
            continue

        logger.info(f"CACHE MISS. DOWNLOADING {lidar_url}")
        response = requests.get(lidar_url)
        response.raise_for_status()

        with open(tile_path, 'wb') as f:
            f.write(response.content)

except Exception as e:
    logger.fatal(f"Something went wrong downloading {lidar_url}. Error: {e}\n{traceback.format_exc()}")
    sys.exit(1)

# |%%--%%| <ayXaetFGQv|WMW03aQclx>
r"""°°°
### PDAL Pipelines
°°°"""
# |%%--%%| <WMW03aQclx|BSvmKyqaZl>
r"""°°°
#### Unprojected (Won't appear in CloudCompare or such because it's not in a Cartesian Coord System) and Uncolored
°°°"""
# |%%--%%| <BSvmKyqaZl|eQ3Xrnu7YV>
r"""°°°
##### Resampled Ground Points Cropped To Extent (we don't care for non-ground points - water can travel through trees or buildings)
°°°"""
# |%%--%%| <eQ3Xrnu7YV|uE4rU6Elha>

try:
    if Path(LC_LIDAR_PATH).is_file():
        logger.info(f"CACHE HIT! CACHE HIT!")
    else:
        bound_string = f"([{MINX_WGS84},{MAXX_WGS84}], [{MINY_WGS84},{MAXY_WGS84}])"

        # Get list of intersecting file names
        intersecting_files = set(os.path.basename(tile[GC_LIDAR_TILE_INDEX_URL_COLUMN_NAME].strip()) for _, tile in intersections.iterrows())

        # Get paths for those files
        laz_files = [os.path.join(GC_LIDAR_ORIGINAL_CACHE_DIR, f) for f in os.listdir(GC_LIDAR_ORIGINAL_CACHE_DIR) if f in intersecting_files]

        # Create the pipeline stages for readers
        readers = [{"type": "readers.las", "filename": f} for f in laz_files]

        # Full pipeline: readers + merge + crop + writer
        pipeline_dict = readers + [
            {"type": "filters.merge"},
            {
                "type": "filters.crop",
                "bounds": bound_string,
            },
            #{
            #    "type":"filters.expression",
            #    "expression":"Classification == 2 && ReturnNumber == 1"
            #},
            {
                "type":"writers.las",
                "extra_dims":"all",
                "scale_x":"auto",
                "scale_y":"auto",
                "scale_z":"auto",
                "offset_x":"auto",
                "offset_y":"auto",
                "offset_z":"auto",
                "filename":f'{LC_LIDAR_PATH[:-4]}.las',
            },
        ]

        # Convert pipeline to JSON string
        pipeline_json = json.dumps(pipeline_dict)

        # Run the pipeline
        pipeline = pdal.Pipeline(pipeline_json)
        pipeline.execute()
        print(f"Successfully processed LAZ files and saved to {LC_LIDAR_PATH}")

except Exception as e:
    logger.fatal(f"Something went wrong during LAZ merging and reprojection: {e}\n{traceback.format_exc()}")
    sys.exit(1)

# |%%--%%| <uE4rU6Elha|SFF5aJOJT9>

from shapely.wkt import loads

try:
    if Path(os.path.join(LC_LIDAR_EXPERIMENTAL_DIR,"db.las")).is_file():
        logger.info(f"CACHE HIT! CACHE HIT!")
    else:
        # Convert GeoDataFrame to WKT polygon string (assumes a single geometry so make sure you've .union_all()'d the buffer)
        crop_polygon_wkt = ROAD_BUFFERED.to_crs("EPSG:4326").geometry.iloc[0].wkt

        pipeline_dict = [
            {
                "type": "readers.las",
                "filename": LC_LIDAR_PATH
            },
            {
                "type": "filters.crop",
                "polygon": crop_polygon_wkt
            },
            {
                "type": "filters.reprojection",
                "out_srs": LIDAR_REPROJECTION_CODE
            },
            {
                "type":"filters.covariancefeatures",
                "feature_set": "Scattering,Verticality",
                "threads": 10,
                "radius": 4,
            },
            {
                "type":"filters.normal",
                "refine":True,
                "knn":8
            },
            {
                "type":"filters.lof",
                "minpts":10
            },
            {
                "type": "filters.lloydkmeans",
                "dimensions":"Scattering,Verticality,Curvature,LocalOutlierFactor,LocalReachabilityDistance,NNDistance",
                "k":3,
                "maxiters":10,
            },
            {
                "type": "writers.las",
                "scale_x": "auto",
                "scale_y": "auto",
                "scale_z": "auto",
                "offset_x": "auto",
                "offset_y": "auto",
                "offset_z": "auto",
                "extra_dims":"all",
                "filename": os.path.join(LC_LIDAR_EXPERIMENTAL_DIR,"db_p1.las"),
            },
            {
                "type":"filters.expression",
                "expression":"ClusterID == 0"
            },
            {
                "type": "writers.las",
                "scale_x": "auto",
                "scale_y": "auto",
                "scale_z": "auto",
                "offset_x": "auto",
                "offset_y": "auto",
                "offset_z": "auto",
                "extra_dims":"all",
                "filename": os.path.join(LC_LIDAR_EXPERIMENTAL_DIR,"db_p2.las"),
            },
            {
                "type":"filters.lof",
                "minpts":10
            },
            {
                "type": "filters.lloydkmeans",
                "dimensions":"LocalOutlierFactor,LocalReachabilityDistance,NNDistance",
                "k":10,
                "maxiters":10,
            },
            {
                "type":"filters.expression",
                "expression":"ClusterID == 0"
            },
            {
                "type": "writers.las",
                "scale_x": "auto",
                "scale_y": "auto",
                "scale_z": "auto",
                "offset_x": "auto",
                "offset_y": "auto",
                "offset_z": "auto",
                "extra_dims":"all",
                "filename": os.path.join(LC_LIDAR_EXPERIMENTAL_DIR,"db_p3.las"),
            },
            {
                "type":"filters.radialdensity",
                "radius":4
            },
            {
                "type": "writers.las",
                "scale_x": "auto",
                "scale_y": "auto",
                "scale_z": "auto",
                "offset_x": "auto",
                "offset_y": "auto",
                "offset_z": "auto",
                "extra_dims":"all",
                "filename": os.path.join(LC_LIDAR_EXPERIMENTAL_DIR,"db_p4.las"),
            },
            {
                "type":"filters.hexbin",
                "tag":"hex_boundary",
                "edge_length":2
            },
            {
                "type": "writers.las",
                "scale_x": "auto",
                "scale_y": "auto",
                "scale_z": "auto",
                "offset_x": "auto",
                "offset_y": "auto",
                "offset_z": "auto",
                "extra_dims":"all",
                "filename": os.path.join(LC_LIDAR_EXPERIMENTAL_DIR,"db_final.las"),
            },
        ]

        # Convert pipeline to JSON string
        pipeline_json = json.dumps(pipeline_dict)

        # Run the pipeline
        pipeline = pdal.Pipeline(pipeline_json)
        pipeline.execute()
        wkt_hex_boundary = pipeline.metadata["metadata"]['filters.hexbin']['boundary']
        shp_geometries = [loads(wkt_hex_boundary)]
        
        # Create a GeoDataFrame
        gdf = gpd.GeoDataFrame(index=[0], crs=LIDAR_REPROJECTION_CODE, geometry=shp_geometries)

        # Save to shapefile
        gdf.to_file(os.path.join(LC_LIDAR_EXPERIMENTAL_DIR,"boundary.shp"))

        print(f"Successfully processed LAZ file and saved to {os.path.join(LC_LIDAR_EXPERIMENTAL_DIR,"db.las")}")

except Exception as e:
    logger.fatal(f"Something went wrong during LAZ processing: {e}\n{traceback.format_exc()}")
    sys.exit(1)

# |%%--%%| <SFF5aJOJT9|Mu1SrgeExM>
r"""°°°
##### Road Buffer Points Only
°°°"""
# |%%--%%| <Mu1SrgeExM|s2uIWDPUgV>

try:
    if Path(LC_LIDAR_ROAD_ONLY_PATH).is_file():
        logger.info(f"CACHE HIT! CACHE HIT!")
    else:
        # Convert GeoDataFrame to WKT polygon string (assumes a single geometry so make sure you've .union_all()'d the buffer)
        crop_polygon_wkt = ROAD_BUFFERED.to_crs("EPSG:4326").geometry.iloc[0].wkt

        pipeline_dict = [
            {
                "type": "readers.las",
                "filename": LC_LIDAR_PATH
            },
            {
                "type": "filters.crop",
                "polygon": crop_polygon_wkt
            },
            {
                "type": "writers.las",
                "scale_x": "auto",
                "scale_y": "auto",
                "scale_z": "auto",
                "offset_x": "auto",
                "offset_y": "auto",
                "offset_z": "auto",
                "filename": LC_LIDAR_ROAD_ONLY_PATH,
            },
        ]

        # Convert pipeline to JSON string
        pipeline_json = json.dumps(pipeline_dict)

        # Run the pipeline
        pipeline = pdal.Pipeline(pipeline_json)
        pipeline.execute()

        print(f"Successfully processed LAZ file and saved to {LC_LIDAR_ROAD_ONLY_PATH}")

except Exception as e:
    logger.fatal(f"Something went wrong during LAZ processing: {e}\n{traceback.format_exc()}")
    sys.exit(1)

# |%%--%%| <s2uIWDPUgV|EoNMo5Rd5I>
r"""°°°
#### Projected and Colored (Usable in CloudCompare and such because the pipeline reprojects into a Cartesian Coordinate System)
°°°"""
# |%%--%%| <EoNMo5Rd5I|jPW6S8rBr6>
r"""°°°
##### All Points
°°°"""
# |%%--%%| <jPW6S8rBr6|wLidxZvj5s>

try:
    if Path(LC_LIDAR_PROJECTED_PATH).is_file():
        logger.info(f"CACHE HIT! CACHE HIT!")
    else:
        pipeline_dict = [
            {
                "type": "readers.las",
                "filename": LC_LIDAR_PATH
            },
            {
                "type": "filters.reprojection",
                "out_srs": LIDAR_REPROJECTION_CODE
            },
            {
                "type": "filters.colorization",
                "dimensions": "Red:1:1.0, Green:2:1.0, Blue:3:1.0",
                "raster": "/home/rbhatt//Projects/RoadConnect/CACHE/__EPSG_6566__EXTENT_-65_303_18_277_-65_281_18_302/LIDAR/TEST_SAT.tif"
            },
            {
                "type": "writers.las",
                "scale_x": "auto",
                "scale_y": "auto",
                "scale_z": "auto",
                "offset_x": "auto",
                "offset_y": "auto",
                "offset_z": "auto",
                "minor_version": "2",
                "dataformat_id": "3",
                "filename": LC_LIDAR_PROJECTED_PATH,
            },
        ]

        # Convert pipeline to JSON string
        pipeline_json = json.dumps(pipeline_dict)

        # Run the pipeline
        pipeline = pdal.Pipeline(pipeline_json)
        pipeline.execute()

        print(f"Successfully processed LAZ files and saved to {LC_LIDAR_PROJECTED_PATH}")

except Exception as e:
    logger.fatal(f"Something went wrong during LAZ merging and reprojection: {e}\n{traceback.format_exc()}")
    sys.exit(1)

# |%%--%%| <wLidxZvj5s|tgOJCT9Vkc>
r"""°°°
##### Road Buffer Points Only
°°°"""
# |%%--%%| <tgOJCT9Vkc|Hp1FdTwGYF>

try:
    if Path(LC_LIDAR_ROAD_ONLY_PROJECTED_PATH).is_file():
        logger.info(f"CACHE HIT! CACHE HIT!")
    else:
        # Convert GeoDataFrame to WKT polygon string (assumes a single geometry so make sure you've .union_all()'d the buffer)
        crop_polygon_wkt = ROAD_BUFFERED.to_crs(LIDAR_REPROJECTION_CODE).geometry.iloc[0].wkt

        pipeline_dict = [
            {
                "type": "readers.las",
                "filename": LC_LIDAR_ROAD_ONLY_PATH
            },
            {
                "type": "filters.reprojection",
                "out_srs": LIDAR_REPROJECTION_CODE
            },
            {
                "type": "filters.colorization",
                "dimensions": "Red:1:1.0, Green:2:1.0, Blue:3:1.0",
                "raster": "World Imagery_FD3963_2.tif"
            },
            {
                "type": "writers.las",
                "scale_x": "auto",
                "scale_y": "auto",
                "scale_z": "auto",
                "offset_x": "auto",
                "offset_y": "auto",
                "offset_z": "auto",
                "minor_version": "2",
                "dataformat_id": "3",
                "filename": LC_LIDAR_ROAD_ONLY_PROJECTED_PATH,
            },
        ]

        # Convert pipeline to JSON string
        pipeline_json = json.dumps(pipeline_dict)

        # Run the pipeline
        pipeline = pdal.Pipeline(pipeline_json)
        pipeline.execute()

        print(f"Successfully processed LAZ files and saved to {LC_LIDAR_ROAD_ONLY_PROJECTED_PATH}")

except Exception as e:
    logger.fatal(f"Something went wrong during LAZ merging and reprojection: {e}\n{traceback.format_exc()}")
    sys.exit(1)

# |%%--%%| <Hp1FdTwGYF|wcj6gHa0xR>
r"""°°°
##### Ground Points Only
°°°"""
# |%%--%%| <wcj6gHa0xR|5FOdYeQYVX>

try:
    if Path(LC_LIDAR_GROUND_ONLY_PROJECTED_PATH).is_file():
        logger.info(f"CACHE HIT! CACHE HIT!")
    else:
        pipeline_dict = [
            {
                "type": "readers.las",
                "filename": LC_LIDAR_PATH
            },
            {
                "type": "filters.reprojection",
                "out_srs": LIDAR_REPROJECTION_CODE
            },
            {
                "type":"filters.elm"
            },
            {
                "type":"filters.outlier"
            },
            {
                "type":"filters.csf"
            },
            {
                "type":"filters.expression",
                "expression":"Classification == 2"
            },
            {
                "type": "filters.colorization",
                "dimensions": "Red:1:1.0, Green:2:1.0, Blue:3:1.0",
                "raster": "World Imagery_FD3963_2.tif"
            },
            {
                "type": "writers.las",
                "scale_x": "auto",
                "scale_y": "auto",
                "scale_z": "auto",
                "offset_x": "auto",
                "offset_y": "auto",
                "offset_z": "auto",
                "minor_version": "2",
                "dataformat_id": "3",
                "filename": LC_LIDAR_GROUND_ONLY_PROJECTED_PATH,
            },
        ]

        # Convert pipeline to JSON string
        pipeline_json = json.dumps(pipeline_dict)

        # Run the pipeline
        pipeline = pdal.Pipeline(pipeline_json)
        pipeline.execute()

        print(f"Successfully processed LAZ files and saved to {LC_LIDAR_GROUND_ONLY_PROJECTED_PATH}")

except Exception as e:
    logger.fatal(f"Something went wrong during LAZ merging and reprojection: {e}\n{traceback.format_exc()}")
    sys.exit(1)

# |%%--%%| <5FOdYeQYVX|ZRu2xtq7P7>
r"""°°°
##### TESTING
°°°"""
# |%%--%%| <ZRu2xtq7P7|hiYORC1qbp>

try:
    pipeline_dict = [
        {
            "type": "readers.las",
            "filename": LC_LIDAR_ROAD_ONLY_PROJECTED_PATH
        },
        {
            "type": "filters.reprojection",
            "out_srs": LIDAR_REPROJECTION_CODE
        },
        {
            "type":"filters.elm"
        },
        {
            "type":"filters.outlier"
        },
        {
            "type":"filters.cluster"
        },
        {
            "type":"writers.bpf",
            "filename": os.path.join(LC_LIDAR_EXPERIMENTAL_DIR,"output_cluster_road_only_projected.bpf"),
            "output_dims":"X,Y,Z,ClusterID"
        },
    ]

    # Convert pipeline to JSON string
    pipeline_json = json.dumps(pipeline_dict)

    # Run the pipeline
    pipeline = pdal.Pipeline(pipeline_json)
    pipeline.execute()

    print(f"Successfully processed LAZ files and saved to {os.path.join(LC_LIDAR_EXPERIMENTAL_DIR,"output_cluster_road_only_projected.bpf")}")

except Exception as e:
    logger.fatal(f"Something went wrong during LAZ merging and reprojection: {e}\n{traceback.format_exc()}")
    sys.exit(1)

# |%%--%%| <hiYORC1qbp|EncpcPNHB7>
r"""°°°
#### Visualizations
°°°"""
# |%%--%%| <EncpcPNHB7|FbcWCc8dOY>

import laspy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the LAS file
las = laspy.read(LC_LIDAR_PROJECTED_PATH)

# Extract x, y, z coordinates
x = las.x
y = las.y
z = las.z

# Downsample for performance (optional)
sample_indices = np.random.choice(len(x), size=300000, replace=False)

# Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[sample_indices], y[sample_indices], z[sample_indices], c=z[sample_indices], cmap='terrain', s=0.5)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()


# |%%--%%| <FbcWCc8dOY|QkbIjGZLLA>

import laspy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the LAS file
las = laspy.read(LC_LIDAR_ROAD_ONLY_PROJECTED_PATH)

# Extract x, y, z coordinates
x = las.x
y = las.y
z = las.z

# Downsample for performance (optional)
sample_indices = np.random.choice(len(x), size=300000, replace=False)

# Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[sample_indices], y[sample_indices], z[sample_indices], c=z[sample_indices], cmap='terrain', s=0.5)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()


# |%%--%%| <QkbIjGZLLA|IivukV31df>
r"""°°°
## Multispectral
°°°"""
# |%%--%%| <IivukV31df|wBqzrHeO2J>

import geopandas as gpd
import requests
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from PIL import Image
import numpy as np
import mercantile
from io import BytesIO

# Mapbox parameters
MAPBOX_TOKEN = "pk.eyJ1IjoiYmhhdHQtcnVzaGktdXQiLCJhIjoiY21iNW0yNmxiMWdmcDJrcHhoejVyYWk2aiJ9.rrfRc0yg3o2HdKhwjR52cw"
MAPBOX_STYLE = "standard-satellite"  # or your custom style ID
TILE_ZOOM = 18  # Higher zoom = more detail, more tiles

# Get bounding box from GeoDataFrame
minx, miny, maxx, maxy = EXTENT_GDF.to_crs(4326).total_bounds

# Get tile coverage for the bounding box
tiles = list(mercantile.tiles(minx, miny, maxx, maxy, TILE_ZOOM))

tile_width = 512
tile_height = 512
num_x_tiles = len(set(t.x for t in tiles))
num_y_tiles = len(set(t.y for t in tiles))

# Create an empty image to stitch tiles
stitched_image = Image.new('RGB', (tile_width * num_x_tiles, tile_height * num_y_tiles))

# Download and paste tiles into final image
for tile in tiles:
    url = f"https://api.mapbox.com/v4/{tileset_id}/{zoom}/{x}/{y}{@2x}.{format}"
    response = requests.get(url)
    tile_img = Image.open(BytesIO(response.content))

    i = tile.x - min(t.x for t in tiles)
    j = tile.y - min(t.y for t in tiles)
    stitched_image.paste(tile_img, (i * tile_width, j * tile_height))

# Convert stitched image to array
image_array = np.array(stitched_image)

# Calculate bounds of full stitched image
bounds_ul = mercantile.ul(min(t.x for t in tiles), min(t.y for t in tiles), TILE_ZOOM)
bounds_lr = mercantile.ul(max(t.x for t in tiles) + 1, max(t.y for t in tiles) + 1, TILE_ZOOM)

transform = from_bounds(bounds_ul.lng, bounds_lr.lat, bounds_lr.lng, bounds_ul.lat,
                        stitched_image.size[0], stitched_image.size[1])

# Save to GeoTIFF
output_path = "mapbox_satellite_image.tif"
with rasterio.open(
    output_path, 'w',
    driver='GTiff',
    height=stitched_image.size[1],
    width=stitched_image.size[0],
    count=3,
    dtype=image_array.dtype,
    crs=CRS.from_epsg(4326),
    transform=transform
) as dst:
    for i in range(3):  # RGB bands
        dst.write(image_array[:, :, i], i + 1)

print(f"Saved georeferenced image to {output_path}")


# |%%--%%| <wBqzrHeO2J|fFy3vaRWNd>

import mercantile
import geopandas as gpd
import requests
from PIL import Image
from io import BytesIO
from IPython.display import display

# Load your shapefile
gdf = gpd.read_file(LC_EXTENT_LAYER_PATH).to_crs("EPSG:4326")
bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]

# Get tile coordinates at zoom 15
tiles = list(mercantile.tiles(bounds[0], bounds[1], bounds[2], bounds[3], zooms=17))

# Example: Download the first tile
x, y, z = tiles[0].x, tiles[0].y, tiles[0].z
u = 2
url = f"https://api.mapbox.com/v4/mapbox.satellite/{z}/{x}/{y}@{u}x.jpg90?access_token=pk.eyJ1IjoiYmhhdHQtcnVzaGktdXQiLCJhIjoiY21iNW0yNmxiMWdmcDJrcHhoejVyYWk2aiJ9.rrfRc0yg3o2HdKhwjR52cw"

# Download the image
response = requests.get(url)
if response.status_code == 200:
    # Load image from response
    img = Image.open(BytesIO(response.content))

    # Save to file
    img_path = f"tile_{z}_{x}_{y}_{u}.jpg"
    img.save(img_path)
    print(f"Image saved to {img_path}")

    # Display in Jupyter notebook
    display(img)
else:
    print(f"Failed to download image: {response.status_code}")

import rasterio
from rasterio.transform import from_bounds
import numpy as np

# Reuse the existing variables: x, y, z
tile_bounds = mercantile.bounds(x, y, z)  # gives (west, south, east, north)

# Convert image to numpy array
img_array = np.array(img)

# If RGB image, rasterio wants channels first: (bands, rows, cols)
if img_array.ndim == 3:
    img_array = np.transpose(img_array, (2, 0, 1))

# Save to GeoTIFF
tif_path = f"tile_{z}_{x}_{y}_{u}.tif"
transform = from_bounds(*tile_bounds, width=img.width, height=img.height)

with rasterio.open(
    tif_path,
    "w",
    driver="GTiff",
    height=img.height,
    width=img.width,
    count=img_array.shape[0],  # 3 for RGB
    dtype=img_array.dtype,
    crs="EPSG:4326",
    transform=transform,
) as dst:
    dst.write(img_array)

print(f"GeoTIFF saved to {tif_path}")


# |%%--%%| <fFy3vaRWNd|S2o3fOSPHA>

import os
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pystac_client import Client
from planetary_computer import sign_inplace

# --- Catalog Setup (High Resolution Imagery) ---
catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=sign_inplace,
)

# Define time range and spatial bounds
time_range = "2018-07-01/2018-12-31"
print(time_range)

bbox = EXTENT_GDF.buffer(100).to_crs(4326).total_bounds

# Search NAIP collection for 1m resolution imagery
search = catalog.search(
    collections=["naip"],
    bbox=bbox,
    datetime=time_range
)

items = search.get_all_items()
if not items:
    print("No results found.")
    exit()

best_item = items[0]  # NAIP imagery usually contains only a few overlapping items

print(f"Available bands: {[b['name'] for b in best_item.assets.values() if 'eo:bands' in b.to_dict() for b in b.to_dict()['eo:bands']]}")

for asset_name, asset in best_item.assets.items():
    gsd = asset.extra_fields.get("gsd", "unknown")
    print(f"{asset_name}: {gsd} m resolution")

# --- Directory setup ---
os.makedirs("masked_bands", exist_ok=True)
os.makedirs("original_bands", exist_ok=True)

# --- Masking setup ---
dst_crs = "EPSG:32161"

for asset_name, asset in best_item.assets.items():
    if asset.media_type != "image/tiff; application=geotiff; profile=cloud-optimized":
        print(f"Skipping {asset_name} (not a GeoTIFF)")
        continue

    original_path = f"original_bands/{asset_name}_reprojected.tif"
    if os.path.exists(original_path):
        print(f"Using cached file: {original_path}")
        with rasterio.open(original_path) as src:
            reprojected = src.read()
            transform = src.transform
            dst_crs = src.crs
            height, width = src.height, src.width
            dst_meta = src.meta
    else:
        with rasterio.open(asset.href) as src:
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds
            )

            dst_meta = src.meta.copy()
            dst_meta.update({
                "crs": dst_crs,
                "transform": transform,
                "width": width,
                "height": height
            })

            reprojected = np.empty((src.count, height, width), dtype=src.meta["dtype"])
            for i in range(src.count):
                reproject(
                    source=rasterio.band(src, i + 1),
                    destination=reprojected[i],
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest
                )

        with rasterio.open(original_path, "w", **dst_meta) as dest:
            dest.write(reprojected)
        print(f"Saved unmasked: {original_path}")


# |%%--%%| <S2o3fOSPHA|tFyLtGetoC>

from sentinelsat import SentinelAPI

user = 'bhattrushi@utexas.edu' 
password = 'Mauve-Spindle-Squiggle-Statute-Immorally9' # Don't worry, this password doesn't work, anymore :)
#api = SentinelAPI(user, password, 'https://scihub.copernicus.eu/dhus')

shapefile_path = LC_EXTENT_LAYER_PATH

# Define date range
date_start = '20180101'
date_end = '20181231'

# Read shapefile and get geometry in WKT format
gdf = gpd.read_file(shapefile_path)
footprint = gdf.unary_union.to_crs("EPSG:4326").wkt  # get combined geometry WKT
print(footprint)

# Query products: Sentinel-2 Level-1C, cloud cover < 30%, date range, footprint polygon
products = api.query(
    footprint,
    date=(date_start, date_end),
    platformname='Sentinel-2',
    processinglevel='Level-1C',
    cloudcoverpercentage=(0, 30)
)

print(f"Found {len(products)} products.")

# Download all products found to a folder named 'downloads'
api.download_all(products, directory_path='./downloads')

# |%%--%%| <tFyLtGetoC|EDHs5i1qkO>

# Collect pixel values by lithology type and band
pixel_data_by_type = {"Chert": [], "sand": [], "Serpentinite": [], "Alluvium": [], "Limestone": []}

# Loop through bands
for asset_name in best_item.assets:
    masked_path = f"masked_bands/{asset_name}_masked.tif"
    if not os.path.exists(masked_path):
        continue

    with rasterio.open(masked_path) as src:
        if asset_name == "visual_masked":
            for i in range(1, src.count + 1):
                band_data = src.read(i).astype(float)
                band_label = f"{asset_name}_B{i}"
                for lith_type in pixel_data_by_type:
                    subset = ROAD_GDF_EDGES[ROAD_GDF_EDGES['Lithology'] == lith_type]
                    if subset.empty:
                        continue

                    shapes = ((geom, 1) for geom in subset.geometry)
                    road_mask = rasterize(
                        shapes=shapes,
                        out_shape=(src.height, src.width),
                        transform=src.transform,
                        fill=0,
                        dtype="float64"
                    )
                    masked_pixels = band_data[road_mask == 1]
                    masked_pixels = masked_pixels[masked_pixels > 0]  # Remove zeros
                    pixel_data_by_type[lith_type].append((band_label, masked_pixels))
        else:
            band_data = src.read(1).astype(float)
            for lith_type in pixel_data_by_type:
                subset = ROAD_GDF_EDGES[ROAD_GDF_EDGES['Lithology'] == lith_type]
                if subset.empty:
                    continue

                shapes = ((geom, 1) for geom in subset.geometry)
                road_mask = rasterize(
                    shapes=shapes,
                    out_shape=(src.height, src.width),
                    transform=src.transform,
                    fill=0,
                    dtype="float64"
                )
                masked_pixels = band_data[road_mask == 1]
                masked_pixels = masked_pixels[masked_pixels > 0]  # Remove zeros
                pixel_data_by_type[lith_type].append((asset_name, masked_pixels))

# Convert to DataFrame for plotting
hist_data = []
for lith_type, bands in pixel_data_by_type.items():
    for band_name, pixels in bands:
        hist_data.append(pd.DataFrame({
            "Band": band_name,
            "Reflectance": pixels,
            "Lith Type": lith_type
        }))

hist_df = pd.concat(hist_data, ignore_index=True)

# Plot
sns.set(style="whitegrid")
g = sns.displot(
    data=hist_df,
    x="Reflectance",
    hue="Lith Type",
    col="Band",
    kind="kde",
    fill=True,
    common_norm=False,
    facet_kws={'sharex': False, 'sharey': False},
    alpha=0.4,
    height=4,
    aspect=1.2
)
g.set_titles(col_template="Band: {col_name}")
plt.tight_layout()
# Save figure
g.savefig("sentinel2_band_histograms.png")
g.savefig("sentinel2_band_histograms.pdf")
plt.show()

# |%%--%%| <EDHs5i1qkO|VssrO92lXE>

# Step 1: Aggregate pixel data per band for all lithologies into a feature matrix and labels
feature_list = []
label_list = []

for lith_type in pixel_data_by_type:
    valid_bands = [item for item in pixel_data_by_type[lith_type] if len(item[1]) > 0]
    if not valid_bands:
        continue

    min_len = min(len(pixels) for _, pixels in valid_bands)

    features_per_band = []
    for band_name, pixels in valid_bands:
        if len(pixels) >= min_len:
            features_per_band.append(pixels[:min_len])
        else:
            features_per_band.append(np.pad(pixels, (0, min_len - len(pixels)), mode='constant'))

    features = np.stack(features_per_band, axis=1)
    labels = np.array([lith_type] * min_len)

    feature_list.append(features)
    label_list.append(labels)

X = np.vstack(feature_list)
y = np.concatenate(label_list)

print(f"Feature matrix shape: {X.shape}")
print(f"Labels shape: {y.shape}")

# Step 2: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Step 3: Hyperparameter tuning using cross-validation
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['sqrt', 'log2', None],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

cv_rf = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1), param_grid, cv=5, n_jobs=-1)
cv_rf.fit(X_train, y_train)

print("Best parameters found:", cv_rf.best_params_)

# Step 4: Evaluate the optimized model
best_rf = cv_rf.best_estimator_
y_pred = best_rf.predict(X_test)

print("Classification Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the trained random forest model
model_path = "random_forest_lithology_model_optimized.joblib"
joblib.dump(best_rf, model_path)
print(f"Model saved to {model_path}")

# |%%--%%| <VssrO92lXE|IBeWqICufx>

# Export the first tree from the random forest
estimator = rf.estimators_[0]

# Export to DOT format
dot_data = export_graphviz(
    estimator,
    out_file=None, 
    feature_names=[band_name for band_name, _ in pixel_data_by_type[next(iter(pixel_data_by_type))]],  # list of band names
    class_names=rf.classes_,
    filled=True,
    rounded=True,
    special_characters=True
)

# Create graph from dot data
graph = graphviz.Source(dot_data)

# Save to PDF
graph.render("random_forest_tree_1", format="pdf", cleanup=True)
print("Random forest tree exported to random_forest_tree_1.pdf")

# |%%--%%| <IBeWqICufx|iwUJvTuxSp>

# Load model
model_path = "random_forest_lithology_model_optimized.joblib"
rf = joblib.load(model_path)

# Load new road layer
new_road_path = os.path.join(CACHE_DIR, 'road_edges.shp')
road_gdf = gpd.read_file(new_road_path)
road_gdf = road_gdf.to_crs(dst_crs)

# Load and resample all bands to match the highest resolution
ref_raster_path = f"original_bands/B04_reprojected.tif"
with rasterio.open(ref_raster_path) as ref_src:
    ref_transform = ref_src.transform
    ref_shape = (ref_src.height, ref_src.width)
    ref_crs = ref_src.crs

band_arrays = []
band_names = []

for asset_name in best_item.assets:
    raster_path = f"original_bands/{asset_name}_reprojected.tif"
    if not os.path.exists(raster_path):
        continue

    with rasterio.open(raster_path) as src:
        if asset_name == "visual_masked":
            for i in range(1, src.count + 1):
                band_data = src.read(i, out_shape=ref_shape, resampling=Resampling.bilinear).astype(float)
                band_arrays.append(band_data)
                band_names.append(f"{asset_name}_B{i}")
        else:
            band_data = src.read(1, out_shape=ref_shape, resampling=Resampling.bilinear).astype(float)
            band_arrays.append(band_data)
            band_names.append(asset_name)

if not band_arrays:
    raise RuntimeError("No bands were loaded after resampling.")

bands_stack = np.stack(band_arrays)

# Function to process each road feature
def process_road(idx, road):
    shape = [mapping(road.geometry)]
    mask = rasterize(
        [(shape[0], 1)],
        out_shape=ref_shape,
        transform=ref_transform,
        fill=0,
        dtype="uint8"
    )

    row_idx, col_idx = np.where(mask == 1)
    if len(row_idx) == 0:
        return idx, "Unknown"

    pixel_values = bands_stack[:, row_idx, col_idx].T
    valid_pixels = pixel_values[~np.all(pixel_values == 0, axis=1)]
    if valid_pixels.shape[0] == 0:
        return idx, "Unknown"

    pred_labels = rf.predict(valid_pixels)
    majority_label = pd.Series(pred_labels).mode().iloc[0]
    return idx, majority_label

# Parallel processing of roads
print("Processing roads in parallel...")
results = Parallel(n_jobs=11)(
    delayed(process_road)(idx, road) for idx, road in tqdm(road_gdf.iterrows(), total=road_gdf.shape[0])
)

# Update GeoDataFrame with results
for idx, label in results:
    road_gdf.at[idx, 'Predicted_Lithology'] = label

output_path = os.path.join(CACHE_DIR, 'road_edges_with_lithology.shp')
road_gdf.to_file(output_path)
print(f"Saved updated road layer with predicted lithology to {output_path}")

# |%%--%%| <iwUJvTuxSp|RWRsKXqrMr>

road_lines.head()

# |%%--%%| <RWRsKXqrMr|zsvtb01Ix1>
r"""°°°
# Flow Metrics
°°°"""
# |%%--%%| <zsvtb01Ix1|hGS2UGAj38>

try:
    # Generate weights

    ## Generate road weights
    if Path(os.path.join(CACHE_DIR, 'road_mask.tif')).is_file():
        logger.info(f"CACHE ACTIVATED FOR road_mask.tif, skipping road masking process!")
        ROAD_MASK = rasterio.open(os.path.join(CACHE_DIR,'road_mask.tif'))
    else:
        meta = DEM_RASTER.meta.copy()
        transform = DEM_RASTER.transform
        out_shape = (DEM_RASTER.height, DEM_RASTER.width)
        crs = DEM_RASTER.crs

        shapes = ((geom, 1) for geom in ROAD_GDF_EDGES.geometry)

        ROAD_MASK = rasterize(
            shapes=shapes,
            out_shape=out_shape,
            transform=transform,
            fill=0,
            masked=True,
            dtype='float64'
        )

        with rasterio.open(os.path.join(CACHE_DIR, "road_mask.tif"), 'w', **meta) as dst:
            dst.write(ROAD_MASK, 1)

    fig, ax = plt.subplots(figsize=(8, 8))
    show(ROAD_MASK, ax=ax, title='Rasterized Road Plot', cmap='gray')
    plt.show()

except Exception as e:
    logger.fatal(f'Could not generate or find road weights. {e}')
    sys.exit(1)

# |%%--%%| <hGS2UGAj38|k8H2yMYDyA>

try: 
    dem = rd.LoadGDAL(os.path.join(DEM_DIR, "merged_dem.tif" ))
    road_weights = rd.LoadGDAL(os.path.join(CACHE_DIR,"road_mask.tif")).__array__().astype('float64')
 
    rd.FillDepressions(dem, epsilon=True, in_place=True)
    #rd.BreachDepressions(dem, epsilon=True, in_place=True)

    def run_rich_flow_masked(method, exponent=None, weights=road_weights):
        if exponent is None:
            m = rd.FlowAccumulation(dem, method=method, weights=weights)
        else:
            m = rd.FlowAccumulation(dem, method=method, exponent=exponent, weights=weights)

        rd.rdShow(m, axes=False, cmap='jet', figsize=(20,15))
        rd.SaveGDAL(os.path.join(CACHE_DIR, f"test_filled_{method}.tif"), m)

    run_rich_flow_masked('D8')
    run_rich_flow_masked('Rho8')
    run_rich_flow_masked('Quinn')
    run_rich_flow_masked('Freeman', exponent=1.1)
    run_rich_flow_masked('Holmgren', exponent=5)
    run_rich_flow_masked('Dinf')

except Exception as e:
    logger.fatal(f'Failed to run flow metrics: {e}')
    sys.exit(1)

# |%%--%%| <k8H2yMYDyA|axXQoN3swm>

import numpy as np

# [elevation, length (m)]
line_segments = np.array([
    [105.2, 1.8],
    [104.7, 2.0],
    [104.1, 1.5],
    [103.8, 1.7],
    [103.0, 2.1]
])

DRAINAGE_AMOUNT = 100.0  # Total initial drainage volume in m^3
VOLUME_TO_BREAKTHROUGH = 2.0  # Required m^3 per meter to proceed to the next segment

# Initialize array to hold water amount after each segment
water_remaining = np.zeros(len(line_segments))

# Find the starting index (segment with the highest elevation)
start_idx = np.argmax(line_segments[:, 0])

# Traverse from the highest elevation to the end of the line
current_water = DRAINAGE_AMOUNT
for i in range(start_idx, len(line_segments)):
    elevation, length = line_segments[i]
    water_remaining[i] = current_water
    volume_required = VOLUME_TO_BREAKTHROUGH * length
    
    if current_water >= volume_required:
        current_water -= volume_required
    else:
        current_water = 0

# Outputs
print("Line segments (original order):")
print(line_segments)
print("\nWater remaining at each segment (original order):")
print(water_remaining)  # m^3 remaining after reaching each segment

# |%%--%%| <axXQoN3swm|t9B1EE747k>

import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point

# Mock data for Road and Flowpath
road_lines = [
    LineString([(2, 0), (2, 5)]),
    LineString([(4, 0), (4, 5)]),
]

flowpath_lines = [
    LineString([(0, 1), (3, 1), (5, 1)]),
    LineString([(0, 3), (3, 3), (5, 3)])
]

roads = gpd.GeoDataFrame(geometry=road_lines, crs="EPSG:4326")
flowpaths = gpd.GeoDataFrame(geometry=flowpath_lines, crs="EPSG:4326")

# Check for intersections and trim flowpaths at roads
terminated_points = []
trimmed_flowpaths = []

for line in flowpaths.geometry:
    intersections = []
    for road in roads.geometry:
        if line.intersects(road):
            inter = line.intersection(road)
            if inter.geom_type == 'Point':
                intersections.append(inter)
            elif inter.geom_type == 'MultiPoint':
                intersections.extend([pt for pt in inter.geoms])
    if intersections:
        # Find the first intersection along the line (smallest distance along the line)
        intersections.sort(key=lambda pt: line.project(pt))
        nearest = intersections[0]
        # Cut the line at the intersection
        distance = line.project(nearest)
        split_line = line.interpolate(distance, normalized=False)
        coords = list(line.coords)
        cut_coords = [pt for pt in coords if line.project(Point(pt)) <= distance]
        cut_coords.append((nearest.x, nearest.y))
        trimmed_flowpaths.append(LineString(cut_coords))
        terminated_points.append(nearest)
    else:
        trimmed_flowpaths.append(line)

# Create output dataframes
trimmed_gdf = gpd.GeoDataFrame(geometry=trimmed_flowpaths, crs="EPSG:4326")
termination_points_gdf = gpd.GeoDataFrame(geometry=terminated_points, crs="EPSG:4326")

# Step 4: Visualization
fig, ax = plt.subplots(figsize=(8, 6))
roads.plot(ax=ax, color='black', label='Roads')
flowpaths.plot(ax=ax, color='blue', linestyle='--', label='Original Flowpaths')
trimmed_gdf.plot(ax=ax, color='green', label='Trimmed Flowpaths')
termination_points_gdf.plot(ax=ax, color='red', marker='o', label='Termination Points')

plt.legend()
plt.title('Flowpath Termination at Road Intersections')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)
plt.show()


# |%%--%%| <t9B1EE747k|XGPNOc1jNg>

import geopandas as gpd
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point
import numpy as np

# Mock data creation
def create_mock_data():
    # Create mock road segments
    road_segments = [
        LineString([(0, 0), (1, 1)]),  # Minima point
        LineString([(1, 1), (2, 2)]),
        LineString([(2, 2), (3, 3)]),
        LineString([(1, 1), (1, 2)]),  # Fork uphill
    ]

    # Elevations for each segment's end point (assume elevation increases)
    elevations = [0, 10, 20, 15]  # Elevation at the end of the segment
    
    road_gdf = gpd.GeoDataFrame({
        'segment_id': range(len(road_segments)),
        'elevation': elevations,
        'geometry': road_segments
    })

    # Minima point (start of flow)
    minima_points = gpd.GeoDataFrame({
        'geometry': [Point(0, 0)]
    }, geometry='geometry')

    # Drainage point intersecting segment 1 and 3
    drainage_points = gpd.GeoDataFrame({
        'value': [5, 10],
        'geometry': [Point(2, 2), Point(1, 2)]
    }, geometry='geometry')

    return road_gdf, minima_points, drainage_points

# Flow accumulation algorithm
def calculate_flow(road_gdf, minima_points, drainage_points, rainfall_per_meter=1):
    road_gdf = road_gdf.copy()
    road_gdf['flow'] = 0

    # Create directed graph from lines
    G = nx.DiGraph()
    for idx, row in road_gdf.iterrows():
        coords = list(row.geometry.coords)
        G.add_edge(coords[0], coords[1], segment_id=row.segment_id)

    # Traverse from minima point and accumulate water
    for minima in minima_points.geometry:
        stack = [(minima.coords[0], 0)]
        visited = set()

        while stack:
            current_point, accumulated = stack.pop()
            for u, v, data in G.out_edges(current_point, data=True):
                if (u, v) in visited:
                    continue
                visited.add((u, v))

                segment_id = data['segment_id']
                line = road_gdf.loc[segment_id].geometry
                length = line.length

                # Water from rainfall + upstream accumulation
                water = accumulated + (rainfall_per_meter * length)

                # Check for drainage points
                intersecting = drainage_points[drainage_points.intersects(line)]
                if not intersecting.empty:
                    water += intersecting['value'].sum()

                road_gdf.at[segment_id, 'flow'] += water
                stack.append((v, water))

    return road_gdf


def plot_network(road_gdf, minima_points, drainage_points):
    fig, ax = plt.subplots(figsize=(8, 6))
    road_gdf.plot(ax=ax, column='flow', cmap='Blues', linewidth=3, legend=True)
    minima_points.plot(ax=ax, color='red', markersize=50, label='Minima')
    drainage_points.plot(ax=ax, color='green', markersize=50, label='Drainage')
    plt.legend()
    plt.title("Uphill Flow Accumulation")
    plt.show()


road_gdf, minima_points, drainage_points = create_mock_data()
road_gdf = calculate_flow(road_gdf, minima_points, drainage_points)
plot_network(road_gdf, minima_points, drainage_points)

# Display resulting flow table
print(road_gdf[['segment_id', 'flow']])

