
import os
import zipfile
import glob
import rasterio
import geopandas as gpd
from rasterio.features import rasterize
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2
import traceback
import csv
import sys
import multiprocessing
from functools import partial

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Assume script is in root of dataset
OUTPUT_DIR = os.path.join(BASE_DIR, "server", "processed_data_multiclass")
CHIP_SIZE = 512
STRIDE = 512
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 2) # Leave 2 cores for OS

# Class Mapping
CLASS_MAP = {
    "background": 0,
    "building": 1,
    "road": 2,
    "water": 3,
    "utility": 4,
    "roof_rcc": 5,
    "roof_tiled": 6
}
# Priority: Higher number overwrites lower
# We draw in order: Road, Water, Utility, Building -> Resulting in correct overlaps? 
# No, usually small things on top. 
# Draw Order: Water(3), Road(2), Utility(4), Building(1/5/6)
# Actually, let's define a draw function that handles layers.

def log_error(village, error_msg):
    with open("processing_errors.log", "a") as f:
        f.write(f"{village}: {error_msg}\n")

def process_village_wrapper(zip_path):
    """Wrapper to catch errors for multiprocessing"""
    try:
        return process_village(zip_path)
    except Exception as e:
        err = traceback.format_exc()
        village = os.path.basename(zip_path)
        log_error(village, str(e))
        return 0

def process_village(zip_path):
    zip_name = os.path.basename(zip_path)
    village_name = os.path.splitext(zip_name)[0]
    unzip_dir = os.path.join(BASE_DIR, "temp_extract", village_name)
    
    # 1. Extract
    if not os.path.exists(unzip_dir):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(unzip_dir)
            
    # 2. Find Ortho
    tifs = list(Path(unzip_dir).rglob("*ORTHO*.tif"))
    if not tifs: tifs = list(Path(unzip_dir).rglob("*.tif"))
    # Filter small
    tifs = [t for t in tifs if os.path.getsize(t) > 50_000_000]
    
    if not tifs:
        return 0 # Skip
        
    ortho_path = tifs[0]
    
    # 3. Find Shapefiles & Merge
    # We need specific shapefiles for specific classes
    # Shapefile Search Strategy: Recursive glob for keywords
    
    layer_map = {
        "building": [],
        "road": [],
        "water": [],
        "utility": []
    }
    
    all_shps = list(Path(unzip_dir).rglob("*.shp"))
    # Also check the global 'PB_training_dataSet_shp_file' if not found locally, 
    # but for "Foolproof" speed, let's assume valid training data has local shps or we skip.
    # (Refining this: You can add the global search path logic here if needed)
    
    for shp in all_shps:
        name = shp.name.lower()
        if "built" in name or "abadi" in name:
            layer_map["building"].append(shp)
        elif "road" in name or "track" in name or "path" in name:
            layer_map["road"].append(shp)
        elif "water" in name or "talab" in name or "pond" in name:
            layer_map["water"].append(shp)
        elif "utility" in name or "infra" in name:
            layer_map["utility"].append(shp)

    # 4. Open Ortho
    with rasterio.open(ortho_path) as src:
        h, w = src.shape
        meta = src.meta.copy()
        
        # Create Master Mask (0 initialized)
        master_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Helper to rasterize


        def burn_layer(shp_list, value_or_col, mode="value"):
            for shp in shp_list:
                try:
                    gdf = gpd.read_file(shp)
                    if gdf.empty: continue
                    if gdf.crs != src.crs:
                        gdf = gdf.to_crs(src.crs)
                        
                    # logic for robust attribute handling
                    final_mode = mode
                    if mode == "attribute" and value_or_col not in gdf.columns:
                        # Fallback if column missing
                        final_mode = "value"
                        effective_val = 1 # Default to Building
                        # print(f"   ⚠️ 'Roof_type' missing in {os.path.basename(shp)}. Defaulting to Building (1).")
                    else:
                        effective_val = value_or_col

                    if final_mode == "value":
                        shapes = ((geom, effective_val) for geom in gdf.geometry)
                        rasterize(shapes, out=master_mask, transform=src.transform, default_value=effective_val)
                        
                    elif final_mode == "attribute":
                        # Handle Roof Type
                        def val_gen():
                            for _, row in gdf.iterrows():
                                r_type = row[value_or_col]
                                # Map 1 -> 5 (RCC), 2 -> 6 (Tiled). Fallback -> 1(Building)
                                val = 5 if r_type in [1, 'RCC', 'R.C.C'] else 6 if r_type in [2, 'Tiled', 'Kavelu'] else 1
                                yield (row.geometry, val)
                        
                        rasterize(val_gen(), out=master_mask, transform=src.transform)
                        
                except Exception as e:
                    print(f"Error rasterizing {shp}: {e}")

        # BURN ORDER: Overwrites previous. Bottom up.
        # 1. Roads (2)
        burn_layer(layer_map["road"], 2)
        # 2. Water (3)
        burn_layer(layer_map["water"], 3)
        # 3. Utility (4)
        burn_layer(layer_map["utility"], 4)
        # 4. Buildings (Typed or Generic)
        # Check if attribute exists
        burn_layer(layer_map["building"], "Roof_type", mode="attribute")
        
        # 5. Chip Generation
        chips_created = 0
        img_dir = os.path.join(OUTPUT_DIR, "images")
        mask_dir = os.path.join(OUTPUT_DIR, "masks")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        
        for row in range(0, h - CHIP_SIZE, STRIDE):
            for col in range(0, w - CHIP_SIZE, STRIDE):
                window = rasterio.windows.Window(col, row, CHIP_SIZE, CHIP_SIZE)
                
                # Check data validity
                mask_chip = master_mask[row:row+CHIP_SIZE, col:col+CHIP_SIZE]
                if np.max(mask_chip) == 0 and np.random.rand() > 0.1: 
                    # Drop 90% of empty chips to save space/time
                    continue
                    
                img_chip = src.read(window=window)
                if np.mean(img_chip) == 0: continue # Empty image
                
                # Save
                chip_id = f"{village_name}_{row}_{col}"
                
                # Update Meta
                meta.update({"driver": "GTiff", "height": CHIP_SIZE, "width": CHIP_SIZE, 
                             "transform": src.window_transform(window)})
                
                with rasterio.open(os.path.join(img_dir, f"{chip_id}.tif"), "w", **meta) as dst:
                    dst.write(img_chip)
                    
                # Save Mask (Uint8)
                meta_mask = meta.copy()
                meta_mask.update({"count": 1, "dtype": "uint8"})
                with rasterio.open(os.path.join(mask_dir, f"{chip_id}.tif"), "w", **meta_mask) as dst:
                    dst.write(mask_chip, 1)
                    
                chips_created += 1
                
    # Cleanup temp
    # shutil.rmtree(unzip_dir) # Optional: keep for debug
    return chips_created

def main():
    print(f"🚀 Starting Foolproof Processing on {NUM_WORKERS} cores")
    print(f"📂 Scanning {BASE_DIR} for zips...")
    
    zips = glob.glob(os.path.join(BASE_DIR, "*.zip"))
    
    # Filter already processed?
    # TODO: Log check
    
    with multiprocessing.Pool(NUM_WORKERS) as pool:
        results = list(tqdm(pool.imap_unordered(process_village_wrapper, zips), total=len(zips)))
        
    print(f"✅ Total Chips Generated: {sum(results)}")

if __name__ == "__main__":
    main()
