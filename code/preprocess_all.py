
import os
import pandas as pd
import zipfile
import glob
import rasterio
import geopandas as gpd
from rasterio.features import rasterize
from pathlib import Path
from tqdm import tqdm
import numpy as np
import shutil
import warnings

warnings.filterwarnings("ignore")

# Configuration
BASE_DIR = r"c:\Users\mahaj\Downloads\GEOAIDATASET"
EXCEL_PATH = os.path.join(BASE_DIR, "Heckathon Data27112025.xlsx")
OUTPUT_DIR = os.path.join(BASE_DIR, "processed_data", "train")
CHIP_SIZE = 512
STRIDE = 512

def find_matching_zip(village_name, all_zips):
    """Finds a zip file that contains the village name (case-insensitive)."""
    for zip_path in all_zips:
        if village_name.lower() in os.path.basename(zip_path).lower():
            return zip_path
    return None

def process_files(ortho_path, shp_dir, village_name):
    """Processes a single village's ortho and shapefiles."""
    print(f"   Using Ortho: {os.path.basename(ortho_path)}")
    
    # Find Shapefiles
    shps = list(Path(shp_dir).rglob("Builtup*.shp"))
    if not shps:
        # Fallback: sometimes shapefiles have different naming?
        shps = list(Path(shp_dir).rglob("*.shp"))
        # Filter out random non-building shapes if possible, or take all?
        # For now, let's look for "Abadi" or "Builtup" or "Lal_dora" if available.
        # But commonly in this dataset they seem to be "Builtup..."
    
    if not shps:
        print(f"   ⚠️ No Shapefiles found in {shp_dir}")
        return

    # Merge Shapefiles
    gdfs = []
    for shp in shps:
        try:
            gdf = gpd.read_file(shp)
            if not gdf.empty:
               gdfs.append(gdf)
        except:
            pass
            
    if not gdfs:
        print("   ⚠️ Valid shapefiles are empty.")
        return
        
    full_gdf = gpd.pd.concat(gdfs, ignore_index=True)
    
    # Open Ortho
    with rasterio.open(ortho_path) as src:
        height, width = src.shape
        transform = src.transform
        crs = src.crs
        
        # Reproject if needed
        if full_gdf.crs != crs:
            try:
                full_gdf = full_gdf.to_crs(crs)
            except:
                print("   ❌ CRS Error.")
                return

        # Rasterize
        shapes = ((geom, 1) for geom in full_gdf.geometry)
        mask = rasterize(shapes, out_shape=(height, width), transform=transform, fill=0, dtype='uint8')
        
        # Chip
        count = 0 
        for row in range(0, height - CHIP_SIZE, STRIDE):
            for col in range(0, width - CHIP_SIZE, STRIDE):
                window = rasterio.windows.Window(col, row, CHIP_SIZE, CHIP_SIZE)
                img_chip = src.read(window=window)
                mask_chip = mask[row:row+CHIP_SIZE, col:col+CHIP_SIZE]
                
                # Filter useful chips (must have some image content)
                if np.mean(img_chip) == 0: continue
                
                # Saving
                chip_id = f"{village_name}_{count:06d}"
                
                img_path = os.path.join(OUTPUT_DIR, "images", f"image_{chip_id}.tif")
                mask_path = os.path.join(OUTPUT_DIR, "masks", f"mask_{chip_id}.tif")
                
                chip_meta = src.meta.copy()
                chip_meta.update({"driver": "GTiff", "height": CHIP_SIZE, "width": CHIP_SIZE, "transform": src.window_transform(window)})
                
                with rasterio.open(img_path, "w", **chip_meta) as dst:
                    dst.write(img_chip)
                    
                mask_meta = chip_meta.copy()
                mask_meta.update({"count": 1})
                with rasterio.open(mask_path, "w", **mask_meta) as dst:
                    dst.write(mask_chip, 1)
                
                count += 1
        print(f"   ✅ Generated {count} chips.")

def main():
    print("🚀 Starting Mass Processing...")
    
    # 1. Read Index
    try:
        # Just grab the first sheet? Or iterate layers?
        # The user said "use heckathon as index".
        xls = pd.read_excel(EXCEL_PATH) # Reads first sheet by default
        # Assuming there's a column like "Village Name" or similar.
        # Let's try to interpret the columns dynamically or just grab the first column if it looks like names.
        print(f"Columns: {xls.columns.tolist()}")
        village_names = []
        if 'Village Name' in xls.columns:
            village_names = xls['Village Name'].dropna().unique().tolist()
        elif 'Village' in xls.columns:
            village_names = xls['Village'].dropna().unique().tolist()
        else:
            # Fallback: Get all zip files and assume they are valid
            print("⚠️ 'Village Name' column not found. Processing all Zips in directory instead.")
    except Exception as e:
        print(f"⚠️ Error reading Excel: {e}. Fallback to processing all Zips.")
        village_names = []

    # Get all Zips
    all_zips = glob.glob(os.path.join(BASE_DIR, "*.zip"))
    processed_count = 0
    
    for zip_path in tqdm(all_zips):
        zip_name = os.path.basename(zip_path)
        folder_name = os.path.splitext(zip_name)[0]
        unzip_dir = os.path.join(BASE_DIR, folder_name)
        
        print(f"\n📦 Processing {zip_name}...")
        
        # Unzip if needed
        if not os.path.exists(unzip_dir):
            print("   Unzipping...")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(unzip_dir)
            except:
                print("   ❌ Unzip failed.")
                continue
        
        # Locate Files
        # Look for TIF (Ortho)
        tifs = list(Path(unzip_dir).rglob("*ORTHO*.tif"))
        if not tifs:
             # Try case insensitive
             tifs = list(Path(unzip_dir).rglob("*.tif"))
             # Filter out small tiles if any
             tifs = [t for t in tifs if os.path.getsize(t) > 100_000_000] # >100MB likely the full ortho

        if not tifs:
            print("   ❌ No Orthophoto found.")
            continue
            
        ortho_path = tifs[0] # Take the first large TIF
        
        # Look for Shapefiles
        # Sometimes shapefiles are inside the zip, sometimes in a separate "Shapefiles" zip?
        # The user provided `PB_training_dataSet_shp_file.zip`. 
        # Check if shapefiles exist LOCALLY in this folder first
        shapes_found = list(Path(unzip_dir).rglob("*.shp"))
        
        if shapes_found:
             process_files(ortho_path, unzip_dir, folder_name)
             processed_count += 1
        else:
             # Check the global 'Shapefiles' directory from before
             # Need to match village name?
             pass 
             # For now, if no local shapefiles, skip (Theme 2 might use these later)
             print("   ⚠️ No Shapefiles found in extracted folder. Skipping for Theme 1.")

    print(f"🎉 Batch Processing Complete. Processed {processed_count} datasets.")

if __name__ == "__main__":
    main()
