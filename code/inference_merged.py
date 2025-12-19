import torch
import cv2
import numpy as np
import os
import glob
from tqdm import tqdm
import segmentation_models_pytorch as smp

# --- CONFIGURATION ---
# User can change this to their test folder
INPUT_DIR = os.path.join("server 2", "processed_data_multiclass", "images") # Defaulting to validation set for demo
OUTPUT_DIR = "final_predictions_moe"     
MAIN_MODEL_PATH = os.path.join("server 2", "best_model_multiclass.pth")
SPECIALIST_MODEL_PATH = "infra_specialist.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Classes: Background=0, Building=1, Road=2, Water=3, Infra=4, RCC=5, Tiled=6
INFRA_CLASS_ID = 4

def load_models():
    print("🔄 Loading Generalist Model (DeepLabV3+)...")
    # Re-instantiate your Main Model Architecture
    # ensuring correct class init
    generalist = smp.DeepLabV3Plus(
        encoder_name="resnet50", 
        encoder_weights=None, 
        in_channels=3, 
        classes=7, 
        activation=None
    ).to(DEVICE)
    
    checkpoint = torch.load(MAIN_MODEL_PATH, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        generalist.load_state_dict(checkpoint['model_state_dict'])
    else:
        generalist.load_state_dict(checkpoint)
    generalist.eval()

    print("🔄 Loading Specialist Model (U-Net ResNet18)...")
    specialist = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None, 
        in_channels=3, 
        classes=1,
        activation=None
    ).to(DEVICE)
    specialist.load_state_dict(torch.load(SPECIALIST_MODEL_PATH, map_location=DEVICE))
    specialist.eval()
    
    return generalist, specialist

def run_inference():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    generalist, specialist = load_models()
    
    # Process only a sample if directory is huge
    image_paths = glob.glob(os.path.join(INPUT_DIR, "*.tif"))[:50] # Taking 50 samples for demo
    if not image_paths:
        # Fallback to jpg/png if tif not found (e.g. if pointing to raw data)
        image_paths = glob.glob(os.path.join(INPUT_DIR, "*.jpg")) + glob.glob(os.path.join(INPUT_DIR, "*.png"))
    
    print(f"🚀 Processing {len(image_paths)} images from {INPUT_DIR}...")
    
    with torch.no_grad():
        for img_path in tqdm(image_paths):
            # 1. Preprocess
            filename = os.path.basename(img_path)
            original_img = cv2.imread(img_path)
            if original_img is None: continue
            
            h, w = original_img.shape[:2]
            
            # Preprocess using Albumentations
            import albumentations as A
            from albumentations.pytorch import ToTensorV2
            
            transform = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
            
            img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            aug = transform(image=img_rgb)
            img_tensor = aug['image'].unsqueeze(0).to(DEVICE)
            
            # 2. Generalist Prediction (The "Base")
            gen_logits = generalist(img_tensor)
            # Resize
            gen_logits = torch.nn.functional.interpolate(gen_logits, size=(h, w), mode='bilinear', align_corners=False)
            gen_preds = torch.argmax(gen_logits, dim=1).squeeze().cpu().numpy()
            
            # 3. Specialist Prediction (The "Overrider")
            spec_logits = specialist(img_tensor)
            spec_logits = torch.nn.functional.interpolate(spec_logits, size=(h, w), mode='bilinear', align_corners=False)
            spec_probs = torch.sigmoid(spec_logits).squeeze().cpu().numpy()
            
            # 4. The Merge Strategy
            # Start with Generalist mask
            final_mask = gen_preds.copy()
            
            # Identify high-confidence Infra pixels from Specialist
            # Threshold: 0.6 (Conservative)
            infra_pixels = spec_probs > 0.6
            
            # OVERWRITE: Where Specialist is confident, force Class 4 (Infra)
            final_mask[infra_pixels] = INFRA_CLASS_ID
            
            # 5. Save Result (Colorized)
            color_map = {
                0: [0, 0, 0],       # Background (Black)
                1: [0, 0, 255],     # Building (Red)
                2: [128, 128, 128], # Road (Gray)
                3: [255, 0, 0],     # Water (Blue)
                4: [0, 255, 255],   # Infra (Yellow) <--- The Star!
                5: [255, 255, 255], # RCC (White)
                6: [0, 255, 0]      # Tiled (Green)
            }
            
            vis_img = np.zeros((h, w, 3), dtype=np.uint8)
            for cls_id, color in color_map.items():
                vis_img[final_mask == cls_id] = color
            
            # Save visual
            save_path = os.path.join(OUTPUT_DIR, f"pred_{filename.replace('.tif', '.png')}")
            cv2.imwrite(save_path, vis_img)

    print(f"✅ All Done! Merged predictions saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    run_inference()
