import torch
import cv2
import numpy as np
import os
import glob
from tqdm import tqdm
import segmentation_models_pytorch as smp

# --- CONFIGURATION ---
INPUT_DIR = os.path.join("server 2", "processed_data_multiclass", "images") 
OUTPUT_DIR = "final_predictions_highlights"     
MAIN_MODEL_PATH = os.path.join("server 2", "best_model_multiclass.pth")
SPECIALIST_MODEL_PATH = "infra_specialist.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INFRA_CLASS_ID = 4

def load_models():
    print("🔄 Loading Generalist Model...")
    generalist = smp.DeepLabV3Plus(
        encoder_name="resnet50", 
        encoder_weights=None, 
        in_channels=3, 
        classes=7, 
        activation=None
    ).to(DEVICE)
    
    ckpt = torch.load(MAIN_MODEL_PATH, map_location=DEVICE)
    if 'model_state_dict' in ckpt: generalist.load_state_dict(ckpt['model_state_dict'])
    else: generalist.load_state_dict(ckpt)
    generalist.eval()

    print("🔄 Loading Specialist Model...")
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

def run_inference_filtered():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    generalist, specialist = load_models()
    
    # Process MORE images since we are filtering many out
    image_paths = glob.glob(os.path.join(INPUT_DIR, "*.tif"))[:200] 
    if not image_paths:
        image_paths = glob.glob(os.path.join(INPUT_DIR, "*.jpg")) + glob.glob(os.path.join(INPUT_DIR, "*.png"))
    
    print(f"🚀 Scanning {len(image_paths)} images from {INPUT_DIR}...")
    print(f"   (Only saving outputs with detected features...)")
    
    saved_count = 0
    
    with torch.no_grad():
        for img_path in tqdm(image_paths):
            filename = os.path.basename(img_path)
            original_img = cv2.imread(img_path)
            if original_img is None: continue
            
            h, w = original_img.shape[:2]
            
            # Preprocess (Albumentations)
            import albumentations as A
            from albumentations.pytorch import ToTensorV2
            transform = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
            img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            aug = transform(image=img_rgb)
            img_tensor = aug['image'].unsqueeze(0).to(DEVICE)
            
            # Inference
            gen_logits = generalist(img_tensor)
            gen_logits = torch.nn.functional.interpolate(gen_logits, size=(h, w), mode='bilinear', align_corners=False)
            gen_preds = torch.argmax(gen_logits, dim=1).squeeze().cpu().numpy()
            
            spec_logits = specialist(img_tensor)
            spec_logits = torch.nn.functional.interpolate(spec_logits, size=(h, w), mode='bilinear', align_corners=False)
            spec_probs = torch.sigmoid(spec_logits).squeeze().cpu().numpy()
            
            final_mask = gen_preds.copy()
            final_mask[spec_probs > 0.6] = INFRA_CLASS_ID
            
            # FILTER: Skipping if purely background (0)
            if np.all(final_mask == 0):
                continue
                
            saved_count += 1
            
            # Visualization
            color_map = {
                0: [0, 0, 0],       # Background
                1: [0, 0, 255],     # Building (Red)
                2: [128, 128, 128], # Road (Gray)
                3: [255, 0, 0],     # Water (Blue)
                4: [0, 255, 255],   # Infra (Yellow)
                5: [255, 255, 255], # RCC (White)
                6: [0, 255, 0]      # Tiled (Green)
            }
            vis_img = np.zeros((h, w, 3), dtype=np.uint8)
            for cls_id, color in color_map.items():
                vis_img[final_mask == cls_id] = color
            
            save_path = os.path.join(OUTPUT_DIR, f"highlight_{filename.replace('.tif', '.png')}")
            cv2.imwrite(save_path, vis_img)

    print(f"✅ Filtered Run Complete.")
    print(f"   Scanned: {len(image_paths)}")
    print(f"   Saved:   {saved_count} (Only images with features)")
    print(f"   Folder:  {OUTPUT_DIR}")

if __name__ == "__main__":
    run_inference_filtered()
