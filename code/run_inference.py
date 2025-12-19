
import os
import torch
import cv2
import segmentation_models_pytorch as smp
import numpy as np
import matplotlib.pyplot as plt

# Configuration
TEST_IMG_PATH = r"c:\Users\mahaj\Downloads\GEOAIDATASET\processed_data\train\images"
MODEL_PATH = r"c:\Users\mahaj\Downloads\GEOAIDATASET\unet_building_model.pth"
OUTPUT_DIR = r"c:\Users\mahaj\Downloads\GEOAIDATASET\inference_results"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def run_inference():
    print(f"🚀 Starting Inference on {DEVICE}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at {MODEL_PATH}")
        return

    # Load Model structure (Same as training)
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None, # Loading our own weights
        in_channels=3,
        classes=1,
        activation="sigmoid"
    )
    model.to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("✅ Model loaded.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Pick random 5 images
    test_images = [f for f in os.listdir(TEST_IMG_PATH) if f.endswith(".tif")]
    if not test_images:
        print("❌ No test images found.")
        return
        
    np.random.shuffle(test_images)
    samples = test_images[:5]
    
    for img_name in samples:
        img_path = os.path.join(TEST_IMG_PATH, img_name)
        
        # Load
        original_img = cv2.imread(img_path)
        img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        input_tensor = img.astype('float32') / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))
        input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).to(DEVICE)
        
        # Infer
        with torch.no_grad():
            output = model(input_tensor)
            pred_mask = output.cpu().numpy()[0, 0]
        
        # Binarize
        pred_mask_bin = (pred_mask > 0.5).astype(np.uint8) * 255
        
        # Visualize
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 3, 1)
        plt.title("Image")
        plt.imshow(img)
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.title("Prediction")
        plt.imshow(pred_mask, cmap='jet')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.title("Binary Overlay")
        # Create red overlay
        overlay = original_img.copy()
        overlay[pred_mask_bin == 255] = [0, 0, 255] # Red BGR
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        save_path = os.path.join(OUTPUT_DIR, f"result_{img_name.replace('.tif', '.png')}")
        plt.savefig(save_path)
        plt.close()
        print(f"saved {save_path}")

if __name__ == "__main__":
    run_inference()
