
import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import segmentation_models_pytorch as smp
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

# Configuration
DATA_DIR = r"c:\Users\mahaj\Downloads\GEOAIDATASET\processed_data\train"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
MASKS_DIR = os.path.join(DATA_DIR, "masks")
MODEL_SAVE_PATH = r"c:\Users\mahaj\Downloads\GEOAIDATASET\unet_building_model.pth"

BATCH_SIZE = 4
EPOCHS = 5
LR = 0.0001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class BuildingDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.images = sorted(os.listdir(images_dir))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        mask_name = img_name.replace("image", "mask")
        
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, mask_name)
        
        # Load Image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load Mask (1 channel)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # Binarize just in case
        mask = np.where(mask > 0, 1, 0).astype('float32')
        mask = np.expand_dims(mask, axis=0) # CHW format

        # Normalize Image
        image = image.astype('float32') / 255.0
        image = np.transpose(image, (2, 0, 1)) # CHW format
        
        return torch.from_numpy(image), torch.from_numpy(mask)

def train_model():
    print(f"🚀 Starting Training on {DEVICE}")
    
    # Dataset
    full_dataset = BuildingDataset(IMAGES_DIR, MASKS_DIR)
    
    # Split
    train_size = int(0.9 * len(full_dataset))
    valid_size = len(full_dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(full_dataset, [train_size, valid_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"📊 Training Samples: {len(train_dataset)}, Validation Samples: {len(valid_dataset)}")

    # Model
    model = smp.Unet(
        encoder_name="resnet34",        # encoder
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,                      # model output channels (number of classes in your dataset)
        activation="sigmoid"
    )
    
    model.to(DEVICE)
    
    loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Loop
    best_iou = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images, masks in loop:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Validation
        model.eval()
        valid_loss = 0
        tp, fp, fn, tn = smp.metrics.get_stats(model(images.to(DEVICE)).round().long(), masks.to(DEVICE).long(), mode='binary')
        iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {train_loss/len(train_loader):.4f} - Val IoU: {iou_score:.4f}")
        
        if iou_score > best_iou:
            best_iou = iou_score
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("💾 Model Saved!")

    print("✅ Training Complete.")

if __name__ == "__main__":
    if not os.path.exists(IMAGES_DIR):
        print("⚠️ Training data not ready yet. Please wait for preprocessing to finish.")
    else:
        train_model()
