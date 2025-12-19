import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import cv2
import os
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- CONFIG ---
DATA_DIR = os.path.join("server 2", "processed_data_multiclass")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
MASKS_DIR = os.path.join(DATA_DIR, "masks")
MODEL_SAVE_PATH = "infra_specialist.pth"

INFRA_CLASS_ID = 4
CROP_SIZE = 256
BATCH_SIZE = 16
LR = 0.001
EPOCHS = 15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- DATASET ---
class InfraCentricDataset(Dataset):
    def __init__(self, images_list, transform=None):
        self.images_list = images_list
        self.transform = transform
        
    def __len__(self):
        # Oversample: Pretend dataset is 4x larger to get more "crops" per epoch
        return len(self.images_list) * 4 

    def __getitem__(self, idx):
        # Map virtual idx to real file
        real_idx = idx % len(self.images_list)
        img_name = self.images_list[real_idx]
        
        img_path = os.path.join(IMAGES_DIR, img_name)
        mask_path = os.path.join(MASKS_DIR, img_name)
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        
        # --- INFRA-CENTRIC CROP ---
        # Find all infra pixels
        y_indices, x_indices = np.where(mask == INFRA_CLASS_ID)
        
        if len(y_indices) > 0:
            # Pick one random infra pixel to center on
            center_idx = np.random.randint(len(y_indices))
            cy, cx = y_indices[center_idx], x_indices[center_idx]
            
            # Calculate crop box
            y1 = max(0, cy - CROP_SIZE // 2)
            x1 = max(0, cx - CROP_SIZE // 2)
            y2 = min(image.shape[0], y1 + CROP_SIZE)
            x2 = min(image.shape[1], x1 + CROP_SIZE)
            
            # Shift if close to edge
            if y2 - y1 < CROP_SIZE:
                y1 = max(0, image.shape[0] - CROP_SIZE)
                y2 = image.shape[0]
            if x2 - x1 < CROP_SIZE:
                x1 = max(0, image.shape[1] - CROP_SIZE)
                x2 = image.shape[1]
                
            image = image[y1:y2, x1:x2]
            mask = mask[y1:y2, x1:x2]
        else:
            # Fallback: Resize if no infra found (shouldn't happen given our filtering)
            image = cv2.resize(image, (CROP_SIZE, CROP_SIZE))
            mask = cv2.resize(mask, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_NEAREST)

        # --- BINARY TARGET ---
        # 1.0 where Infra, 0.0 elsewhere
        target = np.zeros_like(mask, dtype=np.float32)
        target[mask == INFRA_CLASS_ID] = 1.0
        
        if self.transform:
            augmented = self.transform(image=image, mask=target)
            image = augmented['image']
            target = augmented['mask']
            
        return image, target.unsqueeze(0) # (1, H, W)

def get_training_augmentation():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(scale=(0.8, 1.2), rotate=(-20, 20), p=0.5),
        A.ColorJitter(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

# --- TRAINING ---
def train():
    print(f"🚀 Starting Infra Specialist Training (Binary Mode)")
    print(f"   Target: Class {INFRA_CLASS_ID} vs Background")
    
    # 1. Filter Data
    print("🔍 Filtering dataset for Infra images...")
    all_files = [f for f in os.listdir(MASKS_DIR) if f.endswith('.tif')]
    infra_files = []
    
    for f in tqdm(all_files):
        m = cv2.imread(os.path.join(MASKS_DIR, f), cv2.IMREAD_UNCHANGED)
        if INFRA_CLASS_ID in m:
            infra_files.append(f)
            
    if not infra_files:
        print("❌ No Infra images found!")
        return
        
    print(f"✅ Found {len(infra_files)} training images containing Infra.")
    
    # 2. Setup
    dataset = InfraCentricDataset(infra_files, transform=get_training_augmentation())
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    model = smp.Unet(
        encoder_name="resnet18", 
        encoder_weights="imagenet", 
        in_channels=3, 
        classes=1, 
        activation=None
    ).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # Loss: weighted BCE to prioritize foreground slightly + Dice 
    bce_weight = torch.tensor([5.0]).to(DEVICE) # 5x weight for Infra pixels vs non-infra pixels
    criterion_bce = nn.BCEWithLogitsLoss(pos_weight=bce_weight)
    criterion_dice = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
    
    # 3. Loop
    print("\n🏁 Training Start...")
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images, masks in loop:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            loss = criterion_bce(outputs, masks) + criterion_dice(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
    # Save
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n💾 Specialist Model Saved as {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()
