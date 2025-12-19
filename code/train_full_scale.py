
import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import cv2
import segmentation_models_pytorch as smp
import numpy as np
import pandas as pd
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time
import glob

# --- CONFIGURATION ---
DATA_DIR = os.path.join("server", "processed_data_multiclass") # Root dir of chips
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "images")
TRAIN_MASK_DIR = os.path.join(DATA_DIR, "masks")
WEIGHTS_CACHE = os.path.join(DATA_DIR, "sample_weights.pt")

MODEL_SAVE_PATH = os.path.join("server", "best_model_multiclass.pth")
CHECKPOINT_PATH = os.path.join("server", "last_checkpoint.pth")
LOG_FILE = os.path.join("server", "training_log_multiclass.csv")

# Classes: 0:Back, 1:Build, 2:Road, 3:Water, 4:Infra, 5:RCC, 6:Tiled
NUM_CLASSES = 7 
ENCODER = "resnet50"
ENCODER_WEIGHTS = "imagenet"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 16 # Adjust based on Server VRAM (A100 can take 32-64, T4 maybe 16)
EPOCHS = 125
LR = 0.0001
NUM_WORKERS = 0 if os.name == 'nt' else 8 # Windows needs 0, Server can use more

# --- DATASET ---
class MulticlassDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.ids = [f for f in os.listdir(images_dir) if f.endswith('.tif')]
        # Sort to ensure consistent order for Sampler
        self.ids.sort()
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_name = self.ids[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name) # Same name

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask as Index Image (values 0-6)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED) 
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask)
            
        return image, mask.long() # CrossEntropy needs LongTensor

# --- AUGMENTATION ---
def get_training_augmentation():
    train_transform = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(scale=(0.9, 1.1), translate_percent=(-0.1, 0.1), rotate=(-15, 15), p=0.5),
        A.PadIfNeeded(min_height=512, min_width=512, p=1, border_mode=0),
        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.5,
        ),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
    return A.Compose(train_transform)

def get_validation_augmentation():
    test_transform = [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
    return A.Compose(test_transform)

# --- SAMPLER HELPER ---
def get_sampler(dataset):
    """
    Scans dataset for minority classes (Infra=4, Tiled=6) and attempts to oversample them.
    Saves weights to cache to avoid rescanning.
    """
    if os.path.exists(WEIGHTS_CACHE):
        print(f"🔄 Loading cached sample weights from {WEIGHTS_CACHE}")
        return WeightedRandomSampler(torch.load(WEIGHTS_CACHE), len(dataset))
    
    print("⚖️ Scanning dataset to calculate oversampling weights (this happens once)...")
    sample_weights = []
    
    # Priority Weights:
    # Infra (4): 20x
    # Tiled (6): 10x
    # Building (1) / RCC (5): 2x
    # Road (2) / Water (3): 2x
    # Background (0): 1x
    
    for img_name in tqdm(dataset.ids, desc="Scanning Metadata"):
        mask_path = os.path.join(dataset.masks_dir, img_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        
        classes = np.unique(mask)
        weight = 1.0
        
        if 4 in classes: # Infra
            weight = 20.0
        elif 6 in classes: # Tiled
            weight = 10.0
        elif 1 in classes or 5 in classes:
            weight = 2.0
            
        sample_weights.append(weight)
        
    sample_weights = torch.DoubleTensor(sample_weights)
    torch.save(sample_weights, WEIGHTS_CACHE)
    print(f"✅ Weights calculated and saved.")
    
    return WeightedRandomSampler(sample_weights, len(dataset))

# --- TRAINING LOOP ---
def train_full_scale():
    print(f"🚀 Starting High-Scale Training: DeepLabV3+ ({ENCODER}) with Focal+Dice Loss")
    
    # 1. Dataset
    if not os.path.exists(TRAIN_IMG_DIR):
        print(f"❌ Data not found at {TRAIN_IMG_DIR}. Run preprocessing first!")
        return

    # Fix Seed for Reproducibility (Critical for Resume)
    torch.manual_seed(42)
    np.random.seed(42)

    full_dataset = MulticlassDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, transform=get_training_augmentation())
    
    # Split logic must ideally be index-based to support sampling correctly
    # But Sampler works on the *loader*.
    # If we split the dataset using Subset, the Indices change.
    # We need to map global weights to subset indices.
    
    train_size = int(0.9 * len(full_dataset))
    valid_size = len(full_dataset) - train_size
    
    # INDICES based split
    indices = torch.randperm(len(full_dataset)).tolist()
    train_indices, valid_indices = indices[:train_size], indices[train_size:]
    
    train_ds = torch.utils.data.Subset(full_dataset, train_indices)
    valid_ds = torch.utils.data.Subset(full_dataset, valid_indices) # No transform tweak? Wait, Subset keeps original dataset.
    
    # Override transform for validation? 
    # Proper way: separate Datasets with same filtered IDs.
    
    # For now, let's keep it simple. Subset inherits transform. 
    # To fix validation transform, we can wrap the subset.
    class ValidationWrapper(Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
        def __len__(self): return len(self.subset)
        def __getitem__(self, idx):
            # Access underlying dataset
            img, mask = self.subset[idx]
            # Re-apply transform? No, subset[idx] already applies transform of parent.
            # We need parent to NOT have transform, or override it.
            # Hack: Access parent's raw data.
            original_idx = self.subset.indices[idx]
            parent = self.subset.dataset
            
            img_name = parent.ids[original_idx]
            img_path = os.path.join(parent.images_dir, img_name)
            mask_path = os.path.join(parent.masks_dir, img_name)
            
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            
            if self.transform:
                aug = self.transform(image=image, mask=mask)
                image, mask = aug['image'], aug['mask']
            return image, mask.long()

    # Re-init datasets with NO transform for splitting
    raw_dataset = MulticlassDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, transform=None)
    
    # Calculate Weights for SAMPLER (Global)
    # We need weights ONLY for Train Subset.
    # Logic: 
    # 1. Get Global Weights (scan all).
    # 2. Extract weights corresponding to train_indices.
    
    global_weights = []
    if os.path.exists(WEIGHTS_CACHE):
        print(f"🔄 Loading cached weights...")
        global_weights = torch.load(WEIGHTS_CACHE)
    else:
        print("⚖️ Scanning dataset for class balance (One-time)...")
        for img_name in tqdm(raw_dataset.ids):
            mask = cv2.imread(os.path.join(raw_dataset.masks_dir, img_name), cv2.IMREAD_UNCHANGED)
            classes = np.unique(mask)
            w = 1.0
            if 4 in classes: w = 20.0
            elif 6 in classes: w = 10.0
            elif 1 in classes or 5 in classes: w = 2.0
            global_weights.append(w)
        global_weights = torch.DoubleTensor(global_weights)
        torch.save(global_weights, WEIGHTS_CACHE)
        
    # Split
    train_indices, valid_indices = indices[:train_size], indices[train_size:]
    
    # Subset Weights
    train_weights = global_weights[train_indices]
    sampler = WeightedRandomSampler(train_weights, num_samples=len(train_indices))

    # Create Final Datasets
    train_subset = torch.utils.data.Subset(raw_dataset, train_indices) # Raw, no transform
    
    # Wrappers apply transform
    train_wrapped = ValidationWrapper(train_subset, get_training_augmentation())
    valid_wrapped = ValidationWrapper(torch.utils.data.Subset(raw_dataset, valid_indices), get_validation_augmentation())

    train_loader = DataLoader(train_wrapped, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True)
    valid_loader = DataLoader(valid_wrapped, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    print(f"📊 Training on {len(train_wrapped)} images (Oversampled), Validating on {len(valid_wrapped)} images.")

    # 2. Model
    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        in_channels=3, 
        classes=NUM_CLASSES, 
        activation=None
    )
    model.to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3)
    
    # 3. LOSS FUNCTION CHANGE
    # Focal Loss (focus on hard examples) + Dice Loss (focus on IoU)
    loss_fn_focal = smp.losses.FocalLoss(smp.losses.MULTICLASS_MODE, gamma=2.0)
    loss_fn_dice = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)

    # 4. Resume Logic
    start_epoch = 0
    best_iou = 0.0
    
    if os.path.exists(CHECKPOINT_PATH):
        print(f"🔄 Resuming from {CHECKPOINT_PATH}...")
        checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False) # Fix for future warning
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_iou = checkpoint.get('best_iou', 0.0)
        print(f"   Resumed at Epoch {start_epoch}, Best IoU: {best_iou:.4f}")
    
    elif os.path.exists(MODEL_SAVE_PATH):
        print(f"🔄 Fine-tuning from Best Model {MODEL_SAVE_PATH}...")
        checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
             try: model.load_state_dict(checkpoint)
             except: pass
        print("   Loaded weights from best model. Starting Fine-Tuning.")

    # 5. Loop
    history = []
    
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        train_loss = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images, masks in loop:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images) 
            
            # Combined Loss
            l_focal = loss_fn_focal(outputs, masks)
            l_dice = loss_fn_dice(outputs, masks)
            
            loss = l_focal + l_dice # Equal weight often works best with Focal
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        # Validation
        model.eval()
        valid_loss = 0
        iou_scores = []
        
        with torch.no_grad():
            for images, masks in valid_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)
                
                outputs = model(images)
                l_f = loss_fn_focal(outputs, masks)
                l_d = loss_fn_dice(outputs, masks)
                valid_loss += (l_f + l_d).item()
                
                pred_mask = torch.argmax(outputs, dim=1)
                
                # Metrics (Macro IoU)
                tp, fp, fn, tn = smp.metrics.get_stats(pred_mask, masks, mode='multiclass', num_classes=NUM_CLASSES)
                iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
                iou_scores.append(iou.item())
                
        # Per Class IoU Check (Optional debug)
        # ...

        avg_train_loss = train_loss / len(train_loader)
        avg_valid_loss = valid_loss / len(valid_loader)
        avg_iou = np.mean(iou_scores)
        
        print(f"   Epoch {epoch+1} Results: T_Loss={avg_train_loss:.4f}, V_Loss={avg_valid_loss:.4f}, V_IoU={avg_iou:.4f}")
        
        scheduler.step(avg_iou)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_iou': best_iou
        }, CHECKPOINT_PATH)
        
        if avg_iou > best_iou:
            best_iou = avg_iou
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("   💾 New Best Model Saved!")
            
        log_entry = {
            "epoch": epoch+1,
            "train_loss": avg_train_loss,
            "valid_loss": avg_valid_loss,
            "valid_iou": avg_iou,
            "lr": optimizer.param_groups[0]['lr']
        }
        history.append(log_entry)
        pd.DataFrame(history).to_csv(LOG_FILE, mode='a', header=not os.path.exists(LOG_FILE), index=False)

if __name__ == "__main__":
    train_full_scale()
