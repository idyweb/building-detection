#!/usr/bin/env python3
"""
Improved training script based on diagnostic results
Addresses low confidence score issues
"""

import os
import json
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

class BuildingDataset(Dataset):
    """Improved dataset with better preprocessing"""
    
    def __init__(self, image_dir, annotation_file, transforms=None, input_size=(224, 224)):
        self.image_dir = Path(image_dir)
        self.transforms = transforms
        self.input_size = input_size
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Create mappings
        self.image_id_to_info = {img['id']: img for img in self.coco_data['images']}
        self.image_id_to_annotations = {}
        
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[img_id] = []
            self.image_id_to_annotations[img_id].append(ann)
        
        self.image_ids = list(self.image_id_to_info.keys())
        
        # Filter out images with no annotations for better training
        self.image_ids = [img_id for img_id in self.image_ids 
                         if img_id in self.image_id_to_annotations]
        
        print(f"📊 Dataset: {len(self.image_ids)} images with annotations")
        total_buildings = sum(len(self.image_id_to_annotations[img_id]) for img_id in self.image_ids)
        print(f"🏠 Total buildings: {total_buildings}")
        print(f"📈 Average buildings per image: {total_buildings/len(self.image_ids):.1f}")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        # Get image info
        img_id = self.image_ids[idx]
        img_info = self.image_id_to_info[img_id]
        
        # Load image
        img_path = self.image_dir / img_info['file_name']
        image = Image.open(img_path).convert('RGB')
        
        # Get original dimensions
        orig_width, orig_height = image.size
        
        # Resize image to input size (critical fix)
        image = image.resize(self.input_size, Image.BILINEAR)
        
        # Calculate scaling factors
        scale_x = self.input_size[0] / orig_width
        scale_y = self.input_size[1] / orig_height
        
        # Get annotations for this image
        annotations = self.image_id_to_annotations.get(img_id, [])
        
        # Prepare targets
        boxes = []
        labels = []
        masks = []
        areas = []
        
        for ann in annotations:
            # Get bounding box and scale it
            bbox = ann['bbox']  # [x, y, width, height]
            x, y, w, h = bbox
            
            # Scale bounding box to new image size
            x_scaled = x * scale_x
            y_scaled = y * scale_y
            w_scaled = w * scale_x
            h_scaled = h * scale_y
            
            # Convert to [x1, y1, x2, y2] format
            x1 = max(0, x_scaled)
            y1 = max(0, y_scaled)
            x2 = min(self.input_size[0], x_scaled + w_scaled)
            y2 = min(self.input_size[1], y_scaled + h_scaled)
            
            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue
                
            boxes.append([x1, y1, x2, y2])
            labels.append(1)  # Building class
            
            # Create mask from segmentation
            if 'segmentation' in ann and ann['segmentation']:
                mask = self.create_scaled_mask(
                    ann['segmentation'][0], 
                    orig_width, orig_height,
                    scale_x, scale_y
                )
                masks.append(mask)
            else:
                # Create mask from scaled bbox
                mask = np.zeros(self.input_size[::-1], dtype=np.uint8)  # (height, width)
                mask[int(y1):int(y2), int(x1):int(x2)] = 1
                masks.append(mask)
            
            areas.append((x2 - x1) * (y2 - y1))
        
        # Convert to tensors
        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            masks = torch.tensor(np.array(masks), dtype=torch.uint8)
            areas = torch.tensor(areas, dtype=torch.float32)
        else:
            # Handle images with no valid annotations
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
            masks = torch.zeros((0, self.input_size[1], self.input_size[0]), dtype=torch.uint8)
            areas = torch.zeros(0, dtype=torch.float32)
        
        # Create target dictionary
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': torch.tensor([img_id]),
            'area': areas,
            'iscrowd': torch.zeros(len(boxes), dtype=torch.int64)
        }
        
        # Apply transforms
        if self.transforms:
            image = self.transforms(image)
        else:
            image = F.to_tensor(image)
        
        return image, target
    
    def create_scaled_mask(self, segmentation, orig_width, orig_height, scale_x, scale_y):
        """Create binary mask from segmentation polygon and scale it"""
        # Convert segmentation to points
        points = []
        for i in range(0, len(segmentation), 2):
            x = segmentation[i] * scale_x
            y = segmentation[i+1] * scale_y
            points.append([x, y])
        
        # Create mask at new size
        mask = np.zeros(self.input_size[::-1], dtype=np.uint8)  # (height, width)
        points = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [points], 1)
        
        return mask

def collate_fn(batch):
    """Custom collate function for DataLoader"""
    return tuple(zip(*batch))

class ImprovedESRITrainer:
    """Improved trainer with better convergence"""
    
    def __init__(self, output_dir="models", input_size=(224, 224)):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.input_size = input_size
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️ Training on: {self.device}")
        print(f"📐 Input size: {input_size}")
    
    def create_model(self, num_classes=2):
        """Create model with proper initialization"""
        print("🔄 Creating model...")
        
        # Start with pretrained model for better initialization
        model = maskrcnn_resnet50_fpn(weights='DEFAULT')
        
        # Modify for building detection (background + building = 2 classes)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes
        )
        
        # Modify mask predictor
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes
        )
        
        return model
    
    def train_model(self, dataset, num_epochs=30, batch_size=4, learning_rate=0.001, 
                   patience=10, min_lr=1e-6):
        """Improved training with better convergence monitoring"""
        
        print(f"🧠 Starting Improved ESRI Transfer Learning...")
        print(f"   📊 Dataset: {len(dataset)} images")
        print(f"   🔄 Max epochs: {num_epochs}")
        print(f"   📦 Batch size: {batch_size}")
        print(f"   📈 Initial learning rate: {learning_rate}")
        print(f"   ⏰ Patience: {patience}")
        
        # Create model
        model = self.create_model()
        model.to(self.device)
        
        # Create data loader with proper batch size
        data_loader = DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=True, 
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        # Improved optimizer and scheduler
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, 
                                    min_lr=min_lr, verbose=True)
        
        print(f"🎯 Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"🎯 Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        # Training loop with improved monitoring
        model.train()
        losses = []
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            epoch_losses = []
            epoch_loss_components = {'loss_classifier': [], 'loss_box_reg': [], 
                                   'loss_mask': [], 'loss_objectness': [], 
                                   'loss_rpn_box_reg': []}
            
            print(f"\n🔄 Epoch {epoch + 1}/{num_epochs}")
            print(f"📈 Current LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            for batch_idx, (images, targets) in enumerate(data_loader):
                # Move to device
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                # Filter out empty targets
                valid_images = []
                valid_targets = []
                for img, target in zip(images, targets):
                    if len(target['boxes']) > 0:
                        valid_images.append(img)
                        valid_targets.append(target)
                
                if len(valid_images) == 0:
                    continue
                
                # Forward pass
                loss_dict = model(valid_images, valid_targets)
                losses_total = sum(loss for loss in loss_dict.values())
                
                # Check for NaN
                if torch.isnan(losses_total):
                    print(f"⚠️ NaN loss detected, skipping batch")
                    continue
                
                # Backward pass
                optimizer.zero_grad()
                losses_total.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_losses.append(losses_total.item())
                
                # Track loss components
                for key, value in loss_dict.items():
                    if key in epoch_loss_components:
                        epoch_loss_components[key].append(value.item())
                
                if (batch_idx + 1) % max(1, len(data_loader) // 10) == 0:
                    print(f"   Batch {batch_idx + 1}/{len(data_loader)}: Loss = {losses_total.item():.4f}")
            
            if epoch_losses:
                avg_loss = np.mean(epoch_losses)
                losses.append(avg_loss)
                
                print(f"   ✅ Epoch {epoch + 1} average loss: {avg_loss:.4f}")
                
                # Print loss components
                for key, values in epoch_loss_components.items():
                    if values:
                        print(f"      {key}: {np.mean(values):.4f}")
                
                # Learning rate scheduling
                scheduler.step(avg_loss)
                
                # Early stopping and model saving
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                    
                    # Save best model
                    best_model_path = self.output_dir / "best_improved_nigerian_building_model.pth"
                    torch.save(model.state_dict(), best_model_path)
                    print(f"   💾 New best model saved: {best_model_path}")
                else:
                    patience_counter += 1
                    print(f"   ⏰ Patience: {patience_counter}/{patience}")
                
                # Early stopping
                if patience_counter >= patience:
                    print(f"🛑 Early stopping triggered after {epoch + 1} epochs")
                    break
                
                # Stop if learning rate becomes too small
                if optimizer.param_groups[0]['lr'] < min_lr:
                    print(f"🛑 Learning rate too small ({optimizer.param_groups[0]['lr']:.2e}), stopping")
                    break
        
        # Save final model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        final_model_path = self.output_dir / f"improved_nigerian_building_model_{timestamp}.pth"
        torch.save(model.state_dict(), final_model_path)
        print(f"💾 Final model saved: {final_model_path}")
        
        # Plot training curve
        self.plot_training_curve(losses)
        
        return model, str(final_model_path)
    
    def plot_training_curve(self, losses):
        """Plot improved training curve"""
        plt.figure(figsize=(12, 8))
        
        # Main loss plot
        plt.subplot(2, 1, 1)
        plt.plot(losses, 'b-', linewidth=2, marker='o')
        plt.title('Improved Transfer Learning: Training Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        
        # Log scale plot for better analysis
        plt.subplot(2, 1, 2)
        plt.semilogy(losses, 'r-', linewidth=2, marker='s')
        plt.title('Training Loss (Log Scale)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "improved_training_curve.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Training curve saved: {plot_path}")

def check_training_data_quality():
    """Check training data quality"""
    annotation_file = "training_data/annotations.json"
    image_dir = "satellite_images"
    
    if not os.path.exists(annotation_file):
        print("❌ No training data found!")
        return False
    
    try:
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        num_images = len(data['images'])
        num_annotations = len(data['annotations'])
        
        print(f"✅ Training data quality check:")
        print(f"   📸 Images: {num_images}")
        print(f"   🏠 Buildings: {num_annotations}")
        print(f"   📊 Average per image: {num_annotations/num_images:.1f}")
        
        # Check for sufficient data
        if num_annotations < 20:
            print(f"⚠️ Warning: Only {num_annotations} buildings. Consider adding more data.")
        elif num_annotations >= 50:
            print(f"✅ Good: {num_annotations} buildings should be sufficient")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading training data: {e}")
        return False

def main():
    """Main improved training function"""
    print("🚀 IMPROVED ESRI TRANSFER LEARNING TRAINER")
    print("Designed to fix low confidence score issues")
    print("=" * 60)
    
    # Check training data
    if not check_training_data_quality():
        return
    
    # Get training parameters
    print("\n⚙️ Improved Training Configuration:")
    
    try:
        epochs = int(input("Max epochs (default 30): ") or "30")
        batch_size = int(input("Batch size (default 4): ") or "4")
        learning_rate = float(input("Learning rate (default 0.001): ") or "0.001")
        patience = int(input("Early stopping patience (default 10): ") or "10")
    except ValueError:
        print("Using default values...")
        epochs = 30
        batch_size = 4
        learning_rate = 0.001
        patience = 10
    
    print(f"\n📊 Final Training Settings:")
    print(f"   🔄 Max epochs: {epochs}")
    print(f"   📦 Batch size: {batch_size}")
    print(f"   📈 Learning rate: {learning_rate}")
    print(f"   ⏰ Patience: {patience}")
    print(f"   📐 Input size: 224x224 (paper requirement)")
    
    proceed = input("\nStart improved training? (y/n): ").lower().strip()
    if proceed != 'y':
        print("❌ Training cancelled")
        return
    
    # Create improved dataset
    print("\n📚 Loading training data...")
    dataset = BuildingDataset("satellite_images", "training_data/annotations.json", 
                             input_size=(224, 224))
    
    if len(dataset) == 0:
        print("❌ No valid training data found!")
        return
    
    # Create trainer and train model
    trainer = ImprovedESRITrainer(input_size=(224, 224))
    
    print(f"\n🚀 Starting improved training...")
    print("Key improvements:")
    print("- Better model initialization with pretrained weights")
    print("- Adaptive learning rate scheduling")
    print("- Early stopping with patience")
    print("- Gradient clipping for stability")
    print("- Better loss monitoring")
    
    try:
        model, model_path = trainer.train_model(
            dataset, 
            num_epochs=epochs, 
            batch_size=batch_size, 
            learning_rate=learning_rate,
            patience=patience
        )
        
        print(f"\n🎉 IMPROVED TRAINING COMPLETE!")
        print(f"✅ Final model: {model_path}")
        print(f"✅ Best model: models/best_improved_nigerian_building_model.pth")
        print(f"📊 Training curve: models/improved_training_curve.png")
        print()
        print("🎯 Next steps:")
        print("1. Test the improved model")
        print("2. Expected better confidence scores")
        print("3. Compare with previous results")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()