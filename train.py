#!/usr/bin/env python3
"""
Transfer Learning with ESRI Model
Modified from your training script to use ESRI pretrained weights
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

class BuildingDataset(Dataset):
    """Custom dataset for building detection (keeping your original class)"""
    
    def __init__(self, image_dir, annotation_file, transforms=None):
        self.image_dir = Path(image_dir)
        self.transforms = transforms
        
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
        print(f"📊 Dataset: {len(self.image_ids)} images, {len(self.coco_data['annotations'])} buildings")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        # Get image info
        img_id = self.image_ids[idx]
        img_info = self.image_id_to_info[img_id]
        
        # Load image
        img_path = self.image_dir / img_info['file_name']
        image = Image.open(img_path).convert('RGB')
        
        # Get annotations for this image
        annotations = self.image_id_to_annotations.get(img_id, [])
        
        # Prepare targets
        boxes = []
        labels = []
        masks = []
        areas = []
        
        for ann in annotations:
            # Get bounding box
            bbox = ann['bbox']  # [x, y, width, height]
            x, y, w, h = bbox
            boxes.append([x, y, x + w, y + h])
            
            # Label (1 for building)
            labels.append(1)
            
            # Create mask from segmentation
            if 'segmentation' in ann and ann['segmentation']:
                mask = self.create_mask_from_segmentation(
                    ann['segmentation'][0], 
                    img_info['width'], 
                    img_info['height']
                )
                masks.append(mask)
            else:
                # Create mask from bbox if no segmentation
                mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
                mask[y:y+h, x:x+w] = 1
                masks.append(mask)
            
            areas.append(ann['area'])
        
        # Convert to tensors
        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            masks = torch.tensor(np.array(masks), dtype=torch.uint8)
            areas = torch.tensor(areas, dtype=torch.float32)
        else:
            # Handle images with no annotations
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
            masks = torch.zeros((0, img_info['height'], img_info['width']), dtype=torch.uint8)
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
    
    def create_mask_from_segmentation(self, segmentation, width, height):
        """Create binary mask from segmentation polygon"""
        # Convert segmentation to points
        points = []
        for i in range(0, len(segmentation), 2):
            points.append([segmentation[i], segmentation[i+1]])
        
        # Create mask
        mask = np.zeros((height, width), dtype=np.uint8)
        points = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [points], 1)
        
        return mask

def collate_fn(batch):
    """Custom collate function for DataLoader"""
    return tuple(zip(*batch))

class ESRITransferLearningTrainer:
    """Transfer learning trainer using ESRI pretrained model"""
    
    def __init__(self, esri_model_path="extracted_model", output_dir="models"):
        self.esri_model_path = Path(esri_model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️ Training on: {self.device}")
        
        # Load ESRI model configuration
        self.load_esri_config()
    
    def load_esri_config(self):
        """Load ESRI model configuration"""
        emd_file = self.esri_model_path / "usa_building_footprints.emd"
        if emd_file.exists():
            with open(emd_file, 'r') as f:
                self.esri_config = json.load(f)
            print(f"✅ Loaded ESRI config: {self.esri_config.get('ImageHeight')}x{self.esri_config.get('ImageWidth')}")
        else:
            print("⚠️ ESRI config not found, using defaults")
            self.esri_config = {"ImageHeight": 512, "ImageWidth": 512}
    
    def load_esri_pretrained_model(self, num_classes=2):
        """Load ESRI pretrained model and adapt for transfer learning"""
        print("🔄 Loading ESRI pretrained model...")
        
        # Create model architecture (same as ESRI: Mask R-CNN + ResNet50)
        model = maskrcnn_resnet50_fpn(weights=None)  # No pretrained weights yet
        
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
        
        # Load ESRI pretrained weights
        try:
            pth_file = self.esri_model_path / "usa_building_footprints.pth"
            esri_weights = torch.load(pth_file, map_location=self.device)
            
            # Filter weights that match our model architecture
            model_dict = model.state_dict()
            filtered_weights = {}
            
            for key, value in esri_weights.items():
                if key in model_dict and model_dict[key].shape == value.shape:
                    filtered_weights[key] = value
            
            # Load filtered weights
            model_dict.update(filtered_weights)
            model.load_state_dict(model_dict, strict=False)
            
            print(f"✅ Loaded {len(filtered_weights)} ESRI pretrained weights")
            print(f"📊 Transfer learning: {len(filtered_weights)}/{len(model_dict)} layers")
            
        except Exception as e:
            print(f"❌ Failed to load ESRI weights: {e}")
            print("💡 Training from scratch instead...")
        
        return model
    
    def prepare_for_finetuning(self, model, freeze_backbone=True):
        """Prepare model for fine-tuning (following paper methodology)"""
        
        if freeze_backbone:
            # Freeze backbone (recommended for small datasets as per paper)
            for param in model.backbone.parameters():
                param.requires_grad = False
            print("❄️ Backbone frozen (recommended for small datasets)")
        else:
            print("🔥 Training entire model")
        
        # Ensure detection heads are trainable
        for param in model.roi_heads.parameters():
            param.requires_grad = True
        
        for param in model.rpn.parameters():
            param.requires_grad = True
        
        return model
    
    def train_model(self, dataset, num_epochs=20, batch_size=2, learning_rate=0.0001, freeze_backbone=True):
        """Train the model using transfer learning"""
        print(f"🧠 Starting ESRI Transfer Learning...")
        print(f"   📊 Dataset: {len(dataset)} images")
        print(f"   🔄 Epochs: {num_epochs}")
        print(f"   📦 Batch size: {batch_size}")
        print(f"   📈 Learning rate: {learning_rate}")
        print(f"   ❄️ Freeze backbone: {freeze_backbone}")
        
        # Load ESRI pretrained model
        model = self.load_esri_pretrained_model()
        model = self.prepare_for_finetuning(model, freeze_backbone)
        model.to(self.device)
        
        # Create data loader
        data_loader = DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=True, 
            collate_fn=collate_fn,
            num_workers=0  # Avoid multiprocessing issues
        )
        
        # Optimizer - only train unfrozen parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=learning_rate, weight_decay=1e-4)
        
        # Learning rate scheduler (as in paper)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        print(f"🎯 Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
        
        # Training loop
        model.train()
        losses = []
        best_loss = float('inf')
        
        for epoch in range(num_epochs):
            epoch_losses = []
            
            print(f"\n🔄 Epoch {epoch + 1}/{num_epochs}")
            
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
                
                # Backward pass
                optimizer.zero_grad()
                losses_total.backward()
                optimizer.step()
                
                epoch_losses.append(losses_total.item())
                
                if (batch_idx + 1) % 3 == 0:
                    print(f"   Batch {batch_idx + 1}/{len(data_loader)}: Loss = {losses_total.item():.4f}")
                    # Print individual loss components
                    loss_components = ", ".join([f"{k}: {v.item():.3f}" for k, v in loss_dict.items()])
                    print(f"     Components: {loss_components}")
            
            if epoch_losses:
                avg_loss = np.mean(epoch_losses)
                losses.append(avg_loss)
                
                print(f"   ✅ Epoch {epoch + 1} average loss: {avg_loss:.4f}")
                
                # Save best model
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_model_path = self.output_dir / "best_nigerian_building_model.pth"
                    torch.save(model.state_dict(), best_model_path)
                    print(f"   💾 Best model saved: {best_model_path}")
            
            # Update learning rate
            scheduler.step()
        
        # Save final model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        final_model_path = self.output_dir / f"nigerian_building_model_{timestamp}.pth"
        torch.save(model.state_dict(), final_model_path)
        print(f"💾 Final model saved: {final_model_path}")
        
        # Plot training curve
        self.plot_training_curve(losses)
        
        return model, str(final_model_path)
    
    def plot_training_curve(self, losses):
        """Plot training loss curve"""
        plt.figure(figsize=(10, 6))
        plt.plot(losses, 'b-', linewidth=2, marker='o')
        plt.title('Transfer Learning: Training Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "transfer_learning_curve.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Training curve saved: {plot_path}")

def check_esri_model():
    """Check if ESRI model is available"""
    esri_path = Path("extracted_model")
    pth_file = esri_path / "usa_building_footprints.pth"
    emd_file = esri_path / "usa_building_footprints.emd"
    
    if not esri_path.exists():
        print("❌ ESRI model directory not found!")
        print("Please ensure 'extracted_model' directory exists with ESRI model files.")
        return False
    
    if not pth_file.exists():
        print("❌ ESRI model weights not found!")
        print(f"Expected: {pth_file}")
        return False
    
    print("✅ ESRI model found!")
    print(f"   📁 Directory: {esri_path}")
    print(f"   🎯 Model: {pth_file}")
    if emd_file.exists():
        print(f"   ⚙️ Config: {emd_file}")
    
    return True

def check_training_data():
    """Check if training data exists and is valid"""
    annotation_file = "training_data/annotations.json"
    image_dir = "satellite_images"
    
    if not os.path.exists(annotation_file):
        print("❌ No training data found!")
        print(f"Expected: {annotation_file}")
        print("Please make sure your COCO annotations are in the right place.")
        return False
    
    if not os.path.exists(image_dir):
        print("❌ No satellite images found!")
        print(f"Expected: {image_dir}")
        return False
    
    # Load and check annotations
    try:
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        num_images = len(data['images'])
        num_annotations = len(data['annotations'])
        
        print(f"✅ Training data check:")
        print(f"   📸 Images: {num_images}")
        print(f"   🏠 Buildings: {num_annotations}")
        print(f"   📊 Average per image: {num_annotations/num_images:.1f}")
        
        if num_annotations >= 5:
            print(f"✅ Sufficient data for transfer learning (paper used 5-53 buildings)")
        else:
            print(f"⚠️ Warning: Only {num_annotations} buildings. Consider adding more.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading training data: {e}")
        return False

def main():
    """Main transfer learning function"""
    print("🔄 ESRI TRANSFER LEARNING TRAINER")
    print("=" * 50)
    print("This will fine-tune the ESRI model using your Nigerian building data.")
    print("Following the research paper methodology for transfer learning.")
    print()
    
    # Check ESRI model
    if not check_esri_model():
        return
    
    # Check training data
    if not check_training_data():
        return
    
    # Ask user for training parameters
    print("\n⚙️ Transfer Learning Configuration:")
    
    try:
        epochs = int(input("Number of epochs (default 20): ") or "20")
        batch_size = int(input("Batch size (default 2): ") or "2")
        learning_rate = float(input("Learning rate (default 0.0001): ") or "0.0001")
        freeze_response = input("Freeze backbone? (y/n, default y): ").lower().strip()
        freeze_backbone = freeze_response != 'n'
    except ValueError:
        print("Using default values...")
        epochs = 20
        batch_size = 2
        learning_rate = 0.0001
        freeze_backbone = True
    
    print(f"\n📊 Training Settings:")
    print(f"   🔄 Epochs: {epochs}")
    print(f"   📦 Batch size: {batch_size}")
    print(f"   📈 Learning rate: {learning_rate}")
    print(f"   ❄️ Freeze backbone: {freeze_backbone}")
    
    proceed = input("\nStart transfer learning? (y/n): ").lower().strip()
    if proceed != 'y':
        print("❌ Training cancelled")
        return
    
    # Create dataset
    print("\n📚 Loading training data...")
    dataset = BuildingDataset("satellite_images", "training_data/annotations.json")
    
    if len(dataset) == 0:
        print("❌ No valid training data found!")
        return
    
    # Create trainer and train model
    trainer = ESRITransferLearningTrainer()
    
    print(f"\n🚀 Starting transfer learning...")
    print("This may take several minutes...")
    
    try:
        model, model_path = trainer.train_model(
            dataset, 
            num_epochs=epochs, 
            batch_size=batch_size, 
            learning_rate=learning_rate,
            freeze_backbone=freeze_backbone
        )
        
        print(f"\n🎉 TRANSFER LEARNING COMPLETE!")
        print(f"✅ Model saved: {model_path}")
        print(f"✅ Best model: models/best_nigerian_building_model.pth")
        print(f"📊 Training curve: models/transfer_learning_curve.png")
        print()
        print("🎯 Next steps:")
        print("1. Test your model with the adapted weights")
        print("2. Use the best model for building detection")
        print("3. Expected performance: F1-score 0.92-0.96 (as per paper)")
        
    except Exception as e:
        print(f"❌ Transfer learning failed: {e}")
        print("💡 Try reducing batch_size to 1 if you get memory errors")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()