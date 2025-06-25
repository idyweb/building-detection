#!/usr/bin/env python3
"""
Compare old model vs improved model performance
"""

import torch
import torchvision
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_model(model_path, device):
    """Load a trained model"""
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
    
    # Modify for 2 classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 2)
    
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, 256, 2)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    return model

def test_model(model, image_path, input_size, device):
    """Test a model on an image"""
    # Load image
    image = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize to model's expected input size
    image_resized = cv2.resize(image_rgb, input_size)
    
    # Convert to tensor
    image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        predictions = model(image_tensor)
    
    pred = predictions[0]
    scores = pred['scores'].cpu().numpy()
    boxes = pred['boxes'].cpu().numpy()
    
    return scores, boxes, image_resized

def compare_models():
    """Compare old vs new model performance"""
    
    print("🔄 MODEL COMPARISON TOOL")
    print("=" * 40)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Find models
    models_dir = Path("models")
    old_models = [m for m in models_dir.glob("*.pth") if "best_nigerian" in m.name and "improved" not in m.name]
    new_models = [m for m in models_dir.glob("*.pth") if "improved" in m.name]
    
    if not old_models:
        print("❌ No old model found")
        return
    
    if not new_models:
        print("❌ No improved model found. Train the improved model first.")
        return
    
    old_model_path = old_models[0]
    new_model_path = new_models[0]
    
    print(f"📦 Old model: {old_model_path.name}")
    print(f"🆕 New model: {new_model_path.name}")
    
    # Find test image
    image_dirs = ["satellite_images", "."]
    test_image = None
    for img_dir in image_dirs:
        if Path(img_dir).exists():
            images = list(Path(img_dir).glob("*.jpg"))
            if images:
                test_image = images[0]
                break
    
    if not test_image:
        print("❌ No test image found")
        return
    
    print(f"🖼️ Test image: {test_image.name}")
    
    try:
        # Load models
        print("\n📦 Loading models...")
        old_model = load_model(old_model_path, device)
        new_model = load_model(new_model_path, device)
        
        # Test both models
        print("🔍 Testing old model (512x512)...")
        old_scores, old_boxes, old_image = test_model(old_model, test_image, (512, 512), device)
        
        print("🔍 Testing new model (224x224)...")
        new_scores, new_boxes, new_image = test_model(new_model, test_image, (224, 224), device)
        
        # Compare results
        print(f"\n📊 COMPARISON RESULTS")
        print(f"=" * 30)
        
        print(f"Old Model (512x512):")
        print(f"   Total predictions: {len(old_scores)}")
        if len(old_scores) > 0:
            print(f"   Max confidence: {old_scores.max():.4f}")
            print(f"   Mean confidence: {old_scores.mean():.4f}")
            print(f"   >0.1 threshold: {(old_scores > 0.1).sum()}")
            print(f"   >0.3 threshold: {(old_scores > 0.3).sum()}")
        
        print(f"\nNew Model (224x224):")
        print(f"   Total predictions: {len(new_scores)}")
        if len(new_scores) > 0:
            print(f"   Max confidence: {new_scores.max():.4f}")
            print(f"   Mean confidence: {new_scores.mean():.4f}")
            print(f"   >0.1 threshold: {(new_scores > 0.1).sum()}")
            print(f"   >0.3 threshold: {(new_scores > 0.3).sum()}")
        
        # Visualize comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Old model results
        axes[0, 0].imshow(old_image)
        axes[0, 0].set_title(f'Old Model - Raw Image (512x512)\n{len(old_scores)} predictions')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(old_image)
        axes[0, 1].set_title(f'Old Model - Detections >0.05\nMax: {old_scores.max():.3f}')
        
        # Draw old model detections
        old_keep = old_scores > 0.05
        if old_keep.any():
            for box, score in zip(old_boxes[old_keep], old_scores[old_keep]):
                x1, y1, x2, y2 = box
                axes[0, 1].plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 'r-', linewidth=2)
                axes[0, 1].text(x1, y1-2, f'{score:.3f}', color='red', fontsize=8)
        axes[0, 1].axis('off')
        
        # New model results
        axes[1, 0].imshow(new_image)
        axes[1, 0].set_title(f'New Model - Raw Image (224x224)\n{len(new_scores)} predictions')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(new_image)
        axes[1, 1].set_title(f'New Model - Detections >0.05\nMax: {new_scores.max():.3f}')
        
        # Draw new model detections
        new_keep = new_scores > 0.05
        if new_keep.any():
            for box, score in zip(new_boxes[new_keep], new_scores[new_keep]):
                x1, y1, x2, y2 = box
                axes[1, 1].plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 'g-', linewidth=2)
                axes[1, 1].text(x1, y1-2, f'{score:.3f}', color='green', fontsize=8)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig("model_comparison.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"\n💾 Comparison saved: model_comparison.png")
        
        # Analysis
        print(f"\n🎯 ANALYSIS:")
        if len(new_scores) > 0 and new_scores.max() > old_scores.max():
            print("✅ New model shows improvement!")
        elif len(new_scores) > 0 and new_scores.max() > 0.3:
            print("🟡 New model shows reasonable performance")
        else:
            print("🔴 New model may still need more training")
        
        print(f"\n💡 RECOMMENDATIONS:")
        best_score = max(new_scores.max() if len(new_scores) > 0 else 0, 
                        old_scores.max() if len(old_scores) > 0 else 0)
        
        if best_score < 0.3:
            print("1. Consider training with more data")
            print("2. Try different augmentation strategies")
            print("3. Check if annotations are accurate")
            print("4. Consider ensemble approach from paper")
        elif best_score < 0.7:
            print("1. Model shows promise, continue training")
            print("2. Try fine-tuning hyperparameters")
            print("3. Add more diverse training examples")
        else:
            print("1. Model performance looks good!")
            print("2. Test on more diverse images")
            print("3. Consider deployment")
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main comparison function"""
    compare_models()

if __name__ == "__main__":
    main()