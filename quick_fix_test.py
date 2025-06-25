#!/usr/bin/env python3
"""
Quick fix test using exact paper parameters
"""

import torch
import torchvision
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def test_with_paper_params(model_path, test_image_path):
    """Test model using exact parameters from the research paper"""
    
    print("🔧 TESTING WITH EXACT PAPER PARAMETERS")
    print("Input size: 224x224 (as per paper)")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
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
    
    # Load image
    image = cv2.imread(str(test_image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # **CRITICAL: Use 224x224 as per paper, not 512x512**
    image_resized = cv2.resize(image_rgb, (224, 224))
    
    # Convert to tensor
    image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    print(f"✅ Using paper's input size: {image_tensor.shape}")
    
    # Run inference
    with torch.no_grad():
        predictions = model(image_tensor)
    
    pred = predictions[0]
    scores = pred['scores'].cpu().numpy()
    
    print(f"📊 Results with 224x224 input:")
    print(f"   Total predictions: {len(scores)}")
    
    if len(scores) > 0:
        print(f"   Max confidence: {scores.max():.4f}")
        
        # Test different thresholds
        for thresh in [0.1, 0.3, 0.5, 0.7]:
            count = (scores > thresh).sum()
            print(f"   Threshold {thresh}: {count} detections")
        
        if scores.max() > 0.1:
            # Visualize best detections
            visualize_with_correct_size(image_resized, pred, threshold=0.1)
            return True
    
    return False

def visualize_with_correct_size(image, pred, threshold=0.1):
    """Visualize with correct input size"""
    
    scores = pred['scores'].cpu().numpy()
    boxes = pred['boxes'].cpu().numpy()
    
    keep = scores > threshold
    if not keep.any():
        print(f"No detections at threshold {threshold}")
        return
    
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title(f'Results with 224x224 input (Paper size)\nThreshold: {threshold}')
    
    filtered_boxes = boxes[keep]
    filtered_scores = scores[keep]
    
    for box, score in zip(filtered_boxes, filtered_scores):
        x1, y1, x2, y2 = box
        plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 'r-', linewidth=2)
        plt.text(x1, y1-2, f'{score:.3f}', color='red', fontweight='bold', fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("paper_size_test_result.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"💾 Result saved: paper_size_test_result.png")

def main():
    # Find best model
    models_dir = Path("models")
    models = list(models_dir.glob("*.pth"))
    best_models = [m for m in models if "best" in m.name.lower()]
    model_path = best_models[0] if best_models else models[0]
    
    # Find test image
    image_dirs = ["satellite_images", "."]
    test_image = None
    for img_dir in image_dirs:
        if Path(img_dir).exists():
            images = list(Path(img_dir).glob("*.jpg"))
            if images:
                test_image = images[0]
                break
    
    if test_image and model_path:
        success = test_with_paper_params(model_path, test_image)
        
        if not success:
            print("\n🚨 CRITICAL ISSUE: Model likely needs retraining")
            print("\n💡 SOLUTIONS:")
            print("1. Retrain with correct 224x224 input size")
            print("2. Train for more epochs") 
            print("3. Use more training data")
            print("4. Check training convergence")
    else:
        print("❌ Model or image not found")

if __name__ == "__main__":
    main()