#!/usr/bin/env python3
"""
Debug why the new improved model is producing zero predictions
"""

import torch
import torchvision
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def debug_model_issue(model_path, test_image_path):
    """Debug the new model to understand why it produces no predictions"""
    
    print("🔧 DEBUGGING NEW MODEL ISSUE")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load image first
    image = cv2.imread(str(test_image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"📐 Original image shape: {image_rgb.shape}")
    
    # Test different model loading approaches
    print("\n1️⃣ TESTING MODEL LOADING METHODS")
    print("-" * 30)
    
    try:
        # Method 1: Load as in training (with pretrained=True initially)
        print("Testing Method 1: Load with DEFAULT weights first...")
        model1 = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='DEFAULT')
        
        # Modify architecture
        in_features = model1.roi_heads.box_predictor.cls_score.in_features
        model1.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 2)
        
        in_features_mask = model1.roi_heads.mask_predictor.conv5_mask.in_channels
        model1.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, 256, 2)
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=device)
        model1.load_state_dict(checkpoint, strict=False)
        model1.to(device)
        model1.eval()
        
        print("✅ Method 1 successful")
        result1 = test_model_inference(model1, image_rgb, (224, 224), device, "Method 1")
        
    except Exception as e:
        print(f"❌ Method 1 failed: {e}")
        result1 = None
    
    try:
        # Method 2: Load exactly as in old comparison (weights=None)
        print("\nTesting Method 2: Load with weights=None...")
        model2 = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
        
        # Modify architecture
        in_features = model2.roi_heads.box_predictor.cls_score.in_features
        model2.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 2)
        
        in_features_mask = model2.roi_heads.mask_predictor.conv5_mask.in_channels
        model2.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, 256, 2)
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=device)
        model2.load_state_dict(checkpoint, strict=True)
        model2.to(device)
        model2.eval()
        
        print("✅ Method 2 successful")
        result2 = test_model_inference(model2, image_rgb, (224, 224), device, "Method 2")
        
    except Exception as e:
        print(f"❌ Method 2 failed: {e}")
        result2 = None
    
    # Test different input sizes
    print("\n2️⃣ TESTING DIFFERENT INPUT SIZES")
    print("-" * 30)
    
    if result2 is not None:
        model = model2  # Use working model
        
        # Test various input sizes
        test_sizes = [(224, 224), (256, 256), (320, 320), (512, 512)]
        
        for size in test_sizes:
            try:
                result = test_model_inference(model, image_rgb, size, device, f"Size {size}")
            except Exception as e:
                print(f"❌ Size {size} failed: {e}")
    
    # Check model state
    print("\n3️⃣ ANALYZING MODEL STATE")
    print("-" * 30)
    
    if result2 is not None:
        analyze_model_state(model2)
    
    # Test with different confidence thresholds
    print("\n4️⃣ TESTING RAW MODEL OUTPUT")
    print("-" * 30)
    
    if result2 is not None:
        test_raw_output(model2, image_rgb, device)

def test_model_inference(model, image_rgb, input_size, device, method_name):
    """Test model inference with given parameters"""
    
    # Preprocess
    image_resized = cv2.resize(image_rgb, input_size)
    image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    print(f"   📐 Input tensor shape: {image_tensor.shape}")
    print(f"   📊 Input range: {image_tensor.min():.3f} - {image_tensor.max():.3f}")
    
    # Run inference
    with torch.no_grad():
        predictions = model(image_tensor)
    
    pred = predictions[0]
    scores = pred['scores'].cpu().numpy()
    boxes = pred['boxes'].cpu().numpy()
    
    print(f"   📊 {method_name} Results:")
    print(f"      Total predictions: {len(scores)}")
    
    if len(scores) > 0:
        print(f"      Score range: {scores.min():.4f} - {scores.max():.4f}")
        print(f"      Mean score: {scores.mean():.4f}")
        
        # Count at different thresholds
        for thresh in [0.01, 0.05, 0.1, 0.3]:
            count = (scores > thresh).sum()
            print(f"      >={thresh}: {count} detections")
    else:
        print(f"      ❌ NO PREDICTIONS GENERATED!")
    
    return {'scores': scores, 'boxes': boxes}

def analyze_model_state(model):
    """Analyze the model's internal state"""
    
    print("🔍 Model Analysis:")
    
    # Check if model is in eval mode
    print(f"   Training mode: {model.training}")
    
    # Check some key parameters
    try:
        # Check backbone
        backbone_params = sum(p.numel() for p in model.backbone.parameters())
        print(f"   Backbone parameters: {backbone_params:,}")
        
        # Check RPN
        rpn_params = sum(p.numel() for p in model.rpn.parameters())
        print(f"   RPN parameters: {rpn_params:,}")
        
        # Check ROI heads
        roi_params = sum(p.numel() for p in model.roi_heads.parameters())
        print(f"   ROI head parameters: {roi_params:,}")
        
        # Check for frozen parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Frozen parameters: {total_params - trainable_params:,}")
        
    except Exception as e:
        print(f"   Error analyzing model: {e}")

def test_raw_output(model, image_rgb, device):
    """Test raw model output without any filtering"""
    
    # Use 224x224 as intended
    image_resized = cv2.resize(image_rgb, (224, 224))
    image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    print("🔍 Raw Model Output Analysis:")
    
    with torch.no_grad():
        # Get raw predictions
        predictions = model(image_tensor)
    
    pred = predictions[0]
    
    print(f"   Prediction keys: {list(pred.keys())}")
    
    for key, value in pred.items():
        if torch.is_tensor(value):
            print(f"   {key}: shape={value.shape}, dtype={value.dtype}")
            if len(value) > 0:
                if key == 'scores':
                    print(f"      Raw scores: {value.cpu().numpy()[:10]}")  # First 10
                elif key == 'labels':
                    unique_labels, counts = torch.unique(value, return_counts=True)
                    print(f"      Labels: {dict(zip(unique_labels.cpu().numpy(), counts.cpu().numpy()))}")
    
    # Test if model produces any output at all
    scores = pred['scores'].cpu().numpy()
    if len(scores) == 0:
        print("   🚨 CRITICAL: Model produces zero predictions!")
        print("   🔧 Possible causes:")
        print("      1. Model weights didn't load correctly")
        print("      2. Model architecture mismatch")
        print("      3. Input preprocessing issue")
        print("      4. Model completely broken/overfit")
    else:
        print(f"   ✅ Model produces {len(scores)} raw predictions")
        
        # Show very low threshold results
        very_low_thresh = 0.001
        keep = scores > very_low_thresh
        print(f"   At threshold {very_low_thresh}: {keep.sum()} detections")

def check_training_logs():
    """Check if training logs are available"""
    
    print("\n5️⃣ CHECKING TRAINING ARTIFACTS")
    print("-" * 30)
    
    models_dir = Path("models")
    
    # Look for training curve
    training_curve = models_dir / "improved_training_curve.png"
    if training_curve.exists():
        print(f"✅ Training curve found: {training_curve}")
        print("   👀 Check this file to see if training converged properly")
    else:
        print("❌ No training curve found")
    
    # Look for other model files
    improved_models = list(models_dir.glob("*improved*.pth"))
    print(f"📦 Found {len(improved_models)} improved model files:")
    for model in improved_models:
        size_mb = model.stat().st_size / (1024*1024)
        print(f"   {model.name}: {size_mb:.1f} MB")

def main():
    """Main debugging function"""
    
    # Find models and images
    models_dir = Path("models")
    improved_models = [m for m in models_dir.glob("*.pth") if "improved" in m.name]
    
    if not improved_models:
        print("❌ No improved model found!")
        return
    
    # Use best improved model
    best_improved = [m for m in improved_models if "best" in m.name]
    model_path = best_improved[0] if best_improved else improved_models[0]
    
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
    
    print(f"🔧 Debugging model: {model_path.name}")
    print(f"🖼️ Using image: {test_image.name}")
    
    # Run comprehensive debug
    debug_model_issue(model_path, test_image)
    check_training_logs()
    
    print(f"\n" + "="*60)
    print("🎯 DIAGNOSIS SUMMARY")
    print("="*60)
    print("If the new model produces zero predictions:")
    print("1. Check training curve - did training actually work?")
    print("2. Model architecture mismatch during loading")
    print("3. Input size incompatibility")
    print("4. Model completely overfit or broken")
    print(f"\n💡 NEXT STEPS:")
    print("1. If Method 1 works, use that loading approach")
    print("2. If all methods fail, retrain with different parameters")
    print("3. Check training logs for convergence issues")

if __name__ == "__main__":
    main()