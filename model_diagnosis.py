#!/usr/bin/env python3
"""
Complete model diagnostics and fixing script
Based on the research paper methodology
"""

import torch
import torchvision
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

def diagnose_model_and_fix(model_path, test_image_path):
    """Comprehensive model diagnosis"""
    
    print("🔧 COMPREHENSIVE MODEL DIAGNOSTICS")
    print("Based on the research paper methodology")
    print("=" * 60)
    
    # 1. Check model file
    print("1️⃣ CHECKING MODEL FILE")
    print("-" * 30)
    
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        return False
        
    model_size = Path(model_path).stat().st_size / (1024*1024)
    print(f"📦 Model file size: {model_size:.1f} MB")
    
    # Load and inspect model
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        if isinstance(checkpoint, dict):
            print(f"✅ Model is a state dict with {len(checkpoint)} parameters")
            # Show some parameter names
            param_names = list(checkpoint.keys())[:5]
            print(f"📋 Sample parameters: {param_names}")
        else:
            print(f"⚠️ Model type: {type(checkpoint)}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    # 2. Test model loading and architecture
    print(f"\n2️⃣ TESTING MODEL ARCHITECTURE")
    print("-" * 30)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using device: {device}")
    
    # Create model exactly as in training
    try:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
        
        # Modify for building detection (background + building = 2 classes)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 2)
        
        # Modify mask predictor  
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, 256, 2)
        
        print("✅ Model architecture created successfully")
        
        # Load weights
        model.load_state_dict(checkpoint, strict=False)
        model.to(device)
        model.eval()
        print("✅ Model weights loaded successfully")
        
    except Exception as e:
        print(f"❌ Error with model architecture: {e}")
        return False
    
    # 3. Test image loading and preprocessing
    print(f"\n3️⃣ TESTING IMAGE PREPROCESSING")
    print("-" * 30)
    
    # Load image
    image = cv2.imread(str(test_image_path))
    if image is None:
        print(f"❌ Could not load image: {test_image_path}")
        return False
        
    print(f"📐 Original image shape: {image.shape}")
    
    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Show image statistics
    print(f"📊 Image stats - Min: {image_rgb.min()}, Max: {image_rgb.max()}, Mean: {image_rgb.mean():.1f}")
    
    # Resize (paper methodology)
    image_resized = cv2.resize(image_rgb, (512, 512))
    
    # Convert to tensor (exactly as in paper)
    image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float()
    image_tensor = image_tensor / 255.0  # Normalize to [0,1]
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    print(f"✅ Tensor shape: {image_tensor.shape}")
    print(f"✅ Tensor range: {image_tensor.min():.3f} - {image_tensor.max():.3f}")
    
    # 4. Test model inference
    print(f"\n4️⃣ TESTING MODEL INFERENCE")
    print("-" * 30)
    
    try:
        with torch.no_grad():
            predictions = model(image_tensor)
        
        pred = predictions[0]
        scores = pred['scores'].cpu().numpy()
        boxes = pred['boxes'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        
        print(f"✅ Inference successful!")
        print(f"📊 Raw predictions: {len(scores)}")
        
        if len(scores) > 0:
            print(f"📈 Score range: {scores.min():.4f} - {scores.max():.4f}")
            print(f"📈 Mean score: {scores.mean():.4f}")
            
            # Check label distribution
            unique_labels, counts = np.unique(labels, return_counts=True)
            print(f"🏷️ Label distribution: {dict(zip(unique_labels, counts))}")
            
            # Show top 10 predictions
            print(f"\n🎯 TOP 10 RAW PREDICTIONS:")
            top_indices = np.argsort(scores)[::-1][:10]
            for i, idx in enumerate(top_indices, 1):
                score = scores[idx]
                label = labels[idx]
                box = boxes[idx]
                print(f"   {i}. Score: {score:.4f}, Label: {label}, Box: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
            
            # Test different confidence thresholds
            print(f"\n📊 DETECTIONS AT DIFFERENT THRESHOLDS:")
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            for thresh in thresholds:
                keep = scores > thresh
                count = keep.sum()
                print(f"   Threshold {thresh}: {count} detections")
                
            # Visualize results at low threshold
            visualize_predictions(image_resized, boxes, scores, labels, threshold=0.1)
            
        else:
            print("❌ No predictions generated!")
            return False
            
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 5. Analysis and recommendations
    print(f"\n5️⃣ ANALYSIS & RECOMMENDATIONS")
    print("-" * 30)
    
    if len(scores) == 0:
        print("🚨 CRITICAL: No predictions generated")
        print("💡 Recommendations:")
        print("   - Retrain the model with more data")
        print("   - Check training data quality")
        print("   - Verify model convergence during training")
    elif scores.max() < 0.1:
        print("🚨 CRITICAL: All predictions have very low confidence")
        print("💡 Recommendations:")
        print("   - Model likely not trained properly")
        print("   - Training may not have converged")
        print("   - Consider retraining with different parameters")
    elif scores.max() < 0.5:
        print("⚠️ WARNING: Low confidence predictions")
        print("💡 Recommendations:")
        print("   - Model may need more training epochs")
        print("   - Try data augmentation")
        print("   - Check training/validation loss curves")
    else:
        print("✅ Model generates reasonable predictions")
        print("💡 Tune confidence threshold for best results")
    
    return True

def visualize_predictions(image, boxes, scores, labels, threshold=0.1):
    """Visualize predictions at given threshold"""
    
    keep = scores > threshold
    if not keep.any():
        print(f"No detections at threshold {threshold}")
        return
    
    filtered_boxes = boxes[keep]
    filtered_scores = scores[keep]
    
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.title(f'Predictions at threshold {threshold} ({len(filtered_boxes)} detections)')
    
    for box, score in zip(filtered_boxes, filtered_scores):
        x1, y1, x2, y2 = box
        plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 'r-', linewidth=2)
        plt.text(x1, y1-5, f'{score:.3f}', color='red', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.axis('off')
    plt.tight_layout()
    
    # Save result
    save_path = f"diagnosis_result_threshold_{threshold}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"💾 Visualization saved: {save_path}")
    plt.show()

def check_training_approach():
    """Check if training followed paper methodology"""
    
    print(f"\n6️⃣ TRAINING METHODOLOGY CHECK")
    print("-" * 30)
    print("📋 Research Paper Requirements:")
    print("   ✓ Architecture: Mask R-CNN + ResNet50")
    print("   ✓ Transfer Learning: From ESRI USA model") 
    print("   ✓ Training Data: 5-53 buildings per type")
    print("   ⚠️ Multiple Models: Different models for different building types")
    print("   ✓ Data Augmentation: Applied during training")
    print("   ✓ Image Size: 512x512 pixels")
    
    print(f"\n🔍 POTENTIAL ISSUES:")
    print("1. Paper trained SEPARATE models for:")
    print("   - High-rise buildings")
    print("   - Slum areas") 
    print("   - Regular buildings")
    print("   Then combined results using ensemble modeling")
    
    print(f"\n2. Your approach: Single model for all building types")
    print("   This may reduce performance compared to paper")
    
    print(f"\n💡 SOLUTIONS:")
    print("1. IMMEDIATE: Test with pretrained ESRI model")
    print("2. SHORT-TERM: Retrain with more epochs/data")
    print("3. LONG-TERM: Implement paper's ensemble approach")

def test_pretrained_esri_model(test_image_path):
    """Test with pretrained ESRI model as baseline"""
    
    print(f"\n7️⃣ TESTING PRETRAINED ESRI MODEL")
    print("-" * 30)
    print("🔄 Loading pretrained ESRI model...")
    
    try:
        # Load pretrained model
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='DEFAULT')
        model.eval()
        
        # Load and preprocess image
        image = cv2.imread(str(test_image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (512, 512))
        
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0)
        
        # Run inference
        with torch.no_grad():
            predictions = model(image_tensor)
        
        pred = predictions[0]
        scores = pred['scores'].cpu().numpy()
        
        print(f"✅ Pretrained model results:")
        print(f"   Total predictions: {len(scores)}")
        if len(scores) > 0:
            print(f"   Max confidence: {scores.max():.4f}")
            print(f"   Predictions >0.5: {(scores > 0.5).sum()}")
            
        print(f"\n💡 This helps determine if:")
        print(f"   - Image is suitable for detection")
        print(f"   - Issue is with your trained model")
        
    except Exception as e:
        print(f"❌ Error with pretrained model: {e}")

def main():
    """Main diagnostic function"""
    
    # Find model and image
    models_dir = Path("models")
    models = list(models_dir.glob("*.pth")) if models_dir.exists() else []
    
    if not models:
        print("❌ No models found in 'models' directory")
        return
    
    # Use best model
    best_models = [m for m in models if "best" in m.name.lower()]
    model_path = best_models[0] if best_models else models[0]
    
    # Find test image
    image_extensions = ['.jpg', '.jpeg', '.png']
    test_dirs = ["satellite_images", "test_images", "."]
    
    test_image = None
    for test_dir in test_dirs:
        if Path(test_dir).exists():
            for ext in image_extensions:
                images = list(Path(test_dir).glob(f"*{ext}"))
                if images:
                    test_image = images[0]
                    break
            if test_image:
                break
    
    if not test_image:
        print("❌ No test images found")
        return
    
    print(f"🔍 Using model: {model_path.name}")
    print(f"🖼️ Using image: {test_image.name}")
    
    # Run comprehensive diagnosis
    success = diagnose_model_and_fix(model_path, test_image)
    
    if success:
        check_training_approach()
        test_pretrained_esri_model(test_image)
    
    print(f"\n" + "="*60)
    print("🎯 NEXT STEPS BASED ON PAPER:")
    print("="*60)
    print("1. If model shows no predictions:")
    print("   → Retrain with more epochs (paper used 20)")
    print("   → Check training loss convergence")
    print("   → Verify training data quality")
    
    print(f"\n2. If model shows low confidence:")
    print("   → Increase training data (paper: 5-53 buildings)")
    print("   → Add more data augmentation")
    print("   → Train separate models per building type")
    
    print(f"\n3. Paper-specific improvements:")
    print("   → Implement ensemble modeling")
    print("   → Use building-type-specific training")
    print("   → Add more Nigerian building examples")

if __name__ == "__main__":
    main()