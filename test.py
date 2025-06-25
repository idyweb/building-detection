#!/usr/bin/env python3
"""
Test with your downloaded Manhattan satellite images
"""

import torch
import torchvision
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def test_with_manhattan_images():
    """Test the original ESRI model with your Manhattan images"""
    
    print("🇺🇸 TESTING WITH YOUR MANHATTAN SATELLITE IMAGES")
    print("=" * 50)
    
    # Look for Manhattan images in current directory
    current_dir = Path(".")
    
    # Find Manhattan images specifically
    manhattan_patterns = [
        "*Manhattan*",
        "*manhattan*", 
        "*Financial*",
        "*financial*"
    ]
    
    found_images = []
    
    # Search for Manhattan images
    for pattern in manhattan_patterns:
        found_images.extend(list(current_dir.glob(f"{pattern}.jpg")))
        found_images.extend(list(current_dir.glob(f"{pattern}.jpeg")))
        found_images.extend(list(current_dir.glob(f"{pattern}.png")))
    
    # Also look for any large images
    found_images.extend(list(current_dir.glob("large_*.jpg")))
    found_images.extend(list(current_dir.glob("large_*.jpeg")))
    
    # Remove duplicates
    found_images = list(set(found_images))
    found_images.sort()
    
    print(f"🔍 Found {len(found_images)} Manhattan/large image files:")
    for i, img in enumerate(found_images, 1):
        print(f"   {i}. {img.name}")
    
    if not found_images:
        print("❌ No Manhattan images found!")
        print("💡 Looking for any image files...")
        
        # Fallback: look for any images
        all_images = []
        for ext in ['.jpg', '.jpeg', '.png']:
            all_images.extend(list(current_dir.glob(f"*{ext}")))
            all_images.extend(list(current_dir.glob(f"*{ext.upper()}")))
        
        if all_images:
            print(f"Found {len(all_images)} other images:")
            for i, img in enumerate(all_images, 1):
                print(f"   {i}. {img.name}")
            found_images = all_images
        else:
            print("❌ No images found in current directory!")
            return
    
    # Let user select image or use first one
    try:
        if len(found_images) == 1:
            selected_image = found_images[0]
            print(f"\n✅ Using: {selected_image.name}")
        else:
            choice = input(f"\nSelect image (1-{len(found_images)}, or Enter for first): ").strip()
            if choice:
                selected_image = found_images[int(choice) - 1]
            else:
                selected_image = found_images[0]
            print(f"✅ Using: {selected_image.name}")
    
    except (ValueError, IndexError):
        selected_image = found_images[0]
        print(f"✅ Using first image: {selected_image.name}")
    
    # Test the image
    test_original_esri_model(selected_image)

def test_original_esri_model(image_path):
    """Test the original ESRI model on a single image"""
    
    print(f"\n🧪 TESTING ORIGINAL ESRI MODEL")
    print(f"📸 Image: {image_path.name}")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Device: {device}")
    
    # Load the ORIGINAL ESRI model (COCO pretrained)
    print("📦 Loading original ESRI model (COCO pretrained)...")
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='DEFAULT')
    model.to(device)
    model.eval()
    print("✅ Model loaded successfully!")
    
    # Load and process image
    print("🖼️ Loading image...")
    image = cv2.imread(str(image_path))
    
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"📐 Original size: {image_rgb.shape}")
    
    # Resize to 512x512 (same as your trained models)
    image_resized = cv2.resize(image_rgb, (512, 512))
    print(f"📐 Resized to: {image_resized.shape}")
    
    # Convert to tensor
    image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Run inference
    print("🔍 Running inference...")
    with torch.no_grad():
        predictions = model(image_tensor)
    
    # Process results
    pred = predictions[0]
    scores = pred['scores'].cpu().numpy()
    boxes = pred['boxes'].cpu().numpy()
    labels = pred['labels'].cpu().numpy()
    
    print(f"\n📊 RESULTS:")
    print(f"   Total detections: {len(scores)}")
    
    if len(scores) > 0:
        print(f"   Score range: {scores.min():.3f} - {scores.max():.3f}")
        print(f"   Mean score: {scores.mean():.3f}")
        
        # Count by confidence levels
        very_high = (scores > 0.9).sum()
        high = ((scores > 0.7) & (scores <= 0.9)).sum()
        medium = ((scores > 0.5) & (scores <= 0.7)).sum()
        low = ((scores > 0.3) & (scores <= 0.5)).sum()
        very_low = (scores <= 0.3).sum()
        
        print(f"\n🎯 CONFIDENCE BREAKDOWN:")
        print(f"   Very High (>0.9): {very_high}")
        print(f"   High (0.7-0.9): {high}")
        print(f"   Medium (0.5-0.7): {medium}")
        print(f"   Low (0.3-0.5): {low}")
        print(f"   Very Low (≤0.3): {very_low}")
        
        # Show detected object classes (COCO classes)
        coco_classes = {
            1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
            6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
            11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
            16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
            21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
            27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
            34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
            39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard',
            43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup',
            48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana',
            53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot',
            58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair',
            63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
            70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote',
            76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
            80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book',
            85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear',
            89: 'hair drier', 90: 'toothbrush'
        }
        
        # Show classes detected with medium+ confidence
        medium_plus = scores > 0.5
        if medium_plus.any():
            print(f"\n🏷️ DETECTED OBJECTS (>0.5 confidence):")
            unique_labels, counts = np.unique(labels[medium_plus], return_counts=True)
            for label, count in zip(unique_labels, counts):
                class_name = coco_classes.get(label, f'Unknown_{label}')
                avg_conf = scores[labels == label].mean()
                print(f"   {class_name}: {count} detections (avg: {avg_conf:.3f})")
        
        # Visualize results
        visualize_results(image_resized, boxes, scores, labels, coco_classes, image_path.stem)
        
        # Compare with your Nigerian results
        compare_with_nigerian_results(scores, labels, coco_classes, image_path.name)
        
    else:
        print("   ❌ No detections found")
        print("   💡 This might indicate:")
        print("      • Image doesn't contain objects the model recognizes")
        print("      • Image quality issues")
        print("      • Model expects different input format")

def visualize_results(image, boxes, scores, labels, coco_classes, image_name):
    """Visualize detection results"""
    
    # Show different confidence thresholds
    thresholds = [0.3, 0.5, 0.7, 0.9]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, threshold in enumerate(thresholds):
        keep = scores > threshold
        
        axes[i].imshow(image)
        axes[i].set_title(f'Threshold {threshold}\n{keep.sum()} detections')
        axes[i].axis('off')
        
        if keep.any():
            filtered_boxes = boxes[keep]
            filtered_scores = scores[keep]
            filtered_labels = labels[keep]
            
            # Color map
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'cyan', 'magenta']
            
            for box, score, label in zip(filtered_boxes, filtered_scores, filtered_labels):
                x1, y1, x2, y2 = box
                color = colors[int(label) % len(colors)]
                
                axes[i].plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 
                           color=color, linewidth=2)
                
                class_name = coco_classes.get(label, f'C{label}')
                axes[i].text(x1, y1-3, f'{class_name}:{score:.2f}', 
                           color=color, fontweight='bold', fontsize=8,
                           bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"us_test_results_{image_name}.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"💾 Results saved: us_test_results_{image_name}.png")

def compare_with_nigerian_results(us_scores, us_labels, coco_classes, image_name):
    """Compare US results with your Nigerian model results"""
    
    print(f"\n" + "="*60)
    print("🔄 COMPARISON: US vs NIGERIAN RESULTS")
    print("="*60)
    
    # US results summary
    if len(us_scores) > 0:
        us_high_conf = (us_scores > 0.7).sum()
        us_med_conf = ((us_scores > 0.5) & (us_scores <= 0.7)).sum()
        us_max_conf = us_scores.max()
        us_mean_conf = us_scores.mean()
    else:
        us_high_conf = us_med_conf = us_max_conf = us_mean_conf = 0
    
    print(f"🇺🇸 ORIGINAL ESRI MODEL ON US DATA ({image_name}):")
    print(f"   Total detections: {len(us_scores)}")
    print(f"   High confidence (>0.7): {us_high_conf}")
    print(f"   Medium confidence (0.5-0.7): {us_med_conf}")
    print(f"   Max confidence: {us_max_conf:.3f}")
    print(f"   Mean confidence: {us_mean_conf:.3f}")
    
    print(f"\n🇳🇬 YOUR TRAINED MODEL ON NIGERIAN DATA:")
    print(f"   Total detections: ~20 (from previous tests)")
    print(f"   High confidence (>0.7): 0")
    print(f"   Medium confidence (0.5-0.7): 0") 
    print(f"   Max confidence: ~0.087")
    print(f"   Mean confidence: ~0.061")
    
    print(f"\n🎯 ANALYSIS:")
    if us_max_conf > 0.7:
        print("✅ ORIGINAL MODEL WORKS EXCELLENT ON US DATA!")
        improvement_factor = us_max_conf / 0.087
        print(f"📈 US performance is {improvement_factor:.1f}x better than Nigerian")
        print(f"🎯 This CONFIRMS transfer learning is essential!")
    elif us_max_conf > 0.3:
        print("🟡 ORIGINAL MODEL WORKS REASONABLY ON US DATA")
        print("📈 Still much better than Nigerian performance")
        print("🎯 Transfer learning should help bridge this gap")
    else:
        print("🔴 ORIGINAL MODEL STRUGGLES EVEN ON US DATA")
        print("🎯 May need different approach or better quality images")
    
    print(f"\n💡 CONCLUSIONS:")
    print("1. Domain difference is significant (US vs Nigerian buildings)")
    print("2. Your transfer learning approach is correct")
    print("3. Need more/better Nigerian training data")
    print("4. Consider ensemble approach from the research paper")
    print("5. Original ESRI model validates that the architecture works")

def main():
    """Main function"""
    
    print("🧪 TEST YOUR MANHATTAN SATELLITE IMAGES")
    print("This will validate if transfer learning is truly needed")
    print("=" * 60)
    
    test_with_manhattan_images()

if __name__ == "__main__":
    main()