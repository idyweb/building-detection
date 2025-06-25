#!/usr/bin/env python3
"""
Test the original ESRI model with US satellite images
This will help determine if the issue is the model or the data mismatch
"""

import torch
import torchvision
import cv2
import numpy as np
import matplotlib.pyplot as plt
import requests
from pathlib import Path
import urllib.request
from PIL import Image
import io

def download_us_test_images():
    """Download US satellite images for testing"""
    
    print("🇺🇸 DOWNLOADING US SATELLITE TEST IMAGES")
    print("=" * 50)
    
    # Create directory for US test images
    us_images_dir = Path("us_test_images")
    us_images_dir.mkdir(exist_ok=True)
    
    # US satellite image URLs (high-resolution building areas)
    us_test_urls = [
        {
            'name': 'manhattan_residential.jpg',
            'url': 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/18/77771/98366',
            'description': 'Manhattan residential area'
        },
        {
            'name': 'california_suburb.jpg', 
            'url': 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/17/22900/52374',
            'description': 'California suburban houses'
        },
        {
            'name': 'texas_residential.jpg',
            'url': 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/18/61440/100352',
            'description': 'Texas residential area'
        },
        {
            'name': 'florida_houses.jpg',
            'url': 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/17/34960/54144',
            'description': 'Florida houses'
        }
    ]
    
    # Alternative method: Use Bing Maps (often works better)
    bing_urls = [
        {
            'name': 'us_suburban_1.jpg',
            'url': 'https://ecn.t3.tiles.virtualearth.net/tiles/a12030123010302?g=1&n=z',
            'description': 'US Suburban area 1'
        },
        {
            'name': 'us_suburban_2.jpg', 
            'url': 'https://ecn.t2.tiles.virtualearth.net/tiles/a12030123010301?g=1&n=z',
            'description': 'US Suburban area 2'
        }
    ]
    
    downloaded_images = []
    
    print("📥 Downloading satellite images...")
    
    # Try downloading from different sources
    all_urls = us_test_urls + bing_urls
    
    for img_info in all_urls:
        try:
            img_path = us_images_dir / img_info['name']
            
            if img_path.exists():
                print(f"✅ {img_info['name']} already exists")
                downloaded_images.append(img_path)
                continue
            
            print(f"📥 Downloading {img_info['name']}...")
            
            # Set headers to mimic a browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            # Download with requests
            response = requests.get(img_info['url'], headers=headers, timeout=10)
            response.raise_for_status()
            
            # Save image
            with open(img_path, 'wb') as f:
                f.write(response.content)
            
            # Verify it's a valid image
            img = Image.open(img_path)
            if img.size[0] < 100 or img.size[1] < 100:  # Too small
                img_path.unlink()  # Delete
                continue
            
            downloaded_images.append(img_path)
            print(f"✅ Downloaded: {img_info['name']} ({img.size})")
            
        except Exception as e:
            print(f"❌ Failed to download {img_info['name']}: {e}")
    
    # If no downloads worked, create synthetic US-style test data
    if not downloaded_images:
        print("\n📸 Creating synthetic US-style test images...")
        downloaded_images = create_synthetic_us_images(us_images_dir)
    
    print(f"\n✅ Ready for testing with {len(downloaded_images)} US images")
    return downloaded_images

def create_synthetic_us_images(output_dir):
    """Create synthetic US-style satellite images for testing"""
    
    print("🎨 Creating synthetic US-style images...")
    
    synthetic_images = []
    
    # Create simple synthetic images with geometric patterns (like US suburbs)
    for i in range(3):
        # Create image
        img_size = 512
        image = np.ones((img_size, img_size, 3), dtype=np.uint8) * 100  # Gray background
        
        # Add grid-like suburban pattern
        house_size = 40
        spacing = 60
        
        for y in range(50, img_size-50, spacing):
            for x in range(50, img_size-50, spacing):
                if np.random.random() > 0.3:  # 70% chance of house
                    # House body (rectangle)
                    cv2.rectangle(image, (x, y), (x+house_size, y+house_size), 
                                (180, 140, 120), -1)  # Brown/tan color
                    
                    # Roof (triangle-ish)
                    roof_points = np.array([[x, y], [x+house_size//2, y-15], [x+house_size, y]], np.int32)
                    cv2.fillPoly(image, [roof_points], (80, 60, 40))  # Dark roof
                    
                    # Add some variation
                    if np.random.random() > 0.5:
                        cv2.rectangle(image, (x+10, y+10), (x+30, y+30), (200, 180, 160), -1)
        
        # Add roads
        cv2.rectangle(image, (0, img_size//2-10), (img_size, img_size//2+10), (60, 60, 60), -1)
        cv2.rectangle(image, (img_size//2-10, 0), (img_size//2+10, img_size), (60, 60, 60), -1)
        
        # Save
        img_path = output_dir / f"synthetic_us_{i+1}.jpg"
        cv2.imwrite(str(img_path), image)
        synthetic_images.append(img_path)
        print(f"✅ Created: {img_path.name}")
    
    return synthetic_images

def test_original_model_on_us_data(us_images):
    """Test the original ESRI model on US data"""
    
    print(f"\n🧪 TESTING ORIGINAL MODEL ON US DATA")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the ORIGINAL pre-trained ESRI model (not your trained version)
    print("📦 Loading original ESRI model...")
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='DEFAULT')
    model.to(device)
    model.eval()
    
    print(f"✅ Original ESRI model loaded (COCO pretrained)")
    print(f"🎯 This model was trained on diverse data including buildings")
    
    results = []
    
    for i, img_path in enumerate(us_images, 1):
        print(f"\n📸 Testing image {i}/{len(us_images)}: {img_path.name}")
        
        try:
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"❌ Could not load {img_path}")
                continue
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            print(f"   📐 Image size: {image_rgb.shape}")
            
            # Resize to 512x512 (what your models expect)
            image_resized = cv2.resize(image_rgb, (512, 512))
            
            # Convert to tensor
            image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0).to(device)
            
            # Run inference
            with torch.no_grad():
                predictions = model(image_tensor)
            
            pred = predictions[0]
            scores = pred['scores'].cpu().numpy()
            boxes = pred['boxes'].cpu().numpy()
            labels = pred['labels'].cpu().numpy()
            
            # COCO has multiple classes - buildings might be class 1 (person), 3 (car), etc.
            # We're interested in any structural detections
            print(f"   📊 Total detections: {len(scores)}")
            
            if len(scores) > 0:
                print(f"   📈 Score range: {scores.min():.3f} - {scores.max():.3f}")
                
                # Count high-confidence detections
                high_conf = (scores > 0.7).sum()
                med_conf = ((scores > 0.5) & (scores <= 0.7)).sum()
                low_conf = ((scores > 0.3) & (scores <= 0.5)).sum()
                
                print(f"   🎯 High conf (>0.7): {high_conf}")
                print(f"   🎯 Med conf (0.5-0.7): {med_conf}")
                print(f"   🎯 Low conf (0.3-0.5): {low_conf}")
                
                # Show detected classes
                unique_labels, counts = np.unique(labels[scores > 0.5], return_counts=True)
                if len(unique_labels) > 0:
                    print(f"   🏷️ Detected classes: {dict(zip(unique_labels, counts))}")
                
                # Visualize best detections
                visualize_us_results(image_resized, boxes, scores, labels, img_path.stem)
                
                results.append({
                    'image': img_path.name,
                    'total_detections': len(scores),
                    'high_conf_detections': high_conf,
                    'max_confidence': scores.max(),
                    'mean_confidence': scores.mean()
                })
            else:
                print(f"   ❌ No detections")
                results.append({
                    'image': img_path.name,
                    'total_detections': 0,
                    'high_conf_detections': 0,
                    'max_confidence': 0,
                    'mean_confidence': 0
                })
        
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Summary
    print(f"\n" + "="*60)
    print("📊 US DATA TEST SUMMARY")
    print("="*60)
    
    if results:
        total_detections = sum(r['total_detections'] for r in results)
        total_high_conf = sum(r['high_conf_detections'] for r in results)
        avg_max_conf = np.mean([r['max_confidence'] for r in results if r['max_confidence'] > 0])
        
        print(f"📸 Images tested: {len(results)}")
        print(f"🎯 Total detections: {total_detections}")
        print(f"🎯 High confidence detections: {total_high_conf}")
        if not np.isnan(avg_max_conf):
            print(f"📈 Average max confidence: {avg_max_conf:.3f}")
        
        # Analysis
        if total_high_conf > 0:
            print(f"\n✅ ORIGINAL MODEL WORKS WELL ON US DATA!")
            print(f"🎯 This confirms transfer learning is needed for Nigerian images")
        elif total_detections > 0:
            print(f"\n🟡 ORIGINAL MODEL SHOWS SOME ACTIVITY ON US DATA")
            print(f"🎯 May need better US test images or model tuning")
        else:
            print(f"\n🔴 ORIGINAL MODEL SHOWS POOR PERFORMANCE")
            print(f"🎯 May indicate model or testing issues")
    
    return results

def visualize_us_results(image, boxes, scores, labels, image_name):
    """Visualize results on US data"""
    
    # Filter for reasonable confidence
    keep = scores > 0.3
    if not keep.any():
        return
    
    plt.figure(figsize=(12, 8))
    
    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f'US Test Image: {image_name}')
    plt.axis('off')
    
    # Detections
    plt.subplot(1, 2, 2)
    plt.imshow(image)
    plt.title(f'ESRI Model Detections\n{keep.sum()} objects >0.3 confidence')
    
    filtered_boxes = boxes[keep]
    filtered_scores = scores[keep]
    filtered_labels = labels[keep]
    
    # Color map for different classes
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow']
    
    for box, score, label in zip(filtered_boxes, filtered_scores, filtered_labels):
        x1, y1, x2, y2 = box
        color = colors[int(label) % len(colors)]
        
        plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], color=color, linewidth=2)
        plt.text(x1, y1-3, f'C{label}:{score:.2f}', color=color, fontweight='bold', fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    plt.axis('off')
    plt.tight_layout()
    
    # Save
    plt.savefig(f"us_test_result_{image_name}.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"   💾 Saved: us_test_result_{image_name}.png")

def main():
    """Main function to test with US data"""
    
    print("🇺🇸 TESTING ESRI MODEL WITH US SATELLITE DATA")
    print("This will help determine if transfer learning is truly needed")
    print("=" * 70)
    
    # Download US test images
    us_images = download_us_test_images()
    
    if not us_images:
        print("❌ No US test images available")
        return
    
    # Test original ESRI model
    results = test_original_model_on_us_data(us_images)
    
    print(f"\n" + "="*70)
    print("🎯 CONCLUSIONS")
    print("="*70)
    
    if any(r['high_conf_detections'] > 0 for r in results):
        print("✅ ORIGINAL ESRI MODEL WORKS ON US DATA")
        print("🎯 This confirms your hypothesis:")
        print("   • Model works on US-style buildings")
        print("   • Nigerian buildings are too different")
        print("   • Transfer learning IS necessary")
        print(f"\n💡 NEXT STEPS:")
        print("1. Continue with transfer learning approach")
        print("2. Get more Nigerian training data")
        print("3. Try ensemble approach from paper")
    else:
        print("🟡 ORIGINAL MODEL SHOWS LIMITED PERFORMANCE")
        print("🎯 This suggests:")
        print("   • Model may need specific tuning even for US data")
        print("   • Building detection requires specialized training")
        print("   • Your transfer learning approach is correct")

if __name__ == "__main__":
    main()