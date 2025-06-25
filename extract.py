"""
Extract and load PyTorch model from ESRI DLPK for standalone use
"""

import torch
import torchvision
import json
import numpy as np
import cv2
from pathlib import Path

class ESRIModelExtractor:
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_model_config(self):
        """Load the EMD configuration file"""
        emd_file = self.model_path / "usa_building_footprints.emd"
        with open(emd_file, 'r') as f:
            config = json.load(f)
        return config
    
    def load_pytorch_weights(self):
        """Load the PyTorch model weights"""
        pth_file = self.model_path / "usa_building_footprints.pth"
        
        try:
            # Try loading the model state dict
            model_data = torch.load(pth_file, map_location=self.device)
            print("Model loaded successfully!")
            print(f"Model data type: {type(model_data)}")
            
            if isinstance(model_data, dict):
                print("Available keys in model:")
                for key in model_data.keys():
                    print(f"  - {key}")
                    
                # Look for model weights
                if 'model' in model_data:
                    return model_data['model']
                elif 'state_dict' in model_data:
                    return model_data['state_dict']
                else:
                    return model_data
            else:
                return model_data
                
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def create_maskrcnn_model(self, num_classes=2):
        """Create a Mask R-CNN model architecture"""
        # Using torchvision's pre-built Mask R-CNN
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)
        
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
    
    def try_load_esri_model(self):
        """Attempt to load the ESRI model structure"""
        config = self.load_model_config()
        weights = self.load_pytorch_weights()
        
        if weights is None:
            return None, config
            
        # Create model architecture
        model = self.create_maskrcnn_model(num_classes=2)  # background + building
        
        try:
            # Try to load the weights
            if isinstance(weights, dict):
                # Filter weights to match our model
                model_dict = model.state_dict()
                filtered_weights = {}
                
                for key, value in weights.items():
                    if key in model_dict and model_dict[key].shape == value.shape:
                        filtered_weights[key] = value
                        
                model_dict.update(filtered_weights)
                model.load_state_dict(model_dict, strict=False)
                print(f"Loaded {len(filtered_weights)} weight tensors")
            
            model.to(self.device)
            model.eval()
            return model, config
            
        except Exception as e:
            print(f"Error loading weights into model: {e}")
            return None, config

def preprocess_image(image_array, target_size=(512, 512)):
    """Preprocess image to match ESRI model requirements"""
    
    # Resize image
    if len(image_array.shape) == 3:
        h, w, c = image_array.shape
        if c > 3:
            image_array = image_array[:, :, :3]  # Keep only RGB
    
    # Resize to target size
    image_resized = cv2.resize(image_array, target_size)
    
    # Convert to tensor format (C, H, W)
    image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float()
    
    # Normalize (typical ImageNet normalization)
    image_tensor = image_tensor / 255.0
    
    return image_tensor.unsqueeze(0)  # Add batch dimension

def run_inference(model, image_tensor, confidence_threshold=0.5):
    """Run inference on preprocessed image"""
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        predictions = model(image_tensor)
    
    # Filter predictions by confidence
    filtered_predictions = []
    
    for pred in predictions:
        # Filter by score
        keep = pred['scores'] > confidence_threshold
        
        filtered_pred = {
            'boxes': pred['boxes'][keep],
            'scores': pred['scores'][keep],
            'labels': pred['labels'][keep],
            'masks': pred['masks'][keep] if 'masks' in pred else None
        }
        filtered_predictions.append(filtered_pred)
    
    return filtered_predictions

# Example usage
if __name__ == "__main__":
    # Initialize extractor
    extractor = ESRIModelExtractor("extracted_model")
    
    # Try to load the model
    model, config = extractor.try_load_esri_model()
    
    if model is not None:
        print("Model loaded successfully!")
        print(f"Model configuration: {config}")
        
        # Test with dummy image
        dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        image_tensor = preprocess_image(dummy_image)
        
        try:
            predictions = run_inference(model, image_tensor)
            print(f"Inference successful! Found {len(predictions[0]['boxes'])} detections")
        except Exception as e:
            print(f"Inference failed: {e}")
    else:
        print("Failed to load model. You may need to use alternative approach.")
        print(f"Model configuration: {config}")