"""
Building Detection FastAPI Application using ESRI Mask R-CNN Model
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchvision
import cv2
import numpy as np
import json
import uuid
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Optional, List, Dict
import geopandas as gpd
from shapely.geometry import Polygon
import rasterio
from rasterio.transform import from_bounds
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Building Detection API",
    description="Detect buildings from satellite imagery using ESRI Mask R-CNN model",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
device = None
jobs = {}  # Store job status

class BuildingDetector:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        self.model = None
        self.config = None
        self.load_model()
    
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
            model_data = torch.load(pth_file, map_location=self.device)
            logger.info("Model weights loaded successfully!")
            return model_data
        except Exception as e:
            logger.error(f"Error loading model weights: {e}")
            return None
    
    def create_maskrcnn_model(self, num_classes=2):
        """Create a Mask R-CNN model architecture"""
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
        
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
    
    def load_model(self):
        """Load the complete model"""
        try:
            self.config = self.load_model_config()
            weights = self.load_pytorch_weights()
            
            if weights is None:
                raise Exception("Failed to load model weights")
            
            # Create model architecture
            self.model = self.create_maskrcnn_model(num_classes=2)
            
            # Load weights
            model_dict = self.model.state_dict()
            filtered_weights = {}
            
            for key, value in weights.items():
                if key in model_dict and model_dict[key].shape == value.shape:
                    filtered_weights[key] = value
            
            model_dict.update(filtered_weights)
            self.model.load_state_dict(model_dict, strict=False)
            
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded successfully! Loaded {len(filtered_weights)} weight tensors")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def preprocess_image(self, image_array, target_size=(512, 512)):
        """Preprocess image to match ESRI model requirements"""
        # Ensure RGB format
        if len(image_array.shape) == 3:
            h, w, c = image_array.shape
            if c > 3:
                image_array = image_array[:, :, :3]  # Keep only RGB
        
        # Resize to target size
        image_resized = cv2.resize(image_array, target_size)
        
        # Convert to tensor format (C, H, W)
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float()
        
        # Normalize to [0, 1]
        image_tensor = image_tensor / 255.0
        
        return image_tensor.unsqueeze(0)  # Add batch dimension
    
    def postprocess_predictions(self, predictions, confidence_threshold=0.5, 
                              image_shape=None, georef_info=None):
        """Convert predictions to polygons"""
        if not predictions or len(predictions) == 0:
            return []
        
        pred = predictions[0]
        
        # Filter by confidence
        keep = pred['scores'] > confidence_threshold
        
        if not keep.any():
            return []
        
        boxes = pred['boxes'][keep].cpu().numpy()
        scores = pred['scores'][keep].cpu().numpy()
        masks = pred['masks'][keep].cpu().numpy()
        
        polygons = []
        
        for i, (box, score, mask) in enumerate(zip(boxes, scores, masks)):
            # Convert mask to polygon
            mask_binary = (mask[0] > 0.5).astype(np.uint8)
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if len(contour) >= 3:  # Valid polygon needs at least 3 points
                    # Simplify contour
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    simplified = cv2.approxPolyDP(contour, epsilon, True)
                    
                    if len(simplified) >= 3:
                        coords = simplified.reshape(-1, 2)
                        
                        # Convert to geographic coordinates if georef info provided
                        if georef_info and image_shape:
                            coords = self.pixel_to_geo(coords, image_shape, georef_info)
                        
                        try:
                            polygon = Polygon(coords)
                            if polygon.is_valid and polygon.area > 0:
                                polygons.append({
                                    'geometry': polygon,
                                    'confidence': float(score),
                                    'building_id': i + 1
                                })
                        except:
                            continue
        
        return polygons
    
    def pixel_to_geo(self, pixel_coords, image_shape, georef_info):
        """Convert pixel coordinates to geographic coordinates"""
        # This is a simplified conversion - you'd need actual georeference info
        # from the satellite image metadata
        return pixel_coords  # Placeholder
    
    def predict(self, image_array, confidence_threshold=0.5, georef_info=None):
        """Run building detection on image"""
        try:
            # Preprocess
            image_tensor = self.preprocess_image(image_array)
            image_tensor = image_tensor.to(self.device)
            
            # Inference
            with torch.no_grad():
                predictions = self.model(image_tensor)
            
            # Postprocess
            polygons = self.postprocess_predictions(
                predictions, 
                confidence_threshold, 
                image_array.shape, 
                georef_info
            )
            
            return {
                'success': True,
                'detections': len(polygons),
                'polygons': polygons
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'detections': 0,
                'polygons': []
            }

def create_shapefile(polygons: List[Dict], output_path: str, crs: str = 'EPSG:4326'):
    """Create shapefile from polygons"""
    if not polygons:
        raise ValueError("No polygons to save")
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame([
        {
            'geometry': poly['geometry'],
            'confidence': poly['confidence'],
            'building_id': poly['building_id']
        }
        for poly in polygons
    ], crs=crs)
    
    # Save as shapefile
    gdf.to_file(output_path, driver='ESRI Shapefile')
    
    # Create zip file with all shapefile components
    zip_path = output_path.replace('.shp', '.zip')
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        base_name = Path(output_path).stem
        extensions = ['.shp', '.shx', '.dbf', '.prj', '.cpg']
        
        for ext in extensions:
            file_path = output_path.replace('.shp', ext)
            if os.path.exists(file_path):
                zipf.write(file_path, base_name + ext)
                os.remove(file_path)  # Clean up individual files
    
    return zip_path

# Initialize model on startup
@app.on_event("startup")
async def startup_event():
    global model, device
    try:
        model = BuildingDetector("extracted_model")
        device = model.device
        logger.info("Model initialized successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        model = None

@app.get("/")
async def root():
    return {
        "message": "Building Detection API",
        "model_loaded": model is not None,
        "device": str(device) if device else "Not available"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None
    }

@app.post("/detect")
async def detect_buildings(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    confidence_threshold: float = 0.5,
    output_format: str = "shapefile"  # "shapefile" or "geojson"
):
    """Detect buildings in uploaded satellite image"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "processing", "progress": 0}
    
    # Read file contents immediately
    try:
        file_contents = await file.read()
        file_name = file.filename
    except Exception as e:
        jobs[job_id] = {"status": "error", "error": f"Failed to read file: {str(e)}"}
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")
    
    # Start background processing with file contents
    background_tasks.add_task(
        process_image, 
        job_id, 
        file_contents,
        file_name,
        confidence_threshold, 
        output_format
    )
    
    return {
        "job_id": job_id,
        "status": "accepted",
        "message": "Processing started"
    }

async def process_image(job_id: str, file_contents: bytes, file_name: str, confidence_threshold: float, output_format: str):
    """Background task for image processing"""
    try:
        jobs[job_id]["status"] = "reading_image"
        jobs[job_id]["progress"] = 10
        
        # Decode image from bytes
        nparr = np.frombuffer(file_contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = "Could not decode image"
            return
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        jobs[job_id]["status"] = "detecting"
        jobs[job_id]["progress"] = 30
        
        # Run detection
        result = model.predict(image, confidence_threshold)
        
        if not result['success']:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = result.get('error', 'Detection failed')
            return
        
        jobs[job_id]["progress"] = 70
        jobs[job_id]["detections"] = result['detections']
        
        if result['detections'] > 0:
            jobs[job_id]["status"] = "creating_output"
            jobs[job_id]["progress"] = 80
            
            # Create output file
            output_dir = Path("outputs")
            output_dir.mkdir(exist_ok=True)
            
            if output_format == "shapefile":
                output_path = output_dir / f"{job_id}.shp"
                zip_path = create_shapefile(result['polygons'], str(output_path))
                jobs[job_id]["output_file"] = zip_path
            else:  # geojson
                output_path = output_dir / f"{job_id}.geojson"
                gdf = gpd.GeoDataFrame([
                    {
                        'geometry': poly['geometry'],
                        'confidence': poly['confidence'],
                        'building_id': poly['building_id']
                    }
                    for poly in result['polygons']
                ], crs='EPSG:4326')
                gdf.to_file(output_path, driver='GeoJSON')
                jobs[job_id]["output_file"] = str(output_path)
        else:
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["progress"] = 100
            jobs[job_id]["message"] = "No buildings detected"
            return
        
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 100
        
    except Exception as e:
        logger.error(f"Processing failed for job {job_id}: {e}")
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)

@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Get processing status"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return jobs[job_id]

@app.get("/download/{job_id}")
async def download_results(job_id: str):
    """Download detection results"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")
    
    if "output_file" not in job:
        raise HTTPException(status_code=404, detail="No output file available")
    
    output_file = job["output_file"]
    
    if not os.path.exists(output_file):
        raise HTTPException(status_code=404, detail="Output file not found")
    
    filename = os.path.basename(output_file)
    return FileResponse(
        output_file,
        media_type='application/octet-stream',
        filename=filename
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)