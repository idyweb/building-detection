#!/usr/bin/env python3
"""
Nigerian Building Detection - Real API Integration
Streamlit frontend connected to your actual trained model
"""

import streamlit as st
import requests
import json
import time
import io
import zipfile
from PIL import Image
import numpy as np
import base64
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import io

# Image.MAX_IMAGE_PIXELS = None

# Your API configuration
API_BASE_URL = "http://35.227.159.142:8000"

# Set page config
st.set_page_config(
    page_title="Nigerian Building Detection - Live AI Service",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

def add_custom_css():
    """Add custom CSS for better styling"""
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-container {
        background: linear-gradient(90deg, #f0f2f6, #ffffff);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-message {
        background: linear-gradient(90deg, #d4edda, #ffffff);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .processing-animation {
        text-align: center;
        font-size: 1.2rem;
        color: #1f77b4;
        animation: pulse 2s infinite;
    }
    .download-section {
        background: linear-gradient(90deg, #e3f2fd, #ffffff);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2196f3;
        margin: 1rem 0;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    </style>
    """, unsafe_allow_html=True)

def check_api_health():
    """Check if the API is available"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False
    
def draw_bounding_boxes(image, coordinates_data):
    """Draw bounding boxes on image"""
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    
    colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF']
    
    for i, building in enumerate(coordinates_data):
        color = colors[i % len(colors)]
        bbox = building['bbox']
        
        # Draw bounding box
        draw.rectangle(bbox, outline=color, width=3)
        
        # Draw label
        label = f"{building['building_id']}\n{building['area_sqm']:.0f}m²"
        
        # Position text
        text_x, text_y = bbox[0], bbox[1] - 30 if bbox[1] > 30 else bbox[1] + 5
        
        # Draw text with background
        try:
            font = ImageFont.load_default()
            draw.text((text_x, text_y), label, fill=color, font=font)
        except:
            draw.text((text_x, text_y), label, fill=color)
    
    return draw_image

def upload_image_to_api(image_file):
    """Upload image to your API and get job ID"""
    try:
        # Reset file pointer
        image_file.seek(0)
        
        files = {"file": (image_file.name, image_file, image_file.type)}
        response = requests.post(f"{API_BASE_URL}/detect-buildings/", files=files)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Upload failed: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Upload error: {str(e)}")
        return None

def check_job_status(job_id):
    """Check processing status"""
    try:
        response = requests.get(f"{API_BASE_URL}/status/{job_id}")
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def download_results(job_id, file_type="shapefile"):
    """Download results from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/download/{job_id}/{file_type}")
        if response.status_code == 200:
            return response.content
        return None
    except:
        return None

def display_real_time_progress(job_id):
    """Display real-time processing progress"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    while True:
        status = check_job_status(job_id)
        
        if status is None:
            status_text.error("❌ Could not check job status")
            break
        
        progress = status.get('progress', 0)
        message = status.get('message', 'Processing...')
        job_status = status.get('status', 'unknown')
        
        progress_bar.progress(progress / 100)
        status_text.text(f"🤖 {message}")
        
        if job_status == 'completed':
            status_text.success("✅ Processing completed!")
            return status
        elif job_status == 'failed':
            status_text.error(f"❌ Processing failed: {message}")
            return status
        
        time.sleep(2)  # Check every 2 seconds
    
    return None

def main():
    """Main app"""
    
    add_custom_css()
    
    # Header
    st.markdown('<h1 class="main-header">🏠 Nigerian Building Detection - Live AI Service</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Real AI-Powered Building Detection with Your Trained Model</p>', unsafe_allow_html=True)
    
    # Check API status
    api_status = check_api_health()
    
    if api_status:
        st.success("✅ AI Service is online and ready!")
    else:
        st.error("❌ AI Service is offline. Please check your API server.")
        st.info(f"Expected API URL: {API_BASE_URL}")
        st.stop()
    
    # Sidebar
    st.sidebar.header("🎛️ Service Settings")
    st.sidebar.success("🔗 Connected to Live AI Model")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Model Info")
    st.sidebar.info(f"""
    **Architecture:** U-Net with ResNet Backbone
    **Training Data:** Lakowe Area (2,002 buildings)
    **Performance:** 
    - IoU: 0.54
    - Precision: 0.77
    - Recall: 0.66
    - F1-Score: 0.71
    **Status:** Production Ready ✅
    """)
    
    st.sidebar.markdown("### 🗺️ Output Formats")
    st.sidebar.success("""
    ✅ Shapefile (.shp + components)
    ✅ GeoJSON Format
    ✅ Coordinate System: UTM 31N
    ✅ Building Attributes
    ✅ ArcGIS Compatible
    """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Aerial/Satellite Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['png', 'jpg', 'jpeg', 'jp2', 'tif', 'tiff'],
            help="Upload aerial or satellite imagery for building detection"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Show file info
                st.info(f"""
                **File:** {uploaded_file.name}
                **Size:** {uploaded_file.size / (1024*1024):.1f} MB
                **Type:** {uploaded_file.type}
                """)
                
            except Exception as e:
                st.error(f"Error loading image: {e}")
                st.stop()
            
            # Process button
            if st.button("🚀 Detect Buildings with AI", type="primary", use_container_width=True):
                with col2:
                    st.subheader("🤖 Live AI Processing")
                    
                    # Upload to API
                    with st.spinner("Uploading image to AI service..."):
                        upload_result = upload_image_to_api(uploaded_file)
                    
                    if upload_result is None:
                        st.error("❌ Failed to upload image to AI service")
                        st.stop()
                    
                    job_id = upload_result.get('job_id')
                    st.success(f"✅ Image uploaded! Job ID: `{job_id}`")
                    
                    # Real-time progress monitoring
                    st.markdown("### 🔄 Processing Progress")
                    final_status = display_real_time_progress(job_id)
                    
                    if final_status and final_status['status'] == 'completed':
                        results = final_status.get('results', {})
                        
                        # Display results
                        st.markdown("### 📊 Detection Results")
                        
                        # Metrics
                        col2a, col2b, col2c = st.columns(3)
                        
                        with col2a:
                            buildings_count = results.get('buildings_detected', 0)
                            st.metric("🏠 Buildings Detected", buildings_count)
                        
                        with col2b:
                            processing_time = results.get('processing_time_seconds', 0)
                            st.metric("⏱️ Processing Time", f"{processing_time:.1f}s")
                        
                        with col2c:
                            total_area = results.get('total_area_sqm', 0)
                            st.metric("📐 Total Area", f"{total_area:.0f} m²")
                            
                        coordinates_data = results.get('building_coordinates', [])
                        if coordinates_data:
                            st.markdown("### 🎯 Detection Results")
                            
                            col_orig, col_detect = st.columns(2)
                            
                            with col_orig:
                                st.image(image, caption="📷 Original Image", use_column_width=True)
                            
                            with col_detect:
                                detection_image = draw_bounding_boxes(image, coordinates_data)
                                st.image(detection_image, caption=f"🎯 Detected Buildings ({buildings_count})", use_column_width=True)
                                                
                        # Download section
                        st.markdown('<div class="download-section">', unsafe_allow_html=True)
                        st.markdown("### 🗺️ **Download Results**")
                        
                        col_dl1, col_dl2 = st.columns(2)
                        
                        with col_dl1:
                            # Download shapefile
                            shapefile_data = download_results(job_id, "shapefile")
                            if shapefile_data:
                                st.download_button(
                                    label="📥 Download Shapefile Package",
                                    data=shapefile_data,
                                    file_name=f"buildings_{job_id[:8]}.zip",
                                    mime="application/zip",
                                    help="Complete shapefile ready for ArcGIS",
                                    use_container_width=True
                                )
                        
                        with col_dl2:
                            # Download JSON results
                            json_data = download_results(job_id, "json")
                            if json_data:
                                st.download_button(
                                    label="📥 Download JSON Results",
                                    data=json_data,
                                    file_name=f"results_{job_id[:8]}.json",
                                    mime="application/json",
                                    help="Detection results in JSON format",
                                    use_container_width=True
                                )
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Success message
                        st.success(f"✅ Successfully processed! Detected {buildings_count} buildings in {processing_time:.1f} seconds")
                        
                        # Technical details
                        with st.expander("🔬 Technical Details"):
                            st.json(results)
                        
                        # ArcGIS workflow
                        with st.expander("🗺️ ArcGIS Import Instructions"):
                            st.markdown("""
                            **Import into ArcGIS Pro:**
                            
                            1. **Download** the shapefile package above
                            2. **Extract** the ZIP file to your project folder
                            3. **Open ArcGIS Pro** and create/open project
                            4. **Add Data** → Navigate to extracted folder
                            5. **Select** the .shp file and import
                            6. **Projection** will be auto-detected (UTM Zone 31N)
                            7. **Symbolize** buildings by area or confidence
                            
                            **Attribute Fields:**
                            - `bldg_id`: Unique building identifier
                            - `area_sqm`: Building footprint area (m²)
                            - `perim_m`: Building perimeter (m)
                            - `confidence`: AI detection confidence
                            """)
                    
                    elif final_status and final_status['status'] == 'failed':
                        st.error("❌ Processing failed. Please try with a different image.")
                        error_msg = final_status.get('message', 'Unknown error')
                        st.error(f"Error details: {error_msg}")
        
        else:
            with col2:
                st.info("👆 Upload an image to start building detection")
                
                # Example images section
                st.markdown("### 🖼️ Example Test Images")
                st.markdown("""
                **Best Results Expected:**
                - ✅ Urban areas in Nigeria
                - ✅ High-resolution satellite imagery
                - ✅ Clear building boundaries
                - ✅ .jp2, .tiff, or .jpg formats
                
                **Supported Formats:**
                - JPEG 2000 (.jp2)
                - GeoTIFF (.tif, .tiff) 
                - JPEG (.jpg, .jpeg)
                - PNG (.png)
                """)
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <h4>🚀 Live Production Service</h4>
        <p><strong>API Endpoint:</strong> <code>{API_BASE_URL}</code></p>
        <p>Powered by your trained U-Net model with 77% precision</p>
        <p><strong>Ready for:</strong> Commercial Use • Surveyor Services • GIS Workflows</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
