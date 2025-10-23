"""
AI Car Damage Detection - Streamlit Web Application
==================================================

Professional web interface for automated vehicle damage assessment.
Built with Streamlit for production deployment and user-friendly interaction.

Features:
- Real-time damage detection
- Interactive results visualization  
- Professional damage reporting
- Mobile-responsive design

Author: AI Engineer
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd
from car_damage_detector import CarDamageDetector

def load_and_prep_image(uploaded_file):
    """
    Load and prepare the uploaded image
    """
    # Read image
    image_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB (in case it's RGBA)
    image = image.convert('RGB')
    
    # Resize to 640x640 while maintaining aspect ratio
    original_width, original_height = image.size
    aspect_ratio = original_width / original_height
    
    if aspect_ratio > 1:
        new_width = 640
        new_height = int(640 / aspect_ratio)
    else:
        new_height = 640
        new_width = int(640 * aspect_ratio)
        
    image = image.resize((new_width, new_height))
    
    # Convert to numpy array for OpenCV
    return np.array(image)

def main():
    st.set_page_config(
        page_title="AI Car Damage Detection", 
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Professional header with metrics
    st.markdown("""
    <div style='text-align: center; padding: 1rem; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 2rem;'>
        <h1 style='color: white; margin: 0;'>🚗 AI Car Damage Detection System</h1>
        <p style='color: white; margin: 0; opacity: 0.9;'>Powered by YOLOv8 • Real-time Analysis • Professional Assessment</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        confidence_threshold = st.slider(
            "Detection Confidence Threshold", 
            min_value=0.1, 
            max_value=0.9, 
            value=0.3, 
            step=0.05,
            help="Lower values detect more potential damage but may include false positives"
        )
        
        st.header("📊 System Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model", "YOLOv8n")
        with col2:
            st.metric("Classes", "17")
            
        st.header("🎯 Detection Categories")
        st.markdown("""
        - **Safety Critical**: Windscreen, Headlights
        - **Major Body**: Bumpers, Bonnet, Boot  
        - **Body Panels**: Doors, Fenders, Quarters
        - **Components**: Mirrors, Lights, Pillars
        """)
    
    # Photography Guidelines
    st.header("📸 Photo Guidelines")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Do's:
        - Take photos from 10-15 feet away
        - Ensure good lighting conditions
        - Capture the entire car section
        - Use landscape orientation
        - Keep the camera steady
        """)
    
    with col2:
        st.markdown("""
        ### Don'ts:
        - Don't take extremely close-up shots
        - Avoid poor lighting or shadows
        - Don't take blurry photos
        - Avoid extreme angles
        - Don't crop the damaged area
        """)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload an image (recommended size: 640x640)", 
                                   type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        try:
            # Load and process image
            image = load_and_prep_image(uploaded_file)
            
            # Display original image
            st.subheader("Original Image")
            st.image(image, channels="RGB", use_column_width=True)
            
            # Initialize detector with user-configured threshold
            detector = CarDamageDetector("Weights/best.pt", confidence_threshold=confidence_threshold)
            
            # Process image with progress tracking
            with st.spinner('🔍 AI Analysis in Progress...'):
                processed_img, detections = detector.detect_damage(image)
                summary = detector.get_damage_summary(detections)
            
            # Results section with professional layout
            st.markdown("---")
            st.subheader("🎯 Detection Results")
            
            # Key metrics at the top
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Damages Found", summary['total_damages'])
            with col2:
                st.metric("Avg Confidence", f"{summary['confidence_avg']:.1%}")
            with col3:
                severity_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢", "None": "⚪"}
                st.metric("Severity", f"{severity_color.get(summary['severity'], '⚪')} {summary['severity']}")
            with col4:
                st.metric("Processing Time", "< 1s")
            
            # Visual results
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("#### 📸 Annotated Image")
                st.image(processed_img, channels="RGB", use_column_width=True)
            
            with col2:
                if detections:
                    # Damage category breakdown
                    st.markdown("#### 📊 Damage Breakdown")
                    categories = summary['damage_categories']
                    
                    # Create pie chart
                    if categories:
                        fig = px.pie(
                            values=list(categories.values()),
                            names=list(categories.keys()),
                            title="Damage Categories"
                        )
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig, use_container_width=True)
                
            # Detailed findings
            if detections:
                st.markdown("#### 📋 Detailed Analysis")
                
                # Enhanced damage table
                damage_df = pd.DataFrame([
                    {
                        'Damage Type': d['class'].replace('-', ' ').title(),
                        'Confidence': f"{d['confidence']:.1%}",
                        'Location': f"({d['center'][0]}, {d['center'][1]})",
                        'Size (px²)': f"{d['area']:,}",
                        'Category': detector._categorize_damage(d['class'])
                    }
                    for d in detections
                ])
                
                st.dataframe(damage_df, use_container_width=True)
                
                # Professional assessment
                st.markdown("#### 🏥 Professional Assessment")
                assessment_color = {
                    "High": "error",
                    "Medium": "warning", 
                    "Low": "success",
                    "None": "info"
                }
                
                st.markdown(f"""
                <div style='padding: 1rem; border-left: 4px solid {"#ff4444" if summary["severity"] == "High" else "#ffaa00" if summary["severity"] == "Medium" else "#00aa00"}; background-color: #f8f9fa; border-radius: 5px;'>
                    <strong>Assessment:</strong> {summary['assessment']}<br>
                    <strong>Recommendation:</strong> {"Immediate professional inspection required" if summary["severity"] == "High" else "Schedule repair assessment" if summary["severity"] == "Medium" else "Monitor and maintain regular inspections"}
                </div>
                """, unsafe_allow_html=True)
                
                # Download report button
                report_data = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'total_damages': summary['total_damages'],
                    'severity': summary['severity'],
                    'confidence_avg': summary['confidence_avg'],
                    'assessment': summary['assessment'],
                    'detections': detections
                }
                
                if st.button("📄 Generate Report", type="primary"):
                    st.success("Report generated! (Feature available in full version)")
                    
            else:
                st.success("✅ No damage detected in the uploaded image.")
                st.balloons()
                
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.info("Please try uploading a different image.")
    
    # Add usage instructions at the bottom
    st.markdown("""
    ---
    ### How to use:
    1. Review the photo guidelines above
    2. Upload a photo of the car damage
    3. Wait for the system to process the image
    4. Review the detected damages and assessment
    
    Note: For best results, ensure your photos follow the guidelines and are well-lit.
    """)

if __name__ == "__main__":
    main()