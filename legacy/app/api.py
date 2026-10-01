"""
RESTful API for Car Damage Detection System
==========================================

FastAPI-based REST API for integrating damage detection
into enterprise systems and third-party applications.

Author: AI Engineer
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from PIL import Image
import io
import base64
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import uvicorn

from car_damage_detector import CarDamageDetector
from utils import ImageProcessor, ReportGenerator, ValidationUtils
from config import get_config

# Initialize configuration and logging
config = get_config()
logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Car Damage Detection API",
    description="Professional REST API for automated vehicle damage assessment using YOLOv8",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for web integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global detector instance
detector: Optional[CarDamageDetector] = None

@app.on_event("startup")
async def startup_event():
    """Initialize the damage detection model on startup"""
    global detector
    try:
        detector = CarDamageDetector(config.MODEL_PATH)
        logger.info("Car damage detection model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

@app.get("/")
async def root():
    """API root endpoint with system information"""
    return {
        "message": "AI Car Damage Detection API",
        "version": "1.0.0",
        "status": "operational",
        "model": "YOLOv8n",
        "endpoints": {
            "detect": "/detect",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy" if detector is not None else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": detector is not None
    }

@app.post("/detect")
async def detect_damage(
    file: UploadFile = File(...),
    confidence_threshold: float = 0.3,
    return_image: bool = False
):
    """
    Detect damage in uploaded vehicle image
    
    Args:
        file: Image file (JPG, PNG, etc.)
        confidence_threshold: Detection confidence threshold (0.0-1.0)
        return_image: Whether to return annotated image as base64
        
    Returns:
        JSON response with detection results
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate confidence threshold
    if not ValidationUtils.validate_confidence_threshold(confidence_threshold):
        raise HTTPException(
            status_code=400, 
            detail="Confidence threshold must be between 0.0 and 1.0"
        )
    
    try:
        # Read and validate image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image_array = np.array(image.convert('RGB'))
        
        if not ImageProcessor.validate_image(image_array):
            raise HTTPException(status_code=400, detail="Invalid image format or size")
        
        # Update detector confidence threshold
        detector.confidence_threshold = confidence_threshold
        
        # Perform detection
        processed_img, detections = detector.detect_damage(image_array)
        summary = detector.get_damage_summary(detections)
        
        # Prepare response
        response_data = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "detections": detections,
            "processing_info": {
                "confidence_threshold": confidence_threshold,
                "image_size": f"{image_array.shape[1]}x{image_array.shape[0]}",
                "model": "YOLOv8n"
            }
        }
        
        # Add annotated image if requested
        if return_image:
            # Convert processed image to base64
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR))
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            response_data["annotated_image"] = f"data:image/jpeg;base64,{img_base64}"
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.post("/detect/batch")
async def detect_damage_batch(
    files: List[UploadFile] = File(...),
    confidence_threshold: float = 0.3
):
    """
    Batch damage detection for multiple images
    
    Args:
        files: List of image files
        confidence_threshold: Detection confidence threshold
        
    Returns:
        JSON response with batch results
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
    
    results = []
    
    for i, file in enumerate(files):
        try:
            # Process each image
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes))
            image_array = np.array(image.convert('RGB'))
            
            detector.confidence_threshold = confidence_threshold
            processed_img, detections = detector.detect_damage(image_array)
            summary = detector.get_damage_summary(detections)
            
            results.append({
                "filename": file.filename,
                "index": i,
                "success": True,
                "summary": summary,
                "detections": detections
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "index": i,
                "success": False,
                "error": str(e)
            })
    
    return JSONResponse(content={
        "success": True,
        "timestamp": datetime.now().isoformat(),
        "batch_size": len(files),
        "results": results
    })

@app.get("/stats")
async def get_statistics():
    """Get system processing statistics"""
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "statistics": detector.detection_stats,
        "model_info": {
            "classes": len(detector.class_labels),
            "confidence_threshold": detector.confidence_threshold
        }
    }

@app.get("/classes")
async def get_damage_classes():
    """Get list of detectable damage classes"""
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "classes": detector.class_labels,
        "total_classes": len(detector.class_labels),
        "categories": config.DAMAGE_CATEGORIES
    }

if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True if config.DEBUG else False,
        log_level="info"
    )