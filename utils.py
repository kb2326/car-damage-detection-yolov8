"""
Utility Functions for Car Damage Detection System
===============================================

Common utility functions for image processing, validation,
and system operations.

Author: AI Engineer
"""

import cv2
import numpy as np
from PIL import Image
import logging
from typing import Tuple, Optional, Union
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class ImageProcessor:
    """
    Advanced image processing utilities for damage detection
    """
    
    @staticmethod
    def validate_image(image: np.ndarray) -> bool:
        """
        Validate if image is suitable for processing
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            bool: True if image is valid
        """
        if image is None or image.size == 0:
            return False
        
        # Check dimensions
        if len(image.shape) not in [2, 3]:
            return False
            
        # Check minimum size
        height, width = image.shape[:2]
        if height < 100 or width < 100:
            logger.warning(f"Image too small: {width}x{height}")
            return False
            
        return True
    
    @staticmethod
    def preprocess_image(image: np.ndarray, target_size: int = 640) -> np.ndarray:
        """
        Preprocess image for optimal detection performance
        
        Args:
            image (np.ndarray): Input image
            target_size (int): Target size for resizing
            
        Returns:
            np.ndarray: Preprocessed image
        """
        if not ImageProcessor.validate_image(image):
            raise ValueError("Invalid input image")
        
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Assume BGR (OpenCV format), convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize while maintaining aspect ratio
        height, width = image.shape[:2]
        aspect_ratio = width / height
        
        if aspect_ratio > 1:
            new_width = target_size
            new_height = int(target_size / aspect_ratio)
        else:
            new_height = target_size
            new_width = int(target_size * aspect_ratio)
        
        # Resize image
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Pad to square if needed
        if new_width != target_size or new_height != target_size:
            # Create square canvas
            square_image = np.zeros((target_size, target_size, 3), dtype=np.uint8)
            
            # Calculate padding
            y_offset = (target_size - new_height) // 2
            x_offset = (target_size - new_width) // 2
            
            # Place resized image in center
            square_image[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
            
            return square_image
        
        return resized
    
    @staticmethod
    def enhance_image_quality(image: np.ndarray) -> np.ndarray:
        """
        Enhance image quality for better detection
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            np.ndarray: Enhanced image
        """
        # Convert to LAB color space for better enhancement
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Split channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels and convert back to RGB
        enhanced_lab = cv2.merge([l, a, b])
        enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        return enhanced_rgb

class ReportGenerator:
    """
    Generate professional damage assessment reports
    """
    
    @staticmethod
    def generate_json_report(detections: list, summary: dict, metadata: dict = None) -> str:
        """
        Generate JSON format report
        
        Args:
            detections (list): Detection results
            summary (dict): Damage summary
            metadata (dict): Additional metadata
            
        Returns:
            str: JSON report string
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_info': {
                'model': 'YOLOv8n',
                'version': '1.0.0',
                'confidence_threshold': metadata.get('confidence_threshold', 0.3) if metadata else 0.3
            },
            'summary': summary,
            'detections': detections,
            'metadata': metadata or {}
        }
        
        return json.dumps(report, indent=2, ensure_ascii=False)
    
    @staticmethod
    def generate_text_report(detections: list, summary: dict) -> str:
        """
        Generate human-readable text report
        
        Args:
            detections (list): Detection results
            summary (dict): Damage summary
            
        Returns:
            str: Formatted text report
        """
        report_lines = [
            "=" * 60,
            "🚗 AI CAR DAMAGE DETECTION REPORT",
            "=" * 60,
            f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"🔍 Model: YOLOv8n",
            "",
            "📊 SUMMARY",
            "-" * 20,
            f"Total Damages: {summary['total_damages']}",
            f"Severity Level: {summary['severity']}",
            f"Average Confidence: {summary['confidence_avg']:.2%}",
            f"Assessment: {summary['assessment']}",
            ""
        ]
        
        if detections:
            report_lines.extend([
                "📝 DETAILED FINDINGS",
                "-" * 30
            ])
            
            for i, detection in enumerate(detections, 1):
                damage_name = detection['class'].replace('-', ' ').title()
                report_lines.extend([
                    f"{i}. {damage_name}",
                    f"   • Confidence: {detection['confidence']:.2%}",
                    f"   • Location: ({detection['center'][0]}, {detection['center'][1]})",
                    f"   • Area: {detection['area']} pixels²",
                    ""
                ])
        
        report_lines.extend([
            "=" * 60,
            "Report generated by AI Car Damage Detection System",
            "For professional assessment, consult qualified technicians."
        ])
        
        return "\n".join(report_lines)

class ValidationUtils:
    """
    Validation utilities for system inputs and outputs
    """
    
    @staticmethod
    def validate_confidence_threshold(threshold: float) -> bool:
        """
        Validate confidence threshold value
        
        Args:
            threshold (float): Confidence threshold
            
        Returns:
            bool: True if valid
        """
        return 0.0 <= threshold <= 1.0
    
    @staticmethod
    def validate_model_path(path: Union[str, Path]) -> bool:
        """
        Validate model file path
        
        Args:
            path (Union[str, Path]): Path to model file
            
        Returns:
            bool: True if valid
        """
        path = Path(path)
        return path.exists() and path.suffix in ['.pt', '.onnx', '.engine']
    
    @staticmethod
    def validate_image_file(path: Union[str, Path]) -> bool:
        """
        Validate image file
        
        Args:
            path (Union[str, Path]): Path to image file
            
        Returns:
            bool: True if valid
        """
        path = Path(path)
        if not path.exists():
            return False
            
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        return path.suffix.lower() in valid_extensions

def setup_logging(level: str = "INFO") -> None:
    """
    Setup logging configuration
    
    Args:
        level (str): Logging level
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def create_output_filename(prefix: str = "damage_detection", extension: str = "jpg") -> str:
    """
    Create timestamped output filename
    
    Args:
        prefix (str): Filename prefix
        extension (str): File extension
        
    Returns:
        str: Generated filename
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"