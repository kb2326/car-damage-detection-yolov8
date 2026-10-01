"""
Car Damage Detection System using YOLOv8
========================================

A production-ready computer vision system for automated vehicle damage assessment.
Leverages state-of-the-art YOLOv8 object detection for real-time damage classification.

Author: AI Engineer
License: MIT
"""

import cv2
import math
import cvzone
import logging
from typing import List, Tuple, Dict, Any
from ultralytics import YOLO
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CarDamageDetector:
    def __init__(self, model_path: str, confidence_threshold: float = 0.3):
        """
        Initialize the Car Damage Detection System
        
        Args:
            model_path (str): Path to the trained YOLOv8 model weights
            confidence_threshold (float): Minimum confidence score for detections (0.0-1.0)
            
        Raises:
            FileNotFoundError: If model weights file doesn't exist
            ValueError: If confidence threshold is not in valid range
        """
        if not (0.0 <= confidence_threshold <= 1.0):
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
            
        try:
            self.model = YOLO(model_path)
            logger.info(f"Successfully loaded YOLOv8 model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
            
        # Comprehensive damage classification taxonomy
        self.class_labels = [
            'Bodypanel-Dent', 'Front-Windscreen-Damage', 'Headlight-Damage',
            'Rear-windscreen-Damage', 'RunningBoard-Dent', 'Sidemirror-Damage',
            'Signlight-Damage', 'Taillight-Damage', 'bonnet-dent', 'boot-dent',
            'doorouter-dent', 'fender-dent', 'front-bumper-dent', 'pillar-dent',
            'quaterpanel-dent', 'rear-bumper-dent', 'roof-dent'
        ]
        
        self.confidence_threshold = confidence_threshold
        self.detection_stats = {'total_processed': 0, 'total_detections': 0}

    def detect_damage(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Perform damage detection on input image using YOLOv8
        
        Args:
            image (np.ndarray): Input image in BGR format (OpenCV standard)
            
        Returns:
            Tuple[np.ndarray, List[Dict]]: 
                - Annotated image with bounding boxes and labels
                - List of detection dictionaries containing bbox, class, confidence, and metadata
                
        Raises:
            ValueError: If input image is invalid
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid input image")
            
        # Perform inference
        results = self.model(image, verbose=False)
        detections = []
        
        # Update processing statistics
        self.detection_stats['total_processed'] += 1

        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
                
            for box in boxes:
                # Extract bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                
                # Extract confidence and class information
                conf = float(box.conf[0])
                cls_idx = int(box.cls[0])

                if conf > self.confidence_threshold and cls_idx < len(self.class_labels):
                    # Create structured detection object
                    detection = {
                        'bbox': (x1, y1, w, h),
                        'class': self.class_labels[cls_idx],
                        'confidence': conf,
                        'center': (x1 + w//2, y1 + h//2),
                        'area': w * h,
                        'aspect_ratio': w / h if h > 0 else 0
                    }
                    detections.append(detection)
                    self.detection_stats['total_detections'] += 1
                    
                    # Enhanced visualization with professional styling
                    self._draw_detection(image, detection)

        logger.info(f"Detected {len(detections)} damage instances with confidence > {self.confidence_threshold}")
        return image, detections
    
    def _draw_detection(self, image: np.ndarray, detection: Dict[str, Any]) -> None:
        """
        Draw professional-looking bounding box and label for detection
        
        Args:
            image (np.ndarray): Image to draw on
            detection (Dict): Detection information
        """
        x1, y1, w, h = detection['bbox']
        class_name = detection['class']
        confidence = detection['confidence']
        
        # Color coding based on damage severity
        color = self._get_severity_color(class_name)
        
        # Draw enhanced bounding box
        cvzone.cornerRect(image, (x1, y1, w, h), t=3, colorC=color, colorR=color)
        
        # Professional label formatting
        label = f'{class_name.replace("-", " ").title()}'
        conf_text = f'{confidence:.2%}'
        
        cvzone.putTextRect(
            image,
            f'{label} ({conf_text})',
            (x1, y1 - 15),
            scale=0.7,
            thickness=2,
            colorT=(255, 255, 255),
            colorR=color,
            font=cv2.FONT_HERSHEY_SIMPLEX
        )
    
    def _get_severity_color(self, damage_type: str) -> Tuple[int, int, int]:
        """
        Assign color based on damage severity for better visualization
        
        Args:
            damage_type (str): Type of damage detected
            
        Returns:
            Tuple[int, int, int]: BGR color tuple
        """
        # Critical damages (red)
        critical = ['Front-Windscreen-Damage', 'Rear-windscreen-Damage', 'Headlight-Damage']
        # Major damages (orange)
        major = ['front-bumper-dent', 'rear-bumper-dent', 'bonnet-dent', 'boot-dent']
        # Minor damages (yellow)
        
        if any(crit in damage_type for crit in critical):
            return (0, 0, 255)  # Red
        elif any(maj in damage_type for maj in major):
            return (0, 165, 255)  # Orange
        else:
            return (0, 255, 255)  # Yellow

    def get_damage_summary(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate comprehensive damage assessment summary
        
        Args:
            detections (List[Dict]): List of detection results
            
        Returns:
            Dict[str, Any]: Damage summary with statistics and severity assessment
        """
        if not detections:
            return {
                'total_damages': 0,
                'severity': 'None',
                'confidence_avg': 0.0,
                'damage_categories': {},
                'assessment': 'No damage detected'
            }
        
        # Calculate statistics
        total_damages = len(detections)
        avg_confidence = sum(d['confidence'] for d in detections) / total_damages
        
        # Categorize damages
        categories = {}
        for detection in detections:
            category = self._categorize_damage(detection['class'])
            categories[category] = categories.get(category, 0) + 1
        
        # Determine severity
        severity = self._assess_severity(total_damages, avg_confidence, categories)
        
        return {
            'total_damages': total_damages,
            'severity': severity,
            'confidence_avg': avg_confidence,
            'damage_categories': categories,
            'assessment': self._generate_assessment(severity, total_damages, categories)
        }
    
    def _categorize_damage(self, damage_type: str) -> str:
        """Categorize damage type for better analysis"""
        if 'windscreen' in damage_type.lower() or 'headlight' in damage_type.lower():
            return 'Safety Critical'
        elif 'bumper' in damage_type.lower() or 'bonnet' in damage_type.lower():
            return 'Major Body'
        elif 'dent' in damage_type.lower():
            return 'Body Panel'
        else:
            return 'Exterior Component'
    
    def _assess_severity(self, count: int, confidence: float, categories: Dict) -> str:
        """Assess overall damage severity"""
        if 'Safety Critical' in categories or count >= 4:
            return 'High'
        elif count >= 2 or confidence > 0.8:
            return 'Medium'
        else:
            return 'Low'
    
    def _generate_assessment(self, severity: str, count: int, categories: Dict) -> str:
        """Generate human-readable assessment"""
        if severity == 'High':
            return f"Significant damage detected ({count} issues). Immediate attention recommended."
        elif severity == 'Medium':
            return f"Moderate damage found ({count} issues). Professional assessment advised."
        else:
            return f"Minor damage detected ({count} issues). Routine maintenance may be sufficient."

def main():
    """
    Demonstration of the Car Damage Detection System
    """
    try:
        # Initialize detector with custom confidence threshold
        detector = CarDamageDetector("Weights/best.pt", confidence_threshold=0.3)
        
        # Load and process test image
        image_path = "Media/dent_1.jpg"
        img = cv2.imread(image_path)
        
        if img is None:
            logger.error(f"Could not load image from {image_path}")
            return

        logger.info(f"Processing image: {image_path}")
        
        # Perform damage detection
        processed_img, detections = detector.detect_damage(img)
        
        # Generate comprehensive summary
        summary = detector.get_damage_summary(detections)
        
        # Display results
        cv2.imshow("AI Car Damage Detection System", processed_img)
        
        # Print professional analysis
        print("\n" + "="*60)
        print("🚗 AI CAR DAMAGE DETECTION REPORT")
        print("="*60)
        print(f"📊 Total Damages Detected: {summary['total_damages']}")
        print(f"🎯 Average Confidence: {summary['confidence_avg']:.2%}")
        print(f"⚠️  Severity Level: {summary['severity']}")
        print(f"📋 Assessment: {summary['assessment']}")
        
        if detections:
            print(f"\n📝 Detailed Findings:")
            for i, det in enumerate(detections, 1):
                damage_name = det['class'].replace('-', ' ').title()
                print(f"  {i}. {damage_name}")
                print(f"     • Confidence: {det['confidence']:.2%}")
                print(f"     • Location: ({det['center'][0]}, {det['center'][1]})")
                print(f"     • Size: {det['area']} pixels²")
        
        print(f"\n📈 Processing Statistics:")
        print(f"  • Images Processed: {detector.detection_stats['total_processed']}")
        print(f"  • Total Detections: {detector.detection_stats['total_detections']}")
        print("="*60)

        # Interactive display
        print("\n💡 Press 'q' to quit, 's' to save results")
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite('damage_detection_result.jpg', processed_img)
                print("✅ Results saved as 'damage_detection_result.jpg'")

        cv2.destroyAllWindows()
        cv2.waitKey(1)
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()