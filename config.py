"""
Configuration Management for Car Damage Detection System
======================================================

Centralized configuration management for model parameters,
file paths, and system settings.

Author: AI Engineer
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

class Config:
    """
    Configuration class for Car Damage Detection System
    """
    
    # Model Configuration
    MODEL_PATH = "Weights/best.pt"
    DEFAULT_CONFIDENCE_THRESHOLD = 0.3
    INPUT_IMAGE_SIZE = 640
    
    # Damage Classification
    DAMAGE_CLASSES = [
        'Bodypanel-Dent', 'Front-Windscreen-Damage', 'Headlight-Damage',
        'Rear-windscreen-Damage', 'RunningBoard-Dent', 'Sidemirror-Damage',
        'Signlight-Damage', 'Taillight-Damage', 'bonnet-dent', 'boot-dent',
        'doorouter-dent', 'fender-dent', 'front-bumper-dent', 'pillar-dent',
        'quaterpanel-dent', 'rear-bumper-dent', 'roof-dent'
    ]
    
    # Severity Color Mapping (BGR format for OpenCV)
    SEVERITY_COLORS = {
        'critical': (0, 0, 255),    # Red
        'major': (0, 165, 255),     # Orange  
        'minor': (0, 255, 255),     # Yellow
        'default': (255, 0, 0)      # Blue
    }
    
    # Damage Category Mapping
    DAMAGE_CATEGORIES = {
        'Safety Critical': ['Front-Windscreen-Damage', 'Rear-windscreen-Damage', 'Headlight-Damage'],
        'Major Body': ['front-bumper-dent', 'rear-bumper-dent', 'bonnet-dent', 'boot-dent'],
        'Body Panel': ['Bodypanel-Dent', 'doorouter-dent', 'fender-dent', 'quaterpanel-dent'],
        'Exterior Component': ['RunningBoard-Dent', 'Sidemirror-Damage', 'Signlight-Damage', 
                              'Taillight-Damage', 'pillar-dent', 'roof-dent']
    }
    
    # File Paths
    MEDIA_DIR = Path("Media")
    WEIGHTS_DIR = Path("Weights") 
    OUTPUT_DIR = Path("Output")
    
    # Streamlit Configuration
    STREAMLIT_CONFIG = {
        'page_title': 'AI Car Damage Detection',
        'page_icon': '🚗',
        'layout': 'wide',
        'initial_sidebar_state': 'expanded'
    }
    
    # Performance Thresholds
    SEVERITY_THRESHOLDS = {
        'high_damage_count': 4,
        'medium_damage_count': 2,
        'high_confidence': 0.8
    }
    
    @classmethod
    def get_category_for_damage(cls, damage_type: str) -> str:
        """
        Get category for a specific damage type
        
        Args:
            damage_type (str): The damage type to categorize
            
        Returns:
            str: Category name
        """
        for category, damages in cls.DAMAGE_CATEGORIES.items():
            if damage_type in damages:
                return category
        return 'Exterior Component'  # Default category
    
    @classmethod
    def get_severity_color(cls, damage_type: str) -> Tuple[int, int, int]:
        """
        Get color for damage type based on severity
        
        Args:
            damage_type (str): The damage type
            
        Returns:
            Tuple[int, int, int]: BGR color tuple
        """
        category = cls.get_category_for_damage(damage_type)
        
        if category == 'Safety Critical':
            return cls.SEVERITY_COLORS['critical']
        elif category == 'Major Body':
            return cls.SEVERITY_COLORS['major']
        elif category == 'Body Panel':
            return cls.SEVERITY_COLORS['minor']
        else:
            return cls.SEVERITY_COLORS['default']
    
    @classmethod
    def ensure_directories(cls) -> None:
        """
        Ensure all required directories exist
        """
        for directory in [cls.MEDIA_DIR, cls.WEIGHTS_DIR, cls.OUTPUT_DIR]:
            directory.mkdir(exist_ok=True)

# Environment-specific configurations
class DevelopmentConfig(Config):
    """Development environment configuration"""
    DEBUG = True
    LOG_LEVEL = "DEBUG"

class ProductionConfig(Config):
    """Production environment configuration"""
    DEBUG = False
    LOG_LEVEL = "INFO"
    
# Configuration factory
def get_config() -> Config:
    """
    Get configuration based on environment
    
    Returns:
        Config: Configuration instance
    """
    env = os.getenv('ENVIRONMENT', 'development').lower()
    
    if env == 'production':
        return ProductionConfig()
    else:
        return DevelopmentConfig()