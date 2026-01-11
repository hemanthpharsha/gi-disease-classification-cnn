# config.py - Enhanced configuration for GI Disease Classifier
import os
from datetime import timedelta

class Config:
    """Base configuration class"""
    
    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    
    # File Upload Configuration
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB max file size
    UPLOAD_FOLDER = 'static/uploads'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    
    # Model Configuration
    MODEL_PATH = 'model/final_gi_model.h5'
    MODEL_INPUT_SIZE = (224, 224, 3)
    
    # Class Names (update based on your model)
    CLASS_NAMES = [
        'dyed-lifted-polyps',
        'dyed-resection-margins',
        'esophagitis',
        'normal-cecum',
        'normal-pylorus',
        'normal-z-line',
        'polyps',
        'ulcerative-colitis'
    ]
    
    # Prediction Thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.8
    MEDIUM_CONFIDENCE_THRESHOLD = 0.6
    LOW_CONFIDENCE_THRESHOLD = 0.4
    
    # Batch Processing Limits
    MAX_BATCH_SIZE = 10
    BATCH_TIMEOUT = 300  # 5 minutes
    
    # UI Configuration
    APP_NAME = 'GI Disease AI Classifier'
    APP_VERSION = '2.0'
    APP_DESCRIPTION = 'Advanced machine learning for gastrointestinal condition analysis'
    
    # Performance Settings
    SEND_FILE_MAX_AGE_DEFAULT = timedelta(hours=1)
    
    # Medical Information Database
    MEDICAL_CONDITIONS = {
        'dyed-lifted-polyps': {
            'name': 'Dyed Lifted Polyps',
            'description': 'Polyps that have been lifted and marked with dye for endoscopic removal',
            'severity': 'Medium Risk',
            'color': '#f59e0b',
            'icon': 'fa-procedures',
            'treatment_needed': True,
            'urgency': 'Schedule within 2-4 weeks',
            'treatment_options': [
                'Endoscopic polypectomy',
                'Histological examination',
                'Surveillance colonoscopy'
            ],
            'lifestyle_recommendations': [
                'High-fiber diet',
                'Regular exercise',
                'Avoid smoking and excessive alcohol'
            ]
        },
        'dyed-resection-margins': {
            'name': 'Dyed Resection Margins',
            'description': 'Surgical resection margins marked with dye to ensure complete removal',
            'severity': 'Post-Surgical',
            'color': '#8b5cf6',
            'icon': 'fa-cut',
            'treatment_needed': True,
            'urgency': 'Follow-up as scheduled',
            'treatment_options': [
                'Post-surgical monitoring',
                'Histological examination',
                'Healing assessment'
            ],
            'lifestyle_recommendations': [
                'Follow post-operative instructions',
                'Gradual diet progression',
                'Monitor for complications'
            ]
        },
        'esophagitis': {
            'name': 'Esophagitis',
            'description': 'Inflammation of the esophageal lining',
            'severity': 'Medium Risk',
            'color': '#ef4444',
            'icon': 'fa-fire',
            'treatment_needed': True,
            'urgency': 'Consult within 1-2 weeks',
            'treatment_options': [
                'Proton pump inhibitors',
                'Anti-inflammatory medications',
                'Dietary modifications'
            ],
            'lifestyle_recommendations': [
                'Avoid acidic and spicy foods',
                'Eat smaller, frequent meals',
                'Elevate head while sleeping',
                'Avoid late-night eating'
            ]
        },
        'normal-cecum': {
            'name': 'Normal Cecum',
            'description': 'Normal appearance of the cecum (first part of large intestine)',
            'severity': 'Normal',
            'color': '#10b981',
            'icon': 'fa-check-circle',
            'treatment_needed': False,
            'urgency': 'Routine follow-up',
            'treatment_options': [
                'No treatment required',
                'Continue routine screening'
            ],
            'lifestyle_recommendations': [
                'Maintain healthy diet',
                'Regular exercise',
                'Adequate hydration',
                'Follow screening guidelines'
            ]
        },
        'normal-pylorus': {
            'name': 'Normal Pylorus',
            'description': 'Normal appearance of the pylorus (stomach outlet)',
            'severity': 'Normal',
            'color': '#10b981',
            'icon': 'fa-check-circle',
            'treatment_needed': False,
            'urgency': 'Routine follow-up',
            'treatment_options': [
                'No treatment required',
                'Maintain healthy habits'
            ],
            'lifestyle_recommendations': [
                'Balanced nutrition',
                'Regular meal patterns',
                'Limit NSAIDs',
                'Manage stress levels'
            ]
        },
        'normal-z-line': {
            'name': 'Normal Z-Line',
            'description': 'Normal gastroesophageal junction appearance',
            'severity': 'Normal',
            'color': '#10b981',
            'icon': 'fa-check-circle',
            'treatment_needed': False,
            'urgency': 'Routine follow-up',
            'treatment_options': [
                'No treatment required',
                'Continue healthy habits'
            ],
            'lifestyle_recommendations': [
                'Heart-healthy diet',
                'Weight management',
                'Regular physical activity',
                'Avoid excessive alcohol'
            ]
        },
        'polyps': {
            'name': 'Intestinal Polyps',
            'description': 'Growth protruding from the intestinal wall',
            'severity': 'Medium Risk',
            'color': '#f59e0b',
            'icon': 'fa-exclamation-triangle',
            'treatment_needed': True,
            'urgency': 'Schedule within 2-4 weeks',
            'treatment_options': [
                'Endoscopic polypectomy',
                'Histological analysis',
                'Surveillance colonoscopy'
            ],
            'lifestyle_recommendations': [
                'High-fiber diet',
                'Limit red and processed meat',
                'Regular exercise',
                'Maintain healthy weight'
            ]
        },
        'ulcerative-colitis': {
            'name': 'Ulcerative Colitis',
            'description': 'Chronic inflammatory bowel disease affecting the colon',
            'severity': 'High Risk',
            'color': '#dc2626',
            'icon': 'fa-exclamation-circle',
            'treatment_needed': True,
            'urgency': 'Consult immediately',
            'treatment_options': [
                'Anti-inflammatory medications',
                'Immunosuppressive therapy',
                'Biologics if severe',
                'Surgery in severe cases'
            ],
            'lifestyle_recommendations': [
                'Anti-inflammatory diet',
                'Stress management',
                'Regular monitoring',
                'Avoid trigger foods',
                'Adequate rest and sleep'
            ]
        }
    }
    
    # Email Configuration (for future notifications)
    MAIL_SERVER = os.environ.get('MAIL_SERVER')
    MAIL_PORT = int(os.environ.get('MAIL_PORT', 587))
    MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS', 'true').lower() in ['true', 'on', '1']
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')
    
    # Logging Configuration
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = 'logs/gi_classifier.log'
    
    # Cache Configuration
    CACHE_TYPE = 'simple'
    CACHE_DEFAULT_TIMEOUT = 300
    
    # Security Headers
    SECURITY_HEADERS = {
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains'
    }

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False
    
    # Override for development
    SECRET_KEY = 'dev-secret-key'
    
    # Relaxed security for development
    SECURITY_HEADERS = {}

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    
    # Stronger security key required
    SECRET_KEY = os.environ.get('SECRET_KEY')
    if not SECRET_KEY:
        raise ValueError("No SECRET_KEY set for production environment")
    
    # Enhanced security settings
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'

class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True
    
    # Test-specific settings
    WTF_CSRF_ENABLED = False
    UPLOAD_FOLDER = 'tests/uploads'

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config():
    """Get configuration based on environment"""
    return config[os.environ.get('FLASK_ENV', 'default')]

# Utility functions for configuration
def validate_model_path(path):
    """Validate model file exists"""
    return os.path.exists(path)

def create_upload_directory(path):
    """Create upload directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)
    return path

def get_medical_info(condition_key):
    """Get medical information for a condition"""
    return Config.MEDICAL_CONDITIONS.get(condition_key, {
        'name': 'Unknown Condition',
        'description': 'Condition information not available',
        'severity': 'Unknown',
        'color': '#6b7280',
        'icon': 'fa-question-circle',
        'treatment_needed': None,
        'urgency': 'Consult healthcare provider',
        'treatment_options': ['Medical evaluation recommended'],
        'lifestyle_recommendations': ['Follow general health guidelines']
    })

def format_condition_name(condition_key):
    """Format condition key for display"""
    return condition_key.replace('-', ' ').replace('_', ' ').title()

# Application factory configuration helper
def configure_app(app, config_name=None):
    """Configure Flask application with specified configuration"""
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'default')
    
    app.config.from_object(config[config_name])
    
    # Create necessary directories
    create_upload_directory(app.config['UPLOAD_FOLDER'])
    
    # Set up logging
    import logging
    from logging.handlers import RotatingFileHandler
    
    if not app.debug:
        if not os.path.exists('logs'):
            os.mkdir('logs')
        
        file_handler = RotatingFileHandler(
            app.config['LOG_FILE'],
            maxBytes=10240000,  # 10MB
            backupCount=10
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)
        app.logger.setLevel(logging.INFO)
        app.logger.info('GI Disease Classifier startup')
    
    return app