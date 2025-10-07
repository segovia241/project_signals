import logging

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de la aplicación
APP_CONFIG = {
    "app_name": "Sign Language Translator API",
    "version": "1.0.0",
    "host": "0.0.0.0",
    "port": 8000,
    "reload": True,
    "log_level": "info"
}

# Configuración CORS (acceso total)
CORS_CONFIG = {
    "allow_origins": ["*"],
    "allow_credentials": True,
    "allow_methods": ["*"],
    "allow_headers": ["*"],
}

# Configuración del modelo
MODEL_CONFIG = {
    "model_path": "sign_language_model.h5",
    "metadata_path": "model_metadata.json",
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5
}

# Configuración de cámara
CAMERA_CONFIG = {
    "camera_index": 0,
    "frame_width": 640,
    "frame_height": 480,
    "fps": 30,
    "websocket_delay": 0.033  # ~30 FPS
}
