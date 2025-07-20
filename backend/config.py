import os
from pathlib import Path

class Config:
    DEBUG   = False
    TESTING = False

class ModelConfig(Config):
    PROJECT_ROOT = Path(__file__).parent
    MODEL_BASE_PATH = Path(os.getenv("MODEL_BASE_PATH",PROJECT_ROOT / "data" / "models" / "gpt2-large"))
    MODEL_SNAPSHOT = os.getenv("MODEL_SNAPSHOT","32b71b12589c2f8d625668d2335a01cac3249519")

class DevelopmentConfig(Config):
    DEBUG = True
    CORS_ORIGINS = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        # If you need to access the back-end from other devices on your home network, change FLASK_RUN_HOST to 0.0.0.0 inside .flaskenv
    ]

class ProductionConfig(Config):
    DEBUG = False
    CORS_ORIGINS = [
        # "http://10.0.0.0:5173",
        # Set this to your local ip address
    ]