import os
from pathlib import Path

class Config:
    # Set environment variables to avoid warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    LLM_MODEL_NAME = "qwen2.5vl:3b"
    
    EMBEDDINGS_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
    EMBEDDINGS_DEVICE = "cpu"
    EMBEDDINGS_NORMALIZE = False
    
    VECTOR_STORE_K = 2
    # Document Processing Configuration
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 200
    DATA_DIRECTORY = "./data/"
    
    VOSK_MODEL_PATH = "vosk-model-fr-0.22"
    VOSK_SAMPLE_RATE = 16000
    
    TTS_MODEL_PATH = "tts_models/en/ljspeech/tacotron2-DDC"
    LANGUAGE="fr"
    SPEACKER_WAV="data/sample.wav"

    
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    AUDIO_PROCESSING_TIMEOUT = 60  # seconds
    
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    CORS_ORIGINS = ["http://localhost:3000"]
    
    LOG_LEVEL = "INFO"
    
    RAG_PROMPT_TEMPLATE = """Vous êtes un assistant chargé de répondre à des questions. 
Utilisez les éléments de contexte suivants pour répondre à la question. 
Si vous ne connaissez pas la réponse, dites simplement que vous ne savez pas. 
Utilisez au maximum trois phrases et restez concis.

Contexte : {context}

Question : {question}

Réponse :"""
    
    @staticmethod
    def ensure_directories():
        """Create necessary directories if they don't exist"""
        os.makedirs(Config.DATA_DIRECTORY, exist_ok=True)
    
    @staticmethod
    def validate_dependencies():
        import subprocess
        
        errors = []
        
        # Check FFmpeg
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            errors.append("FFmpeg not found. Please install FFmpeg.")
        
        # Check Vosk model
        if not os.path.exists(Config.VOSK_MODEL_PATH):
            errors.append(f"Vosk model not found at {Config.VOSK_MODEL_PATH}")
        
        # Check data directory
        if not os.path.exists(Config.DATA_DIRECTORY):
            Config.ensure_directories()
        
        return errors