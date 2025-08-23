import os
import json
import tempfile
import subprocess
import logging
from typing import Optional
from fastapi import UploadFile, HTTPException
from vosk import Model, KaldiRecognizer

from config import Config

logger = logging.getLogger(__name__)

class SpeechService:
    def __init__(self):
        """ summary_
        """
        self.model: Optional[Model] = None
        self.is_initialized = False
    
    async def initialize(self):
        
        logger.info("Initializing Vosk speech recognition model...")
        try:
            if not os.path.exists(Config.VOSK_MODEL_PATH):
                raise FileNotFoundError(f"Vosk model not found at {Config.VOSK_MODEL_PATH}")
            
            self.model = Model(Config.VOSK_MODEL_PATH)
            self.is_initialized = True
            logger.info("Vosk model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Vosk model: {e}")
            raise
    
    def _check_ffmpeg(self) -> bool:
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    async def transcribe_audio(self, audio_file: UploadFile) -> str:
        
        if not self.is_initialized:
            raise HTTPException(status_code=503, detail="Speech service not initialized")
        
        if not self._check_ffmpeg():
            raise HTTPException(status_code=500, detail="FFmpeg not available")
        
        temp_file_path = None
        pcm_file_path = None
        
        try:
            file_content = await audio_file.read()
            
            if len(file_content) > Config.MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=400, 
                    detail=f"File too large. Maximum size is {Config.MAX_FILE_SIZE // (1024*1024)}MB"
                )
            
            logger.info(f"Processing audio file: {audio_file.filename} ({len(file_content)} bytes)")
            
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name
            
            # Convert to PCM format
            pcm_file_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pcm").name

            await self._convert_to_pcm(temp_file_path, pcm_file_path)
            logger.info(f"Converting audio to PCM format: {pcm_file_path}")
            
            # Transcribe using Vosk
            transcribed_text = await self._transcribe_with_vosk(pcm_file_path)

            logger.info(f"Transcription successful: '{transcribed_text}'")
            return transcribed_text
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
        finally:
            # Clean up temporary files
            self._cleanup_files([temp_file_path, pcm_file_path])
    
    async def _convert_to_pcm(self, input_path: str, output_path: str):
        """Convert audio file to PCM format using FFmpeg"""
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-ar", str(Config.VOSK_SAMPLE_RATE),
            "-ac", "1",  # Mono
            "-f", "s16le",  # PCM 16-bit little-endian
            output_path
        ]
        
        try:
            result = subprocess.run(
                ffmpeg_cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=Config.AUDIO_PROCESSING_TIMEOUT
            )
            
            if result.stderr:
                logger.debug(f"FFmpeg stderr: {result.stderr}")
            
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                raise HTTPException(status_code=400, detail="Failed to convert audio file")
                
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=408, detail="Audio processing timeout")
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e}")
            raise HTTPException(status_code=400, detail="Failed to process audio file")
    
    async def _transcribe_with_vosk(self, pcm_file_path: str) -> str:
        try:
            recognizer = KaldiRecognizer(self.model, Config.VOSK_SAMPLE_RATE)
            
            with open(pcm_file_path, "rb") as f:
                while True:
                    data = f.read(4096)
                    if not data:
                        break
                    recognizer.AcceptWaveform(data)
            
            final_result = json.loads(recognizer.FinalResult())
            return final_result.get("text", "")
            
        except json.JSONDecodeError:
            logger.error("Failed to parse Vosk result")
            raise HTTPException(status_code=500, detail="Transcription result parsing error")
        except Exception as e:
            logger.error(f"Vosk transcription error: {e}")
            raise HTTPException(status_code=500, detail="Transcription processing error")
    
    def _cleanup_files(self, file_paths: list):
        """Clean up temporary files"""
        for file_path in file_paths:
            if file_path and os.path.exists(file_path):
                try:
                    os.unlink(file_path)
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file {file_path}: {e}")
    
    def get_status(self) -> dict:
        return {
            "initialized": self.is_initialized,
            "model_path": Config.VOSK_MODEL_PATH,
            "ffmpeg_available": self._check_ffmpeg()
        }