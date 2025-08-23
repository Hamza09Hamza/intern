import io
import logging
import pyttsx3
import tempfile
import os
import subprocess
import time
from fastapi import HTTPException
from typing import Optional

logger = logging.getLogger(__name__)

class TTSService:
    def __init__(self):
        self.is_initialized = False

    async def initialize(self):
        try:
            logger.info("Initializing local TTS engine (pyttsx3)...")
            # Just test that pyttsx3 works - don't keep a persistent engine
            test_engine = pyttsx3.init()
            
            # Test voice selection
            found_french = False
            for voice in test_engine.getProperty('voices'):
                if "fr" in voice.id.lower() or "french" in voice.name.lower():
                    logger.info(f"Found French voice: {voice.name} ({voice.id})")
                    found_french = True
                    break
            
            if not found_french:
                logger.warning("No French voice found. Default voice will be used.")
            
            # Clean up test engine
            test_engine.stop()
            del test_engine
            
            self.is_initialized = True
            logger.info("Local TTS engine initialized successfully.")
            
        except Exception as e:
            logger.error(f"Local TTS initialization failed: {e}")
            raise RuntimeError("Failed to initialize local TTS engine.")

    async def synthesize_stream(self, text: str) -> io.BytesIO:
        if not self.is_initialized:
            raise RuntimeError("TTS engine not initialized")

        logger.info(f"🔊 TTS Request: '{text}' ({len(text)} chars)")

        # Create fresh engine for each request to avoid state issues
        raw_path = None
        final_path = None
        
        try:
            # Create fresh engine
            engine = pyttsx3.init()
            
            # Configure engine
            engine.setProperty("rate", 150)
            
            # Set French voice
            for voice in engine.getProperty('voices'):
                if "fr" in voice.id.lower() or "french" in voice.name.lower():
                    engine.setProperty('voice', voice.id)
                    break

            # Create temp files
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_raw:
                raw_path = tmp_raw.name

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_final:
                final_path = tmp_final.name

            logger.info(f"📁 Generating audio: {raw_path}")

            # Generate audio
            engine.save_to_file(text, raw_path)
            engine.runAndWait()
            
            # IMPORTANT: Properly cleanup engine
            try:
                engine.stop()
            except:
                pass
            del engine

            # Wait for file to be written
            time.sleep(0.2)

            # Verify raw file was created
            if not os.path.exists(raw_path):
                raise Exception("TTS engine failed to create audio file")
            
            raw_size = os.path.getsize(raw_path)
            if raw_size == 0:
                raise Exception("TTS engine created empty audio file")
                
            logger.info(f"📏 Raw audio size: {raw_size} bytes")

            # Convert using ffmpeg
            logger.info("🔄 Converting with ffmpeg...")
            subprocess.run([
                "ffmpeg", "-y", "-i", raw_path,
                "-ac", "1", "-ar", "16000",
                "-sample_fmt", "s16",
                "-f", "wav", final_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Verify converted file
            if not os.path.exists(final_path):
                raise Exception("FFmpeg failed to create converted audio file")
                
            final_size = os.path.getsize(final_path)
            if final_size < 1000:  # Audio should be at least 1KB
                raise Exception(f"Converted audio file too small: {final_size} bytes")
                
            logger.info(f"📏 Final audio size: {final_size} bytes")

            # Read file into buffer
            with open(final_path, "rb") as f:
                buffer = io.BytesIO(f.read())

            # Clean up temp files
            self._safe_cleanup([raw_path, final_path])

            buffer.seek(0)
            logger.info(f"✅ TTS completed: {len(buffer.getvalue())} bytes")
            return buffer

        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg failed: {e}")
            self._safe_cleanup([raw_path, final_path])
            raise HTTPException(status_code=500, detail="Failed to convert audio format")
        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")
            self._safe_cleanup([raw_path, final_path])
            raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {str(e)}")

    def _safe_cleanup(self, file_paths):
        """Safely cleanup temporary files"""
        for file_path in file_paths:
            try:
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)
                    logger.debug(f"🗑️ Cleaned up: {file_path}")
            except Exception as e:
                logger.warning(f"⚠️ Cleanup failed for {file_path}: {e}")