import io
import logging
import pyttsx3
import tempfile
import os
import subprocess
from fastapi import HTTPException
from typing import Optional

logger = logging.getLogger(__name__)

class TTSService:
    def __init__(self):
        self.engine: Optional[pyttsx3.Engine] = None
        self.is_initialized = False

    async def initialize(self):
        try:
            logger.info("Initializing local TTS engine (pyttsx3)...")
            self.engine = pyttsx3.init()
            rate = self.engine.getProperty("rate")
            logger.info(f"Default speech rate: {rate}")
            self.engine.setProperty("rate", 150) 

            # Look for a French voice
            found = False
            for voice in self.engine.getProperty('voices'):
                if "fr" in voice.id.lower() or "french" in voice.name.lower():
                    self.engine.setProperty('voice', voice.id)
                    logger.info(f"Selected French voice: {voice.name} ({voice.id})")
                    found = True
                    break

            if not found:
                logger.warning("No French voice found. Default voice will be used.")
            else:
                self.is_initialized = True
                logger.info("Local TTS engine initialized successfully with French voice.")
        except Exception as e:
            logger.error(f"Local TTS initialization failed: {e}")
            raise RuntimeError("Failed to initialize local TTS engine.")

    async def synthesize_stream(self, text: str) -> io.BytesIO:
        if not self.is_initialized or not self.engine:
            raise RuntimeError("TTS engine not initialized")

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_raw:
                raw_path = tmp_raw.name

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_final:
                final_path = tmp_final.name

            # Save initial audio
            self.engine.save_to_file(text, raw_path)
            self.engine.runAndWait()

            # Re-encode using ffmpeg to mono, 16kHz, PCM S16LE
            subprocess.run([
                "ffmpeg", "-y", "-i", raw_path,
                "-ac", "1", "-ar", "16000",
                "-sample_fmt", "s16",
                "-f", "wav", final_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            with open(final_path, "rb") as f:
                buffer = io.BytesIO(f.read())

            # Clean up
            os.remove(raw_path)
            os.remove(final_path)

            buffer.seek(0)
            return buffer

        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg failed: {e}")
            raise HTTPException(status_code=500, detail="Failed to convert audio format")
        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")
            raise HTTPException(status_code=500, detail="TTS synthesis failed")
