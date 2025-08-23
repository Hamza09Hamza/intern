import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from uuid import uuid4
import io

from service.speech_service import SpeechService
from service.rag_service import RAGService
from service.tts_service import TTSService


# in this section we, import the NL2SQL services 
from mysqlservice.nlusystem import NLUSystem
from config import Config 

logger = logging.getLogger(__name__)

router = APIRouter()
audio_store: dict[str, io.BytesIO] = {}

# Services to be set by main.py
speech_service: SpeechService | None = None
# rag_service:    RAGService    | None = None
tts_service:    TTSService    | None = None
# Optional NLU service, can be None if not used
nlu_service:    NLUSystem     | None = None

def set_services(
    svc_speech: SpeechService,
    # svc_rag: RAGService,
    svc_tts: TTSService,
    svc_nlu: NLUSystem,  
):
    global speech_service, tts_service, nlu_service
    speech_service = svc_speech
    tts_service    = svc_tts
    nlu_service    = svc_nlu
    # rag_service    = svc_rag

@router.post("/process_audio")
async def process_audio(file: UploadFile = File(...)):
    if not all([speech_service, nlu_service, tts_service]):
        raise HTTPException(status_code=503, detail="Services not initialized")

    # 1) Transcribe
    transcription = await speech_service.transcribe_audio(file)
    if not transcription.strip():
        return JSONResponse(
            status_code=200,
            content={
                "success": False,
                "transcription": "",
                "answer": "",
                "error": "No speech detected"
            }
        )

    # 2) RAG answer
    #answer = await rag_service.process_question(transcription)
    
    
    #2.1) Optional NLU processing
    answer = await nlu_service.query(transcription)
    
    
            

    # 3) TTS synthesis → returns BytesIO
    try:
        audio_io = await tts_service.synthesize_stream(answer)
    except Exception as e:
        logger.error(f"TTS synthesis error: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "transcription": transcription,
                "answer": answer,
                "error": "TTS synthesis failed"
            }
        )

    # 4) Store audio buffer and return metadata
    key = uuid4().hex
    audio_store[key] = audio_io

    return {
        "success": True,
        "transcription": transcription,
        "answer": answer,
        "audio_id": key
    }

@router.get("/stream_audio/{audio_id}")
async def stream_audio(audio_id: str):
    buffer = audio_store.get(audio_id)
    if not buffer:
        raise HTTPException(status_code=404, detail="Audio not found")

    buffer.seek(0)

    logger.info(f"Streaming audio for ID {audio_id} - {len(buffer.getvalue())} bytes")

    # Return browser-safe WAV
    return StreamingResponse(buffer, media_type="audio/wav")

@router.get("/health")
async def health_check():
    if not all([speech_service, nlu_service, tts_service]):
        raise HTTPException(status_code=503, detail="Services not initialized")

    return JSONResponse(content={"status": "ok"}, status_code=200)
