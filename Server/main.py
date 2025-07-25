import logging
from fastapi import FastAPI
from api.routes import router, set_services
from service.speech_service import SpeechService
from service.rag_service import RAGService
from service.tts_service import TTSService
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://192.168.*", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

speech_service = None
rag_service = None
tts_service = None

@app.on_event("startup")
async def startup_event():
    global speech_service, rag_service, tts_service
    try:
        logger.info("Starting service initialization...")

        speech_service = SpeechService()
        rag_service = RAGService()
        tts_service = TTSService()

        await speech_service.initialize()
        await rag_service.initialize()
        await tts_service.initialize()

        set_services(speech_service, rag_service, tts_service)

        logger.info("Services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        speech_service = None
        rag_service = None
        tts_service = None

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
