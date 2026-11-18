from fastapi import Depends, FastAPI, Query, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import get_db
from app.routes import api_router
from app.routes.devices import (
    device_ack_tts_job,
    device_next_tts_job,
    device_stream_tts_audio,
)
from app.schemas.device import DeviceTTSAckRequest, DeviceTTSAckResponse, DeviceTTSJob
from app.websocket.routes import router as ws_router
from app.websocket import api_router as ws_api_router
from app.tasks.scheduler import start_scheduler, stop_scheduler
from app.db.init_db import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    start_scheduler()
    yield
    stop_scheduler()


app = FastAPI(
    title="LumoHub API",
    description="Backend for LumoHub calendar & event management system",
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
app.include_router(ws_router)
app.include_router(ws_api_router)

# ── Public audio endpoint cho ESP32 ────────────────────────────────
# ESP32 không có JWT → không thể đi qua /api/v1/lumo/audio/ (auth-required).
# Wrapper endpoint này gọi chung logic pipeline bên dưới, không cần auth.
async def public_audio_endpoint(request: Request, audio: UploadFile = File(...)):
    # Import tại thời điểm gọi để tránh circular import với app.routes
    from app.routes.lumo import lumo_audio
    return await lumo_audio(request, audio)

app.add_api_route(
    "/audio/",
    public_audio_endpoint,
    methods=["POST"],
    summary="LUMO voice pipeline (ESP32 public, no auth)",
    description="Upload WAV → Groq STT → Gemini TTT → Fast TTS/Cache → raw PCM/WAV binary. "
                 "No auth required. ESP32 calls this directly.",
)


# ── Tương thích đường dẫn DSOFT-AudioTTS cho ESP32 (/v1/devices/...) ──
@app.get(
    "/v1/devices/{device_code}/next",
    tags=["ESP32 TTS"],
    response_model=DeviceTTSJob,
    responses={204: {"description": "No queued TTS job"}},
)
def compat_device_next_tts_job(device_code: str, db: Session = Depends(get_db)):
    return device_next_tts_job(device_code=device_code, db=db)


@app.get("/v1/devices/{device_code}/audio/{job_id}", tags=["ESP32 TTS"])
def compat_device_stream_tts_audio(
    device_code: str,
    job_id: str,
    lease: str = Query(..., min_length=1, max_length=64),
    db: Session = Depends(get_db),
):
    return device_stream_tts_audio(device_code=device_code, job_id=job_id, lease=lease, db=db)


@app.post(
    "/v1/devices/{device_code}/ack",
    tags=["ESP32 TTS"],
    response_model=DeviceTTSAckResponse,
)
def compat_device_ack_tts_job(
    device_code: str,
    body: DeviceTTSAckRequest,
    db: Session = Depends(get_db),
):
    return device_ack_tts_job(device_code=device_code, body=body, db=db)


@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "ok",
        "service": "LumoHub API",
        "version": settings.APP_VERSION,
    }
