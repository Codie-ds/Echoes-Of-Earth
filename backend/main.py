import os
import uuid
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from audio_analyzer import YAMNetAnalyzer

logger = logging.getLogger("echoes_of_earth")

# Base and temporary directory paths anchored to backend directory
BASE_DIR = Path(__file__).resolve().parent
TEMP_DIR = BASE_DIR / "temp_uploads"
STATIC_DIR = BASE_DIR / "static"

# Global analyzer instance
analyzer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global analyzer
    TEMP_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    
    # Privacy safeguard: clean up any orphaned temporary files from prior interrupted processes
    for stale_file in TEMP_DIR.glob("*"):
        try:
            if stale_file.is_file():
                stale_file.unlink()
        except Exception:
            pass

    try:
        analyzer = YAMNetAnalyzer()
    except Exception as e:
        logger.error("Failed to load YAMNetAnalyzer during startup: %s", type(e).__name__)
    yield

app = FastAPI(title="Echoes of Earth API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB
CHUNK_SIZE = 64 * 1024  # 64 KB

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model_ready": analyzer is not None
    }

@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    if not analyzer:
        raise HTTPException(status_code=503, detail="Model is not ready.")
        
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")
    
    # Path traversal protection: extract basename only and strip directory separators
    raw_name = Path(file.filename.replace("\\", "/")).name.strip()
    safe_filename = raw_name if raw_name and raw_name not in (".", "..") else "uploaded_audio.wav"
    
    # Clamp extension to valid audio suffix to prevent arbitrary file typing
    safe_ext = Path(safe_filename).suffix.lower()
    if safe_ext not in [".wav", ".ogg", ".flac", ".mp3", ".webm", ".m4a", ".aac"]:
        safe_ext = ".wav"

    temp_path = TEMP_DIR / f"{uuid.uuid4()}{safe_ext}"
    total_bytes = 0
    
    try:
        # Bounded streaming read: stream chunks directly to disk to prevent RAM exhaustion
        with open(temp_path, "wb") as f:
            while True:
                chunk = await file.read(CHUNK_SIZE)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_SIZE:
                    raise HTTPException(status_code=413, detail="File too large. Maximum size is 10MB.")
                f.write(chunk)
                
        if total_bytes == 0:
            raise HTTPException(status_code=400, detail="File is empty.")

        # Analyze in threadpool to keep event loop responsive
        try:
            result = await run_in_threadpool(analyzer.analyze_audio, str(temp_path), filename=safe_filename)
        except ValueError as ve:
            err_msg = str(ve)
            if "exceeds the maximum allowed 120 seconds" in err_msg:
                raise HTTPException(status_code=400, detail="Audio duration exceeds the maximum allowed 120 seconds.")
            # Format/decoding rejection without exposing internal paths
            logger.warning("Audio processing rejected: invalid or unparseable format")
            raise HTTPException(status_code=422, detail="Failed to process audio format. Ensure audio is a valid, supported format.")
        except HTTPException:
            raise
        except Exception as e:
            # Internal server error without leaking internal filesystem paths, stack traces, or secrets
            logger.error("Unexpected error during audio inference: %s", type(e).__name__)
            raise HTTPException(status_code=500, detail="An internal error occurred during audio analysis.")
            
        return result
    finally:
        # Reliable cleanup on both success and failure
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass

# Create static directory if it doesn't exist
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
