import os
import uuid
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from audio_analyzer import YAMNetAnalyzer

from contextlib import asynccontextmanager

# Global analyzer instance
analyzer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global analyzer
    try:
        analyzer = YAMNetAnalyzer()
    except Exception as e:
        print(f"Warning: Failed to load YAMNetAnalyzer during startup: {e}")
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
        raise HTTPException(status_code=503, detail="Model is not ready")
        
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Prepare temporary file path
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    
    ext = Path(file.filename).suffix
    if ext == "":
        ext = ".wav"

    temp_path = temp_dir / f"{uuid.uuid4()}{ext}"
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
            result = await run_in_threadpool(analyzer.analyze_audio, str(temp_path), filename=file.filename)
        except ValueError as ve:
            err_msg = str(ve)
            if "exceeds the maximum allowed 120 seconds" in err_msg:
                raise HTTPException(status_code=400, detail=err_msg)
            raise HTTPException(status_code=422, detail=f"Failed to process audio format: {err_msg}")
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Failed to process audio format or perform inference. Librosa supports WAV/OGG natively, and others via ffmpeg. Ensure audio is a valid format. Error: {str(e)}")
            
        return result
    finally:
        # Reliable cleanup on both success and failure
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass

# Create static directory if it doesn't exist
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
