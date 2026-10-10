import os
import uuid
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from audio_analyzer import YAMNetAnalyzer

app = FastAPI(title="Echoes of Earth API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global analyzer instance
analyzer = None

@app.on_event("startup")
async def startup_event():
    global analyzer
    try:
        analyzer = YAMNetAnalyzer()
    except Exception as e:
        print(f"Warning: Failed to load YAMNetAnalyzer during startup: {e}")

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
    
    # Restrict file size to roughly 10MB
    file_bytes = await file.read()
    if len(file_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 10MB.")
        
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="File is empty.")

    # Write to temp file
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    
    ext = Path(file.filename).suffix
    if ext == "":
        ext = ".webm" # Assume webm if no extension (common for MediaRecorder blob)

    temp_path = temp_dir / f"{uuid.uuid4()}{ext}"
    
    try:
        with open(temp_path, "wb") as f:
            f.write(file_bytes)
            
        # Analyze
        try:
            result = await run_in_threadpool(analyzer.analyze_audio, str(temp_path), filename=file.filename)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Failed to process audio format or perform inference. Librosa supports WAV/OGG natively, and others via ffmpeg. Ensure audio is a valid format. Error: {str(e)}")
            
        return result
    finally:
        # Cleanup
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass

# Create static directory if it doesn't exist
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
