# Echoes of Earth

AI-powered environmental sound exploration.

## Project Phases
- **Phase 1**: Initial inference (YAMNet loading and audio analysis).
- **Phase 2**: The Evidence Engine (frame-level timestamped events).
- **Phase 3**: MVP Application (FastAPI backend and Web UI).
- **Phase 4**: Privacy and Reliability Hardening (bounded streaming, sanitized error responses, zero persistence).

## Privacy & Data Processing Lifecycle

- **Local-Only Audio Inference**: All audio decoding, feature extraction, and YAMNet model inference execute entirely on-device (CPU in-memory). No audio recordings, waveforms, embeddings, or predictions are ever transmitted to external APIs or cloud services.
- **Model Acquisition**: YAMNet weights are fetched once from TensorFlow Hub (`tfhub.dev`) during initialization if not already cached locally. Runtime inference runs 100% offline.
- **Zero Long-Term Audio Persistence**: Uploaded and recorded audio is streamed to isolated temporary files with random UUID filenames. Temporary files are unconditionally removed in a `finally` block upon completion, validation rejection, decode error, or server exception. Stale files from prior process terminations are wiped on startup.
- **Path Traversal Protection**: Uploaded filenames are stripped of directory paths and traversal sequences prior to temporary handling and response generation.
- **Sanitized Error Handling**: Client error responses return generic, actionable messages without leaking internal filesystem paths, stack traces, or exception internals. Server-side logs capture operational diagnostics without recording audio payloads or secrets.

## Local Run Instructions

### 1. Activate the Virtual Environment
```bash
# From the repository root
source .venv/bin/activate
```

### 2. Install Dependencies
If you haven't already:
```bash
cd backend
pip install -r requirements.txt
pip install fastapi uvicorn python-multipart httpx
```

### 3. Start the Backend Server
```bash
cd backend
uvicorn main:app --reload
```
The server will start on `http://127.0.0.1:8000`.

### 4. Open the Frontend
Since the frontend is served as static files by FastAPI, simply navigate to:
[http://127.0.0.1:8000/](http://127.0.0.1:8000/)

### 5. Running the Test Suite
From the `backend` directory:
```bash
cd backend
pytest tests/ -v
```

### 6. Testing the API
- **Health check**:
  ```bash
  curl http://127.0.0.1:8000/health
  ```
- **Upload via curl**:
  ```bash
  # From repository root:
  curl -F "file=@backend/samples/Clip_06.wav" http://127.0.0.1:8000/analyze
  # Or from backend directory:
  curl -F "file=@samples/Clip_06.wav" http://127.0.0.1:8000/analyze
  ```
- **In-browser Recording**: Open `http://127.0.0.1:8000/`, click "Start Recording", speak or play environmental audio, click "Stop Recording", then click "Analyze Audio".
  *(Microphone capture uses the Web Audio API to produce standard 16-bit PCM WAV audio, ensuring cross-platform support without requiring FFmpeg).*

### Supported Audio Formats & Host Constraints
- **Native Formats**: WAV, OGG, and FLAC are decoded natively on the backend via `soundfile` and `librosa` without external system dependencies.
- **Microphone Capture**: The web UI directly records to 16 kHz 16-bit mono WAV, decodable on any platform out of the box.
- **Host Limitations**: Formats requiring FFmpeg (such as containerized WebM or MP4 uploads) require an installed `ffmpeg` binary on the host machine. If FFmpeg is not installed, uploading those formats returns HTTP 422 with an explanatory error.
- **Resource Limits**: Uploads are restricted to 10 MB (enforced via streaming chunk checks), and audio duration is strictly limited to 120 seconds (with bounded decoding).