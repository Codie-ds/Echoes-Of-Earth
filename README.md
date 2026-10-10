# Echoes of Earth

AI-powered environmental sound exploration.

## Project Phases
- **Phase 1**: Initial inference (YAMNet loading and audio analysis).
- **Phase 2**: The Evidence Engine (frame-level timestamped events).
- **Phase 3**: MVP Application (FastAPI backend and Web UI).

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

### 5. Testing the API
- **Health check**: Visit `http://127.0.0.1:8000/health`
- **Upload via browser**: Open the web UI, select a WAV file, and click "Analyze Audio".
- **Upload via curl**:
  ```bash
  curl -F "file=@backend/samples/Clip_06.wav" http://127.0.0.1:8000/analyze
  ```
- **Record via browser**: Open the web UI, click "Start Recording", speak or play a sound, then click "Stop Recording" and "Analyze Audio". 
  *(Note: Microphone access requires localhost or HTTPS).*

### Supported Audio Formats
The backend natively supports formats decodable by `librosa` and `soundfile` (such as WAV, OGG, and FLAC). When you record from the browser, it typically sends a `.webm` or `.ogg` blob. The application writes it to a temporary file and analyzes it. If ffmpeg is installed on the host machine, `librosa` (via soundfile/audioread) can handle `.webm`. If not, `.webm` analysis might fail, but `.wav` uploads will succeed.