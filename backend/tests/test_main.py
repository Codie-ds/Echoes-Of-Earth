import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import json

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import app, analyzer

client = TestClient(app)

def test_health_check():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_ready"] is True

def test_analyze_empty_file():
    with TestClient(app) as client:
        response = client.post("/analyze", files={"file": ("empty.wav", b"")})
        assert response.status_code == 400
        assert response.json()["detail"] == "File is empty."

def test_analyze_large_file():
    large_bytes = b"0" * (10 * 1024 * 1024 + 1)
    with TestClient(app) as client:
        response = client.post("/analyze", files={"file": ("large.wav", large_bytes)})
        assert response.status_code == 413
        assert "too large" in response.json()["detail"]

def test_analyze_invalid_audio():
    # Send non-audio text file
    with TestClient(app) as client:
        response = client.post("/analyze", files={"file": ("invalid.wav", b"Not an audio file")})
        assert response.status_code == 422
        assert "Failed to process audio" in response.json()["detail"]

def test_analyze_valid_audio():
    # Use one of the real WAV files from samples to test the full pipeline
    sample_path = Path(__file__).resolve().parent.parent / "samples" / "Clip_06.wav"
    
    if not sample_path.exists():
        pytest.skip(f"Sample file {sample_path} not found")
        
    with open(sample_path, "rb") as f:
        file_bytes = f.read()
        
    with TestClient(app) as client:
        response = client.post("/analyze", files={"file": ("Clip_06.wav", file_bytes)})
        
        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "Clip_06.wav"
    assert "duration_seconds" in data
    assert data["model"] == "YAMNet"
    assert "events" in data
    assert "top_predictions" in data
    assert "limitations" in data

    # Check cleanup
    temp_dir = Path(__file__).resolve().parent.parent / "temp_uploads"
    if temp_dir.exists():
        # Any file created should be deleted by the finally block
        assert len(list(temp_dir.glob("*.wav"))) == 0
