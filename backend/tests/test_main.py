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

def test_analyze_duration_limit():
    import io
    import soundfile as sf
    import numpy as np
    
    # Synthesize a 125-second silent WAV file in memory (exceeds 120s limit)
    buf = io.BytesIO()
    samples = np.zeros(16000 * 125, dtype=np.float32)
    sf.write(buf, samples, 16000, format="WAV")
    audio_bytes = buf.getvalue()
    
    with TestClient(app) as client:
        response = client.post("/analyze", files={"file": ("too_long.wav", audio_bytes)})
        assert response.status_code == 400
        assert "exceeds the maximum allowed 120 seconds" in response.json()["detail"]
        
    temp_dir = Path(__file__).resolve().parent.parent / "temp_uploads"
    if temp_dir.exists():
        assert len(list(temp_dir.glob("*.wav"))) == 0

def test_temp_cleanup_on_all_failures():
    temp_dir = Path(__file__).resolve().parent.parent / "temp_uploads"
    with TestClient(app) as client:
        # 1. Empty file
        client.post("/analyze", files={"file": ("empty.wav", b"")})
        # 2. Oversized file
        client.post("/analyze", files={"file": ("large.wav", b"0" * (10 * 1024 * 1024 + 1))})
        # 3. Invalid audio
        client.post("/analyze", files={"file": ("invalid.wav", b"corrupted bytes")})
        
    if temp_dir.exists():
        # Confirm no orphaned files remain in temp_uploads
        assert len(list(temp_dir.glob("*.*"))) == 0

def test_analyze_browser_wav_recording():
    import io
    import soundfile as sf
    import numpy as np
    
    # Synthesize a 2-second 16 kHz mono WAV file mimicking in-browser Web Audio API recording
    buf = io.BytesIO()
    samples = np.zeros(16000 * 2, dtype=np.float32)
    sf.write(buf, samples, 16000, format="WAV", subtype="PCM_16")
    wav_bytes = buf.getvalue()
    
    with TestClient(app) as client:
        response = client.post("/analyze", files={"file": ("browser_recording.wav", wav_bytes)})
        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "browser_recording.wav"
        assert data["duration_seconds"] == 2.0
        assert data["model"] == "YAMNet"
        assert isinstance(data["events"], list)
        assert len(data["top_predictions"]) == 5

def test_concurrent_requests():
    import concurrent.futures
    
    sample_path = Path(__file__).resolve().parent.parent / "samples" / "Clip_06.wav"
    with open(sample_path, "rb") as f:
        clip_bytes = f.read()
        
    with TestClient(app) as client:
        def make_request(i):
            if i % 2 == 0:
                resp = client.get("/health")
                return resp.status_code == 200 and resp.json()["status"] == "ok"
            else:
                resp = client.post("/analyze", files={"file": (f"req_{i}.wav", clip_bytes)})
                return resp.status_code == 200 and resp.json()["model"] == "YAMNet"
                
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(make_request, range(4)))
            
        assert all(results)

def test_path_traversal_filename_sanitization():
    from unittest.mock import patch
    import io
    import soundfile as sf
    import numpy as np

    # Synthesize small valid WAV
    buf = io.BytesIO()
    samples = np.zeros(16000, dtype=np.float32)
    sf.write(buf, samples, 16000, format="WAV", subtype="PCM_16")
    wav_bytes = buf.getvalue()

    # Attempt directory traversal filenames
    traversal_filenames = [
        "../../../../etc/passwd.wav",
        "..\\..\\secret\\exploit.wav",
        "....//....//nested.wav"
    ]

    with TestClient(app) as client:
        for fname in traversal_filenames:
            resp = client.post("/analyze", files={"file": (fname, wav_bytes)})
            assert resp.status_code == 200
            data = resp.json()
            # Ensure path traversal components are completely removed from response
            assert "/" not in data["filename"]
            assert "\\" not in data["filename"]
            assert ".." not in data["filename"]
            
    # Verify no leaked files in parent or root directories
    temp_dir = Path(__file__).resolve().parent.parent / "temp_uploads"
    if temp_dir.exists():
        assert len(list(temp_dir.glob("*.wav"))) == 0

def test_generic_error_response_no_stack_trace_leakage():
    from unittest.mock import patch
    import io
    import soundfile as sf
    import numpy as np

    buf = io.BytesIO()
    samples = np.zeros(16000, dtype=np.float32)
    sf.write(buf, samples, 16000, format="WAV", subtype="PCM_16")
    wav_bytes = buf.getvalue()

    with TestClient(app) as client:
        # Mock an unexpected internal exception with internal paths and sensitive text
        with patch("main.analyzer.analyze_audio", side_effect=RuntimeError("Internal C++ crash at /var/private/secrets/core.cpp:99")):
            resp = client.post("/analyze", files={"file": ("test.wav", wav_bytes)})
            assert resp.status_code == 500
            data = resp.json()
            # Ensure generic response
            assert data["detail"] == "An internal error occurred during audio analysis."
            # Confirm no internal paths or exception details leaked in response body
            assert "/var/private/secrets" not in resp.text
            assert "core.cpp" not in resp.text
            assert "RuntimeError" not in resp.text

def test_cleanup_on_unexpected_server_error():
    from unittest.mock import patch
    import io
    import soundfile as sf
    import numpy as np

    buf = io.BytesIO()
    samples = np.zeros(16000, dtype=np.float32)
    sf.write(buf, samples, 16000, format="WAV", subtype="PCM_16")
    wav_bytes = buf.getvalue()

    temp_dir = Path(__file__).resolve().parent.parent / "temp_uploads"

    with TestClient(app) as client:
        with patch("main.analyzer.analyze_audio", side_effect=Exception("Database or model died")):
            resp = client.post("/analyze", files={"file": ("crash.wav", wav_bytes)})
            assert resp.status_code == 500
            
    # Verify cleanup occurred even when inference crashed
    if temp_dir.exists():
        assert len(list(temp_dir.glob("*.*"))) == 0

def test_stale_temp_cleanup_on_lifespan_startup():
    temp_dir = Path(__file__).resolve().parent.parent / "temp_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    stale_file = temp_dir / "orphaned_crash_file.wav"
    stale_file.write_bytes(b"leftover from previous crash")
    assert stale_file.exists()

    # Initializing a new TestClient executes the lifespan context manager
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200

    # Lifespan must have wiped the stale file
    assert not stale_file.exists()

def test_model_not_ready_returns_503():
    from unittest.mock import patch
    with TestClient(app) as client:
        with patch("main.analyzer", None):
            resp = client.post("/analyze", files={"file": ("test.wav", b"dummy")})
            assert resp.status_code == 503
            assert "Model is not ready" in resp.json()["detail"]
