from pathlib import Path
import json
import time
import csv

import librosa
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf

from evidence_engine import EvidenceEngine, asdict

MODEL_URL = "https://tfhub.dev/google/yamnet/1"

class YAMNetAnalyzer:
    def __init__(self, model_url=MODEL_URL):
        print("Loading YAMNet model...")
        self.model = hub.load(model_url)
        self.evidence_engine = EvidenceEngine(score_threshold=0.3, min_consecutive_frames=2)
        
        # Load labels
        class_map_path = self.model.class_map_path().numpy()
        if isinstance(class_map_path, bytes):
            class_map_path = class_map_path.decode("utf-8")

        with tf.io.gfile.GFile(class_map_path, "r") as f:
            reader = csv.DictReader(f)
            self.labels = [row["display_name"] for row in reader]

    def analyze_audio(self, file_path: str, filename: str = None):
        if filename is None:
            filename = Path(file_path).name

        start_time = time.time()
        
        # Load audio (this will resample to 16000 and convert to mono)
        try:
            audio, _ = librosa.load(str(file_path), sr=16000, mono=True)
            duration = round(len(audio) / 16000, 2)
            if duration > 120.0:
                raise ValueError(f"Audio duration ({duration}s) exceeds the maximum allowed 120 seconds.")
        except Exception as e:
            raise ValueError(f"Failed to load audio: {e}")

        waveform = tf.convert_to_tensor(audio, dtype=tf.float32)

        scores, embeddings, spectrogram = self.model(waveform)
        scores = scores.numpy()
        
        duration = round(len(audio) / 16000, 2)

        # Average predictions
        mean_scores = scores.mean(axis=0)
        top_indices = np.argsort(mean_scores)[::-1][:5]
        predictions = [
            {"label": self.labels[int(i)], "score": round(float(mean_scores[i]), 4)}
            for i in top_indices
        ]
        
        # Evidence Engine
        events = self.evidence_engine.process_frames(scores, self.labels, duration_seconds=duration)

        return {
            "filename": filename,
            "duration_seconds": duration,
            "model": "YAMNet",
            "events": [asdict(e) for e in events],
            "top_predictions": predictions,
            "limitations": [
                "Scores are model outputs, not calibrated probabilities.",
                "Event timestamps are approximate."
            ]
        }

def main():
    SAMPLES_DIR = Path("samples")
    RESULTS_DIR = Path("results")
    RESULTS_DIR.mkdir(exist_ok=True)
    
    analyzer = YAMNetAnalyzer()
    
    all_results = []
    csv_results = []
    
    for file in sorted(SAMPLES_DIR.glob("*.wav")):
        print(f"\nAnalyzing {file.name}...")
        start_time = time.time()
        result = analyzer.analyze_audio(str(file))
        
        all_results.append(result)
        
        predictions = result["top_predictions"]
        events = result["events"]
        
        top_pred_label = predictions[0]["label"] if predictions else ""
        top_pred_score = predictions[0]["score"] if predictions else 0.0
        events_str = "; ".join([f"{e['label']} ({e['start_seconds']}s-{e['end_seconds']}s, {e['peak_score']})" for e in events])
        
        csv_results.append({
            "filename": file.name,
            "verified_source_category": "Unverified",
            "top_prediction": top_pred_label,
            "top_score": top_pred_score,
            "relevant_top_five_match": "Unverified",
            "detected_events": events_str,
            "processing_time_seconds": round(time.time() - start_time, 2),
            "observations": f"{len(events)} events detected"
        })
        print(json.dumps(result, indent=2))

    output_path = RESULTS_DIR / "yamnet_evidence.json"
    output_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved evidence results to {output_path}")
    
    csv_path = RESULTS_DIR / "yamnet_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filename", "verified_source_category", "top_prediction", "top_score", 
            "relevant_top_five_match", "detected_events", "processing_time_seconds", "observations"
        ])
        writer.writeheader()
        writer.writerows(csv_results)
    print(f"Saved CSV results to {csv_path}")

if __name__ == "__main__":
    main()
