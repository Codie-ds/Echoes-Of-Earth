
from pathlib import Path
import json
import time

import librosa
import numpy as np
import tensorflow_hub as hub

from evidence_engine import EvidenceEngine, asdict

SAMPLES_DIR = Path("samples")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

MODEL_URL = "https://tfhub.dev/google/yamnet/1"


def main():
    print("Loading YAMNet model...")
    model = hub.load(MODEL_URL)
    class_names = (
        model.class_map_path().numpy()
        if hasattr(model.class_map_path(), "numpy")
        else model.class_map_path()
    )

    # YAMNet's class map is a CSV file, so read its labels.
    import csv
    import tensorflow as tf

    class_map_path = model.class_map_path().numpy()
    if isinstance(class_map_path, bytes):
        class_map_path = class_map_path.decode("utf-8")

    with tf.io.gfile.GFile(class_map_path, "r") as f:
        reader = csv.DictReader(f)
        labels = [row["display_name"] for row in reader]

    all_results = []
    csv_results = []
    
    # Initialize Evidence Engine
    evidence_engine = EvidenceEngine(score_threshold=0.3, min_consecutive_frames=2)

    for file in sorted(SAMPLES_DIR.glob("*.wav")):
        print(f"\nAnalyzing {file.name}...")
        start_time = time.time()

        audio, _ = librosa.load(
            str(file),
            sr=16000,
            mono=True,
        )
        waveform = tf.convert_to_tensor(audio, dtype=tf.float32)

        scores, embeddings, spectrogram = model(waveform)
        scores = scores.numpy()
        
        process_time = time.time() - start_time

        # Average predictions across audio frames.
        mean_scores = scores.mean(axis=0)
        top_indices = np.argsort(mean_scores)[::-1][:5]

        predictions = [
            {
                "label": labels[int(i)],
                "score": round(float(mean_scores[i]), 4),
            }
            for i in top_indices
        ]
        
        # New Evidence Engine logic
        duration = round(len(audio) / 16000, 2)
        events = evidence_engine.process_frames(scores, labels, duration_seconds=duration)

        result = {
            "filename": file.name,
            "duration_seconds": duration,
            "model": "YAMNet",
            "events": [asdict(e) for e in events],
            "top_predictions": predictions,
            "limitations": [
                "Scores are model outputs, not calibrated probabilities.",
                "Event timestamps are approximate."
            ]
        }
        all_results.append(result)

        # CSV row preparation
        top_pred_label = predictions[0]["label"] if predictions else ""
        top_pred_score = predictions[0]["score"] if predictions else 0.0
        events_str = "; ".join([f"{e.label} ({e.start_seconds}s-{e.end_seconds}s, {e.peak_score})" for e in events])
        
        # Find verified source category if we had a mapping (assume Unverified for now, based on instructions)
        verified_source = "Unverified"
        
        csv_results.append({
            "filename": file.name,
            "verified_source_category": verified_source,
            "top_prediction": top_pred_label,
            "top_score": top_pred_score,
            "relevant_top_five_match": "Unverified",  # Can't easily automate without ground truth
            "detected_events": events_str,
            "processing_time_seconds": round(process_time, 2),
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
