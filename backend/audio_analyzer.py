
from pathlib import Path
import json

import librosa
import numpy as np
import tensorflow_hub as hub

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

    for file in sorted(SAMPLES_DIR.glob("*.wav")):
        print(f"\nAnalyzing {file.name}...")

        audio, _ = librosa.load(
            str(file),
            sr=16000,
            mono=True,
        )
        waveform = tf.convert_to_tensor(audio, dtype=tf.float32)

        scores, embeddings, spectrogram = model(waveform)
        scores = scores.numpy()

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

        result = {
            "filename": file.name,
            "duration_seconds": round(len(audio) / 16000, 2),
            "top_predictions": predictions,
        }
        all_results.append(result)

        print(json.dumps(result, indent=2))

    output_path = RESULTS_DIR / "yamnet_results.json"
    output_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
