import csv
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any

@dataclass
class SoundEvent:
    label: str
    category: str
    start_seconds: float
    end_seconds: float
    peak_score: float
    mean_score: float
    status: str = "detected"

class EvidenceEngine:
    """
    Analyzes frame-level predictions to identify sustained sound events.
    """
    # Define broad categories for mapping YAMNet labels
    CATEGORY_MAPPING = {
        "Animal": "Birds and animals",
        "Bird": "Birds and animals",
        "Dog": "Birds and animals",
        "Cat": "Birds and animals",
        "Wind": "Wind and weather",
        "Thunderstorm": "Wind and weather",
        "Rain": "Water and rain",
        "Water": "Water and rain",
        "Speech": "Human activity",
        "Footsteps": "Human activity",
        "Car": "Transport",
        "Vehicle": "Transport",
        "Engine": "Transport",
        "Train": "Transport",
        "Fire": "Fire and crackling",
        "Crackle": "Fire and crackling",
        "Music": "Music and other sounds",
    }
    
    DEFAULT_CATEGORY = "Unknown or unclassified"

    def __init__(self, score_threshold: float = 0.3, min_consecutive_frames: int = 2):
        self.score_threshold = score_threshold
        self.min_consecutive_frames = min_consecutive_frames
        self.frame_step_seconds = 0.48
        self.frame_duration_seconds = 0.96

    def get_category(self, label: str) -> str:
        label_lower = label.lower()
        for key, category in self.CATEGORY_MAPPING.items():
            if key.lower() in label_lower:
                return category
        return self.DEFAULT_CATEGORY

    def process_frames(self, scores: Any, labels: List[str], duration_seconds: Optional[float] = None) -> List[SoundEvent]:
        """
        Process frame-level scores and return detected sound events.
        `scores` should be an array of shape (num_frames, num_classes).
        `labels` is a list of class names corresponding to the columns in `scores`.
        """
        events = []
        num_frames = len(scores)
        
        # Process each class independently
        for class_idx, label in enumerate(labels):
            class_scores = [scores[i][class_idx] for i in range(num_frames)]
            
            # Find intervals where score >= score_threshold
            current_interval = []
            
            for frame_idx, score in enumerate(class_scores):
                if score >= self.score_threshold:
                    current_interval.append((frame_idx, score))
                else:
                    if len(current_interval) >= self.min_consecutive_frames:
                        events.append(self._create_event(label, current_interval, duration_seconds))
                    current_interval = []
            
            # Check last interval
            if len(current_interval) >= self.min_consecutive_frames:
                events.append(self._create_event(label, current_interval, duration_seconds))

        # Sort events by start time, then peak score (descending)
        events.sort(key=lambda e: (e.start_seconds, -e.peak_score))
        return events
        
    def _create_event(self, label: str, interval: List[tuple], duration_seconds: Optional[float]) -> SoundEvent:
        frame_indices = [idx for idx, score in interval]
        scores = [score for idx, score in interval]
        
        start_frame = frame_indices[0]
        end_frame = frame_indices[-1]
        
        start_time = start_frame * self.frame_step_seconds
        end_time = end_frame * self.frame_step_seconds + self.frame_duration_seconds
        
        if duration_seconds is not None:
            start_time = min(start_time, duration_seconds)
            end_time = min(end_time, duration_seconds)
        
        return SoundEvent(
            label=label,
            category=self.get_category(label),
            start_seconds=round(start_time, 2),
            end_seconds=round(end_time, 2),
            peak_score=round(float(max(scores)), 4),
            mean_score=round(float(sum(scores)/len(scores)), 4)
        )
