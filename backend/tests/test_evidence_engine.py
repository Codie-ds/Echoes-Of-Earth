import pytest
import sys
from pathlib import Path

# Add backend directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evidence_engine import EvidenceEngine, SoundEvent

def test_get_category():
    engine = EvidenceEngine()
    assert engine.get_category("Dog") == "Birds and animals"
    assert engine.get_category("Rain") == "Water and rain"
    assert engine.get_category("RandomSound") == "Unknown or unclassified"

def test_sustained_detection():
    engine = EvidenceEngine(score_threshold=0.3, min_consecutive_frames=2)
    scores = [[0.1], [0.4], [0.5], [0.1]]
    labels = ["Dog"]
    events = engine.process_frames(scores, labels)
    assert len(events) == 1
    event = events[0]
    assert event.label == "Dog"
    assert event.start_seconds == 0.48
    assert event.end_seconds == 2 * 0.48 + 0.96
    assert event.peak_score == 0.5
    assert event.mean_score == 0.45

def test_fails_score_threshold():
    engine = EvidenceEngine(score_threshold=0.6, min_consecutive_frames=2)
    scores = [[0.1], [0.4], [0.5], [0.1]]
    labels = ["Dog"]
    events = engine.process_frames(scores, labels)
    assert len(events) == 0

def test_fails_persistence_requirement():
    engine = EvidenceEngine(score_threshold=0.3, min_consecutive_frames=3)
    scores = [[0.1], [0.4], [0.5], [0.1]]
    labels = ["Dog"]
    events = engine.process_frames(scores, labels)
    assert len(events) == 0

def test_separating_events_gap():
    engine = EvidenceEngine(score_threshold=0.3, min_consecutive_frames=2)
    scores = [[0.4], [0.4], [0.1], [0.4], [0.4]]
    labels = ["Dog"]
    events = engine.process_frames(scores, labels)
    assert len(events) == 2
    assert events[0].start_seconds == 0.0
    assert events[1].start_seconds == 3 * 0.48

def test_overlapping_labels():
    engine = EvidenceEngine(score_threshold=0.3, min_consecutive_frames=2)
    scores = [[0.4, 0.1], [0.4, 0.4], [0.1, 0.4]]
    labels = ["Dog", "Rain"]
    events = engine.process_frames(scores, labels)
    assert len(events) == 2
    # Check that we got one event for Dog and one for Rain
    labels_found = [e.label for e in events]
    assert "Dog" in labels_found
    assert "Rain" in labels_found

def test_empty_input():
    engine = EvidenceEngine()
    events = engine.process_frames([], [])
    assert len(events) == 0
