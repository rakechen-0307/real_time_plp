import librosa
import numpy as np
import mir_eval
from realtimeplp_new import RealTimeBeatTracker, BeatAnalyzer

# Configuration
DRUM_WAV = "/data/ssd0/rakec/slakh2100/processed/test/Track01876/percussive.wav"
GUITAR_WAV = "/data/ssd0/rakec/slakh2100/processed/test/Track01876/guitar_0.wav"

SR = 48000
HOP = 1600
LOW = 60
HIGH = 180
LOOKAHEAD = 20

START_SEC = 20
DURATION_SEC = 30

# Helper: Extract Beat Times with Tracker
def extract_beats_from_audio(audio_path, sr=48000, hop=1600, start_sec=None, duration_sec=None):
    """Loads an audio file and runs the real-time tracker to get beat timestamps (seconds)."""
    audio, _ = librosa.load(audio_path, sr=sr)

    if start_sec is not None and duration_sec is not None:
        audio = audio[start_sec * sr : (start_sec + duration_sec) * sr]

    tracker = RealTimeBeatTracker.from_args(
        N=2 * hop,
        H=hop,
        samplerate=sr,
        N_time=6,
        Theta=np.arange(LOW, HIGH + 1, 1),
        lookahead=LOOKAHEAD,
    )
    analyzer = BeatAnalyzer(tracker)

    # Process frame by frame
    for i in range(0, len(audio), hop):
        frame = audio[i : i + hop]
        if len(frame) < hop:
            frame = np.pad(frame, (0, hop - len(frame)), mode="constant")
        analyzer.process(audio_frame=frame)

    # Filter frame times where a beat was detected
    beat_times = np.array([
        t for t, is_beat in zip(analyzer.frame_times, analyzer.beat_detection_frames)
        if is_beat
    ], dtype=float)

    return beat_times

# Extract Ground Truth & Predictions
print("Extracting reference (GT) beats from drums...")
gt_beat_times = extract_beats_from_audio(
    DRUM_WAV, sr=SR, hop=HOP, start_sec=START_SEC, duration_sec=DURATION_SEC
)

print("Extracting predicted beats from guitar...")
pred_beat_times = extract_beats_from_audio(
    GUITAR_WAV, sr=SR, hop=HOP, start_sec=START_SEC, duration_sec=DURATION_SEC
)

print(f"\nTotal Reference (GT) Beats : {len(gt_beat_times)}")
print(f"Total Predicted Beats      : {len(pred_beat_times)}")

# Compute Metrics
# Ensure inputs are valid 1D float arrays with >= 2 beats
if len(gt_beat_times) < 2 or len(pred_beat_times) < 2:
    raise ValueError("At least 2 beats are required in both GT and prediction to compute metrics.")

mir_eval.beat.validate(gt_beat_times, pred_beat_times)
scores = mir_eval.beat.evaluate(gt_beat_times, pred_beat_times)
f_measure = mir_eval.beat.f_measure(gt_beat_times, pred_beat_times, f_measure_threshold=0.070)

print("\n" + "=" * 45)
print("             EVALUATION RESULTS              ")
print("=" * 45)
print(f"F-measure (±70 ms)   : {f_measure:.4f}")
print(f"Cemgil Accuracy      : {scores['Cemgil']:.4f}")
print(f"Information Gain     : {scores['Information gain']:.4f}")
print(f"P-Score              : {scores['P-score']:.4f}")
print("-" * 45)
print(f"Continuity - CMLc    : {scores['Correct Metric Level Continuous']:.4f}")
print(f"Continuity - CMLt    : {scores['Correct Metric Level Total']:.4f}")
print(f"Continuity - AMLc    : {scores['Any Metric Level Continuous']:.4f}")
print(f"Continuity - AMLt    : {scores['Any Metric Level Total']:.4f}")
print("=" * 45)