import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from realtimeplp_new import RealTimeBeatTracker, BeatAnalyzer

# ==========================================
# 1. Configuration
# ==========================================
DRUM_WAV = "/data/ssd0/rakec/slakh2100/processed/test/Track01876/percussive.wav"
GUITAR_WAV = "/data/ssd0/rakec/slakh2100/processed/test/Track01876/guitar_0.wav"

SR = 48000
HOP = 1600
LOW = 90
HIGH = 180
LOOKAHEAD = 7
START_SEC = 20
DURATION_SEC = 30

# ==========================================
# 2. Extract Offline Ground Truth (Drums)
# ==========================================
def get_offline_gt_beats(audio_path, sr=48000, start_sec=0, duration_sec=None):
    """Uses librosa's offline, non-causal tracker to establish the perfect reference grid."""
    audio, _ = librosa.load(audio_path, sr=sr)
    if duration_sec is not None:
        audio = audio[start_sec * sr : (start_sec + duration_sec) * sr]

    # Offline tracking looks at the whole file to find the true mathematical beat
    _, beat_frames = librosa.beat.beat_track(y=audio, sr=sr, hop_length=512)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=512)
    return beat_times

# ==========================================
# 3. Extract Real-Time Predictions (Guitar)
# ==========================================
def get_realtime_predicted_beats(audio_path, sr=48000, hop=1600, start_sec=0, duration_sec=None):
    """Runs your custom real-time algorithm on the target stem."""
    audio, _ = librosa.load(audio_path, sr=sr)
    if duration_sec is not None:
        audio = audio[start_sec * sr : (start_sec + duration_sec) * sr]

    tracker = RealTimeBeatTracker.from_args(
        N=2 * hop,
        H=hop,
        samplerate=sr,
        Theta=np.arange(LOW, HIGH + 1, 1),
        lookahead=LOOKAHEAD
    )
    analyzer = BeatAnalyzer(tracker)

    for i in range(0, len(audio), hop):
        frame = audio[i : i + hop]
        if len(frame) < hop:
            frame = np.pad(frame, (0, hop - len(frame)), mode="constant")
        analyzer.process(audio_frame=frame)

    beat_times = np.array([
        t for t, is_beat in zip(analyzer.frame_times, analyzer.beat_detection_frames)
        if is_beat
    ], dtype=float)
    return beat_times

# ==========================================
# 4. Execution
# ==========================================
print("Extracting Offline GT beats from drums...")
gt_beat_times = get_offline_gt_beats(DRUM_WAV, SR, START_SEC, DURATION_SEC)

print("Extracting Real-Time predicted beats from guitar...")
pred_beat_times = get_realtime_predicted_beats(GUITAR_WAV, SR, HOP, START_SEC, DURATION_SEC)

if len(gt_beat_times) < 2 or len(pred_beat_times) < 2:
    raise ValueError("Not enough beats detected to perform evaluation.")

# ==========================================
# 5. Visual Sanity Check: Plot & Save Image
# ==========================================
print("Generating waveform plot with overlaid beats...")

# Load the guitar audio slice for the background waveform
audio_plot, _ = librosa.load(GUITAR_WAV, sr=SR)
if DURATION_SEC is not None:
    audio_plot = audio_plot[START_SEC * SR : (START_SEC + DURATION_SEC) * SR]

time_axis = np.arange(len(audio_plot)) / SR

# Create the plot
plt.figure(figsize=(16, 4))
plt.plot(time_axis, audio_plot, color='lightgray', label='Guitar Waveform')

# Plot Ground Truth Beats (Green solid lines)
plt.vlines(gt_beat_times, ymin=-1.1, ymax=1.1, color='green', alpha=0.8,
           linestyle='-', linewidth=2, label='Ground Truth (Offline Drums)')

# Plot Predicted Beats (Red dashed lines)
plt.vlines(pred_beat_times, ymin=-1.1, ymax=1.1, color='red', alpha=0.8,
           linestyle='--', linewidth=2, label='Predicted (Real-Time Guitar)')

plt.title("Beat Tracker Evaluation: Real-Time vs Offline Ground Truth")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.xlim(0, max(time_axis))
plt.ylim(-1.1, 1.1)

# Put the legend outside the plot so it doesn't cover the waveform
plt.legend(loc='upper right', bbox_to_anchor=(1, 1.15), ncol=3)
plt.tight_layout()

# Save the image to your directory
output_image = "compare.png"
plt.savefig(output_image, dpi=300)
print(f"Plot saved successfully as '{output_image}'")

# ==========================================
# 6. Audio Sonification (Click Tracks)
# ==========================================
print("Generating sonified audio with click tracks...")

# Mix 1: Predicted Beats
clicks_pred = librosa.clicks(times=pred_beat_times, sr=SR, length=len(audio_plot))
mixed_audio_pred = (audio_plot * 0.7) + clicks_pred
pred_output = "sonified_beats_pred.wav"
sf.write(pred_output, mixed_audio_pred, SR)
print(f"Predicted sonification saved successfully as '{pred_output}'")

# Mix 2: Ground Truth Beats
clicks_gt = librosa.clicks(times=gt_beat_times, sr=SR, length=len(audio_plot))
mixed_audio_gt = (audio_plot * 0.7) + clicks_gt
gt_output = "sonified_beats_gt.wav"
sf.write(gt_output, mixed_audio_gt, SR)
print(f"Ground truth sonification saved successfully as '{gt_output}'")