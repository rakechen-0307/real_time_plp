import pathlib
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from tqdm import tqdm

from realtimeplp_new import RealTimeBeatTracker, BeatAnalyzer

# === Configurations ===
DATA_PATH = pathlib.Path("/data/ssd0/rakec/slakh2100/processed/test")
OUT_DIR = pathlib.Path("/data/hdd0/rakec/test/realtimeplp_eval")
OUT_DIR.mkdir(exist_ok=True, parents=True)

SR = 48000
HOP = 1600
LOW = 90
HIGH = 180
LOOKAHEAD = 7

DURATION_SEC = 30
STRIDE_SEC = 15

def _crop(audio, sr, start_sec, duration_sec):
    """Crops a slice of audio based on start time and duration."""
    if duration_sec is None:
        return audio
    s = int(round(start_sec * sr))
    return audio[s : s + int(round(duration_sec * sr))]

# === Extract Offline Ground Truth (Drums) ===
def get_offline_gt_beats(audio, sr=48000, start_sec=0, duration_sec=None):
    """Uses librosa's offline tracker on the local cropped window."""
    audio = _crop(audio, sr, start_sec, duration_sec)
    _, beat_frames = librosa.beat.beat_track(y=audio, sr=sr, hop_length=512)
    return librosa.frames_to_time(beat_frames, sr=sr, hop_length=512)

# === Extract Real-Time Predictions (Guitar) ===
def get_realtime_predicted_beats(audio, sr=48000, hop=1600, start_sec=0, duration_sec=None):
    """Runs the real-time algorithm on the cropped stem and extracts soft/hard beats."""
    audio = _crop(audio, sr, start_sec, duration_sec)

    tracker = RealTimeBeatTracker.from_args(
        N=2*hop,
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

    frame_times = np.asarray(analyzer.frame_times, dtype=float)
    is_beat = np.asarray(analyzer.beat_detection_frames, dtype=bool)

    extras = dict(
        frame_times=frame_times,
        gamma_lfo=np.asarray(analyzer.gamma_lfos, dtype=float)
    )
    return frame_times[is_beat], extras

def main():
    tracks = sorted([t.name for t in DATA_PATH.iterdir() if t.is_dir()])

    for track in tqdm(tracks, ncols=140):
        track_path = DATA_PATH / track
        drum_path = track_path / "percussive.wav"
        guitar_paths = sorted(track_path.glob("guitar_*.wav"))

        if not drum_path.exists() or not guitar_paths:
            print(f"[INFO] Skipping {track}: Missing drum or guitar files.")
            continue

        drum_audio, _ = librosa.load(drum_path, sr=SR)
        guitar_audios = {p.stem: librosa.load(p, sr=SR)[0] for p in guitar_paths}
        audio_length = len(drum_audio)

        for start_sample in range(0, audio_length - DURATION_SEC * SR + 1, STRIDE_SEC * SR):
            start_sec = start_sample / SR

            try:
                gt_beat_times = get_offline_gt_beats(drum_audio, SR, start_sec, DURATION_SEC)
            except Exception as e:
                print(f"[WARN] GT failed {track}@{start_sec}: {e}")
                continue

            if len(gt_beat_times) < 4:  # drums silent / GT unreliable
                continue

            for gname, guitar_audio in guitar_audios.items():
                if len(guitar_audio) < start_sample + DURATION_SEC * SR:
                    continue

                pred_beat_times, extras = get_realtime_predicted_beats(
                    guitar_audio, SR, HOP, start_sec, DURATION_SEC
                )

                # Setup identifiers and background audio
                basename = f"{track}_{gname}_start{int(start_sec)}s"
                audio_crop = _crop(guitar_audio, SR, start_sec, DURATION_SEC)
                time_axis = np.arange(len(audio_crop)) / SR

                # --- Plotting ---
                fig, axs = plt.subplots(2, 1, figsize=(16, 8), sharex=True, height_ratios=[1.5, 1])

                # Subplot 1: Waveform + Beats
                axs[0].set_title(f"Beat Tracker Evaluation: {basename}")
                axs[0].plot(time_axis, audio_crop, color='lightgray', label='Guitar Waveform')
                axs[0].vlines(gt_beat_times, ymin=-1.1, ymax=1.1, color='green', alpha=0.8,
                              linestyle='-', linewidth=2, label='Ground Truth (Offline Drums)')
                axs[0].vlines(pred_beat_times, ymin=-1.1, ymax=1.1, color='red', alpha=0.8,
                              linestyle='--', linewidth=2, label='Predicted (Real-Time Guitar)')
                axs[0].set_ylabel("Amplitude")
                axs[0].set_ylim(-1.1, 1.1)
                axs[0].legend(loc='upper right')

                # Subplot 2: Gamma LFO
                axs[1].plot(extras["frame_times"], extras["gamma_lfo"], color='C2', label=r'$\gamma$-LFO')
                axs[1].axhline(0, color='gray', linestyle='--')
                axs[1].set_xlabel("Time (seconds)")
                axs[1].set_ylabel(r'$\gamma$-LFO')
                axs[1].set_ylim(-1.1, 1.1)
                axs[1].set_xlim(0, max(time_axis))
                axs[1].legend(loc='upper right')

                plt.tight_layout()
                plt.savefig(OUT_DIR / f"{basename}.png", dpi=150)
                plt.close(fig)

                # --- Audio Sonification ---
                # Generate click track at the predicted times and mix it over the original audio
                clicks = librosa.clicks(times=pred_beat_times, sr=SR, length=len(audio_crop))
                mixed_audio = (audio_crop * 0.7) + clicks
                sf.write(OUT_DIR / f"{basename}.wav", mixed_audio, SR)

if __name__ == "__main__":
    main()