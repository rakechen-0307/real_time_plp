"""
eval_beat_this.py  --  standalone ceiling test: Beat This! (offline, non-causal) vs MIDI ground truth.

Usage:
  python eval_beat_this.py                                    # 100 random guitar stems from the test split
  python eval_beat_this.py --plp_csv /data/hdd0/rakec/test/realtimeplp_eval_gt/tracks.csv
  python eval_beat_this.py --stem_glob percussive.wav         # sanity check (should be good; offset ~ 0 ms)
  python eval_beat_this.py --stem_glob mixture.wav            # full-mix reference

Outputs (OUT_DIR): tracks.csv, windows.csv, summary.png, renders/*.png|wav, cache/*.npz
"""
import json
import pathlib
import argparse
import warnings

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import pretty_midi
import mir_eval
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

warnings.filterwarnings("ignore", module="mir_eval")

# === Configuration ===
PROCESSED = pathlib.Path("/data/ssd0/rakec/slakh2100/processed")
OUT_DIR = pathlib.Path("/data/hdd0/rakec/test/beat_this_eval")
LOW_BPM, HIGH_BPM = 90, 180          # tracker range, used for octave folding of the GT
MIN_GRID_SCORE = 1.3                 # skip tracks whose MIDI tempo map looks wrong (convert.py beat check)
DURATION_SEC, STRIDE_SEC = 30, 15
TRIM_SEC = 5.0                       # ignore the first seconds (same as compare_full.py)
BT_SR = 22050                        # Beat This! native sample rate
EVENT_KEYS = ["F", "Cemgil", "CMLc", "CMLt", "AMLc", "AMLt"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test")
    p.add_argument("--stem_glob", default="guitar_*.wav", help="e.g. guitar_*.wav, percussive.wav, mixture.wav")
    p.add_argument("--num_stems", type=int, default=100)
    p.add_argument("--all_stems", action="store_true", help="allow several stems per track (default: one per track)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint", default="final0")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dbn", action="store_true", help="use madmom DBN postprocessing (default: peak picking)")
    p.add_argument("--plp_csv", default=None, help="tracks.csv written by compare_full.py (same split/stems)")
    p.add_argument("--render", type=int, default=6, help="number of stems to plot / sonify (worst by raw AMLt)")
    p.add_argument("--no_skip_bad_gt", action="store_true")
    return p.parse_args()


# === Ground truth from MIDI (same math as pre_encode.py / compare_full.py) ===
def fold_factor(beats, low_bpm, high_bpm):
    median_bpm = 60.0 / max(float(np.median(np.diff(beats))), 1e-6)
    f = 1.0
    while median_bpm * f < low_bpm:
        f *= 2.0
    while median_bpm * f > high_bpm:
        f *= 0.5
    return f


def folded_beat_times(beats, f, t_max):
    idx = np.arange(beats.size, dtype=np.float64)
    x = np.arange(0.0, beats.size - 1 + 1e-9, 1.0 / f)
    t = np.interp(x, idx, beats)
    return t[t <= t_max]


def load_gt(midi_path, duration):
    beats = np.asarray(pretty_midi.PrettyMIDI(str(midi_path)).get_beats(), dtype=np.float64)
    if beats.size < 2:
        raise ValueError("fewer than 2 beats in MIDI")
    f = fold_factor(beats, LOW_BPM, HIGH_BPM)
    return beats[beats <= duration], folded_beat_times(beats, f, duration), f


# === Beat This! ===
def load_beat_this(checkpoint, device, dbn):
    try:
        from beat_this.inference import Audio2Beats
    except ImportError as e:
        raise SystemExit(
            "beat_this is not installed:\n  pip install https://github.com/CPJKU/beat_this/archive/main.zip"
        ) from e
    return Audio2Beats(checkpoint_path=checkpoint, device=device, dbn=dbn)


def run_beat_this(model, mono, sr, cache_path):
    if cache_path.exists():
        d = np.load(cache_path)
        return d["beats"], d["downbeats"]
    y = librosa.resample(mono.astype(np.float32), orig_sr=sr, target_sr=BT_SR)
    beats, downbeats = model(y, BT_SR)
    beats, downbeats = np.asarray(beats, dtype=np.float64), np.asarray(downbeats, dtype=np.float64)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, beats=beats, downbeats=downbeats)
    return beats, downbeats


# === Metrics (same definitions as compare_full.py) ===
def event_metrics(gt, est, t0, t1, prefix):
    g = gt[(gt >= t0) & (gt < t1)]
    e = est[(est >= t0) & (est < t1)]
    out = {prefix + k: np.nan for k in EVENT_KEYS + ["off_med_ms", "off_mad_ms", "ibi_ratio"]}
    out[prefix + "n_gt"], out[prefix + "n_est"] = int(len(g)), int(len(e))
    if len(g) < 4:
        return out
    if len(e) < 2:                                   # nothing detected counts as failure
        for k in EVENT_KEYS:
            out[prefix + k] = 0.0
        return out
    try:
        out[prefix + "F"] = float(mir_eval.beat.f_measure(g, e))             # +-70 ms
        out[prefix + "Cemgil"] = float(mir_eval.beat.cemgil(g, e)[0])
        cmlc, cmlt, amlc, amlt = mir_eval.beat.continuity(g, e)
        out[prefix + "CMLc"], out[prefix + "CMLt"] = float(cmlc), float(cmlt)
        out[prefix + "AMLc"], out[prefix + "AMLt"] = float(amlc), float(amlt)
    except Exception:
        pass
    j = np.clip(np.searchsorted(g, e), 1, len(g) - 1)
    nearest = np.where(np.abs(e - g[j - 1]) < np.abs(e - g[j]), g[j - 1], g[j])
    d = e - nearest
    d = d[np.abs(d) < 0.15]                          # signed offset, > 0 means the estimate is late
    if d.size:
        med = float(np.median(d))
        out[prefix + "off_med_ms"] = med * 1000.0
        out[prefix + "off_mad_ms"] = float(np.median(np.abs(d - med))) * 1000.0
    out[prefix + "ibi_ratio"] = float(np.median(np.diff(e)) / max(np.median(np.diff(g)), 1e-6))
    return out


def evaluate_range(est, gt_raw, gt_fold, t0, t1):
    row = {}
    row.update(event_metrics(gt_raw, est, t0, t1, "raw_"))
    row.update(event_metrics(gt_fold, est, t0, t1, "fold_"))
    return row


def active_frames(mono, sr, fps=30.0, silence_db=-50.0):
    """Per-frame 'is there sound' flag at the latent frame rate (same level threshold as the tracker's gate)."""
    hop = int(round(sr / fps))
    n = len(mono) // hop
    x = mono[:n * hop].reshape(n, hop)
    db = 20 * np.log10(np.sqrt((x ** 2).mean(axis=1)) + 1e-12)
    return db > silence_db, hop / sr


# === Reporting ===
def summarize(trk, win, plp):
    cols = ["raw_F", "raw_AMLt", "raw_CMLt", "fold_F", "fold_AMLt", "fold_CMLt",
            "raw_off_med_ms", "chance_raw_F", "chance_raw_AMLt", "chance_fold_F", "chance_fold_AMLt", "active_frac"]
    cols = [c for c in cols if c in trk.columns]
    pd.set_option("display.width", 170)
    print(f"\n=== Beat This! vs MIDI GT: {len(trk)} stems, {len(win)} windows ===")
    print(trk[cols].agg(["mean", "median"]).T.round(3))

    r = trk["raw_ibi_ratio"].dropna()
    if len(r):
        print("\nMetrical level (median estimated beat interval / true beat interval):")
        print(f"  same level  : {(np.abs(r - 1) < 0.1).mean():.1%}")
        print(f"  half-time   : {(np.abs(r - 2) < 0.2).mean():.1%}")
        print(f"  double-time : {(np.abs(r - 0.5) < 0.1).mean():.1%}")
        print(f"  other       : {1 - ((np.abs(r - 1) < 0.1) | (np.abs(r - 2) < 0.2) | (np.abs(r - 0.5) < 0.1)).mean():.1%}")

    if len(win):
        bins = pd.cut(win["active_frac"], [-0.01, 0.5, 0.9, 1.01], labels=["<50% active", "50-90%", ">90%"])
        print("\nWindow scores by fraction of the window with sound:")
        print(win.groupby(bins, observed=True)[["raw_F", "raw_AMLt", "fold_F", "fold_AMLt"]]
              .agg(["mean", "count"]).round(3))

    if plp is not None:
        m = trk.merge(plp, on=["track", "stem"], how="inner", suffixes=("", "_plp"))
        print(f"\n=== Paired comparison with PLP on {len(m)} matched stems ===")
        if len(m):
            pairs = [("fold_F", "pk_F"), ("fold_AMLt", "pk_AMLt"), ("fold_CMLt", "pk_CMLt")]
            for a, b in pairs:
                if b in m.columns:
                    ok = m[[a, b]].dropna()
                    print(f"  Beat This! {a:10s} mean {ok[a].mean():.3f} median {ok[a].median():.3f} | "
                          f"PLP {b:10s} mean {ok[b].mean():.3f} median {ok[b].median():.3f} | "
                          f"Beat This! better on {(ok[a] > ok[b]).mean():.0%} of stems")
            if "feat_corr_best" in m.columns:
                print(f"  PLP feature correlation with GT LFO: mean {m['feat_corr_best'].mean():.3f}, "
                      f"median {m['feat_corr_best'].median():.3f}")
        return m
    return None


def summary_plot(trk, merged, path):
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.5))
    b = np.linspace(0, 1, 21)
    axs[0].hist(trk["raw_F"].dropna(), b, alpha=0.7, label="Beat This! vs true beats")
    axs[0].hist(trk["fold_F"].dropna(), b, alpha=0.5, label="Beat This! vs folded GT")
    axs[0].hist(trk["chance_raw_F"].dropna(), b, alpha=0.4, label="chance")
    axs[0].set_title("F-measure (+-70 ms)"); axs[0].legend()
    axs[1].hist(trk["raw_AMLt"].dropna(), b, alpha=0.7, label="vs true beats")
    axs[1].hist(trk["chance_raw_AMLt"].dropna(), b, alpha=0.4, label="chance")
    axs[1].set_title("AMLt"); axs[1].legend()
    if merged is not None and "pk_AMLt" in merged.columns and len(merged):
        axs[2].scatter(merged["pk_AMLt"], merged["fold_AMLt"], s=12, alpha=0.6)
        axs[2].plot([0, 1], [0, 1], "k--", lw=0.8)
        axs[2].set_xlabel("PLP AMLt"); axs[2].set_ylabel("Beat This! AMLt (folded GT)")
        axs[2].set_title("Per-stem comparison")
    else:
        axs[2].hist(trk["raw_off_med_ms"].dropna(), np.linspace(-100, 100, 41))
        axs[2].set_title("Median signed offset (ms, > 0 = late)")
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close(fig)


def render(row, payload, sr_audio, path_dir):
    p = payload[(row["track"], row["stem"])]
    mono, est, gt = p["mono"], p["est"], p["gt_raw"]
    t0 = 30.0 if p["duration"] > 65 else 0.0
    t1 = min(t0 + 30.0, p["duration"])
    crop = mono[int(t0 * sr_audio):int(t1 * sr_audio)]
    ta = t0 + np.arange(len(crop)) / sr_audio
    g = gt[(gt >= t0) & (gt < t1)]
    e = est[(est >= t0) & (est < t1)]
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(ta, crop / (np.abs(crop).max() + 1e-9), color="lightgray")
    ax.vlines(g, 0, 1.1, color="green", lw=2, label="MIDI beats")
    ax.vlines(e, -1.1, 0, color="blue", lw=2, ls="--", label="Beat This!")
    ax.set_ylim(-1.1, 1.1); ax.legend(loc="upper right")
    ax.set_title(f"{row['track']}/{row['stem']}  raw_F={row['raw_F']:.2f}  raw_AMLt={row['raw_AMLt']:.2f}")
    plt.tight_layout()
    base = f"{row['track']}_{row['stem']}"
    path_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(path_dir / f"{base}.png", dpi=120); plt.close(fig)
    c_gt = librosa.clicks(times=g - t0, sr=sr_audio, length=len(crop), click_freq=1800.0, click_duration=0.05)
    c_es = librosa.clicks(times=e - t0, sr=sr_audio, length=len(crop), click_freq=900.0, click_duration=0.05)
    mix = crop * 0.7 + 0.5 * c_gt + 0.5 * c_es
    sf.write(path_dir / f"{base}.wav", (mix / max(1.0, np.abs(mix).max())).astype(np.float32), sr_audio)


def main():
    args = parse_args()
    data_dir = PROCESSED / args.split
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_dir = OUT_DIR / "cache"

    # --- choose stems ---
    bad = set()
    report = PROCESSED / "metadata" / f"beat_check_{args.split}.json"
    if report.exists() and not args.no_skip_bad_gt:
        for k, v in json.load(open(report)).items():
            gs = v.get("grid_score")
            if v.get("status") in ("bad_midi", "no_midi") or (gs is not None and gs < MIN_GRID_SCORE):
                bad.add(k)
        print(f"[INFO] Skipping {len(bad)} tracks flagged by the MIDI beat check.")
    items = []
    for t in sorted(d.name for d in data_dir.iterdir() if d.is_dir()):
        if t in bad:
            continue
        paths = sorted((data_dir / t).glob(args.stem_glob))
        if paths:
            items += [(t, p) for p in (paths if args.all_stems else paths[:1])]
    rng = np.random.default_rng(args.seed)
    rng.shuffle(items)
    items = items[:args.num_stems]
    print(f"[INFO] Evaluating {len(items)} stems ({args.stem_glob}) with Beat This! [{args.checkpoint}, dbn={args.dbn}]")

    model = load_beat_this(args.checkpoint, args.device, args.dbn)

    rows_t, rows_w, payload, skipped = [], [], {}, []
    sr_audio = None
    for track, path in tqdm(items, ncols=120):
        stem = path.stem
        try:
            y, sr = sf.read(str(path), dtype="float32", always_2d=True)
            sr_audio = sr
            mono = y.mean(axis=1)
            duration = len(mono) / sr
            midi = data_dir / track / "all_src.mid"
            if not midi.exists():
                raise FileNotFoundError("all_src.mid missing (run convert.py --midi_only)")
            gt_raw, gt_fold, fold = load_gt(midi, duration)
            est, _ = run_beat_this(model, mono, sr, cache_dir / f"{track}_{stem}_{args.checkpoint}_{int(args.dbn)}.npz")
        except Exception as e:
            skipped.append(f"{track}/{stem}: {type(e).__name__}: {e}")
            continue

        act, frame_s = active_frames(mono, sr)
        for start in range(0, int(duration) - DURATION_SEC + 1, STRIDE_SEC):
            t0e = max(float(start), TRIM_SEC)
            r = evaluate_range(est, gt_raw, gt_fold, t0e, start + DURATION_SEC)
            f0, f1 = int(t0e / frame_s), int((start + DURATION_SEC) / frame_s)
            r.update(track=track, stem=stem, start_sec=start,
                     active_frac=float(act[f0:f1].mean()) if f1 > f0 else np.nan)
            if not np.isnan(r["raw_F"]):
                rows_w.append(r)

        r = evaluate_range(est, gt_raw, gt_fold, TRIM_SEC, duration)
        if np.isnan(r["raw_F"]):
            skipped.append(f"{track}/{stem}: too few GT beats")
            continue
        r.update(track=track, stem=stem, duration=duration, gt_fold=fold,
                 gt_bpm_raw=60.0 / float(np.median(np.diff(gt_raw))),
                 active_frac=float(act[int(TRIM_SEC / frame_s):].mean()))
        rows_t.append(r)
        payload[(track, stem)] = dict(est=est, gt_raw=gt_raw, gt_fold=gt_fold, duration=duration, mono=mono)

    for s in skipped[:15]:
        print(f"[WARNING] skipped {s}")
    if len(skipped) > 15:
        print(f"[WARNING] ... and {len(skipped) - 15} more skipped.")
    if not rows_t:
        print("[ERROR] Nothing evaluated.")
        return

    # --- chance baseline: the same estimates scored against another track's GT ---
    keys = list(payload.keys())
    crng = np.random.default_rng(args.seed + 1)
    for r in rows_t:
        others = [k for k in keys if k[0] != r["track"]]
        r.update({"chance_raw_F": np.nan, "chance_raw_AMLt": np.nan, "chance_fold_F": np.nan, "chance_fold_AMLt": np.nan})
        if not others:
            continue
        p, o = payload[(r["track"], r["stem"])], payload[others[int(crng.integers(len(others)))]]
        dur = min(p["duration"], o["duration"])
        m = evaluate_range(p["est"], o["gt_raw"], o["gt_fold"], TRIM_SEC, dur)
        r["chance_raw_F"], r["chance_raw_AMLt"] = m["raw_F"], m["raw_AMLt"]
        r["chance_fold_F"], r["chance_fold_AMLt"] = m["fold_F"], m["fold_AMLt"]

    trk, win = pd.DataFrame(rows_t), pd.DataFrame(rows_w)
    trk.to_csv(OUT_DIR / "tracks.csv", index=False)
    win.to_csv(OUT_DIR / "windows.csv", index=False)

    plp = None
    if args.plp_csv and pathlib.Path(args.plp_csv).exists():
        plp = pd.read_csv(args.plp_csv)
        plp = plp[plp.get("level", "track") == "track"] if "level" in plp.columns else plp
        keep = [c for c in ["track", "stem", "pk_F", "pk_AMLt", "pk_CMLt", "feat_corr_best"] if c in plp.columns]
        plp = plp[keep]
    merged = summarize(trk, win, plp)
    summary_plot(trk, merged, OUT_DIR / "summary.png")
    print(f"\n[INFO] Wrote tracks.csv, windows.csv, summary.png to {OUT_DIR}")

    if args.render > 0 and sr_audio:
        worst = trk.sort_values("raw_AMLt").head(args.render)
        for _, r in worst.iterrows():
            render(r, payload, sr_audio, OUT_DIR / "renders")
        print(f"[INFO] Rendered {len(worst)} worst stems (plots + click tracks) to {OUT_DIR / 'renders'}")


if __name__ == "__main__":
    main()