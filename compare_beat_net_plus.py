"""
compare_beat_net_plus.py

Evaluates BeatNet+ (50 FPS causal CRNN + cascade particle filter) projected onto the 30 FPS VAE latent
grid (hop = 1600 at 48 kHz) with the same `lookahead` convention as `compare_full.py`.
BEAT (beat_sin, beat_cos) and DOWNBEAT (db_sin, db_cos) phase signals are scored against Slakh MIDI
beats / downbeats (all_src.mid).

No octave folding: GT beats / tempo and BeatNet+ tempo are both used at their natural rate. Metrical-level
ambiguity is handled by the level search in `phase_metrics` (LEVELS), AMLt and `tempo_acc2`.

Pipeline
  1. run BeatNet+ once per stem and record the raw 50 FPS streams (cached; independent of lookahead/latency)
  2. project the streams onto the 30 FPS grid (cheap, recomputed on every run)
  3. score against the MIDI ground truth

Downbeat phase
  phi_db = ((beat_index_in_bar) + phi_beat) / bar_beats, where beat_index_in_bar is a counter driven by wraps
  of the beat phase (not by the event stream), and is re-anchored by BeatNet+ downbeat events. The bar length
  comes from the number of beats between consecutive downbeat events and is stored per frame, so the
  extrapolation to the 30 FPS grid advances the bar phase at omega / bar_beats.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import json
import math
import hashlib
import pathlib
import argparse
import tempfile
import warnings
import multiprocessing as mp
import concurrent.futures

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import pretty_midi
import mir_eval
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from tqdm import tqdm
from omegaconf import OmegaConf

warnings.filterwarnings("ignore", module="mir_eval")

# === Configuration (defaults, all overridable on the command line) ===
DEFAULT_SPLIT = "test"
DEFAULT_PROCESSED = "/data/ssd0/rakec/slakh2100/processed"
DEFAULT_OUT_DIR = "/data/hdd0/rakec/test/beatnet_plus_eval_sincos"
DEFAULT_PRE_ENCODE_CFG = "configs/pre_encode_slakh2100.yaml"
DEFAULT_WEIGHTS_PATH = "beatnet_plus/src/BeatNetPlus/models/af_non_percussive_weights.pt"

MIN_GRID_SCORE = 1.3
SKIP_STATUSES = ("bad_midi", "no_midi", "bad_tempo", "tempo_glitch")

DURATION_SEC = 30
STRIDE_SEC = 15
TRIM_SEC = 5.0
LAG_MAX_FRAMES = 6
MIN_EVENT_GAP_FRAMES = 6        # beat events from phase wraps (0.2 s at 30 FPS)
MIN_DB_EVENT_GAP_FRAMES = 12    # downbeat events from phase wraps (0.4 s at 30 FPS)
GT_MIN_BEAT_GAP_S = 0.15
GT_MIN_DB_GAP_S = 0.45
LEVELS = (1.0, 2.0, 0.5, 3.0, 1.0 / 3.0)
N_PLOT_WORST = 50
N_PLOT_BEST = 50
N_PLOT_RANDOM = 100

BN_FPS = 50.0
MIN_BAR_BEATS, MAX_BAR_BEATS = 2, 4   # bar lengths BeatNet+ is assumed to choose between
CACHE_VERSION = "beatnet_plus_raw50_v2"
RAW_KEYS = ("phi_beat", "phi_db", "bpm", "bar_beats", "db_ok", "act", "coh", "beat_events", "down_events")

DEFAULT_BEAT_CFG = {
    "lookahead": 7,            # in 30 FPS VAE frames
    "latency_ms": 0.0,         # extra phase advance; > 0 compensates a tracker that is late
    "hop": 1600,               # 30 FPS at 48 kHz
    "silence_db": -40.0,
    "hold_frames": 30,
    "lock_coh": 0.5,           # particle coherence above which the tracker counts as locked
    "gate": True,              # zero sin/cos where the stem is silent or the tracker is unlocked
    "weights_path": DEFAULT_WEIGHTS_PATH,
    "bn_mode": "realtime",     # 'realtime' (causal LSTM + PF) or 'online'
    "bn_device": "cpu",
}
TRACKER_KEYS = ["hop", "lookahead", "latency_ms", "silence_db", "hold_frames",
                "lock_coh", "gate", "weights_path", "bn_mode"]
EVENT_KEYS = ["F", "Cemgil", "CMLc", "CMLt", "AMLc", "AMLt"]

# Per-worker cached BeatNet+ instance to avoid reloading weights on every track
_WORKER_ESTIMATOR = None
_WORKER_CFG_KEY = None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", type=str, default=DEFAULT_SPLIT)
    p.add_argument("--processed", type=str, default=DEFAULT_PROCESSED)
    p.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    p.add_argument("--pre_encode_cfg", type=str, default=DEFAULT_PRE_ENCODE_CFG)
    p.add_argument("--weights_path", type=str, default=DEFAULT_WEIGHTS_PATH,
                   help="Path to BeatNet+ checkpoint (e.g. generic_weights.pt or AF/GF non-percussive weights)")
    p.add_argument("--bn_mode", type=str, default="realtime", choices=["realtime", "online"],
                   help="BeatNet+ causal inference mode")
    p.add_argument("--bn_device", type=str, default="cpu", help="Device for BeatNet+ inference ('cpu' or 'cuda:0')")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_tracks", type=int, default=None)
    p.add_argument("--recompute", action="store_true", help="ignore cached raw BeatNet+ outputs")
    p.add_argument("--no_skip_bad_gt", action="store_true")
    p.add_argument("--no_render", action="store_true")
    p.add_argument("--lookahead", type=int, default=None, help="tracker lookahead in 30 FPS frames")
    p.add_argument("--latency_ms", type=float, default=None,
                   help="extra phase advance in ms (> 0 compensates a tracker that is late); no --recompute needed")
    p.add_argument("--lock_coh", type=float, default=None, help="particle coherence threshold for 'locked'")
    p.add_argument("--no_gate", action="store_true", help="do not zero sin/cos when inactive / unlocked")
    return p.parse_args()


def make_paths(args):
    processed = pathlib.Path(args.processed)
    out = pathlib.Path(args.out_dir)
    return dict(
        data=processed / args.split,
        latent=processed / "latents" / args.split,
        report=processed / "metadata" / f"beat_check_{args.split}.json",
        out=out,
        cache=out / "cache",
    )


def load_cfg(path, overrides):
    cfg = dict(DEFAULT_BEAT_CFG)
    p = pathlib.Path(path)
    if p.exists():
        y = OmegaConf.to_container(OmegaConf.load(p), resolve=True)
        cfg.update(y.get("beat", {}) or {})
    else:
        print(f"[WARNING] {path} not found, using built-in defaults.")
    cfg["hop"] = 1600  # 30 FPS directly at 48 kHz
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    cfg["lookahead"] = int(cfg["lookahead"])
    cfg["latency_ms"] = float(cfg["latency_ms"])
    cfg["lock_coh"] = float(cfg["lock_coh"])
    cfg["gate"] = bool(cfg["gate"])
    return cfg


def grid_offset_sec(cfg, sr):
    """Time (s) that frame n's phase refers to is n / fps + this offset."""
    return (cfg["lookahead"] + 0.5) * cfg["hop"] / sr


# ----------------------------------------------------------------------------------------------------------
# Ground truth
# ----------------------------------------------------------------------------------------------------------
def _interp_positions(events, times):
    idx = np.arange(events.size, dtype=np.float64)
    pos = np.interp(times, events, idx)

    intervals = np.maximum(np.diff(events), 1e-6)
    first_iv = max(float(np.median(intervals[:4])), 1e-6)
    last_iv = max(float(np.median(intervals[-4:])), 1e-6)
    pos = np.where(times < events[0], (times - events[0]) / first_iv, pos)
    pos = np.where(times > events[-1], (events.size - 1) + (times - events[-1]) / last_iv, pos)

    smooth_iv = median_filter(intervals, size=5, mode="mirror") if intervals.size >= 5 else intervals
    seg = np.clip(np.searchsorted(events, times, side="right") - 1, 0, smooth_iv.size - 1)
    bpm = 60.0 / smooth_iv[seg]
    return pos, bpm


def midi_gt(midi_path, times):
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    beats = np.asarray(pm.get_beats(), dtype=np.float64)
    if beats.size < 2:
        raise ValueError("fewer than 2 beats in MIDI")

    downbeats = np.asarray(pm.get_downbeats(), dtype=np.float64)
    if downbeats.size < 2:
        downbeats = beats[::4]

    pos, bpm = _interp_positions(beats, times)
    db_pos, db_bpm = _interp_positions(downbeats, times)
    n_tempo = int(len(pm.get_tempo_changes()[1]))
    return beats, pos, bpm, downbeats, db_pos, db_bpm, n_tempo


def clean_beats(beats, duration, min_gap=GT_MIN_BEAT_GAP_S):
    b = beats[beats <= duration]
    if b.size == 0:
        return b
    keep = [b[0]]
    for x in b[1:]:
        if x - keep[-1] >= min_gap:
            keep.append(x)
    return np.asarray(keep)


# ----------------------------------------------------------------------------------------------------------
# BeatNet+ (raw 50 FPS streams)
# ----------------------------------------------------------------------------------------------------------
def _get_beatnet_estimator(cfg):
    global _WORKER_ESTIMATOR, _WORKER_CFG_KEY
    key = (cfg["weights_path"], cfg["bn_mode"], cfg["bn_device"])
    if _WORKER_ESTIMATOR is None or _WORKER_CFG_KEY != key:
        from beatnet_plus.src.BeatNetPlus.inference import BeatNetPlusInference

        _WORKER_ESTIMATOR = BeatNetPlusInference(
            cfg["weights_path"],
            mode=cfg["bn_mode"],
            inference_model="PF",
            device=cfg["bn_device"],
        )
        _WORKER_CFG_KEY = key
    return _WORKER_ESTIMATOR


def _run_pf_pass(estimator, audio):
    """
    One complete BeatNet+ pass with a FRESH particle filter and fresh recording state.
    Hooks `process` of the PF class and records, for every 50 FPS frame:
      phi_beat  circular-mean beat phase of the particles in [0, 1)  (0 = beat)
      phi_db    bar phase in [0, 1): ((beat index in bar) + phi_beat) / bar_beats
      bpm       60 * 50 / mean particle interval
      bar_beats current bar length estimate (beats per bar)
      db_ok     True once a first downbeat event has anchored the beat counter
      act       CRNN beat + downbeat activation
      coh       particle coherence |mean(exp(2*pi*i*pos))|
    plus the beat / downbeat event times emitted by BeatNet+.
    """
    pf_cls = type(estimator.estimator)
    orig_process = pf_cls.process
    estimator.estimator = pf_cls(beats_per_bar=[], fps=int(BN_FPS), plot=[], mode=estimator.mode)

    rec = {k: [] for k in ("phi_beat", "phi_db", "bpm", "bar_beats", "db_ok", "act", "coh")}
    beat_events, down_events = [], []
    # wraps: number of beat-phase wraps since the downbeat wrap (0 = first beat of the bar)
    s = dict(frame=0, prev_phi=None, wraps=0, pending=False, M=MAX_BAR_BEATS, have_db=False)

    def hooked_process(pf_self, preds):
        preds_arr = np.atleast_2d(np.asarray(preds))
        out = []
        for i in range(len(preds_arr)):
            frame_pred = preds_arr[i:i + 1]
            m = s["frame"]
            s["frame"] += 1

            act = float(np.sum(frame_pred[0, :2])) if frame_pred.shape[1] >= 2 else float(frame_pred[0, 0])
            ev = orig_process(pf_self, frame_pred)

            pos = np.asarray(pf_self.st.state_positions[pf_self.particles], dtype=np.float64)
            itv = np.asarray(pf_self.st.state_intervals[pf_self.particles], dtype=np.float64)
            z = np.mean(np.exp(2j * np.pi * pos))
            phi_b = float((np.angle(z) / (2.0 * np.pi)) % 1.0)
            coh = float(np.abs(z))
            bpm = (60.0 * BN_FPS) / max(float(np.mean(itv)), 1e-6)

            # 1) beat counter driven by wraps of the beat phase
            if s["prev_phi"] is not None and (s["prev_phi"] - phi_b) > 0.5:
                if s["pending"]:
                    s["wraps"], s["pending"] = 0, False
                else:
                    s["wraps"] += 1
            s["prev_phi"] = phi_b

            # 2) BeatNet+ events; a downbeat event re-anchors the counter. The event and the wrap of the
            #    circular-mean phase can be a few frames apart in either order: phi_b < 0.5 means the wrap
            #    belonging to this beat has already happened, otherwise it is still coming.
            if ev is not None and len(ev) > 0:
                ev_arr = np.atleast_2d(ev)
                out.append(ev_arr)
                t_ev = (m + 1) / BN_FPS
                beat_events.append(t_ev)
                if int(ev_arr[-1, -1]) == 1:
                    down_events.append(t_ev)
                    if phi_b < 0.5:
                        bar_len, s["wraps"] = s["wraps"], 0
                    else:
                        bar_len, s["pending"] = s["wraps"] + 1, True
                    if s["have_db"] and MIN_BAR_BEATS <= bar_len <= MAX_BAR_BEATS:
                        s["M"] = bar_len
                    s["have_db"] = True

            M = s["M"]
            phi_db = (((s["wraps"] % M) + phi_b) / M) % 1.0

            rec["phi_beat"].append(phi_b)
            rec["phi_db"].append(phi_db)
            rec["bpm"].append(bpm)
            rec["bar_beats"].append(float(M))
            rec["db_ok"].append(bool(s["have_db"]))
            rec["act"].append(act)
            rec["coh"].append(coh)

        return np.vstack(out) if out else np.empty((0, 2))

    pf_cls.process = hooked_process
    try:
        estimator.process(audio)
    finally:
        pf_cls.process = orig_process

    if not rec["phi_beat"]:
        raise RuntimeError("BeatNet+ produced 0 frames")
    raw = {k: np.asarray(v, dtype=np.float64) for k, v in rec.items()}
    raw["act"] = raw["act"].astype(np.float32)
    raw["coh"] = raw["coh"].astype(np.float32)
    raw["db_ok"] = raw["db_ok"].astype(bool)
    raw["beat_events"] = np.asarray(beat_events, dtype=np.float64)
    raw["down_events"] = np.asarray(down_events, dtype=np.float64)
    return raw


def run_beatnet_raw(mono, sr, cfg):
    estimator = _get_beatnet_estimator(cfg)
    mono_22k = librosa.resample(mono.astype(np.float32), orig_sr=sr, target_sr=22050)
    try:
        return _run_pf_pass(estimator, mono_22k)
    except Exception as e1:
        warnings.warn(f"BeatNet+ failed on an in-memory array ({e1!r}); retrying through a temporary wav")
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                sf.write(tmp.name, mono_22k, 22050)
                return _run_pf_pass(estimator, tmp.name)  # fresh PF and fresh recording state
        except Exception as e2:
            raise RuntimeError(f"BeatNet+ failed on array ({e1!r}) and on wav ({e2!r})") from e2


def project_to_grid(mono, sr, raw, cfg):
    """
    Causal projection of the 50 FPS streams onto the 30 FPS grid. At the end of 30 FPS frame n only 50 FPS
    frames with (m + 1) / 50 <= (n + 1) / 30 exist. The phase is extrapolated to the time that frame n is
    scored against, T_n = (n + 0.5 + lookahead) / 30 (+ latency), the same time the GT is sampled at.
    """
    hop = int(cfg["hop"])
    if sr != 48000:
        raise ValueError(f"tracker expects sr=48000, got {sr}")
    vae_fps = sr / hop
    n_model = int(math.ceil(len(mono) / hop))
    n_50 = len(raw["phi_beat"])
    if n_50 == 0:
        raise RuntimeError("BeatNet+ produced 0 frames")

    x = mono
    if x.shape[0] < n_model * hop:
        x = np.pad(x, (0, n_model * hop - x.shape[0]))
    frames = x[: n_model * hop].reshape(n_model, hop)
    rms_db = 20.0 * np.log10(np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-12)
    loud = rms_db > cfg["silence_db"]
    active = np.zeros(n_model, dtype=bool)
    hold = 0
    for n in range(n_model):
        hold = int(cfg["hold_frames"]) if loud[n] else max(hold - 1, 0)
        active[n] = hold > 0

    n_idx = np.arange(n_model)
    t_n = (n_idx + 1) / vae_fps
    m_idx = np.clip(np.floor(t_n * BN_FPS).astype(int) - 1, 0, n_50 - 1)   # latest causal 50 FPS frame
    t_m = (m_idx + 1) / BN_FPS
    target_t = (n_idx + 0.5 + cfg["lookahead"]) / vae_fps
    dt = target_t + cfg["latency_ms"] / 1000.0 - t_m

    omega_hz = raw["bpm"][m_idx] / 60.0
    bar_beats = np.maximum(raw["bar_beats"][m_idx], 1.0)
    phi_b = (raw["phi_beat"][m_idx] + dt * omega_hz) % 1.0
    phi_db = (raw["phi_db"][m_idx] + dt * omega_hz / bar_beats) % 1.0   # bar phase advances at omega / M

    coh = raw["coh"][m_idx].astype(np.float32)
    lock = coh > cfg["lock_coh"]
    if cfg["gate"]:
        g_b = active & lock
        g_db = g_b & raw["db_ok"][m_idx]
    else:
        g_b = g_db = np.ones(n_model, dtype=bool)

    sin_ = (np.sin(2.0 * np.pi * phi_b) * g_b).astype(np.float32)
    cos_ = (np.cos(2.0 * np.pi * phi_b) * g_b).astype(np.float32)
    db_sin_ = (np.sin(2.0 * np.pi * phi_db) * g_db).astype(np.float32)
    db_cos_ = (np.cos(2.0 * np.pi * phi_db) * g_db).astype(np.float32)

    return dict(
        sin=sin_, cos=cos_, db_sin=db_sin_, db_cos=db_cos_,
        tempo=np.where(g_b, raw["bpm"][m_idx], 0.0).astype(np.float32),
        bar_beats=bar_beats.astype(np.float32),
        rms_db=rms_db.astype(np.float32), active=active,
        trusted=(active & (coh > 0.3)), lock=lock, lock_score=coh,
        flag_times=raw["beat_events"], db_flag_times=raw["down_events"],
        act_50=raw["act"], num_frames=n_model, vae_fps=vae_fps,
    )


# ----------------------------------------------------------------------------------------------------------
# Events / metrics
# ----------------------------------------------------------------------------------------------------------
def raw_cache_path(paths, track, stem, cfg):
    key = json.dumps({"weights": str(cfg["weights_path"]), "mode": cfg["bn_mode"], "v": CACHE_VERSION},
                     sort_keys=True)
    return paths["cache"] / f"{track}_{stem}_{hashlib.md5(key.encode()).hexdigest()[:8]}.npz"


def phase_events(sin, cos, offset_sec, fps, min_gap_frames=MIN_EVENT_GAP_FRAMES):
    s, c = sin.astype(np.float64), cos.astype(np.float64)
    ok = np.hypot(s, c) > 1e-6
    ph = np.arctan2(s, c)
    n = np.where(ok[:-1] & ok[1:] & (ph[:-1] < 0) & (ph[1:] >= 0) & (c[:-1] > 0) & (c[1:] > 0))[0]
    if n.size == 0:
        return np.array([])
    frac = n + (-ph[n]) / np.maximum(ph[n + 1] - ph[n], 1e-9)
    keep = [0]
    for i in range(1, len(frac)):
        if frac[i] - frac[keep[-1]] >= min_gap_frames:
            keep.append(i)
    return (frac[keep] / fps) + offset_sec


def prepare_track(track, stem, cfg, paths, recompute=False):
    y, sr = sf.read(str(paths["data"] / track / f"{stem}.wav"), dtype="float32", always_2d=True)
    mono = y.mean(axis=1).astype(np.float64)

    cache = raw_cache_path(paths, track, stem, cfg)
    raw = None
    if cache.exists() and not recompute:
        try:
            with np.load(cache) as z:
                raw = {k: z[k] for k in z.files}
        except Exception:
            raw = None
        if raw is not None and any(k not in raw for k in RAW_KEYS):
            raw = None
    if raw is None:
        raw = run_beatnet_raw(mono, sr, cfg)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **raw)

    tr = project_to_grid(mono, sr, raw, cfg)

    midi = paths["data"] / track / "all_src.mid"
    if not midi.exists():
        raise FileNotFoundError("all_src.mid missing")

    vae_fps = float(tr["vae_fps"])
    num_frames = int(tr["num_frames"])
    offset_sec = grid_offset_sec(cfg, sr)
    frame_times = np.arange(num_frames) / vae_fps + offset_sec
    duration = len(mono) / sr

    beats, pos, bpm, downbeats, db_pos, db_bpm, n_tempo = midi_gt(midi, frame_times)
    ang = 2.0 * np.pi * pos
    db_ang = 2.0 * np.pi * db_pos

    P = dict(
        mono=mono, sr=sr, duration=duration,
        frame_times=frame_times, beats_raw=beats, downbeats_raw=downbeats, n_tempo_msgs=n_tempo,
        gt_beats=clean_beats(beats, duration),
        gt_pos=pos, gt_bpm=bpm,
        gt_sin=np.sin(ang).astype(np.float32), gt_cos=np.cos(ang).astype(np.float32),
        pk_times=phase_events(tr["sin"], tr["cos"], offset_sec, vae_fps),
        gt_downbeats=clean_beats(downbeats, duration, min_gap=GT_MIN_DB_GAP_S),
        gt_db_pos=db_pos, gt_db_bpm=db_bpm,
        gt_db_sin=np.sin(db_ang).astype(np.float32), gt_db_cos=np.cos(db_ang).astype(np.float32),
        db_pk_times=phase_events(tr["db_sin"], tr["db_cos"], offset_sec, vae_fps,
                                 min_gap_frames=MIN_DB_EVENT_GAP_FRAMES),
    )
    P.update(tr)
    P["vae_fps"], P["num_frames"] = vae_fps, num_frames
    return P


def signed_nearest(e, ref):
    """e - (nearest element of the sorted array ref); empty if ref has fewer than 2 elements."""
    e = np.asarray(e, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if e.size == 0 or ref.size < 2:
        return np.array([])
    j = np.clip(np.searchsorted(ref, e), 1, len(ref) - 1)
    nearest = np.where(np.abs(e - ref[j - 1]) < np.abs(e - ref[j]), ref[j - 1], ref[j])
    return e - nearest


def event_metrics(gt, est, t0, t1, prefix):
    g = gt[(gt >= t0) & (gt < t1)]
    e = est[(est >= t0) & (est < t1)]
    out = {prefix + k: np.nan for k in EVENT_KEYS + ["off_med_ms", "off_mad_ms", "ibi_ratio"]}
    out[prefix + "n_gt"] = int(len(g))
    out[prefix + "n_est"] = int(len(e))
    offs = np.array([])
    if len(g) < 4:
        return out, offs
    if len(e) < 2:
        for k in EVENT_KEYS:
            out[prefix + k] = 0.0
        return out, offs
    try:
        out[prefix + "F"] = float(mir_eval.beat.f_measure(g, e))
        out[prefix + "Cemgil"] = float(mir_eval.beat.cemgil(g, e)[0])
        cmlc, cmlt, amlc, amlt = mir_eval.beat.continuity(g, e)
        out[prefix + "CMLc"], out[prefix + "CMLt"] = float(cmlc), float(cmlt)
        out[prefix + "AMLc"], out[prefix + "AMLt"] = float(amlc), float(amlt)
    except Exception:
        pass
    d = signed_nearest(e, g)
    d = d[np.abs(d) < 0.15]
    if d.size:
        med = float(np.median(d))
        out[prefix + "off_med_ms"] = med * 1000.0
        out[prefix + "off_mad_ms"] = float(np.median(np.abs(d - med))) * 1000.0
        offs = d * 1000.0
    out[prefix + "ibi_ratio"] = float(np.median(np.diff(e)) / max(np.median(np.diff(g)), 1e-6))
    return out, offs


def _pearson(a, b):
    if a.size < 10 or a.std() < 1e-6 or b.std() < 1e-6:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def phase_metrics(sin, cos, pos, gt_bpm):
    out = {
        "plv": np.nan, "plv_k1": np.nan, "level_k": np.nan,
        "phase_std_deg": np.nan, "phase_off_ms": np.nan,
        "locked_frac": float((np.hypot(sin, cos) > 1e-6).mean()) if len(sin) else np.nan
    }
    amp = np.hypot(sin.astype(np.float64), cos.astype(np.float64))
    ok = amp > 1e-6
    if ok.sum() < 30:
        return out
    z = (cos[ok].astype(np.float64) + 1j * sin[ok].astype(np.float64)) / amp[ok]
    p = pos[ok]
    best = None
    for k in LEVELS:
        m = (z * np.exp(-2j * np.pi * k * p)).mean()
        r = float(abs(m))
        if k == 1.0:
            out["plv_k1"] = r
        if best is None or r > best[0]:
            best = (r, k, m)
    r, k, m = best
    out["plv"], out["level_k"] = float(r), float(k)
    out["phase_std_deg"] = float(np.degrees(np.sqrt(max(-2.0 * np.log(max(r, 1e-9)), 0.0))))
    bpm = gt_bpm[ok]
    bpm_med = float(np.median(bpm[bpm > 0])) if (bpm > 0).any() else np.nan
    if np.isfinite(bpm_med):
        out["phase_off_ms"] = float(-np.angle(m) / (2.0 * np.pi) * 60000.0 / (bpm_med * k))
    return out


def _lagged(a, b, d):
    if d >= 0:
        return a[d:], b[:len(b) - d]
    return a[:d], b[-d:]


def feature_corr(sin, cos, pos, k, frame_ms):
    ang = 2.0 * np.pi * k * pos
    gs, gc = np.sin(ang), np.cos(ang)

    def corr(d):
        a1, b1 = _lagged(cos, gc, d)
        a2, b2 = _lagged(sin, gs, d)
        return 0.5 * (_pearson(a1, b1) + _pearson(a2, b2))

    best_c, best_d = -2.0, 0
    for d in range(-LAG_MAX_FRAMES, LAG_MAX_FRAMES + 1):
        c = corr(d)
        if c > best_c:
            best_c, best_d = c, d
    return {"feat_corr0": corr(0), "feat_corr_best": best_c, "feat_lag_ms": best_d * frame_ms}


def tempo_metrics(pred, gt):
    v = (pred > 0) & (gt > 0)
    out = {
        "tempo_valid_frac": float(v.mean()), "tempo_acc1": np.nan, "tempo_acc2": np.nan,
        "tempo_ratio_med": np.nan, "tempo_pred_med": np.nan
    }
    if v.sum() == 0:
        return out
    r = pred[v] / gt[v]
    out["tempo_acc1"] = float((np.abs(r - 1) < 0.04).mean())
    ok2 = np.zeros(r.shape, dtype=bool)
    for k in (1, 2, 0.5, 3, 1 / 3):
        ok2 |= np.abs(r / k - 1) < 0.04
    out["tempo_acc2"] = float(ok2.mean())
    out["tempo_ratio_med"] = float(np.median(r))
    out["tempo_pred_med"] = float(np.median(pred[v]))
    return out


def _in(a, t0, t1):
    return a[(a >= t0) & (a < t1)]


def evaluate_range(P, t0, t1):
    ft = P["frame_times"]
    t0e = max(t0, TRIM_SEC)
    f0, f1 = int(np.searchsorted(ft, t0e, "left")), int(np.searchsorted(ft, t1, "left"))
    gt_ev = P["gt_beats"]
    if f1 - f0 < 30 or ((gt_ev >= t0e) & (gt_ev < t1)).sum() < 4:
        return None
    sl = slice(f0, f1)
    frame_ms = (1.0 / P["vae_fps"]) * 1000.0

    row = {}
    # 1) Beat metrics
    m, off_pk = event_metrics(gt_ev, P["pk_times"], t0e, t1, "pk_")
    row.update(m)
    m, off_flag = event_metrics(gt_ev, P["flag_times"], t0e, t1, "flag_")
    row.update(m)

    pm = phase_metrics(P["sin"][sl], P["cos"][sl], P["gt_pos"][sl], P["gt_bpm"][sl])
    row.update(pm)
    k = pm["level_k"] if np.isfinite(pm["level_k"]) else 1.0
    row.update(feature_corr(P["sin"][sl], P["cos"][sl], P["gt_pos"][sl], k, frame_ms))

    # 2) Downbeat metrics
    gt_db_ev = P["gt_downbeats"]
    m_db_pk, off_db_pk = event_metrics(gt_db_ev, P["db_pk_times"], t0e, t1, "db_pk_")
    row.update(m_db_pk)
    m_db_fl, off_db_flag = event_metrics(gt_db_ev, P["db_flag_times"], t0e, t1, "db_flag_")
    row.update(m_db_fl)

    pm_db = phase_metrics(P["db_sin"][sl], P["db_cos"][sl], P["gt_db_pos"][sl], P["gt_db_bpm"][sl])
    row.update({f"db_{k_name}": val for k_name, val in pm_db.items()})
    k_db = pm_db["level_k"] if np.isfinite(pm_db["level_k"]) else 1.0
    fm_db = feature_corr(P["db_sin"][sl], P["db_cos"][sl], P["gt_db_pos"][sl], k_db, frame_ms)
    row.update({f"db_{k_name}": val for k_name, val in fm_db.items()})

    # 3) Internal consistency (no GT): predicted downbeat events should coincide with predicted beat events
    d = signed_nearest(_in(P["db_pk_times"], t0e, t1), _in(P["pk_times"], t0e, t1))
    row["db_vs_beat_off_ms"] = float(np.median(d) * 1000.0) if d.size else np.nan   # > 0: downbeat after beat
    row["db_on_beat_frac"] = float((np.abs(d) < 0.07).mean()) if d.size else np.nan

    # 4) Tempo & activity metrics (raw rates, no folding)
    row.update(tempo_metrics(P["tempo"][sl], P["gt_bpm"][sl]))
    row["tempo_gt_med"] = float(np.median(P["gt_bpm"][sl]))
    row["active_frac"] = float(P["active"][sl].mean())
    row["trusted_frac"] = float(P["trusted"][sl].mean())
    okdb = np.hypot(P["db_sin"][sl], P["db_cos"][sl]) > 1e-6
    row["bar_beats_med"] = float(np.median(P["bar_beats"][sl][okdb])) if okdb.any() else np.nan
    return row, {"pk": off_pk, "flag": off_flag, "db_pk": off_db_pk, "db_flag": off_db_flag}


def lock_stats(lock, fps):
    lock = np.asarray(lock, dtype=bool)
    first = np.nonzero(lock)[0]
    t0 = int(math.ceil(TRIM_SEC * fps))
    return {
        "lock_on_sec": float(first[0] / fps) if first.size else np.nan,
        "lock_frac": float(lock[t0:].mean()) if lock.size > t0 else np.nan,
        "n_unlocks": int(np.sum(lock[:-1] & ~lock[1:]))
    }


def process_item(args):
    track, stem, cfg, paths, recompute = args
    try:
        P = prepare_track(track, stem, cfg, paths, recompute)
    except Exception as e:
        return {"skip": f"{track}/{stem}: {type(e).__name__}: {e}"}

    sr = P["sr"]
    windows = []
    for start in range(0, len(P["mono"]) - DURATION_SEC * sr + 1, STRIDE_SEC * sr):
        res = evaluate_range(P, start / sr, start / sr + DURATION_SEC)
        if res is None:
            continue
        row, _ = res
        row.update(track=track, stem=stem, start_sec=start / sr, level="window")
        windows.append(row)

    track_row, payload = None, None
    res = evaluate_range(P, 0.0, P["duration"])
    if res is not None:
        row, offs = res
        row.update(
            track=track, stem=stem, level="track", duration=P["duration"],
            gt_bpm_raw=60.0 / float(np.median(np.diff(P["beats_raw"]))),
            **lock_stats(P["lock"], P["vae_fps"]),
            n_tempo_changes=max(int(P["n_tempo_msgs"]) - 1, 0),
        )
        track_row = row
        payload = dict(
            gt_beats=P["gt_beats"], gt_pos=P["gt_pos"], gt_bpm=P["gt_bpm"], pk_times=P["pk_times"],
            sin=P["sin"], cos=P["cos"],
            gt_downbeats=P["gt_downbeats"], gt_db_pos=P["gt_db_pos"], gt_db_bpm=P["gt_db_bpm"],
            db_pk_times=P["db_pk_times"], db_sin=P["db_sin"], db_cos=P["db_cos"],
            duration=P["duration"], fps=P["vae_fps"],
            frame_ms=(1.0 / P["vae_fps"]) * 1000.0,
            off_pk=offs["pk"], off_flag=offs["flag"], off_db_pk=offs["db_pk"]
        )
    return {"windows": windows, "track": track_row, "payload": payload}


def add_chance(track_rows, payloads, seed=0):
    rng = np.random.default_rng(seed)
    keys = list(payloads.keys())
    for r in track_rows:
        r.update({
            "chance_pk_F": np.nan, "chance_pk_AMLt": np.nan,
            "chance_plv": np.nan, "chance_plv_k1": np.nan, "chance_feat_corr_best": np.nan,
            "chance_db_pk_F": np.nan, "chance_db_plv": np.nan, "chance_db_plv_k1": np.nan,
        })
        others = [k for k in keys if k[0] != r["track"]]
        if not others:
            continue
        p, o = payloads[(r["track"], r["stem"])], payloads[others[int(rng.integers(len(others)))]]
        dur = min(p["duration"], o["duration"])
        m, _ = event_metrics(o["gt_beats"], p["pk_times"], TRIM_SEC, dur, "pk_")
        f0 = int(math.ceil(TRIM_SEC * p["fps"]))
        L = min(len(p["sin"]), len(o["gt_pos"]))
        pm = phase_metrics(p["sin"][f0:L], p["cos"][f0:L], o["gt_pos"][f0:L], o["gt_bpm"][f0:L])
        k = pm["level_k"] if np.isfinite(pm["level_k"]) else 1.0
        fm = feature_corr(p["sin"][f0:L], p["cos"][f0:L], o["gt_pos"][f0:L], k, p["frame_ms"])
        r["chance_pk_F"], r["chance_pk_AMLt"] = m["pk_F"], m["pk_AMLt"]
        r["chance_plv"], r["chance_plv_k1"] = pm["plv"], pm["plv_k1"]
        r["chance_feat_corr_best"] = fm["feat_corr_best"]

        m_db, _ = event_metrics(o["gt_downbeats"], p["db_pk_times"], TRIM_SEC, dur, "db_pk_")
        pm_db = phase_metrics(p["db_sin"][f0:L], p["db_cos"][f0:L], o["gt_db_pos"][f0:L], o["gt_db_bpm"][f0:L])
        r["chance_db_pk_F"] = m_db["db_pk_F"]
        r["chance_db_plv"], r["chance_db_plv_k1"] = pm_db["plv"], pm_db["plv_k1"]


def summarize(trk, win, cfg):
    cols = [
        "pk_F", "pk_AMLt", "pk_CMLt", "flag_F", "plv", "plv_k1", "phase_std_deg", "phase_off_ms",
        "feat_corr0", "feat_corr_best", "feat_lag_ms", "pk_off_med_ms", "pk_ibi_ratio",
        "db_pk_F", "db_pk_AMLt", "db_flag_F", "db_plv", "db_plv_k1", "db_phase_off_ms",
        "db_feat_corr0", "db_feat_corr_best", "db_pk_ibi_ratio",
        "db_vs_beat_off_ms", "db_on_beat_frac", "bar_beats_med",
        "tempo_acc1", "tempo_acc2", "locked_frac", "active_frac", "trusted_frac",
        "chance_pk_F", "chance_plv", "chance_db_pk_F", "chance_db_plv"
    ]
    cols = [c for c in cols if c in trk.columns]
    pd.set_option("display.width", 160)
    pd.set_option("display.max_rows", 80)
    print(f"=== BeatNet+ Track-level summary ({len(trk)} tracks/stems, {len(win)} windows) ===")
    print(trk[cols].agg(["mean", "median"]).T.round(3))

    if "bar_beats_med" in trk.columns and trk["bar_beats_med"].notna().any():
        print("Median beats/bar chosen by the tracker (track count):",
              trk["bar_beats_med"].round().value_counts().sort_index().to_dict())

    med = float(trk["pk_off_med_ms"].median()) if trk["pk_off_med_ms"].notna().any() else np.nan
    pho = float(trk["phase_off_ms"].median()) if trk["phase_off_ms"].notna().any() else np.nan
    print(f"Median signed offset of beat events: {med:+.1f} ms | median phase offset: {pho:+.1f} ms "
          f"(> 0 = tracker late), lookahead = {cfg['lookahead']}, latency_ms = {cfg['latency_ms']}.")
    if np.isfinite(med) and abs(med) > 15:
        if abs(med) < 60:
            print(f"  -> consistent latency: re-run with  --latency_ms {cfg['latency_ms'] + med:+.0f}  "
                  f"(no --recompute needed)")
        else:
            print("  -> offset is too large to be latency; check for off-beat tracking "
                  "(listen to the *_gt.wav / *_pr.wav renders) before compensating.")


def summary_plot(trk, win, offs_pk, path, frame_ms):
    fig, axs = plt.subplots(4, 3, figsize=(17, 17))
    b = np.linspace(0, 1, 21)
    axs[0, 0].hist(trk["pk_F"].dropna(), b, alpha=0.7, label="Beat F1")
    if "db_pk_F" in trk:
        axs[0, 0].hist(trk["db_pk_F"].dropna(), b, alpha=0.6, label="Downbeat F1")
    axs[0, 0].hist(trk["chance_pk_F"].dropna(), b, alpha=0.4, label="chance (beat)")
    axs[0, 0].set_title("Beat & Downbeat F-measure (+-70 ms), phase-wrap events"); axs[0, 0].legend()

    axs[0, 1].hist(trk["plv"].dropna(), b, alpha=0.7, label="Beat PLV (best)")
    axs[0, 1].hist(trk["plv_k1"].dropna(), b, alpha=0.5, label="Beat PLV (k=1)")
    if "db_plv_k1" in trk:
        axs[0, 1].hist(trk["db_plv_k1"].dropna(), b, alpha=0.5, label="Downbeat PLV (k=1)")
    axs[0, 1].set_title("Phase-locking value of (sin, cos) vs GT phase"); axs[0, 1].legend()

    if offs_pk.size:
        axs[0, 2].hist(offs_pk, np.linspace(-150, 150, 61))
    axs[0, 2].axvline(0, color="k", lw=0.8)
    axs[0, 2].set_title("Signed timing offset of predicted beats (ms, > 0 = late)")

    for col, lab in (("pk_ibi_ratio", "beat"), ("db_pk_ibi_ratio", "downbeat")):
        if col in trk:
            r = trk[col].dropna()
            r = r[r > 0]
            axs[1, 0].hist(np.log2(r), np.linspace(-2, 2, 41), alpha=0.6, label=lab)
    axs[1, 0].legend()
    axs[1, 0].set_title("log2(tracker interval / GT)  (0 ok, +1 half-time, -1 double-time)")

    if len(win):
        axs[1, 1].scatter(win["active_frac"], win["pk_F"], s=6, alpha=0.3, label="Beat F1")
        if "db_pk_F" in win:
            axs[1, 1].scatter(win["active_frac"], win["db_pk_F"], s=6, alpha=0.3, label="Downbeat F1")
        axs[1, 1].legend()
    axs[1, 1].set_xlabel("fraction of window with active guitar"); axs[1, 1].set_ylabel("F-measure")

    axs[1, 2].hist(trk["phase_off_ms"].dropna(), np.linspace(-500, 500, 51), alpha=0.7, label="Beat")
    if "db_phase_off_ms" in trk:
        axs[1, 2].hist(trk["db_phase_off_ms"].dropna(), np.linspace(-500, 500, 51), alpha=0.5, label="Downbeat")
        axs[1, 2].legend()
    axs[1, 2].axvline(0, color="k", lw=0.8)
    axs[1, 2].set_title("Phase timing offset at best level (ms, > 0 = late)")

    g = trk[["tempo_gt_med", "tempo_pred_med"]].dropna()
    if len(g):
        axs[2, 0].scatter(g["tempo_gt_med"], g["tempo_pred_med"], s=8, alpha=0.4)
        xs = np.array([20.0, 460.0])
        for kk, ls in ((1.0, "-"), (2.0, "--"), (0.5, "--")):
            axs[2, 0].plot(xs, xs * kk, color="k", lw=0.7, ls=ls)
        axs[2, 0].set_xscale("log"); axs[2, 0].set_yscale("log")
    axs[2, 0].set_xlabel("GT tempo (BPM)"); axs[2, 0].set_ylabel("BeatNet+ tempo (BPM)")
    axs[2, 0].set_title("Tempo, no folding (lines: k = 1, 2, 0.5)")

    lk = trk["level_k"].dropna()
    ks = sorted(LEVELS)
    axs[2, 1].bar([f"{k:.3g}" for k in ks], [np.isclose(lk, k).mean() if len(lk) else 0 for k in ks])
    axs[2, 1].set_title("Best metrical level k (tracker rate = k x GT rate)")

    if len(win):
        axs[2, 2].hist(
            win["feat_lag_ms"].dropna(),
            np.arange(-LAG_MAX_FRAMES - 0.5, LAG_MAX_FRAMES + 1.5) * frame_ms
        )
    axs[2, 2].set_title("Best lag of tracker sin/cos vs GT (ms, > 0 = late)")

    if "db_vs_beat_off_ms" in trk:
        axs[3, 0].hist(trk["db_vs_beat_off_ms"].dropna(), np.linspace(-300, 300, 61))
    axs[3, 0].axvline(0, color="k", lw=0.8)
    axs[3, 0].set_title("Predicted downbeat - nearest predicted beat (ms); should be ~0")

    if "bar_beats_med" in trk:
        bb = trk["bar_beats_med"].dropna().round()
        axs[3, 1].bar(["2", "3", "4"], [(bb == v).mean() if len(bb) else 0 for v in (2, 3, 4)])
    axs[3, 1].set_title("Beats per bar chosen by the tracker (median per stem)")

    if "db_on_beat_frac" in trk:
        axs[3, 2].hist(trk["db_on_beat_frac"].dropna(), b)
    axs[3, 2].set_title("Fraction of predicted downbeats within 70 ms of a predicted beat")

    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)


def render_window(row, cfg, paths, tag, rank):
    P = prepare_track(row["track"], row["stem"], cfg, paths, recompute=False)
    sr = P["sr"]
    t0 = float(row["start_sec"])
    t1 = t0 + DURATION_SEC
    crop = P["mono"][int(round(t0 * sr)):int(round(t1 * sr))]
    t_axis = t0 + np.arange(len(crop)) / sr
    ft = P["frame_times"]
    m = (ft >= t0) & (ft < t1)

    gt_in = _in(P["gt_beats"], t0, t1)
    pk_in = _in(P["pk_times"], t0, t1)
    gt_db_in = _in(P["gt_downbeats"], t0, t1)
    db_pk_in = _in(P["db_pk_times"], t0, t1)

    fig, axs = plt.subplots(6, 1, figsize=(16, 16), sharex=True, height_ratios=[1.4, 1, 1, 0.9, 0.5, 0.8])
    axs[0].plot(t_axis, crop / (np.abs(crop).max() + 1e-9), color="lightgray", label="guitar waveform")
    axs[0].vlines(gt_in, 0, 0.75, color="green", lw=1.2, alpha=0.7, label="GT beats")
    axs[0].vlines(gt_db_in, 0, 1.1, color="darkgreen", lw=2.5, label="GT downbeats")
    axs[0].vlines(pk_in, -0.75, 0, color="red", lw=1.2, ls="--", alpha=0.7, label="BeatNet+ beats")
    axs[0].vlines(db_pk_in, -1.1, 0, color="darkred", lw=2.5, ls="--", label="BeatNet+ downbeats")
    axs[0].set_ylim(-1.15, 1.15); axs[0].legend(loc="upper right", ncol=3)
    axs[0].set_title(
        f"{row['track']}/{row['stem']} start={t0:.0f}s | Beat F1={row.get('pk_F', np.nan):.2f} "
        f"PLV={row.get('plv_k1', np.nan):.2f} | Downbeat F1={row.get('db_pk_F', np.nan):.2f} "
        f"DB_PLV={row.get('db_plv_k1', np.nan):.2f} | beat off={row.get('phase_off_ms', np.nan):.0f}ms "
        f"db-beat={row.get('db_vs_beat_off_ms', np.nan):.0f}ms"
    )

    axs[1].plot(ft[m], P["gt_cos"][m], color="green", label="GT beat cos")
    axs[1].plot(ft[m], P["cos"][m], color="red", ls="--", label="BeatNet+ beat cos")
    axs[1].plot(ft[m], P["gt_sin"][m], color="green", lw=0.8, ls=":", alpha=0.7, label="GT beat sin")
    axs[1].plot(ft[m], P["sin"][m], color="red", lw=0.8, ls=":", alpha=0.7, label="BeatNet+ beat sin")
    axs[1].set_ylim(-1.1, 1.1); axs[1].legend(loc="upper right", ncol=4); axs[1].set_ylabel("Beat sin/cos")

    axs[2].plot(ft[m], P["gt_db_cos"][m], color="darkgreen", label="GT downbeat cos")
    axs[2].plot(ft[m], P["db_cos"][m], color="darkred", ls="--", label="BeatNet+ downbeat cos")
    axs[2].plot(ft[m], P["gt_db_sin"][m], color="darkgreen", lw=0.8, ls=":", alpha=0.7, label="GT downbeat sin")
    axs[2].plot(ft[m], P["db_sin"][m], color="darkred", lw=0.8, ls=":", alpha=0.7, label="BeatNet+ downbeat sin")
    axs[2].set_ylim(-1.1, 1.1); axs[2].legend(loc="upper right", ncol=4); axs[2].set_ylabel("Downbeat sin/cos")

    axs[3].plot(ft[m], P["gt_bpm"][m], color="green", label="GT tempo")
    axs[3].plot(ft[m], np.where(P["tempo"][m] > 0, P["tempo"][m], np.nan), color="red", ls="--",
                label="BeatNet+ tempo")
    axs[3].set_ylabel("BPM"); axs[3].legend(loc="upper right")
    ax3b = axs[3].twinx()
    ax3b.plot(ft[m], P["lock_score"][m], color="tab:blue", lw=0.8, alpha=0.6)
    ax3b.set_ylim(0, 1.05); ax3b.set_ylabel("particle coherence")

    axs[4].step(ft[m], np.where(np.hypot(P["db_sin"][m], P["db_cos"][m]) > 1e-6, P["bar_beats"][m], np.nan),
                where="post", color="purple")
    axs[4].set_ylim(1, 5); axs[4].set_yticks([2, 3, 4]); axs[4].set_ylabel("beats/bar")

    t50 = (np.arange(len(P["act_50"])) + 1) / BN_FPS   # activation of frame m is available at (m + 1) / 50
    m50 = (t50 >= t0) & (t50 < t1)
    axs[5].plot(t50[m50], P["act_50"][m50], color="purple", lw=0.8, label="BeatNet+ CRNN activation (50 FPS)")
    axs[5].vlines(gt_in, 0, 1.0, color="green", lw=1, alpha=0.6, label="GT beats")
    axs[5].set_ylim(0, 1.05); axs[5].set_ylabel("activation"); axs[5].set_xlabel("Time (seconds)")
    axs[5].legend(loc="upper right"); axs[5].set_xlim(t0, t1)

    plt.tight_layout()
    base = f"{tag}{rank:02d}_{row['track']}_{row['stem']}_start{int(t0)}s"
    (paths["out"] / "plots").mkdir(parents=True, exist_ok=True)
    plt.savefig(paths["out"] / "plots" / f"{base}.png", dpi=120)
    plt.close(fig)

    # Render dual-pitch clicks: 1200 Hz on downbeats, 800 Hz on regular beats
    def clicks(beat_t, db_t):
        return (
            librosa.clicks(times=beat_t - t0, sr=sr, length=len(crop), click_freq=800.0, click_duration=0.04) * 0.35
            + librosa.clicks(times=db_t - t0, sr=sr, length=len(crop), click_freq=1200.0, click_duration=0.06) * 0.45
        )

    for suffix, c in (("gt", clicks(gt_in, gt_db_in)), ("pr", clicks(pk_in, db_pk_in))):
        mix = crop + c
        sf.write(paths["out"] / "plots" / f"{base}_{suffix}.wav",
                 (mix / max(1.0, float(np.abs(mix).max()))).astype(np.float32), sr)


def main():
    args = parse_args()
    paths = make_paths(args)
    paths["out"].mkdir(parents=True, exist_ok=True)
    paths["cache"].mkdir(parents=True, exist_ok=True)
    cfg = load_cfg(args.pre_encode_cfg, dict(
        lookahead=args.lookahead, latency_ms=args.latency_ms, lock_coh=args.lock_coh,
        gate=False if args.no_gate else None,
        weights_path=args.weights_path, bn_mode=args.bn_mode, bn_device=args.bn_device,
    ))
    print(f"[INFO] BeatNet+ settings: { {k: cfg[k] for k in TRACKER_KEYS} } device={cfg['bn_device']}")

    grid_scores, bad = {}, set()
    if paths["report"].exists():
        for k, v in json.load(open(paths["report"])).items():
            gs = v.get("grid_score")
            grid_scores[k] = gs
            if v.get("status") in SKIP_STATUSES or (gs is not None and gs < MIN_GRID_SCORE):
                bad.add(k)

    tracks = sorted(t.name for t in paths["data"].iterdir() if t.is_dir())
    if not args.no_skip_bad_gt:
        tracks = [t for t in tracks if t not in bad]
    if args.max_tracks:
        tracks = tracks[:args.max_tracks]

    items = [
        (t, gp.stem, cfg, paths, args.recompute)
        for t in tracks
        for gp in sorted((paths["data"] / t).glob("guitar_*.wav"))
    ]
    print(f"[INFO] Evaluating {len(items)} guitar stems ({args.num_workers} workers).")

    win_rows, trk_rows, payloads, skipped = [], [], {}, []
    mp_ctx = mp.get_context("spawn") if "cuda" in cfg["bn_device"] else None

    def collect(res):
        if "skip" in res:
            skipped.append(res["skip"])
            return
        win_rows.extend(res["windows"])
        if res["track"] is not None:
            trk_rows.append(res["track"])
            payloads[(res["track"]["track"], res["track"]["stem"])] = res["payload"]

    if args.num_workers <= 1:
        for it in tqdm(items, total=len(items), ncols=140):
            collect(process_item(it))
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers, mp_context=mp_ctx) as ex:
            futs = [ex.submit(process_item, it) for it in items]
            for fut in tqdm(concurrent.futures.as_completed(futs), total=len(futs), ncols=140):
                collect(fut.result())

    if skipped:
        print(f"[WARNING] Skipped {len(skipped)} stems (first 3: {skipped[:3]})")
    if not trk_rows:
        return

    add_chance(trk_rows, payloads)
    trk = pd.DataFrame(trk_rows)
    win = pd.DataFrame(win_rows)
    for df in (trk, win):
        if len(df):
            df["grid_score"] = df["track"].map(grid_scores)
    trk.to_csv(paths["out"] / "tracks.csv", index=False)
    win.to_csv(paths["out"] / "windows.csv", index=False)
    frame_ms = next(iter(payloads.values()))["frame_ms"]

    summarize(trk, win, cfg)
    offs = np.concatenate([p["off_pk"] for p in payloads.values() if p["off_pk"].size] or [np.array([])])
    summary_plot(trk, win, offs, paths["out"] / "summary.png", frame_ms)

    if not args.no_render and len(win):
        w = win.dropna(subset=["pk_F"]).sort_values(["pk_F", "feat_corr_best"])
        best = w.tail(N_PLOT_BEST).iloc[::-1]
        worst = w.head(N_PLOT_WORST)
        rest = w.drop(best.index.union(worst.index))
        rand = rest.sample(min(N_PLOT_RANDOM, len(rest)), random_state=0) if len(rest) else rest
        for tag, df in (("best", best), ("worst", worst), ("random", rand)):
            for rank, (_, r) in enumerate(tqdm(df.iterrows(), total=len(df), desc=f"render {tag}", ncols=140)):
                render_window(r, cfg, paths, tag, rank)


if __name__ == "__main__":
    main()