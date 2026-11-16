"""
Module Name: realtimeplp.py
Author:      Peter Meier
Email:       peter.meier@audiolabs-erlangen.de
Date:        2024-10-01
Version:     0.0.1
Description: Single File Real-Time PLP Implementation.
License:     MIT License (https://opensource.org/licenses/MIT)

Local modifications
  * ControlSignals exposes beat_phase / beat_sin / beat_cos / tempo_bpm, read from the current PLP kernel
    (valid in the trusted AND the flywheel branch).
  * Acquire -> lock -> re-acquire state machine in PredominantLocalPulse.  Everything is per-frame
    and causal.
  * Time convention of the pulse read-out: the tracker runs one step per 30 FPS hop. The newest activation is
    at buffer index N//2 - 1 and the read cursor is N//2 + lookahead. `lookahead` is in TRACKER frames.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import signal


@dataclass
class Peaks:
    """Represents detected peaks with frame indices and time positions."""

    i: np.ndarray  # frame indices
    t: np.ndarray  # time positions

    @classmethod
    def pick(cls, signal_x: np.ndarray, signal_t: np.ndarray, prominence: float = 0.01):
        """Detects peaks in a signal based on prominence."""
        peaks_i, _ = signal.find_peaks(signal_x, prominence=prominence)
        peaks_t = signal_t[peaks_i]
        return cls(peaks_i, peaks_t)


@dataclass
class BeatActivation:
    """Beat Activation (Spectral Flux) from Audio Input"""

    N: int = 3200  # window size
    H: int = 1600  # hop size (30 FPS at 48 kHz)
    samplerate: int = 48000
    gamma: int = 1000  # parameter for logarithmic compression
    M: int = 10  # size of half centric local average window (no future values)

    _window_buffer: np.ndarray = field(init=False, repr=False)  # of size N samples
    _la_buffer: np.ndarray = field(init=False, repr=False)  # of size M frames
    _Y_last: np.ndarray = field(init=False, repr=False)
    _stft_window: np.ndarray = field(init=False, repr=False)

    activation_frame: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self._window_buffer = np.zeros(self.N)
        self._la_buffer = np.zeros(self.M)
        self.activation_frame = np.array([])
        # Initialize last spectrogram slice with zeros (matches librosa.stft output shape: 1 + N/2)
        self._Y_last = np.zeros((self.N // 2 + 1, 1))
        # Pre-compute the STFT window for exact librosa numerical parity (fftbins=True)
        self._stft_window = signal.get_window("hann", self.N, fftbins=True)

    def process(self, audio_frame: np.ndarray) -> np.ndarray:
        """Process input frame to output frame."""
        # STFT
        X_frame = self.stft(audio_frame)
        # Spectrogram
        Y_frame = self.spectrogram(X_frame)
        # Novelty (Discrete Derivative)
        Y_diff = Y_frame - self._Y_last
        Y_diff[Y_diff < 0] = 0  # Half-Wave Rectification
        nov = np.sum(Y_diff, axis=0)  # Accumulation
        # store last spectrogram Y_frame
        self._Y_last = Y_frame
        # local average
        la = self.local_average(nov)
        # normalization
        nov_norm = self.normalization(nov, la)
        # store current activation frame
        if nov_norm.size == 0:
            self.activation_frame = np.array([0.0])
        else:
            self.activation_frame = nov_norm
        return nov_norm

    def stft(self, audio_frame):
        """Process audio frame and compute STFT."""
        self.roll_window_buffer(audio_frame)
        X_frame = np.fft.rfft(self._window_buffer * self._stft_window)
        return X_frame.reshape(-1, 1)

    def spectrogram(self, X_frame):
        """Compute online spectrogram."""
        Y_frame = np.log(1 + self.gamma * np.abs(X_frame))
        return Y_frame

    def roll_window_buffer(self, audio_frame):
        """Roll window_buffer with new audio_frame input."""
        self._window_buffer = np.roll(self._window_buffer, -self.H)
        self._window_buffer[-self.H :] = audio_frame

    def local_average(self, nov):
        """Compute the local average."""
        if len(nov) > 0:
            self._la_buffer = np.roll(self._la_buffer, -1)
            self._la_buffer[-1] = nov[0]
        la = (1 / self.M) * np.sum(self._la_buffer)
        return la

    def normalization(self, nov, la):
        """Normalized novelty curve. Post-processing step."""
        nov_norm = nov - la
        nov_norm[nov_norm < 0] = 0
        return nov_norm


@dataclass
class Tempogram:
    """Tempogram from Beat Activation Input."""

    N_time: int = 10
    framerate: float = 0
    Theta: np.ndarray = field(default_factory=lambda: np.arange(90, 180, 1))
    H: int = field(default=1, repr=False)
    winfunc: Any = field(default_factory=lambda: np.hanning, repr=False)
    harmonics: tuple = (2,)

    _tempo_buffer: np.ndarray = field(init=False, repr=False)
    _E: np.ndarray = field(init=False, repr=False)
    _win: np.ndarray = field(init=False, repr=False)

    tempogram_frame: np.ndarray = field(init=False, repr=False)
    harm_frames: list = field(init=False, repr=False)

    def __post_init__(self):
        self._tempo_buffer = np.zeros(self.N)
        assert (
            self.framerate != 1
        ), f"Tempogram samplerate is not set: fs = {self.framerate}"
        self.tempogram_frame = np.array([])
        self.harm_frames = []
        m = np.arange(self.N) / self.framerate
        bpm = np.concatenate([np.asarray(self.Theta, dtype=np.float64)] +
                             [h * np.asarray(self.Theta, dtype=np.float64) for h in self.harmonics])
        self._E = np.exp(-2j * np.pi * (bpm[:, None] / 60.0) * m[None, :])
        self._win = self.winfunc(self.N)

    @property
    def N(self):
        return round(self.N_time * self.framerate)

    @property
    def win(self):
        return self._win if hasattr(self, "_win") else self.winfunc(self.N)

    def process(self, activation_frame: np.ndarray) -> np.ndarray:
        self.half_window_method(activation_frame)
        K = len(self.Theta)
        Xall = self._E @ (self._tempo_buffer * self._win)
        X = Xall[:K].reshape(K, 1).astype(np.complex128)
        self.harm_frames = [Xall[(i + 1) * K:(i + 2) * K] for i in range(len(self.harmonics))]
        self.tempogram_frame = X
        return X

    def half_window_method(self, activation_frame):
        if not len(activation_frame) == 0:
            nov_half = self._tempo_buffer[: self.N // 2]
            nov_half = np.roll(nov_half, -self.H)
            nov_half[-self.H :] = activation_frame
            self._tempo_buffer[: len(nov_half)] = nov_half


@dataclass
class Kernel:
    """Kernel class."""

    n: int
    k: Any
    tempo: int
    omega: float
    c: Any
    phase: float
    t_start: int
    t_end: int
    t: np.ndarray
    x: np.ndarray

    @classmethod
    def from_plp(
        cls,
        N: int,
        H: int,
        win: np.ndarray,
        Theta: np.ndarray,
        framerate: float,
        X: np.ndarray,
        n: int,
        tempogram: np.ndarray,
        k: Any = None,
    ):
        if k is None:
            k = int(np.argmax(tempogram[:, n]))
        tempo = Theta[k]
        omega = (tempo / 60) / framerate
        c = X[k, n]
        phase = -np.angle(c) / (2 * np.pi)
        t_start = n * H
        t_end = t_start + N
        t = np.arange(t_start, t_end)
        x = win * np.cos(2 * np.pi * (t * omega - phase))
        return cls(n, k, tempo, omega, c, phase, t_start, t_end, t, x)


@dataclass
class PredominantLocalPulse:
    """Real-time predominant local pulse: tempo inertia, silence gating and an acquire -> lock -> re-acquire machine."""

    N_time: int = 10
    framerate: float = 0
    Theta: np.ndarray = field(default_factory=lambda: np.arange(90, 180, 1))
    lookahead: int = 0
    H: int = field(default=1, repr=False)
    energy_threshold: float = 0.05
    conf_threshold: float = 2.0
    switch_ratio: float = 1.25
    switch_persist_sec: float = 3.0
    harmonic_weight: float = 1.0

    lock_enable: bool = True
    lock_sec: float = 4.0
    lock_tol_bpm: float = 4.0
    lose_rho: float = 0.5
    lose_conf: float = 1.3
    lose_err: float = 0.4
    phase_gate: float = 0.15
    reacquire_sec: float = 4.0
    pll_tau_sec: float = 1.5
    pll_conf_hi: float = 3.0

    stability: float = field(default=0, init=False, repr=False)
    current_tempo: float = field(default=0, init=False, repr=False)
    _pulse_buffer: np.ndarray = field(init=False, repr=False)
    _t: np.ndarray = field(init=False, repr=False)
    _cursor: int = field(default=0, init=False, repr=False)
    _last_beat_distance: int = field(default=1_000_000_000, init=False, repr=False)
    _max_peak_amplitude: float = field(default=-1, init=False, repr=False)

    _has_locked_tempo: bool = field(default=False, init=False, repr=False)
    _confident_once: bool = field(default=False, init=False, repr=False)
    _extrapolated_phase: float = field(default=0.0, init=False, repr=False)
    _decay_factor: float = field(default=1.0, init=False, repr=False)

    _locked: bool = field(default=False, init=False, repr=False)
    _lock_bpm: float = field(default=0.0, init=False, repr=False)
    _theta: float = field(default=0.0, init=False, repr=False)
    _cand_bpm: float = field(default=0.0, init=False, repr=False)
    _cand_count: int = field(default=0, init=False, repr=False)
    _chal_bpm: float = field(default=0.0, init=False, repr=False)
    _chal_count: int = field(default=0, init=False, repr=False)
    _bad_count: float = field(default=0.0, init=False, repr=False)
    _err_ema: float = field(default=0.0, init=False, repr=False)
    _evidence: float = field(default=0.0, init=False, repr=False)
    _m: np.ndarray = field(init=False, repr=False)
    _k_peak: int = field(default=0, init=False, repr=False)
    n_locks: int = field(default=0, init=False, repr=False)
    n_unlocks: int = field(default=0, init=False, repr=False)

    last_confidence: float = field(default=0.0, init=False, repr=False)
    last_trusted: bool = field(default=False, init=False, repr=False)

    current_kernel: Kernel = field(init=False, repr=False)

    def __post_init__(self):
        self._pulse_buffer = np.zeros(self.N)
        self._t = np.arange(self.N) / self.framerate
        self._cursor = (self.N // 2) + self.lookahead
        self._m = np.arange(self.N, dtype=np.float64)

    @property
    def cursor(self):
        return self._cursor

    @property
    def N(self):
        return round(self.N_time * self.framerate)

    @property
    def win(self):
        return np.hanning(self.N) if self.N else None

    @property
    def current_buffer(self):
        return self._pulse_buffer

    @property
    def lock_score(self):
        if self._locked:
            need = max(self.reacquire_sec * self.framerate, 1.0)
            return 0.5 + 0.5 * (1.0 - min(self._bad_count / need, 1.0))
        if self.lock_enable and self._has_locked_tempo:
            need = max(self.lock_sec * self.framerate, 1.0)
            return 0.5 * min(self._cand_count / need, 1.0)
        return 0.0

    @staticmethod
    def pick_peak(mag: np.ndarray) -> int:
        m = np.asarray(mag, dtype=np.float64).reshape(-1)
        if m.size < 3:
            return int(np.argmax(m))
        inner = (m[1:-1] > m[:-2]) & (m[1:-1] >= m[2:])
        if not inner.any():
            return int(np.argmax(m))
        idx = np.nonzero(inner)[0] + 1
        return int(idx[np.argmax(m[idx])])

    def process(self, tempogram_frame: np.ndarray, active: bool = True,
                activation_buffer: np.ndarray = None, activation_win: np.ndarray = None,
                harmonic_frames: list = None) -> np.ndarray:
        self._pulse_buffer = np.roll(self._pulse_buffer, -self.H)
        self._pulse_buffer[-self.H :] = 0

        score = np.abs(tempogram_frame[:, 0]).astype(np.float64)
        if harmonic_frames is not None and self.harmonic_weight > 0:
            for hf in harmonic_frames:
                score = score + self.harmonic_weight * np.abs(hf)
        tempogram = score[:, None]
        self._k_peak = self.pick_peak(tempogram[:, 0])
        peak_energy = float(tempogram[self._k_peak, 0])
        mean_energy = float(np.mean(tempogram))
        confidence = peak_energy / (mean_energy + 1e-12)
        self.last_confidence = float(confidence)

        confident = bool(confidence > self.conf_threshold)
        trusted = bool(
            active
            and peak_energy > self.energy_threshold
            and (confident or not self._confident_once)
        )
        self.last_trusted = trusted

        if self._locked:
            lost = self._step_locked(tempogram_frame, tempogram, peak_energy, mean_energy, active,
                                     activation_buffer, activation_win)
            if not lost:
                return self._pulse_buffer

        if trusted:
            self._step_acquire(tempogram_frame, tempogram, confident)

        elif self._has_locked_tempo and self.current_tempo > 0:
            self._decay_factor *= 0.995

            omega = (self.current_tempo / 60) / self.framerate
            self._extrapolated_phase = (self._extrapolated_phase - self.H * omega) % 1.0

            t_start = 0
            t_end = self.N
            t = np.arange(t_start, t_end)

            x = self.win * np.cos(2 * np.pi * (t * omega - self._extrapolated_phase))
            x *= self._decay_factor

            kernel = Kernel(
                n=0,
                k=np.argmin(np.abs(self.Theta - self.current_tempo)),
                tempo=int(self.current_tempo),
                omega=omega,
                c=0.0,
                phase=self._extrapolated_phase,
                t_start=t_start,
                t_end=t_end,
                t=t,
                x=x,
            )
            self.current_kernel = kernel
            self._pulse_buffer += kernel.x

        else:
            self.current_tempo = 0.0
            self.current_kernel = Kernel(
                n=0, k=0, tempo=0, omega=0.0, c=0.0, phase=0.0,
                t_start=0, t_end=self.N, t=np.arange(self.N), x=np.zeros(self.N)
            )

        return self._pulse_buffer

    def _step_acquire(self, tempogram_frame, tempogram, confident):
        k_best = self._k_peak
        k = k_best

        if self._confident_once and self._has_locked_tempo and self.current_tempo > 0:
            k_cur = int(np.argmin(np.abs(self.Theta - self.current_tempo)))
            if abs(self.Theta[k_best] - self.current_tempo) > self.lock_tol_bpm:
                if abs(self.Theta[k_best] - self._chal_bpm) <= self.lock_tol_bpm:
                    self._chal_count += 1
                else:
                    self._chal_bpm, self._chal_count = float(self.Theta[k_best]), 1
                persisted = self._chal_count >= self.switch_persist_sec * self.framerate
                if not (persisted and tempogram[k_best, 0] >= self.switch_ratio * tempogram[k_cur, 0]):
                    k = k_cur
            else:
                self._chal_count = 0
                if tempogram[k_best, 0] < self.switch_ratio * tempogram[k_cur, 0]:
                    k = k_cur
            if k != k_cur:
                self._chal_count = 0

        self._has_locked_tempo = True
        self._decay_factor = 1.0

        kernel = Kernel.from_plp(
            N=self.N,
            H=self.H,
            win=self.win,
            Theta=self.Theta,
            framerate=self.framerate,
            X=tempogram_frame,
            n=0,
            tempogram=tempogram,
            k=k,
        )
        self.current_tempo = kernel.tempo
        self._extrapolated_phase = kernel.phase
        self.current_kernel = kernel
        self._pulse_buffer += kernel.x
        if confident:
            self._confident_once = True

        if self.lock_enable and confident:
            if abs(kernel.tempo - self._cand_bpm) <= self.lock_tol_bpm:
                self._cand_count += 1
            else:
                self._cand_bpm, self._cand_count = float(kernel.tempo), 1
            if self._cand_count >= self.lock_sec * self.framerate:
                self._enter_lock(tempogram, k, kernel)

    def _enter_lock(self, tempogram, k, kernel):
        bpm = float(self.Theta[k])
        if 0 < k < len(self.Theta) - 1:
            y0, y1, y2 = float(tempogram[k - 1, 0]), float(tempogram[k, 0]), float(tempogram[k + 1, 0])
            den = y0 - 2.0 * y1 + y2
            if den < 0:
                bpm += float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5)) * float(self.Theta[1] - self.Theta[0])
        self._locked = True
        self._lock_bpm = bpm
        self._theta = float((self.cursor * kernel.omega - kernel.phase) % 1.0)
        self._bad_count = 0.0
        self._err_ema = 0.0
        self._chal_count = 0
        self.n_locks += 1

    def _step_locked(self, tempogram_frame, tempogram, peak_energy, mean_energy, active, buf, win):
        fps = self.framerate
        omega = self._lock_bpm / 60.0 / fps
        self._theta = (self._theta + self.H * omega) % 1.0

        k_lock = int(np.argmin(np.abs(self.Theta - self._lock_bpm)))
        if buf is not None and win is not None:
            c = complex(np.sum(buf * win * np.exp(-2j * np.pi * omega * self._m)))
        else:
            c = complex(tempogram_frame[k_lock, 0])
        mag_lock = abs(c)
        if buf is not None and win is not None and self.harmonic_weight > 0:
            mag_lock += self.harmonic_weight * abs(complex(np.sum(buf * win * np.exp(-4j * np.pi * omega * self._m))))
        conf_lock = mag_lock / (mean_energy + 1e-12)
        gate = bool(active and peak_energy > self.energy_threshold)
        evidence = float(np.clip((conf_lock - 1.0) / max(self.pll_conf_hi - 1.0, 1e-6), 0.0, 1.0)) if gate else 0.0
        self._evidence = evidence

        if evidence > 0.0:
            phi_obs = (self.cursor * omega + np.angle(c) / (2.0 * np.pi)) % 1.0
            err = ((phi_obs - self._theta + 0.5) % 1.0) - 0.5
            self._err_ema += (1.0 / (3.0 * fps)) * (abs(err) - self._err_ema)
            if abs(err) <= self.phase_gate:
                a = 1.0 / (self.pll_tau_sec * fps)
                b = 0.25 * a * a
                self._theta = (self._theta + a * evidence * err) % 1.0
                omega = omega + b * evidence * err
                lo, hi = float(self.Theta.min()) - 2.0, float(self.Theta.max()) + 2.0
                self._lock_bpm = float(np.clip(omega * 60.0 * fps, lo, hi))
                omega = self._lock_bpm / 60.0 / fps

        if gate:
            rho = float(tempogram[k_lock, 0]) / (peak_energy + 1e-12)
            bad = (rho < self.lose_rho) or (conf_lock < self.lose_conf) or (self._err_ema > self.lose_err)
            self._bad_count = self._bad_count + 1.0 if bad else max(self._bad_count - 1.0, 0.0)
        if self._bad_count >= self.reacquire_sec * fps:
            self._locked = False
            self.current_tempo = self._lock_bpm
            self._confident_once = True
            self._cand_bpm, self._cand_count = 0.0, 0
            self._chal_bpm, self._chal_count = 0.0, 0
            self._bad_count = 0.0
            self.n_unlocks += 1
            return True

        phase_param = float((self.cursor * omega - self._theta) % 1.0)
        self._decay_factor = 1.0 if gate else self._decay_factor * 0.999
        t = np.arange(self.N)
        x = self.win * np.cos(2.0 * np.pi * (t * omega - phase_param)) * self._decay_factor
        self.current_kernel = Kernel(n=0, k=k_lock, tempo=self._lock_bpm, omega=omega, c=c, phase=phase_param,
                                     t_start=0, t_end=self.N, t=t, x=x)
        self.current_tempo = self._lock_bpm
        self._extrapolated_phase = phase_param
        self._has_locked_tempo = True
        self.last_trusted = gate and evidence > 0.0
        self._pulse_buffer += x
        return False

    def detect_beat(self, pulse_buffer: np.ndarray) -> bool:
        if not self._has_locked_tempo or np.max(np.abs(pulse_buffer)) < 0.1:
            self._last_beat_distance = 1_000_000_000
            return False

        buffer_peaks = Peaks.pick(pulse_buffer, self._t)
        future_beats = buffer_peaks.i[buffer_peaks.i > self._cursor]
        if len(future_beats) > 0:
            closest_future_beat = future_beats[0]
        else:
            closest_future_beat = 0
        distance_to_closest_future_beat = closest_future_beat - self._cursor
        beat_detected = distance_to_closest_future_beat > self._last_beat_distance
        self._last_beat_distance = distance_to_closest_future_beat
        if beat_detected:
            peak_amplitude = pulse_buffer[self._cursor]
            if peak_amplitude > self._max_peak_amplitude:
                self._max_peak_amplitude = peak_amplitude
            self.stability = peak_amplitude / max(self._max_peak_amplitude, 1e-6)
        return beat_detected


@dataclass
class ControlSignals:
    """Beat-Synchronous Control Signals Based on Real-Time PLP."""

    plp: PredominantLocalPulse

    def __post_init__(self):
        self._max_window_sum = self.calc_max_window_sum()
        self._alpha = self.calc_alpha()

    @property
    def tempo_bpm(self):
        p = self.plp
        if not p._has_locked_tempo or p.current_tempo <= 0:
            return 0.0
        return float(p.current_tempo)

    @property
    def beat_phase(self):
        p = self.plp
        if not p._has_locked_tempo or p.current_tempo <= 0:
            return None
        k = getattr(p, "current_kernel", None)
        if k is None or k.omega <= 0:
            return None
        return float((p.cursor * k.omega - k.phase) % 1.0)

    @property
    def lock_state(self):
        return int(self.plp._locked)

    @property
    def lock_score(self):
        return float(self.plp.lock_score)

    @property
    def beat_sin(self):
        ph = self.beat_phase
        return 0.0 if ph is None else float(np.sin(2.0 * np.pi * ph))

    @property
    def beat_cos(self):
        ph = self.beat_phase
        return 0.0 if ph is None else float(np.cos(2.0 * np.pi * ph))

    @property
    def normalized_buffer(self):
        return self.normalize_buffer(self.plp.current_buffer)

    @property
    def alpha_lfo(self):
        alpha_plp = self.calc_alpha_plp(self.plp.current_buffer)
        return alpha_plp[self.plp.cursor]

    @property
    def gamma_lfo(self):
        gamma_plp = self.calc_gamma_plp(self.plp.current_buffer)
        return gamma_plp[self.plp.cursor]

    @property
    def beta_confidence(self):
        beta_envelope = self.calc_beta_envelope(self.plp.current_buffer)
        return beta_envelope[self.plp.cursor]

    @property
    def gamma_confidence(self):
        gamma_envelope = self.calc_gamma_envelope(self.plp.current_buffer)
        return gamma_envelope[self.plp.cursor]

    @property
    def alpha_plp(self):
        return self.calc_alpha_plp(self.plp.current_buffer)

    @property
    def gamma_plp(self):
        return self.calc_gamma_plp(self.plp.current_buffer)

    @property
    def beta_envelope(self):
        return self.calc_beta_envelope(self.plp.current_buffer)

    @property
    def gamma_envelope(self):
        return self.calc_gamma_envelope(self.plp.current_buffer)

    def overlap_add_kernel_windows(self, window: np.ndarray, H: int, L: int):
        N = len(window)
        M = np.floor((L - N) / H).astype(int) + 1
        w_sum = np.zeros(L)
        for m in range(M):
            w_shifted = np.zeros(L)
            w_shifted[m * H : m * H + N] = window
            w_sum = w_sum + w_shifted
        return w_sum

    def calc_max_window_sum(self):
        window = signal.get_window(window="hann", Nx=self.plp.N)
        return np.sum(window).astype(int)

    def normalize_buffer(self, buffer):
        return buffer / self._max_window_sum

    def calc_alpha(self):
        H = 1
        L = 2 * self.plp.N
        window = signal.get_window(window="hann", Nx=self.plp.N)
        window_sum = self.overlap_add_kernel_windows(window, H, L)
        alpha = window_sum[(len(window_sum) // 2) :]
        alpha = self.normalize_buffer(alpha)
        return alpha

    def calc_alpha_plp(self, buffer):
        norm_buffer = self.normalize_buffer(buffer)
        alpha_plp = norm_buffer / self._alpha
        return alpha_plp

    def calc_gamma_plp(self, buffer):
        alpha_plp = self.calc_alpha_plp(buffer)
        beta_envelope = self.calc_beta_envelope(buffer)

        gamma_plp = np.divide(
            alpha_plp,
            beta_envelope,
            out=np.zeros_like(alpha_plp),
            where=(beta_envelope > 1e-8)
        )
        return gamma_plp

    def calc_beta_envelope(self, buffer):
        alpha_plp = self.calc_alpha_plp(buffer)
        beta_envelope = np.abs(signal.hilbert(alpha_plp))
        return beta_envelope

    def calc_gamma_envelope(self, buffer):
        beta_envelope = self.calc_beta_envelope(buffer)
        gamma_envelope = self._alpha * beta_envelope
        return gamma_envelope

    def calc_gamma_envelope2(self, buffer):
        norm_buffer = self.normalize_buffer(buffer)
        gamma_envelope = np.abs(signal.hilbert(norm_buffer))
        return gamma_envelope


@dataclass
class RealTimeBeatTracker:
    """Real-Time Beat Tracker with Predominant Local Pulse (PLP)"""

    activation: BeatActivation
    tempogram: Tempogram
    plp: PredominantLocalPulse
    cs: ControlSignals

    silence_db: float = -40.0
    hold_frames: int = 30

    _hold: int = field(default=0, init=False, repr=False)
    last_rms_db: float = field(default=-120.0, init=False, repr=False)
    active: bool = field(default=False, init=False, repr=False)

    def process(self, audio_frame: np.ndarray) -> bool:
        rms = np.sqrt(np.mean(np.asarray(audio_frame, dtype=np.float64) ** 2))
        self.last_rms_db = float(20 * np.log10(rms + 1e-12))
        if self.last_rms_db > self.silence_db:
            self._hold = self.hold_frames
        else:
            self._hold = max(self._hold - 1, 0)
        self.active = self._hold > 0

        activation_frame = self.activation.process(audio_frame)
        tempogram_frame = self.tempogram.process(activation_frame)
        pulse_buffer = self.plp.process(tempogram_frame, self.active,
                                        self.tempogram._tempo_buffer, self.tempogram.win,
                                        self.tempogram.harm_frames)
        beat_detected = self.plp.detect_beat(pulse_buffer)
        return beat_detected

    @classmethod
    def from_args(
        cls,
        N=3200,
        H=1600,
        samplerate=48000,
        N_time=10,
        Theta=np.arange(90, 180, 1),
        lookahead=0,
        silence_db=-40.0,
        hold_frames=30,
        conf_threshold=2.0,
        switch_ratio=1.25,
        energy_threshold=0.05,
        lock_enable=True,
        lock_sec=4.0,
        lock_tol_bpm=4.0,
        lose_rho=0.5,
        lose_conf=1.3,
        lose_err=0.4,
        phase_gate=0.15,
        reacquire_sec=4.0,
        pll_tau_sec=1.5,
        pll_conf_hi=3.0,
        switch_persist_sec=3.0,
        harmonic_weight=1.0,
    ):
        act = BeatActivation(N=N, H=H, samplerate=samplerate)
        tempo = Tempogram(N_time=N_time, framerate=(samplerate / H), Theta=Theta,
                          harmonics=(2,) if harmonic_weight > 0 else ())
        pulse = PredominantLocalPulse(
            N_time=N_time,
            framerate=(samplerate / H),
            Theta=Theta,
            lookahead=lookahead,
            conf_threshold=conf_threshold,
            switch_ratio=switch_ratio,
            energy_threshold=energy_threshold,
            lock_enable=lock_enable,
            lock_sec=lock_sec,
            lock_tol_bpm=lock_tol_bpm,
            lose_rho=lose_rho,
            lose_conf=lose_conf,
            lose_err=lose_err,
            phase_gate=phase_gate,
            reacquire_sec=reacquire_sec,
            pll_tau_sec=pll_tau_sec,
            pll_conf_hi=pll_conf_hi,
            switch_persist_sec=switch_persist_sec,
            harmonic_weight=harmonic_weight,
        )
        cs = ControlSignals(plp=pulse)
        return cls(
            activation=act,
            tempogram=tempo,
            plp=pulse,
            cs=cs,
            silence_db=silence_db,
            hold_frames=hold_frames,
        )

@dataclass
class BeatAnalyzer:
    """Beat analyzer class wrapper for storing all beat tracker frames over time."""
    tracker: RealTimeBeatTracker
    frame_indices: list = field(default_factory=list, init=False)
    frame_times: list = field(default_factory=list, init=False)
    audio_frames: list = field(default_factory=list, init=False)
    activation_frames: list = field(default_factory=list, init=False)
    tempogram_frames: list = field(default_factory=list, init=False)
    tempo_values: list = field(default_factory=list, init=False)
    kernels: list = field(default_factory=list, init=False)
    plp_buffers: list = field(default_factory=list, init=False)

    alpha_lfos: list = field(default_factory=list, init=False)
    gamma_lfos: list = field(default_factory=list, init=False)
    beta_confs: list = field(default_factory=list, init=False)
    gamma_confs: list = field(default_factory=list, init=False)

    alpha_plps: list = field(default_factory=list, init=False)
    gamma_plps: list = field(default_factory=list, init=False)
    beta_envs: list = field(default_factory=list, init=False)
    gamma_envs: list = field(default_factory=list, init=False)

    beat_sins: list = field(default_factory=list, init=False)
    beat_coss: list = field(default_factory=list, init=False)
    tempo_bpms: list = field(default_factory=list, init=False)

    beat_detection_frames: list = field(default_factory=list, init=False)

    rms_dbs: list = field(default_factory=list, init=False)
    tempogram_confs: list = field(default_factory=list, init=False)
    trusted_flags: list = field(default_factory=list, init=False)

    _frame_count: int = field(default=0, init=False)

    def process(self, audio_frame: np.ndarray) -> bool:
        beat_detected = self.tracker.process(audio_frame)
        self.frame_indices.append(self._frame_count)
        self.frame_times.append(self.get_frame_time(self._frame_count))
        self.audio_frames.append(audio_frame)
        self.activation_frames.append(self.tracker.activation.activation_frame)
        self.tempogram_frames.append(self.tracker.tempogram.tempogram_frame)
        self.tempo_values.append(self.tracker.plp.current_tempo)
        self.kernels.append(self.tracker.plp.current_kernel)
        self.plp_buffers.append(self.tracker.cs.normalized_buffer)
        self.beat_detection_frames.append(beat_detected)

        self.alpha_lfos.append(self.tracker.cs.alpha_lfo)
        self.gamma_lfos.append(self.tracker.cs.gamma_lfo)
        self.beta_confs.append(self.tracker.cs.beta_confidence)
        self.gamma_confs.append(self.tracker.cs.gamma_confidence)

        self.alpha_plps.append(self.tracker.cs.alpha_plp)
        self.gamma_plps.append(self.tracker.cs.gamma_plp)
        self.beta_envs.append(self.tracker.cs.beta_envelope)
        self.gamma_envs.append(self.tracker.cs.gamma_envelope)

        self.beat_sins.append(self.tracker.cs.beat_sin)
        self.beat_coss.append(self.tracker.cs.beat_cos)
        self.tempo_bpms.append(self.tracker.cs.tempo_bpm)

        self.rms_dbs.append(self.tracker.last_rms_db)
        self.tempogram_confs.append(self.tracker.plp.last_confidence)
        self.trusted_flags.append(self.tracker.plp.last_trusted)

        self._frame_count += 1
        return beat_detected

    def get_frame_time(self, frame_count: int) -> float:
        return frame_count / self.tracker.plp.framerate


if __name__ == "__main__":
    tracker = RealTimeBeatTracker.from_args()
    analyzer = BeatAnalyzer(tracker)