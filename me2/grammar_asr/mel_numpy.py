"""Torch-free 40-band log-mel feature extraction (NumPy only).

Drop-in replacement for features.log_mel on the RPi5, where installing
torchaudio is impractical. Matches the training pipeline's parameters:
  SAMPLE_RATE=16000, N_FFT=512, HOP=160 (10 ms), WIN=512, N_MELS=40,
  Hann window, power->dB->log, per-utterance mean/std normalization.

NOTE: torchaudio's MelSpectrogram defaults to a *linear* (power) scale
before .log(). We replicate that exactly: |STFT|^2 -> mel -> clamp -> log.
"""
from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000
N_MELS = 40
N_FFT = 512
HOP = 160
WIN = 512


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(n_mels: int, n_fft: int, sample_rate: int,
                    fmin: float = 0.0, fmax: float | None = None) -> np.ndarray:
    """Triangular mel filterbank, shape (n_mels, n_fft//2+1)."""
    if fmax is None:
        fmax = sample_rate / 2.0
    mel_min = _hz_to_mel(np.array([fmin]))
    mel_max = _hz_to_mel(np.array([fmax]))
    mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts = _mel_to_hz(mel_pts)
    bins = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    bin_pts = np.searchsorted(bins, hz_pts)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    bp = np.asarray(bin_pts, dtype=int).ravel()
    for m in range(n_mels):
        left, center, right = int(bp[m]), int(bp[m + 1]), int(bp[m + 2])
        for i in range(left, center):
            if center > left:
                fb[m, i] = (i - left) / (center - left)
        for i in range(center, right):
            if right > center:
                fb[m, i] = (right - i) / (right - center)
    return fb


_MEL_FB = None


def _get_mel_fb() -> np.ndarray:
    global _MEL_FB
    if _MEL_FB is None:
        _MEL_FB = _mel_filterbank(N_MELS, N_FFT, SAMPLE_RATE)
    return _MEL_FB


def stft_power(audio: np.ndarray) -> np.ndarray:
    """Hann-windowed STFT power magnitude, shape (N_FFT//2+1, T).

    Matches torchaudio MelSpectrogram framing: T = 1 + (N-1)//HOP, with the
    last frame zero-padded to WIN samples."""
    win = np.hanning(WIN).astype(np.float32)
    # torchaudio MelSpectrogram pads one full window (WIN) at the end
    audio = np.pad(audio, (0, WIN))
    N = len(audio)
    n_frames = 1 + (N - WIN) // HOP
    starts = np.arange(n_frames) * HOP
    # gather frames, zero-padding the tail
    idx = starts[:, None] + np.arange(WIN)[None, :]   # (T, WIN)
    valid = idx < N
    idx = np.where(valid, idx, 0)
    frames = audio[idx] * win[None, :] * valid         # (T, WIN)
    spec = np.fft.rfft(frames, n=N_FFT, axis=1)        # (T, n_bins)
    power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
    return power.T  # (n_bins, T)


def log_mel(audio: np.ndarray) -> np.ndarray:
    """40-band log-mel features, shape (T, 40)."""
    power = stft_power(audio)              # (n_bins, T)
    mel = _get_mel_fb() @ power            # (40, T)
    logspec = np.clip(mel, 1e-5, None)
    logspec = np.log(logspec).astype(np.float32)
    return logspec.T                       # (T, 40)


def normalize_feat(feat: np.ndarray) -> np.ndarray:
    """Per-utterance mean/std normalization over time."""
    mu = feat.mean(axis=0, keepdims=True)
    sd = feat.std(axis=0, keepdims=True) + 1e-8
    return ((feat - mu) / sd).astype(np.float32)


def _resample_linear(audio: np.ndarray, sr: int, target: int = SAMPLE_RATE) -> np.ndarray:
    """Simple linear resampler (torch-free). Good enough for 12k->16k."""
    if sr == target:
        return audio
    n_out = int(len(audio) * target / sr)
    if n_out < 2:
        return np.zeros(2, dtype=np.float32)
    x_old = np.arange(len(audio), dtype=np.float64)
    x_new = np.linspace(0.0, len(audio) - 1, n_out)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def wav_to_feat(path_or_audio, sr: int = SAMPLE_RATE,
                max_seconds: float = 6.0) -> np.ndarray:
    """Load a WAV path (or float32 array @ SAMPLE_RATE) -> normalized (T, 40).

    Trims/pads to max_seconds (same as the training dataset). Resamples to
    16 kHz if the source sample rate differs (some dataset WAVs are 12 kHz)."""
    if isinstance(path_or_audio, (str,)):
        import soundfile as sf
        audio, src_sr = sf.read(path_or_audio, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if src_sr != SAMPLE_RATE:
            audio = _resample_linear(audio, src_sr, SAMPLE_RATE)
    else:
        audio = np.asarray(path_or_audio, dtype=np.float32)
    max_samples = int(max_seconds * SAMPLE_RATE)
    if len(audio) > max_samples:
        audio = audio[:max_samples]
    elif len(audio) < max_samples:
        audio = np.pad(audio, (0, max_samples - len(audio)))
    return normalize_feat(log_mel(audio))


if __name__ == "__main__":
    import sys, time
    p = sys.argv[1]
    t0 = time.time()
    f = wav_to_feat(p)
    print(f"{p}: audio->feat {f.shape} range [{f.min():.2f},{f.max():.2f}] "
          f"in {time.time()-t0:.3f}s")
