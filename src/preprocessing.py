"""STFT/MFCC preprocessing for Speech Commands v0.02.

Dependencies: numpy, scipy, python_speech_features.
Features have shape (time, feature, channel). No model is selected here.
"""
from pathlib import Path
from math import gcd
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

SAMPLE_RATE = 16000
NUM_SAMPLES = 16000


def load_audio(path):
    """Read WAV as mono float32 and resample to 16 kHz if necessary."""
    rate, wave = wavfile.read(path)
    if np.issubdtype(wave.dtype, np.signedinteger):
        scale = float(2 ** (np.iinfo(wave.dtype).bits - 1))
        wave = wave.astype(np.float32) / scale
    elif wave.dtype == np.uint8:
        wave = (wave.astype(np.float32) - 128.0) / 128.0
    elif np.issubdtype(wave.dtype, np.floating):
        wave = wave.astype(np.float32)
    else:
        raise ValueError(f"Unsupported WAV format: {wave.dtype}")
    if wave.ndim == 2:
        wave = wave.mean(axis=1)
    if wave.ndim != 1 or wave.size == 0 or not np.isfinite(wave).all():
        raise ValueError(f"Expected a nonempty, finite audio signal: {path}")
    if rate != SAMPLE_RATE:
        divisor = gcd(int(rate), SAMPLE_RATE)
        wave = resample_poly(wave, SAMPLE_RATE // divisor, int(rate) // divisor)
    return wave.astype(np.float32)


def pad_or_trim(wave):
    """Center-pad short clips; keep the first second of longer recordings."""
    wave = np.asarray(wave, dtype=np.float32)
    if wave.ndim != 1 or wave.size == 0 or not np.isfinite(wave).all():
        raise ValueError("Expected a nonempty, finite mono waveform.")
    wave = wave[:NUM_SAMPLES]
    missing = NUM_SAMPLES - len(wave)
    return np.pad(wave, (missing // 2, missing - missing // 2))


def add_background_noise(wave, noise_path, scale=0.2, rng=None):
    """Mix a random noise crop. Scale is an amplitude multiplier, not SNR.

    Use for training augmentation; keep evaluation clean unless explicitly
    running a separate, reproducible noise robustness experiment.
    """
    if not np.isfinite(scale) or scale < 0:
        raise ValueError("Noise scale must be finite and nonnegative.")
    wave = pad_or_trim(wave)
    if scale == 0:
        return wave.copy()
    rng = np.random.default_rng() if rng is None else rng
    noise = load_audio(noise_path)
    if len(noise) < len(wave):
        noise = np.tile(noise, int(np.ceil(len(wave) / len(noise))))
    start = int(rng.integers(0, len(noise) - len(wave) + 1))
    return np.clip(wave + scale * noise[start:start + len(wave)], -1, 1).astype(np.float32)


def extract_stft(wave):
    """Magnitude STFT: 256-sample periodic Hann window, 128-sample hop.

    Uses the framing/window convention of the original tf.signal.stft call,
    without end padding. Returns (124, 129, 1) for a one-second signal.
    """
    wave = pad_or_trim(wave)
    frames = np.lib.stride_tricks.sliding_window_view(wave, 256)[::128]
    window = np.hanning(257)[:-1]  # periodic Hann
    magnitude = np.abs(np.fft.rfft(frames * window, n=256, axis=-1))
    return magnitude.astype(np.float32)[..., None]


def extract_mfcc(wave):
    """Original MFCC settings, with corrected channel position: (99, 13, 1)."""
    from python_speech_features import mfcc
    coefficients = mfcc(
        pad_or_trim(wave), samplerate=SAMPLE_RATE,
        winlen=0.025, winstep=0.01, numcep=13, nfilt=26, nfft=512,
        lowfreq=0, highfreq=8000, preemph=0.97, ceplifter=22,
        appendEnergy=False, winfunc=np.hamming,
    )
    return coefficients.astype(np.float32)[..., None]


def build_manifest(data_dir):
    """Use official validation/testing lists; put all remaining WAVs in train.

    Return records with relative path, label, label_id, speaker_id, and split,
    plus the sorted class list. Reject missing files and speaker/file overlap.
    """
    root = Path(data_dir)
    files = sorted(p for p in root.glob('*/*.wav') if not p.parent.name.startswith('_'))
    if not files:
        raise FileNotFoundError(f"No class WAV files found under {root}")
    lists = {}
    for split, filename in [('validation', 'validation_list.txt'), ('test', 'testing_list.txt')]:
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Keep the official split file in the dataset root: {path}")
        lists[split] = {line.strip().replace('\\', '/') for line in path.read_text().splitlines() if line.strip()}
    relative = {p.relative_to(root).as_posix() for p in files}
    if lists['validation'] & lists['test']:
        raise ValueError('Validation and test file lists overlap.')
    missing = (lists['validation'] | lists['test']) - relative
    if missing:
        raise ValueError(f"Official split lists reference {len(missing)} missing audio files.")
    labels = sorted({p.parent.name for p in files})
    label_ids = {name: i for i, name in enumerate(labels)}
    records = []
    speakers = {split: set() for split in ['train', 'validation', 'test']}
    for path in files:
        rel = path.relative_to(root).as_posix()
        if '_nohash_' not in path.name:
            raise ValueError(f"Cannot identify Speech Commands speaker: {rel}")
        split = 'validation' if rel in lists['validation'] else 'test' if rel in lists['test'] else 'train'
        speaker = path.name.split('_nohash_')[0]
        speakers[split].add(speaker)
        records.append(dict(path=rel, label=path.parent.name, label_id=label_ids[path.parent.name], speaker_id=speaker, split=split))
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        if speakers[a] & speakers[b]:
            raise ValueError(f"Speakers overlap between {a} and {b}.")
    if any(not group for group in speakers.values()):
        raise ValueError('All three dataset splits must be nonempty.')
    return records, labels
