import os
import numpy as np
import pywt
import librosa
import scipy
import pandas as pd
from math import ceil, sqrt
from tqdm import tqdm

# === CONFIGURATION ===
WAVELET = 'db1'
LEVEL = 3
SAMPLE_RATE = 16000
DURATION = 2.0
EXPECTED_LENGTH = int(SAMPLE_RATE * DURATION)

def _preEmphasis(wave: np.ndarray, p=0.97) -> np.ndarray:
    """
    Optional pre-emphasis filter to enhance high frequencies.
    """
    return scipy.signal.lfilter([1.0, -p], 1, wave)

def _extract_label(protocol_df: pd.DataFrame) -> np.ndarray:
    """
    Converts protocol labels to binary: 0 = bonafide, 1 = spoof.
    """
    labels = np.ones(len(protocol_df))
    labels[protocol_df["Label"] == "bonafide"] = 0
    return labels.astype(int)

def _extract_dwt(signal: np.ndarray, wavelet=WAVELET, level=LEVEL) -> np.ndarray:
    """
    Perform full DWT and reshape to square 2D matrix.
    """
    coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level)
    coeffs = np.concatenate(coeffs)
    total_len = len(coeffs)
    side = ceil(sqrt(total_len))
    padded_len = side * side

    if total_len < padded_len:
        coeffs = np.pad(coeffs, (0, padded_len - total_len), mode='constant')

    return coeffs.reshape((side, side))

def _calc_dwt(path: str) -> np.ndarray:
    """
    Load audio from path, apply optional preemphasis, extract and reshape DWT.
    """
    signal, sr = librosa.load(path, sr=SAMPLE_RATE, duration=DURATION)

    # Pad/trim to fixed length
    if len(signal) < EXPECTED_LENGTH:
        signal = np.pad(signal, (0, EXPECTED_LENGTH - len(signal)), mode='constant')
    elif len(signal) > EXPECTED_LENGTH:
        signal = signal[:EXPECTED_LENGTH]

    signal = _preEmphasis(signal)

    return _extract_dwt(signal)

def calc_dwt(protocol_df: pd.DataFrame, audio_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute DWT features from raw waveform for all files listed in the protocol.
    """
    features = []
    for utt_id in tqdm(protocol_df["utt_id"], desc="Extracting DWT features"):
        file_path = os.path.join(audio_dir, utt_id + ".flac")
        feat = _calc_dwt(file_path)
        features.append(feat)

    features = np.stack(features)
    labels = _extract_label(protocol_df)

    return features, labels
