import os
import numpy as np
import pywt
import librosa
import pandas as pd
import scipy.signal
from tqdm import tqdm

def _preEmphasis(wave: np.ndarray, p=0.97) -> np.ndarray:
    """Pre-Emphasis
    apply a pre-emphasis filter to the audio signal.
    can enhance the signal-to-noise ratio and make the features extracted from the audio more robust and informative
    """
    return scipy.signal.lfilter([1.0, -p], 1, wave)
def _extract_label(protocol: pd.DataFrame) -> np.ndarray:
    """Extract labels from ASVSpoof2019 protocol

    Args:
        protocol (pd.DataFrame): ASVSpoof2019 protocol

    Returns:
        np.ndarray: Labels.
    """
    labels = np.ones(len(protocol))
    labels[protocol["Label"] == "bonafide"] = 0
    return labels.astype(int)

def _calc_wavelets(path: str) -> np.ndarray:
    """Calculate wavelets .

    Args:
        path (str): Path to audio file

    Returns:
        np.ndarray: 2d wavelets .
    """
    MAX_LENGTH = 200
    wave, sr = librosa.load(path, sr=16000)
    wave = _preEmphasis(wave)
    wave = librosa.feature.melspectrogram(y=wave, sr=sr, n_fft=2048, hop_length=160, win_length=320, n_mels=40,fmax=8000)
    wave = librosa.power_to_db(wave, ref=np.max)
    gd_mod = (wave-np.min(wave))/(np.max(wave)-np.min(wave))
    wp = pywt.WaveletPacket2D(data=gd_mod, wavelet='db1', mode='symmetric')
    gd_mod = wp.data

    # Trim or pad the spectrogram to ensure it has the specific maximum length
    if gd_mod.shape[1] > MAX_LENGTH:
        gd_mod = gd_mod[:, :MAX_LENGTH]
    else:
        padding = MAX_LENGTH - gd_mod.shape[1]
        gd_mod = np.pad(gd_mod, ((0, 0), (0, padding)), mode='constant')

    gd_mod = gd_mod.astype("float32")
    return gd_mod[..., np.newaxis]

def calc_wavelets(protocol_df: pd.DataFrame, path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    This function extracts wavelets from mel spec.

    Args:
        protocol_df (pd.DataFrame): ASVspoof2019 protocol.
        path (str): Path to ASVSpoof2019

    Returns:
        Tuple[np.ndarray, np.ndarray]: Data and labels.
            - data: wavelets with 4 dimensions like (n_samples, height, width, 1)
            - labels: 0 = Genuine, 1 = Spoof
    """

    data = []
    for audio in tqdm(protocol_df["utt_id"]):
        file = path + audio + ".flac"
        # Calculate wavelets
        grp_spec = _calc_wavelets(file)
        data.append(grp_spec)

    # Extract labels from protocol
    labels = _extract_label(protocol_df)

    stacked_data = np.stack(data, axis=0)
    
    return stacked_data, labels