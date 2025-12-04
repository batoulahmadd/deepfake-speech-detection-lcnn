import pickle
from typing import Tuple
import gc
import librosa
import numpy as np
import pandas as pd
import scipy
from tqdm import tqdm
from scipy.fftpack import dct
import os
import psutil

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



def _calc_stft(path: str) -> np.ndarray:
    """Calculate STFT with librosa.

    Args:
        path (str): Path to audio file

    Returns:
        np.ndarray: A STFT spectrogram.
    """
    MAX_LENGTH = 200
    wave, sr = librosa.load(path, sr=16000)
    wave = _preEmphasis(wave)
    stft = librosa.core.stft(wave, n_fft=512, win_length=320, hop_length=160, window="hamming")
    amp_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    
    # Trim or pad the spectrogram to ensure it has the specific maximum length
    if amp_db.shape[1] > MAX_LENGTH:
        amp_db = amp_db[:, :MAX_LENGTH]
    else:
        padding = MAX_LENGTH - amp_db.shape[1]
        amp_db = np.pad(amp_db, ((0, 0), (0, padding)), mode='constant')
    
    amp_db = amp_db.astype("float32")
    return amp_db[..., np.newaxis]


def calc_stft(protocol_df: pd.DataFrame, path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function extracts spectrograms from raw audio data by using FFT.

    Args:
        protocol_df (pd.DataFrame): ASVspoof2019 protocol.
        path (str): Path to ASVSpoof2019

    Returns:
        Tuple[np.ndarray, np.ndarray]: Data and labels.
            - data: Spectrograms with 4 dimensions like (n_samples, height, width, 1)
            - labels: 0 = Genuine, 1 = Spoof
    """

    data = []
    for audio in tqdm(protocol_df["utt_id"]):
        file = path + audio + ".flac"
        # Calculate STFT
        stft_spec = _calc_stft(file)
        data.append(stft_spec)

    # Extract labels from protocol
    labels = _extract_label(protocol_df)

    stacked_data = np.stack(data, axis=0)
    
    return stacked_data, labels
