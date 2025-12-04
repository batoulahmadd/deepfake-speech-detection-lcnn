import librosa
import scipy
import numpy as np
import pandas as pd
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

def _calc_melspectrogram(path: str) -> np.ndarray:
    """ Calculate melspectrogram.

    Args:
        path (str): Path to audio file

    Returns:
        np.ndarray: melspectrogram.
    """

    MAX_LENGTH = 200
    wave, sr = librosa.load(path, sr=16000)
    wave = _preEmphasis(wave)
    wave = librosa.feature.melspectrogram(y=wave, sr=sr, n_fft=2048, hop_length=160, win_length=320, n_mels=40,fmax=8000)
    wave = librosa.power_to_db(wave, ref=np.max)
    mel_norm = (wave-np.min(wave))/(np.max(wave)-np.min(wave))

    # Trim or pad the coeffs to ensure it has the specific maximum length
    if mel_norm.shape[1] > MAX_LENGTH:
        mel_norm = mel_norm[:, :MAX_LENGTH]
    else:
        padding = MAX_LENGTH - mel_norm.shape[1]
        mel_norm = np.pad(mel_norm, ((0, 0), (0, padding)), mode='constant')

    mel_norm = mel_norm.astype("float32")
    return mel_norm[..., np.newaxis]
    
def calc_melspectrogram(protocol_df: pd.DataFrame, path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    This function extracts melspectrogram from raw audio data.

    Args:
        protocol_df (pd.DataFrame): ASVspoof2019 protocol.
        path (str): Path to ASVSpoof2019

    Returns:
        Tuple[np.ndarray, np.ndarray]: Data and labels.
            - data: _calc_wavelets
            - labels: 0 = Genuine, 1 = Spoof
    """

    data = []
    for audio in tqdm(protocol_df["utt_id"]):
        file = path + audio + ".flac"
        # Calculate melspectrogram
        mel_feature = _calc_melspectrogram(file)
        data.append(mel_feature)

    # Extract labels from protocol
    labels = _extract_label(protocol_df)

    stacked_data = np.stack(data, axis=0)

    return stacked_data, labels