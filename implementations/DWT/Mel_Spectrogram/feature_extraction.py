import pywt
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

def fix_width(mat, target_width):
    current_width = mat.shape[1]
    if current_width > target_width:
        return mat[:, :target_width]
    elif current_width < target_width:
        pad_width = target_width - current_width
        return np.pad(mat, ((0, 0), (0, pad_width)), mode='constant')
    else:
        return mat

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
    """ Calculate wavelets on mel.

    Args:
        path (str): Path to audio file

    Returns:
        np.ndarray: wavelets on mel.
    """

    MAX_LENGTH = 200
    wave, sr = librosa.load(path, sr=16000)
    wave = _preEmphasis(wave)
    wave = librosa.feature.melspectrogram(y=wave, sr=sr, n_fft=2048, hop_length=160, win_length=320, n_mels=40,fmax=8000)
    wave = librosa.power_to_db(wave, ref=np.max)
    mel_norm = (wave-np.min(wave))/(np.max(wave)-np.min(wave))

    # 3. Wavelet decomposition (2D DWT)
    coeffs = pywt.wavedec2(mel_norm, wavelet='db1', level=3, mode='symmetric')
    # A, (H, V, D) = coeffs  # A=Approximation, H/V/D=Horizontal/Vertical/Diagonal details

    A = coeffs[0]
    A_width = A.shape[1]

    Hs = []
    for details in coeffs[1:]:
        H = details[0]
        # Fix width (axis=1) to match A
        if H.shape[1] != A_width:
            H = fix_width(H, A.shape[1])
        Hs.append(H)

    gd_mod = np.concatenate([A] + Hs, axis=0)
    # gd_mod = np.concatenate([A, H], axis=0)  # Shape: (n_mels, time)

    # Trim or pad the coeffs to ensure it has the specific maximum length
    if gd_mod.shape[1] > MAX_LENGTH:
        gd_mod = gd_mod[:, :MAX_LENGTH]
    else:
        padding = MAX_LENGTH - gd_mod.shape[1]
        gd_mod = np.pad(gd_mod, ((0, 0), (0, padding)), mode='constant')

    gd_mod = gd_mod.astype("float32")
    return gd_mod[..., np.newaxis]
    
def calc_wavelets(protocol_df: pd.DataFrame, path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    This function extracts wavelets on mel from raw audio data.

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
        # Calculate STFT
        grp_spec = _calc_wavelets(file)
        data.append(grp_spec)

    # Extract labels from protocol
    labels = _extract_label(protocol_df)

    stacked_data = np.stack(data, axis=0)

    return stacked_data, labels