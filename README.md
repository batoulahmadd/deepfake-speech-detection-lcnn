# deepfake-speech-detection-lcnn

## Project Overview
This project develops a deepfake speech detection system using a Light Convolutional Neural Network (LCNN) trained on wavelet-based features. Audio signals were processed using both     traditional time-frequency representations (STFT, Mel-spectrogram) and multi-resolution wavelet transforms (DWT & WPT). The system was evaluated on the official ASVspoof2019-LA dataset.

## Motivation
AI-generated speech is becoming increasingly realistic, creating risks in security, fraud, and identity verification. This project aims to build a lightweight yet robust classifier capable of distinguishing real vs. synthetic speech under real-world conditions such as:
  - different languages (Arabic & English)
  - speaker gender (male/female)
  - noise variations (different SNR levels)

## Methods & System Architecture
  - Input normalized to 2-second audio segments
  - Zero-padding / trimming to enforce dimension consistency
  - Wavelet-based features extracted from:
      -- Raw waveform
      -- Mel-spectrogram
  - Features saved as .npy for fast training
  - ML Model: Light CNN (LCNN)
  - Training: AdamW optimizer, 50 epochs, batch size 32
  - Evaluation Metrics: Equal Error Rate (EER) and min t-DCF

## Technologies Used
- Python 3.9
- PyTorch (or TensorFlow, whichever you used)
- NumPy, SciPy
- Librosa
- Scikit-learn
- Wavelets toolbox (PyWavelets)

## Key Results (Benchmarking on ASVspoof2019-LA)
   Feature Type	          Wavelet	        Levels	    EER	       min t-DCF
 DWT on Raw Audio	      Daubechies-4	      3	       8.89%	      0.0499
 DWT on Raw Audio      	Daubechies-4       	1	       11.35%	      0.0489
 Mel-Spectrogram	             –            –	       10.6%      	0.0500
      STFT	                   –	          –      	 18.92%     	0.0503

- Wavelet-based features significantly outperform baseline spectral methods.
- Db-4 at level 3 = best accuracy
- Db-4 at level 1 = best sensitivity


## Repository Structure
deepfake-speech-detection-lcnn/

│── implementations/

│   ├── STFT/

│   ├── Mel_spectrogram/

│   ├── DWT/

│   │   ├── Raw_data/

│   │   └── Mel_spectrogram/

│   └── WPT/

│── src/

│   ├── models/

│   │   ├── lcnn.py

│   │   └── layers.py

│   └── metrics.py

│── notebooks/

│   └── deepfake_demo.ipynb

│── docs/

│   ├── block_diagram.png

│   └── lcnn_architecture.png

│── data/

│   └── dummy_samples/

│── requirements.txt

│── README.md


## Dataset
This project uses the official ASVspoof2019-LA dataset:

🔗 https://datashare.ed.ac.uk/handle/10283/3336

Due to licensing and size restrictions, the dataset is not included.

A few short dummy audio samples are provided in `data/dummy_samples/`

for demonstration purposes

## System Architecture
The full processing pipeline consists of:

1️⃣ Preprocessing → 2️⃣ Feature Extraction (STFT / Mel / DWT / WPT) → 3️⃣ LCNN Model → 4️⃣ Evaluation

Block diagram available in `docs/block_diagram.png`

LCNN layer configuration available in `docs/lcnn_architecture.png`


## 🚀 How to Run
1- Install dependencies:

  pip install -r requirements.txt

2- Choose a feature extraction method:

Example: run pipeline for DWT on raw audio

  - cd implementations/DWT/Raw_data/
  - python run_pipeline.py
  
3- mathematica:

This will:

  - Extract features
  - Save .npy files
  - Train LCNN
  - Evaluate model performance


## Publications & Reports
Full Research Report (Arabic PDF) and presentation Slides:

🔗 https://drive.google.com/file/d/1BcEi5XujzH0SQSV9J6WYLFmzKmcxARH-/view


## Skills Demonstrated
  - Speech signal processing
  - Wavelet feature engineering
  - Deep learning for security
  - Benchmark experiment design
  - Scientific analysis & evaluation


## Author
Albatoul Ahmad
Telecom Engineer | Speech & Machine Learning
📩 batoulahmad292@gmail.com
