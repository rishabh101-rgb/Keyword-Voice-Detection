

# Keyword Spotting with CNN-RNN using MFCC Features

This project implements a **Keyword Spotting** system using a combination of **Convolutional Neural Networks (CNN)** and **Recurrent Neural Networks (RNN)** built with **TensorFlow**. The model is trained to recognize four specific keywords from audio input:  
**"on"**, **"off"**, **"start"**, and **"stop"**.  

All other words are categorized as **unknown**, and background noise is labeled as **noise**.

## 🔍 Project Overview

- **Input:** Raw audio waveforms (.wav files)
- **Feature Extraction:** MFCCs (Mel Frequency Cepstral Coefficients) using `librosa`
- **Model Architecture:** Hybrid CNN + RNN model for sequence modeling
- **Output Classes:**  
  - `on`  
  - `off`  
  - `start`  
  - `stop`  
  - `unknown`  
  - `noise`




## 🧠 Feature Extraction

Audio preprocessing is handled using the `librosa` library:
- Sample rate normalization
- Silence trimming
- MFCC feature extraction (typically 13–40 coefficients)

```python
import librosa
y, sr = librosa.load('file.wav', sr=16000)
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
```

## 🏗️ Model Architecture

The model combines:
- **CNN Layers**: Capture local temporal-spectral features
- **RNN Layers (LSTM/GRU)**: Capture long-term temporal dependencies
- **Dense Output Layer**: Softmax activation for classifying into 6 categories

Key libraries:
- `tensorflow`
- `librosa`
- `numpy`
- `scikit-learn`
- `matplotlib` (for visualization)

## 📊 Results

The model achieves high accuracy in detecting the specified keywords, with robustness to noise and unknown words.

## 📝 Future Work

- Extend to more keywords
- Real-time inference using microphone input
- Optimize for edge devices (e.g., TensorFlow Lite)


