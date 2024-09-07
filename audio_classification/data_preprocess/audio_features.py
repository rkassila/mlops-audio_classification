import torchaudio
import torch
import librosa
import numpy as np

# Function to preprocess audio by extracting MFCCs
def preprocess_audio(audio_path, target_length=100, n_mfcc=13):
    waveform, sample_rate = torchaudio.load(audio_path)

    # Resample if necessary to 16kHz
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

    # Extract MFCCs
    mfcc = librosa.feature.mfcc(y=waveform.numpy()[0], sr=16000, n_mfcc=n_mfcc)

    # Pad or truncate to target_length
    if mfcc.shape[1] > target_length:
        mfcc = mfcc[:, :target_length]
    else:
        padding = target_length - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, padding)), mode='constant')

    return torch.tensor(mfcc.flatten(), dtype=torch.float32)
