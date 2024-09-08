import torchaudio
import torch
import librosa
import numpy as np

# Function to preprocess audio by extracting Mel-Spectrograms
def preprocess_audio(audio_path, target_length=128, n_mels=40):
    waveform, sample_rate = torchaudio.load(audio_path)

    # Resample if necessary to 16kHz
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

    # Convert to NumPy for further processing
    waveform = waveform.numpy()[0]

    # Extract Mel-spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=waveform, sr=16000, n_mels=n_mels, fmax=8000)

    # Convert Mel-spectrogram to log scale (dB)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Pad or truncate to target length
    if mel_spectrogram_db.shape[1] > target_length:
        mel_spectrogram_db = mel_spectrogram_db[:, :target_length]
    else:
        padding = target_length - mel_spectrogram_db.shape[1]
        mel_spectrogram_db = np.pad(mel_spectrogram_db, ((0, 0), (0, padding)), mode='constant')

    # Flatten the Mel-spectrogram for input to the model
    return torch.tensor(mel_spectrogram_db.flatten(), dtype=torch.float32)
