import librosa
import numpy as np

def preprocess_single_audio(audio_data, sample_rate):
    """
    Preprocess a single audio file for inference.
    You can extract features similar to the ones used during training, e.g., MFCCs.
    """
    # Example: extract MFCCs with 40 coefficients
    n_mfcc = 40
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=n_mfcc)

    # Check the shape of mfccs to ensure it matches the expected input
    # Reshape to match the model's expected input: (1, 40, time_steps)
    mfccs = np.expand_dims(mfccs, axis=0)  # Add batch dimension (1, 40, time_steps)

    # Optionally, pad or trim to match the expected time_steps (5120 // 40 = 128)
    expected_time_steps = 128
    if mfccs.shape[2] > expected_time_steps:
        mfccs = mfccs[:, :, :expected_time_steps]  # Trim if too long
    elif mfccs.shape[2] < expected_time_steps:
        mfccs = np.pad(mfccs, ((0, 0), (0, 0), (0, expected_time_steps - mfccs.shape[2])), mode='constant')

    return mfccs
