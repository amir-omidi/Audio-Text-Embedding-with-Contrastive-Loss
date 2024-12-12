import librosa
import numpy as np
import pandas as pd
import os

# Load the paired data
data = pd.read_csv('audio_text_pairs.csv')

audio_folder = 'ESC-50-master/audio'  # Update this if needed

# Lists to store features and labels
mfcc_features = []
spectrogram_features = []
labels = []

for index, row in data.iterrows():
    audio_path = os.path.join(audio_folder, row['Audio File'])
    label = row['Text Label']

    try:
        # Load the audio file
        y, sr = librosa.load(audio_path, sr=None)

        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)  # Take the mean across time

        # Extract Mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  # Convert to dB scale

        # Append features and label
        mfcc_features.append(mfccs_mean)
        spectrogram_features.append(mel_spec_db)
        labels.append(label)
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")

# Save MFCCs, spectrograms, and labels to files
np.save('mfcc_features.npy', np.array(mfcc_features))
np.save('spectrogram_features.npy', np.array(spectrogram_features))
np.save('audio_labels.npy', np.array(labels))

print("Feature extraction complete. Saved MFCCs, spectrograms, and labels.")
