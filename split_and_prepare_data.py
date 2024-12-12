import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import librosa

# Load the paired data
data = pd.read_csv('audio_text_pairs.csv')

# Load the saved features and labels from the previous step
mfcc_features = np.load('mfcc_features.npy', allow_pickle=True)
spectrogram_features = np.load('spectrogram_features.npy', allow_pickle=True)
labels = np.load('audio_labels.npy', allow_pickle=True)

# Combine MFCCs and spectrograms (ensure matching dimensions)
features = []
for mfcc, spectrogram in zip(mfcc_features, spectrogram_features):
    # Ensure the MFCC features match the number of frames in the spectrogram
    n_frames = spectrogram.shape[1]  # Number of time frames in spectrogram

    # If MFCC has only one dimension (mean across time), we need to reshape it to match the frames
    if len(mfcc.shape) == 1:
        mfcc = np.expand_dims(mfcc, axis=-1)  # Add a time dimension (assume 1 frame)

    # Ensure MFCC frames match spectrogram frames (either pad or trim)
    if mfcc.shape[1] < n_frames:
        mfcc = np.pad(mfcc, ((0, 0), (0, n_frames - mfcc.shape[1])), mode='constant')
    elif mfcc.shape[1] > n_frames:
        mfcc = mfcc[:, :n_frames]

    # Combine MFCC and spectrogram along the feature axis (axis=0)
    feature = np.concatenate([mfcc, spectrogram], axis=0)  # Combine both features along the frequency/time axis
    features.append(feature)

features = np.array(features)

# Split the dataset (80% training, 10% validation, 10% test)
X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Save labels as numpy arrays (optional, depends on your needs)
np.save('y_train.npy', y_train)
np.save('y_val.npy', y_val)
np.save('y_test.npy', y_test)

# Normalize features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1))  # Flatten the features
X_val = scaler.transform(X_val.reshape(X_val.shape[0], -1))
X_test = scaler.transform(X_test.reshape(X_test.shape[0], -1))

# Save the preprocessed features
np.save('X_train.npy', X_train)
np.save('X_val.npy', X_val)
np.save('X_test.npy', X_test)
