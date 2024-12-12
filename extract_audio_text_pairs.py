import pandas as pd
import csv

# Load the ESC-50 metadata
esc50_metadata = pd.read_csv('ESC-50-master/meta/esc50.csv')

# Extract relevant columns
audio_files = esc50_metadata['filename'].tolist()
labels = esc50_metadata['category'].tolist()

# Save the pairs to a new CSV file
with open('audio_text_pairs.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Audio File', 'Text Label'])  # Write header
    for audio, label in zip(audio_files, labels):
        writer.writerow([audio, label])  # Write each pair


