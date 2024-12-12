# Audio-Text Alignment with Deep Learning

**Author: Amirhossein Omidi**

This project explores the alignment between audio and text data using deep learning techniques. The focus is on extracting features from audio files, creating meaningful embeddings for both audio and text modalities, and aligning them using a contrastive loss function. The project leverages Mel-frequency cepstral coefficients (MFCC) and Mel-spectrogram features from the audio data, and a simple neural network model is trained to generate embeddings for both audio and text. The embeddings are then aligned using contrastive loss to improve similarity prediction.

## Dataset

This project uses the ESC-50: Dataset for Environmental Sound Classification dataset, which consists of paired audio and text data. Each pair contains an audio clip and its corresponding textual description. The audio data is preprocessed to extract features, and the text data is used to generate embeddings for comparison.

## Project Structure

- **`extract_audio_features.py`**: Contains the code for extracting MFCC and Mel-spectrogram features from the audio data.
- **`split_and_prepare_data.py`**: Handles the preprocessing of data, splitting it into training and validation sets, and preparing the audio and text pairs for training.
- **`Contrastive Loss Model Training.py`**: Main script for training the model using the contrastive loss function to align audio and text embeddings.
- **`extract_audio_text_pairs.py`**: Extracts and prepares audio-text pairs for training, ensuring that each audio clip is paired with its corresponding textual description.
- **`requirements.txt`**: Lists the required Python packages to run the project.

## Features

- **Feature Extraction**: 
   - MFCC and Mel-spectrogram features are extracted from the audio files using `librosa` and other audio processing libraries.
   
- **Model**:
   - A simple fully connected neural network model is designed to generate embeddings for both audio and text data.
   
- **Contrastive Loss**:
   - The model is trained using a contrastive loss function, which encourages similar audio-text pairs to be close in the embedding space and dissimilar pairs to be farther apart.

- **Evaluation**:
   - The model is evaluated using the following metrics:
     - **Accuracy**
     - **Precision**
     - **Recall**
     - **F1 Score**

## Requirements

To run this project, you'll need the following Python libraries:

- Python 3.x
- PyTorch
- NumPy
- scikit-learn
- librosa
- Matplotlib

You can install the required libraries with:

```bash
pip install -r requirements.txt
```

## Contact Me

For any questions or inquiries, please contact Amirhossein Omidi at [65mirhossein@gmail.com](mailto:65mirhossein@gmail.com)

