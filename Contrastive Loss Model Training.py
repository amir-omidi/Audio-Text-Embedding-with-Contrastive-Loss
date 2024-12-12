import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Assuming you have a simple model to extract embeddings for audio and text
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        # Define your neural network here (example: a simple fully connected layer)
        self.fc = nn.Linear(512, 128)

    def forward(self, x):
        return torch.relu(self.fc(x))


# Contrastive loss function (updated with binary labels)
def contrastive_loss(y, embedding_audio, embedding_text, margin=1.0):
    # Normalize the embeddings to have unit length (L2 norm)
    embedding_audio = torch.nn.functional.normalize(embedding_audio, p=2, dim=1)
    embedding_text = torch.nn.functional.normalize(embedding_text, p=2, dim=1)

    # Compute the Euclidean distance between the embeddings
    distance = torch.norm(embedding_audio - embedding_text, p=2, dim=1)

    # Calculate the positive (similar) loss: when y == 1
    positive_loss = y * distance ** 2

    # Calculate the negative (dissimilar) loss: when y == 0
    negative_loss = (1 - y) * torch.clamp(margin - distance, min=0) ** 2

    # Total loss: sum of positive and negative losses
    total_loss = 0.5 * (positive_loss + negative_loss)

    return total_loss.mean()  # Return the mean loss to make it scalar


# Evaluation function to compute accuracy, precision, recall, and F1 score
def evaluate(model, data_loader):
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for batch_audio, batch_text, batch_labels in data_loader:
            # Get embeddings
            audio_embeddings = model(batch_audio)
            text_embeddings = model(batch_text)
            cosine_similarity = torch.cosine_similarity(audio_embeddings, text_embeddings)
            predictions = (cosine_similarity > 0.5).float()  # 0.5 is the threshold for similarity
            all_labels.extend(batch_labels.cpu().numpy())  # Flatten to a 1D array
            all_predictions.extend(predictions.cpu().numpy())  # Flatten to a 1D array
    all_labels = np.array(all_labels).flatten()
    all_predictions = np.array(all_predictions).flatten()
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    return accuracy, precision, recall, f1


# Training loop
def train(model, train_loader, val_loader, epochs=10, learning_rate=0.001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loss = []
    val_accuracy = []
    val_precision = []
    val_recall = []
    val_f1 = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_audio, batch_text, batch_labels in train_loader:
            optimizer.zero_grad()

            # Get embeddings
            audio_embeddings = model(batch_audio)
            text_embeddings = model(batch_text)

            # Convert labels into binary (1 for same, 0 for different)
            y = (batch_labels[:, None] == batch_labels).float()  # Binary labels: 1 if same, 0 if different

            # Compute contrastive loss
            loss = contrastive_loss(y, audio_embeddings, text_embeddings)

            # Backpropagate
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation
        val_accuracy_epoch, val_precision_epoch, val_recall_epoch, val_f1_epoch = evaluate(model, val_loader)

        train_loss.append(running_loss / len(train_loader))
        val_accuracy.append(val_accuracy_epoch)
        val_precision.append(val_precision_epoch)
        val_recall.append(val_recall_epoch)
        val_f1.append(val_f1_epoch)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}, "
              f"Validation Accuracy: {val_accuracy_epoch}, Precision: {val_precision_epoch}, "
              f"Recall: {val_recall_epoch}, F1 Score: {val_f1_epoch}")

    # Plot metrics after training
    plot_metrics(train_loss, val_accuracy, val_precision, val_recall, val_f1)


# Example data generation (replace this with your actual dataset)
def generate_fake_data(num_samples=1000):
    # Generate random audio and text embeddings 
    audio_data = torch.randn(num_samples, 512)  # Example: 512-dimensional audio embeddings
    text_data = torch.randn(num_samples, 512)  # Example: 512-dimensional text embeddings

    # Random labels (replace with actual labels)
    labels = np.random.randint(0, 2, size=num_samples)  # Binary labels (0 or 1 for simplicity)

    # Create a simple train-test split
    train_audio, val_audio, train_text, val_text, train_labels, val_labels = train_test_split(
        audio_data, text_data, labels, test_size=0.2, random_state=42)

    train_data = [(train_audio[i], train_text[i], torch.tensor(train_labels[i])) for i in range(len(train_labels))]
    val_data = [(val_audio[i], val_text[i], torch.tensor(val_labels[i])) for i in range(len(val_labels))]

    return train_data, val_data


# Function to plot metrics
def plot_metrics(train_loss, val_accuracy, val_precision, val_recall, val_f1):
    epochs = range(1, len(train_loss) + 1)

    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label="Training Loss", color="b")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss over Epochs")
    plt.legend()
    plt.show()

    # Plot validation metrics
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_accuracy, label="Accuracy", color="g")
    plt.plot(epochs, val_precision, label="Precision", color="r")
    plt.plot(epochs, val_recall, label="Recall", color="orange")
    plt.plot(epochs, val_f1, label="F1 Score", color="purple")
    plt.xlabel("Epochs")
    plt.ylabel("Metric Value")
    plt.title("Validation Metrics over Epochs")
    plt.legend()
    plt.show()


# Example usage
train_data, val_data = generate_fake_data(num_samples=1000)

# Convert data into DataLoader
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=False)

# Initialize and train the model
model = SimpleModel()
train(model, train_loader, val_loader, epochs=50, learning_rate=0.001)
