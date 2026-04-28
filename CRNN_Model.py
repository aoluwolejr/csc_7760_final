import os, torch, torchaudio
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from torch.utils.tensorboard.writer import SummaryWriter
from typing import cast

# Define the dataset class
class GTZANDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: Subset):
        self.dataset = dataset
        self.mel_spectrogram = MelSpectrogram(n_fft=256, hop_length=128)
        self.amplitude_to_db = AmplitudeToDB()

        # Generate label mapping dynamically
        self.label_mapping = self.generate_label_mapping()
        print("Generated label mapping:", self.label_mapping)

    def generate_label_mapping(self):
        """Create a mapping from unique labels to integers."""
        unique_labels = set()
        for _, _, label, *_ in self.dataset:  # Iterate through the dataset
            unique_labels.add(label)
        return {label: idx for idx, label in enumerate(sorted(unique_labels))}

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int | list[int]) -> tuple[torch.Tensor, int]:
        """ TODO Load and preprocess a single sample from the dataset."""
        waveform, sample_rate, label, *_ = self.dataset[idx]

        # Map label to integer
        if label in self.label_mapping:
            label = self.label_mapping[label]
        else:
            raise ValueError(f"Unexpected label: {label}")

        # Pad or truncate waveform to the same length
        waveform = self.pad_waveform(waveform)

        # Generate spectrogram
        spectrogram = self.amplitude_to_db(self.mel_spectrogram(waveform)) # Convert amplitude to decibels

        # pad spectrogram from (1, 128, 126) to (1, 128, 128)
        spectrogram = torch.nn.functional.pad(spectrogram, (0, 2))
        return spectrogram, label

    @staticmethod
    def pad_waveform(waveform: torch.Tensor, length: int = 16000) -> torch.Tensor:
        """ TODO Pad or truncate waveform to the same length."""
        if waveform.size(-1) < length:
            waveform = torch.nn.functional.pad(waveform, (0, length - waveform.size(-1)))
        else:
            waveform = waveform[:, :length]

        return waveform

# Load the dataset
data_path = os.path.normpath("data")
os.makedirs(data_path, exist_ok=True)
#Data must be in a genres folder
dataset_path = "./gtzan_300ms_150ms_overlap/data"
dataset = torchaudio.datasets.GTZAN(root=dataset_path, download=False)
total_size = len(dataset)

# Split dataset into train and test sets
train_size = int(0.8 * total_size)  # Fill in the train size fraction
test_size = total_size - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Apply the custom preprocessing pipeline
train_dataset = GTZANDataset(train_dataset)
test_dataset = GTZANDataset(test_dataset)

dl_batch_size = 32

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=dl_batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=dl_batch_size, shuffle=True)

class CRNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.crnn_stack_1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(32), #1
            nn.Conv2d(1, 64, kernel_size=3, padding='same'), #2
            nn.ReLU(),
            nn.BatchNorm2d(64), #3
            nn.MaxPool2d(kernel_size=2), #4
            nn.Conv2d(64, 64, kernel_size=3, padding='same'), #5
            nn.ReLU(),
            nn.BatchNorm2d(64), #6
            nn.MaxPool2d(kernel_size=2) #7
        )

        self.gru_layer = nn.GRU(512, 512, 2)

        self.crnn_stack_2 = nn.Sequential(
            nn.Dropout(p=0.5), #12
            nn.Linear(512, num_classes) #13
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #1 - 7
        x = self.crnn_stack_1(x)
        #8: permute - (0,2,1,3)
        x = torch.permute(x,(0,2,1,3))
        #9: reshape - (shape[0], shape[1], -1)
        x_shape = x.size()
        x = torch.reshape(x, (x_shape[0], x_shape[1], -1))
        #10, 11
        x, _ = self.gru_layer(x)
        #12, 13
        x = self.crnn_stack_2(x)
        return x

# Dynamically determine the number of classes
num_classes = len(train_dataset.label_mapping)
model = CRNN(num_classes)

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

# Initialize TensorBoard
writer = SummaryWriter()

# Train the model
num_epochs = 10

device = torch.device("mps")
model.to(device)
print("Model Ready")

for epoch in range(num_epochs):
    model.train()
    total_loss: float = 0
    total_correct: int = 0
    total_samples: int = 0

    for spectrograms, labels in train_loader:
        spectrograms, labels = spectrograms.to(device), labels.to(device)

        # Forward Pass
        pred_labels = model(spectrograms)
        pred_labels = pred_labels.mean(dim=1) #averaging to change shape from [256, 8, 35] -> [256, 35]

        loss = criterion(pred_labels, labels)

        # Backward Pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculating Loss and Accuracy       
        _, predicted = torch.max(pred_labels.data, 1)

        total_loss += loss.item()
        total_correct += (predicted == labels).sum()
        total_samples += len(labels)

    train_accuracy = total_correct/total_samples
    writer.add_scalar("Loss/train", total_loss / len(train_loader), epoch)
    writer.add_scalar("Accuracy/train", train_accuracy, epoch)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}, Accuracy: {train_accuracy:.4f}")


# Evaluate the model
model.eval()
total_correct: int = 0
total_samples: int = 0

with torch.no_grad():
    for spectrograms, labels in test_loader:
        spectrograms, labels = spectrograms.to(device), labels.to(device)

        #Forward Pass
        test_pred_labels = model(spectrograms)
        test_pred_labels = test_pred_labels.mean(dim=1) #averaging to change shape from [256, 8, 35] -> [256, 35]

        #Accuracy Compilation
        _, predicted = torch.max(test_pred_labels.data, 1)

        total_correct += (predicted == labels).sum()
        total_samples += len(labels)

test_accuracy = total_correct/total_samples
print(f"Test Accuracy: {test_accuracy:.4f}")
