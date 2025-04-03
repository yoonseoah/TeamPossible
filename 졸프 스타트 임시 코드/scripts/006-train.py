import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# ==========================
# Dataset Filtering & Splitting
# ==========================
def filter_and_split_dataset(video_features_dir, audio_features_dir, output_dir, split_ratios=(0.7, 0.15, 0.15)):
    valid_data = []

    # Iterate over classes in video features directory
    for class_name in os.listdir(video_features_dir):
        video_class_dir = os.path.join(video_features_dir, class_name)
        audio_class_dir = os.path.join(audio_features_dir, class_name)

        if not os.path.isdir(video_class_dir) or not os.path.isdir(audio_class_dir):
            continue

        # Match video and audio features
        for video_file in os.listdir(video_class_dir):
            video_path = os.path.join(video_class_dir, video_file)
            audio_path = os.path.join(audio_class_dir, video_file)

            if os.path.exists(video_path) and os.path.exists(audio_path):
                valid_data.append((video_path, audio_path))

    # Shuffle and split data
    np.random.shuffle(valid_data)
    train_size = int(len(valid_data) * split_ratios[0])
    val_size = int(len(valid_data) * split_ratios[1])
    test_size = len(valid_data) - train_size - val_size

    train_data, val_data, test_data = np.split(valid_data, [train_size, train_size + val_size])

    # Save split data
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "train_data.npy"), train_data)
    np.save(os.path.join(output_dir, "validation_data.npy"), val_data)
    np.save(os.path.join(output_dir, "test_data.npy"), test_data)

    print(f"Dataset split completed: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test.")

# ==========================
# Dataset Definition
# ==========================
class MultiTaskDataset(Dataset):
    def __init__(self, split_data, label_file):
        self.data = np.load(split_data, allow_pickle=True)
        self.labels_df = pd.read_csv(label_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_path, audio_path = self.data[idx]

        # Load video and audio features
        video_features = np.load(video_path)  # Shape: [256]
        audio_features = np.load(audio_path)  # Shape: [256, 1024]

        # Extract label
        video_id = os.path.basename(video_path).replace('.npy', '')
        label_row = self.labels_df[self.labels_df['video_id'] == video_id]
        if label_row.empty:
            raise ValueError(f"Label for video ID {video_id} not found in the label file.")
        label = label_row['encoded_label'].values[0]

        return (
            torch.tensor(video_features, dtype=torch.float32),
            torch.tensor(audio_features, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long)
        )

# ==========================
# Multi-Task Learning Model
# ==========================
class MultiTaskModel(nn.Module):
    def __init__(self, video_input_dim=256, audio_output_dim=(256, 1024), num_classes=10):
        super(MultiTaskModel, self).__init__()

        # Shared Fully Connected Layer
        self.fc_shared = nn.Sequential(
            nn.Linear(video_input_dim, 512),
            nn.ReLU()
        )

        # Classification Branch
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

        # Reconstruction Branch
        self.reconstructor = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, np.prod(audio_output_dim)),
            nn.Unflatten(1, audio_output_dim)
        )

    def forward(self, x):
        shared_output = self.fc_shared(x)

        cls_output = self.classifier(shared_output)  # Classification
        rec_output = self.reconstructor(shared_output)  # Reconstruction

        return cls_output, rec_output

# ==========================
# Training Function
# ==========================
def train(model, train_loader, val_loader, criterion_cls, criterion_rec, optimizer, device, num_epochs=30):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for video_features, audio_features, labels in train_loader:
            video_features, audio_features, labels = video_features.to(device), audio_features.to(device), labels.to(device)

            # Forward pass
            cls_output, rec_output = model(video_features)

            # Compute losses
            loss_cls = criterion_cls(cls_output, labels)
            loss_rec = criterion_rec(rec_output, audio_features)
            loss = loss_cls + loss_rec

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Validation Step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for video_features, audio_features, labels in val_loader:
                video_features, audio_features, labels = video_features.to(device), audio_features.to(device), labels.to(device)
                cls_output, rec_output = model(video_features)
                loss_cls = criterion_cls(cls_output, labels)
                loss_rec = criterion_rec(rec_output, audio_features)
                val_loss += loss_cls.item() + loss_rec.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}")

# ==========================
# Main Training Code
# ==========================
if __name__ == "__main__":
    # 경로 설정
    video_features_dir = r"D:/vid_lstmfeatures"
    audio_features_dir = r"D:/aud-features"
    label_file = r"Audio-Feature-Generation/results/lstm_labels.csv"
    split_data_dir = r"D:/splited-data"

    # 데이터셋 스플릿
    filter_and_split_dataset(video_features_dir, audio_features_dir, split_data_dir)

    # 데이터셋 및 데이터로더
    train_dataset = MultiTaskDataset(
        split_data=os.path.join(split_data_dir, "train_data.npy"),
        label_file=label_file
    )
    val_dataset = MultiTaskDataset(
        split_data=os.path.join(split_data_dir, "validation_data.npy"),
        label_file=label_file
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # 모델, 손실 함수, 최적화 기법
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskModel(num_classes=10)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_rec = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    # Training
    print("Starting training...")
    train(model, train_loader, val_loader, criterion_cls, criterion_rec, optimizer, device, num_epochs=30)

    # 모델 저장
    torch.save(model.state_dict(), "trained_model.pth")
    print("Training completed and model saved as 'trained_model.pth'!")
