import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# ==========================
# Dataset 정의
# ==========================
class MultiTaskDataset(Dataset):
    def __init__(self, split_data, video_features_dir, audio_features_dir, label_file):
        self.data = np.load(split_data, allow_pickle=True)
        self.video_features_dir = video_features_dir
        self.audio_features_dir = audio_features_dir
        
        # 비디오 라벨 로드
        self.labels_df = pd.read_csv(label_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_path, audio_path = self.data[idx]
        
        # Video Features
        video_features = np.load(video_path)
        
        # Audio Features
        audio_features = np.load(audio_path)
        
        # Extract video label
        video_id = os.path.basename(video_path).replace('.npy', '')
        label_row = self.labels_df[self.labels_df['video_id'] == video_id]
        
        # 라벨 가져오기
        if label_row.empty:
            raise ValueError(f"Label for video ID {video_id} not found in the label file.")
        label = label_row['encoded_label'].values[0]

        return torch.tensor(video_features, dtype=torch.float32), \
               torch.tensor(audio_features, dtype=torch.float32), \
               torch.tensor(label, dtype=torch.long)


# ==========================
# Multi-Task Learning Model
# ==========================
class MultiTaskModel(nn.Module):
    def __init__(self, video_input_dim=256, audio_output_dim=(256, 1024), num_classes=10):
        super(MultiTaskModel, self).__init__()
        
        # Shared LSTM Layer
        self.lstm = nn.LSTM(input_size=video_input_dim, hidden_size=512, num_layers=2, batch_first=True)
        
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
        # Shared LSTM
        lstm_out, _ = self.lstm(x.unsqueeze(1))
        lstm_out = lstm_out[:, -1, :]  # 마지막 타임스텝 출력
        
        # Classification Branch
        cls_output = self.classifier(lstm_out)
        
        # Reconstruction Branch
        rec_output = self.reconstructor(lstm_out)
        
        return cls_output, rec_output

# ==========================
# Training Function
# ==========================
def train(model, train_loader, val_loader, criterion_cls, criterion_rec, optimizer, device, num_epochs=10):
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
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")

# ==========================
# Main Training Code
# ==========================
if __name__ == "__main__":
    # 경로 설정
    video_features_dir = r"C:/Users/swu/Desktop/AudioFeatureGeneration/Audio-Feature-Generation/data/lstm_features"
    audio_features_dir = r"C:/Users/swu/Desktop/AudioFeatureGeneration/Audio-Feature-Generation/data/audio-features"
    label_file = r"C:/Users/swu/Desktop/AudioFeatureGeneration/lstm_labels.csv"
    split_data_dir = r"C:/Users/swu/Desktop/AudioFeatureGeneration/Audio-Feature-Generation/data/splited-data"

    # 데이터셋 및 데이터로더
    train_dataset = MultiTaskDataset(
        split_data=os.path.join(split_data_dir, "train_data.npy"),
        video_features_dir=video_features_dir,
        audio_features_dir=audio_features_dir,
        label_file=label_file
    )
    val_dataset = MultiTaskDataset(
        split_data=os.path.join(split_data_dir, "validation_data.npy"),
        video_features_dir=video_features_dir,
        audio_features_dir=audio_features_dir,
        label_file=label_file
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # 모델, 손실 함수, 최적화 기법
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskModel(num_classes=10)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_rec = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    print("Starting training...")
    train(model, train_loader, val_loader, criterion_cls, criterion_rec, optimizer, device, num_epochs=10)
    # 모델 저장
    torch.save(model.state_dict(), "final_model.pth")
    print("Training completed and model saved as 'final_model.pth'!")

