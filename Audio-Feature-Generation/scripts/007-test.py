import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# ==========================
# Dataset 정의 (테스트용)
# ==========================
class TestDataset(Dataset):
    def __init__(self, split_data, video_features_dir):
        self.data = np.load(split_data, allow_pickle=True)
        self.video_features_dir = video_features_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_path, _ = self.data[idx]  # audio_path는 무시

        # Video Features
        video_features = np.load(video_path)
        return torch.tensor(video_features, dtype=torch.float32), os.path.basename(video_path)


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
# Test Function
# ==========================
def test(model, test_loader, device, output_dir, results_file):
    model.to(device)
    model.eval()

    predictions = []
    reconstructed_audio_paths = []

    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for video_features, video_file in test_loader:
            video_features = video_features.to(device)

            # 모델 예측
            cls_output, rec_output = model(video_features)

            # Classification 결과
            preds = torch.argmax(cls_output, dim=1).cpu().numpy()
            predictions.extend(list(zip(video_file, preds)))

            # Reconstruction 결과 저장
            for i, rec_feat in enumerate(rec_output):
                output_file = os.path.join(output_dir, f"reconstructed_{video_file[i]}")
                np.save(output_file, rec_feat.cpu().numpy())
                reconstructed_audio_paths.append(output_file)

    # Classification 결과 저장
    results_df = pd.DataFrame(predictions, columns=["video_file", "predicted_class"])
    results_df.to_csv(results_file, index=False)
    print(f"Classification results saved to: {results_file}")

    return predictions, reconstructed_audio_paths

# ==========================
# Main Testing Code
# ==========================
if __name__ == "__main__":
    # 경로 설정
    video_features_dir = r"C:/Users/swu/Desktop/AudioFeatureGeneration/Audio-Feature-Generation/data/lstm_features"
    test_split_file = r"C:/Users/swu/Desktop/AudioFeatureGeneration/Audio-Feature-Generation/data/splited-data/test_data.npy"
    output_dir = r"C:/Users/swu/Desktop/AudioFeatureGeneration/Audio-Feature-Generation/data/reconstructed-audio"
    results_file = r"C:/Users/swu/Desktop/AudioFeatureGeneration/Audio-Feature-Generation/data/classification_results.csv"
    model_path = "final_model.pth"  # 학습된 모델 경로

    # 데이터셋 및 데이터로더
    test_dataset = TestDataset(split_data=test_split_file, video_features_dir=video_features_dir)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 장치 설정 및 모델 로드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskModel(num_classes=10)
    model.load_state_dict(torch.load(model_path))
    
    print("Starting testing...")
    predictions, reconstructed_audio_paths = test(model, test_loader, device, output_dir, results_file)
    print("Testing completed!")

    # 결과 출력
    print("\nReconstructed Audio Files:")
    for path in reconstructed_audio_paths:
        print(f"Saved: {path}")
        
    print("Classification Results:")
    for video_file, pred in predictions:
        print(f"Video: {video_file}, Predicted Class: {pred}")
