import os
import numpy as np
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader, Dataset

# 백엔드 설정
torchaudio.set_audio_backend("soundfile")
print("Using audio backend:", torchaudio.get_audio_backend())

# AENet 모델 클래스 정의
class AENet(nn.Module):
    def __init__(self, output_size=1024):
        super(AENet, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(output_size)  # 최종 1024 차원 피처
        )

    def forward(self, x):
        return self.cnn(x)

# Dataset 클래스 (손상된 파일 처리)
class AudioDataset(Dataset):
    def __init__(self, audio_dir, max_length=48000):
        self.audio_files = []
        self.max_length = max_length
        for root, _, files in os.walk(audio_dir):
            for file in files:
                if file.endswith('.wav'):
                    self.audio_files.append(os.path.join(root, file))

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        try:
            waveform, sample_rate = torchaudio.load(audio_path)

            # NaN 체크 및 무효 데이터 확인
            if torch.isnan(waveform).any() or waveform.abs().sum() == 0:
                print(f"Skipping invalid audio file: {audio_path}")
                return None, None

            # 패딩 처리
            if waveform.shape[1] > self.max_length:
                waveform = waveform[:, :self.max_length]
            else:
                padding = self.max_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))

            return waveform, audio_path
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return None, None

# collate_fn: None 값을 필터링
def collate_fn(batch):
    batch = [item for item in batch if item[0] is not None]
    if not batch:
        return torch.Tensor([]), []
    waveforms, audio_paths = zip(*batch)
    return torch.stack(waveforms), audio_paths

# 오디오 피처 추출 함수
def extract_audio_features(audio_dir, output_dir, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 모델 초기화
    model = AENet(output_size=1024).to(device)
    model.eval()

    # Dataset 및 DataLoader 설정
    dataset = AudioDataset(audio_dir, max_length=48000)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)

    os.makedirs(output_dir, exist_ok=True)

    # 피처 추출 시작
    with torch.no_grad():
        for waveforms, audio_paths in dataloader:
            if len(waveforms) == 0:
                continue  # 모든 배치가 None일 경우 스킵
            waveforms = waveforms.to(device)  # [batch_size, 1, max_length]

            # 모델로 피처 추출
            features = model(waveforms)  # [batch_size, 1024]

            # 피처 저장
            for i, audio_path in enumerate(audio_paths):
                feature = features[i].cpu().numpy()
                class_name = os.path.basename(os.path.dirname(audio_path))
                output_class_dir = os.path.join(output_dir, class_name)
                os.makedirs(output_class_dir, exist_ok=True)

                file_name = os.path.basename(audio_path).replace('.wav', '.npy')
                np.save(os.path.join(output_class_dir, file_name), feature)
                print(f"Saved feature: {os.path.join(output_class_dir, file_name)}")

# 실행 코드
if __name__ == "__main__":
    audio_input_dir = r"C:/Users/swu/Desktop/AudioFeatureGeneration/Audio-Feature-Generation/data/audio-processed"
    audio_output_dir = r"C:/Users/swu/Desktop/AudioFeatureGeneration/Audio-Feature-Generation/data/audio-features"

    print("Starting audio feature extraction...")
    extract_audio_features(audio_input_dir, audio_output_dir, batch_size=32)
    print("Audio feature extraction completed!")
