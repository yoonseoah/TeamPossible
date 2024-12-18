import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# ==========================
# Device 설정
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================
# LSTM 모델 정의
# ==========================
class LSTMExtractor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        """
        LSTM 기반 Feature 추출기
        :param input_size: 각 프레임의 Feature 크기 (예: 4096)
        :param hidden_size: LSTM 은닉 상태 크기
        :param num_layers: LSTM 레이어 수
        :param output_size: 최종 출력 Feature 크기
        """
        super(LSTMExtractor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)  # 최종 출력 크기를 조정하는 Fully Connected Layer

    def forward(self, x):
        """
        :param x: [batch_size, sequence_length, feature_size]
        :return: [batch_size, output_size]
        """
        _, (hidden, _) = self.lstm(x)  # hidden: [num_layers, batch_size, hidden_size]
        hidden = hidden[-1]  # 마지막 레이어의 hidden state 사용
        x = self.fc(hidden)  # Fully Connected Layer 통과
        return x

# ==========================
# Feature 추출 함수
# ==========================
def extract_lstm_features(feature_dir, output_dir, model, batch_size=1):
    """
    LSTM을 통해 비디오 Feature를 추출하고 저장합니다.
    :param feature_dir: 입력 Feature 디렉토리
    :param output_dir: 출력 Feature 디렉토리
    :param model: LSTM 모델
    :param batch_size: 처리 배치 크기 (현재 1)
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    action_classes = os.listdir(feature_dir)
    for action_class in tqdm(action_classes, desc="Processing Classes"):
        class_feature_path = os.path.join(feature_dir, action_class)
        output_class_path = os.path.join(output_dir, action_class)
        os.makedirs(output_class_path, exist_ok=True)

        video_files = [f for f in os.listdir(class_feature_path) if f.endswith('.npy')]
        for video_file in tqdm(video_files, desc=f"Processing {action_class}", leave=False):
            video_feature_path = os.path.join(class_feature_path, video_file)
            output_file_path = os.path.join(output_class_path, video_file)

            if not os.path.exists(video_feature_path):
                print(f"Feature file not found: {video_feature_path}")
                continue

            try:
                # Feature 로드 및 Tensor 변환
                feature = np.load(video_feature_path)  # [18, 4096]
                feature_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device)  # [1, 18, 4096]

                # LSTM 모델 통과
                with torch.no_grad():
                    lstm_feature = model(feature_tensor).cpu().numpy().squeeze()  # [256]

                # LSTM Feature 저장
                np.save(output_file_path, lstm_feature)
                print(f"Saved LSTM feature for video: {video_file}")

            except Exception as e:
                print(f"Error processing {video_file}: {e}")

# ==========================
# Main Execution
# ==========================
if __name__ == "__main__":
    # Feature 디렉토리 및 저장 경로 설정
    feature_dir = r"D:/vid-concatfeatures"
    output_feature_dir = r"D:/vid_lstmfeatures"

    # LSTM 모델 초기화
    input_size = 4096
    hidden_size = 512
    num_layers = 2
    output_size = 256
    model = LSTMExtractor(input_size, hidden_size, num_layers, output_size).to(device)

    # Feature 추출
    print("Starting LSTM feature extraction...")
    extract_lstm_features(feature_dir, output_feature_dir, model)
    print("LSTM feature extraction completed!")
