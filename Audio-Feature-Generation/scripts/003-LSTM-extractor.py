#LSTM에 concat feature를 넣어 x를  추출

'''
import os
import numpy as np
import torch
import torch.nn as nn

# GPU 또는 CPU 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# LSTM 모델 정의
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

# Feature 저장 함수
def extract_lstm_features(feature_dir, output_feature_path, model, batch_size=1):
    action_classes = os.listdir(feature_dir)
    
    for action_class in action_classes:
        class_feature_path = os.path.join(feature_dir, action_class)
        output_class_path = os.path.join(output_feature_path, action_class)
        os.makedirs(output_class_path, exist_ok=True)

        # 비디오별 Feature 추출
        video_files = [f for f in os.listdir(class_feature_path) if f.endswith('.npy')]
        for video_file in video_files:
            video_feature_path = os.path.join(class_feature_path, video_file)
            output_file_path = os.path.join(output_class_path, video_file)

            try:
                # Feature 로드 및 LSTM 입력
                feature = np.load(video_feature_path)  # [18, 4096]
                feature_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device)  # [1, 18, 4096]

                # LSTM 모델 통과
                with torch.no_grad():
                    lstm_feature = model(feature_tensor).cpu().numpy().squeeze()  # [256]

                # LSTM Feature 저장
                np.save(output_file_path, lstm_feature)
                print(f"Saved LSTM feature for video: {video_file}, Path: {output_file_path}")

            except Exception as e:
                print(f"Error processing {video_file}: {e}")

if __name__ == "__main__":
    # Feature 디렉토리 및 저장 경로 설정
    feature_dir = r"C:\Users\swu\Desktop\AudioFeatureGeneration\Audio-Feature-Generation\data\features"
    output_feature_path = r"C:\Users\swu\Desktop\AudioFeatureGeneration\Audio-Feature-Generation\data\lstm_features"

    # LSTM 모델 초기화
    input_size = 4096
    hidden_size = 512
    num_layers = 2
    output_size = 256
    model = LSTMExtractor(input_size, hidden_size, num_layers, output_size).to(device)
    model.eval()

    # LSTM Feature 추출
    extract_lstm_features(feature_dir, output_feature_path, model)
'''

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# LSTM Feature 저장 경로
feature_dir = "C:/Users/swu/Desktop/AudioFeatureGeneration/Audio-Feature-Generation/data/lstm_features"

# Feature와 레이블을 저장할 리스트 초기화
features = []
labels = []
video_ids = []

# 액션 클래스 목록 가져오기
action_classes = os.listdir(feature_dir)

# 각 클래스별로 Feature와 레이블 로드
for action_class in action_classes:
    class_dir = os.path.join(feature_dir, action_class)
    feature_files = [f for f in os.listdir(class_dir) if f.endswith('.npy')]

    for feature_file in feature_files:
        feature_path = os.path.join(class_dir, feature_file)
        # Feature 로드
        feature = np.load(feature_path)  # Shape: [256]
        features.append(feature)
        # 레이블 및 비디오 ID 저장
        labels.append(action_class)
        video_ids.append(feature_file.replace('.npy', ''))

# Feature와 레이블을 배열로 변환
features = np.array(features)  # Shape: [num_samples, 256]
labels = np.array(labels)      # Shape: [num_samples]

# 레이블 인코딩 (문자열 -> 숫자)
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# 클래스 이름과 인코딩된 값 매핑 확인
class_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Class Mapping:", class_mapping)

# CSV 파일로 저장
output_csv_path = "C:/Users/swu/Desktop/AudioFeatureGeneration/lstm_labels.csv"
df = pd.DataFrame({
    'video_id': video_ids,
    'class_label': labels,
    'encoded_label': encoded_labels
})
df.to_csv(output_csv_path, index=False)
print(f"CSV file saved to {output_csv_path}")

"""
{'autographing': 0, 'baking': 1, 'balancing': 2, 'barbecuing': 3, 'barking': 4, 
 'bending': 5, 'bicycling': 6, 'biting': 7, 'blocking': 8, 'blowing': 9}
"""