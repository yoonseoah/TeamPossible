import os
import numpy as np
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from PIL import Image

# GPU 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ResNet50 모델 로드 및 GPU로 이동
resnet50_model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
resnet50_model.fc = torch.nn.Identity()  # Fully Connected Layer 제거
resnet50_model.eval()

# 데이터 전처리 변환 설정
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(base_rgb_path, base_flow_path, output_path, target_count=18):
    """
    RGB 및 Optical Flow 데이터를 ResNet50으로 Feature Extraction 후 시퀀스 형태로 저장
    """
    action_classes = os.listdir(base_rgb_path)
    for action_class in action_classes:
        rgb_class_path = os.path.join(base_rgb_path, action_class)
        flow_class_path = os.path.join(base_flow_path, action_class)
        output_class_path = os.path.join(output_path, action_class)

        os.makedirs(output_class_path, exist_ok=True)

        if os.path.isdir(rgb_class_path):
            video_folders = os.listdir(rgb_class_path)
            for video_folder in video_folders:
                rgb_video_path = os.path.join(rgb_class_path, video_folder)
                flow_video_path = os.path.join(flow_class_path, video_folder)
                output_file_path = os.path.join(output_class_path, f"{video_folder}.npy")

                try:
                    # RGB 및 Optical Flow 파일 로드
                    rgb_frame_files = sorted([os.path.join(rgb_video_path, f) for f in os.listdir(rgb_video_path) if f.endswith(('.jpg', '.png'))])
                    flow_frame_files = sorted([os.path.join(flow_video_path, f) for f in os.listdir(flow_video_path) if f.endswith(('.jpg', '.png'))])

                    if not rgb_frame_files or not flow_frame_files:
                        print(f"Skipping {video_folder}: Missing frames in one of the directories")
                        continue

                    # 프레임 읽기 및 변환
                    rgb_frames = torch.stack([transform(Image.open(f).convert("RGB")) for f in rgb_frame_files]).to(device)
                    flow_frames = torch.stack([transform(Image.open(f).convert("RGB")) for f in flow_frame_files]).to(device)

                    # ResNet50로 Feature Extraction
                    with torch.no_grad():
                        rgb_features = resnet50_model(rgb_frames).cpu().numpy()  # Shape: [18, 2048]
                        flow_features = resnet50_model(flow_frames).cpu().numpy()  # Shape: [18, 2048]

                    # 시퀀스 형태로 결합
                    concat_features = np.concatenate((rgb_features, flow_features), axis=1)  # Shape: [18, 4096]
                    np.save(output_file_path, concat_features)
                    print(f"Saved features for video: {video_folder}, Path: {output_file_path}")

                except Exception as e:
                    print(f"Error processing {video_folder}: {e}")

if __name__ == "__main__":
    # 경로 설정
    rgb_dir = r"D:/vid-processed/rgb"
    flow_dir = r"D:/vid-processed/flow"
    output_dir = r"D:/vid-concatfeatures"

    # Feature Extraction 실행
    extract_features(rgb_dir, flow_dir, output_dir)
