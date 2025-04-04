import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from einops import rearrange
import numpy as np

# -------------------------------------------------------
# 1. 기존 모듈 정의 (VideoPatchEmbed, VideoEmbed, VideoViTFeatureExtractor, VideoFeatureExtractor)
# -------------------------------------------------------

# VideoPatchEmbed: 비디오를 패치 임베딩으로 변환
class VideoPatchEmbed(nn.Module):
    def __init__(self, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.proj(x)  # (B*T, embed_dim, H_patch, W_patch)
        _, embed_dim, H_patch, W_patch = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B*T, N, embed_dim) with N = H_patch * W_patch
        return x, T, W_patch

# VideoEmbed: 패치 임베딩에 시간 임베딩을 추가하여 spatiotemporal embedding 생성
class VideoEmbed(nn.Module):
    def __init__(self, patch_embed, embed_dim=768, num_frames=18):
        super().__init__()
        self.patch_embed = patch_embed
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        self.drop = nn.Dropout(0.1)
        nn.init.trunc_normal_(self.time_embed, std=0.02)
    
    def forward(self, x):
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        x, T, W_patch = self.patch_embed(x)  # (B*T, N, embed_dim)
        N = x.shape[1]
        x = x.view(B, T, N, self.embed_dim)
        if T != self.time_embed.shape[1]:
            time_embed = self.time_embed.transpose(1, 2)
            new_time_embed = F.interpolate(time_embed, size=T, mode='linear', align_corners=False)
            new_time_embed = new_time_embed.transpose(1, 2)
        else:
            new_time_embed = self.time_embed
        x = x + new_time_embed.unsqueeze(2)
        x = self.drop(x)
        x = x.reshape(B, T * N, self.embed_dim)
        return x

# VideoViTFeatureExtractor: Hugging Face의 ViTModel을 활용하여 피처 추출
from transformers import ViTModel, ViTConfig

class VideoViTFeatureExtractor(nn.Module):
    def __init__(self, embed_dim=768, depth=12, num_heads=12, mlp_dim=3072, num_tokens=196, dropout=0.1):
        super().__init__()
        config = ViTConfig(
            hidden_size=embed_dim,
            num_hidden_layers=depth,
            num_attention_heads=num_heads,
            intermediate_size=mlp_dim,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            image_size=224,   # 더미 값
            patch_size=1,     # 더미 값
            num_channels=3,   # 더미 값
        )
        self.vit = ViTModel(config)
        # patch embedding을 bypass하기 위해 Identity로 변경
        self.vit.embeddings.patch_embeddings = nn.Identity()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        num_total_tokens = num_tokens + 1  # CLS 포함
        self.vit.embeddings.position_embeddings = nn.Parameter(torch.zeros(1, num_total_tokens, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.vit.embeddings.position_embeddings, std=0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens):
        # tokens: (B, num_tokens, embed_dim)
        B = tokens.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, tokens), dim=1)
        x = x + self.vit.embeddings.position_embeddings
        x = self.dropout(x)
        encoder_outputs = self.vit.encoder(x, return_dict=True)
        cls_output = self.vit.layernorm(encoder_outputs.last_hidden_state[:, 0])
        return cls_output

# VideoFeatureExtractor: 전체 파이프라인 구성 (RGB 또는 Optical Flow)
class VideoFeatureExtractor(nn.Module):
    def __init__(self, in_chans=3, patch_size=16, embed_dim=768, num_frames=18, vit_depth=12, vit_heads=12, vit_mlp_dim=3072):
        super().__init__()
        self.video_embed = VideoEmbed(
            VideoPatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim),
            embed_dim=embed_dim,
            num_frames=num_frames
        )
        num_patches_per_frame = (224 // patch_size) ** 2
        total_tokens = num_frames * num_patches_per_frame
        self.vit_extractor = VideoViTFeatureExtractor(
            embed_dim=embed_dim,
            depth=vit_depth,
            num_heads=vit_heads,
            mlp_dim=vit_mlp_dim,
            num_tokens=total_tokens,
            dropout=0.1
        )
    
    def forward(self, x):
        tokens = self.video_embed(x)
        features = self.vit_extractor(tokens)
        return features

# -------------------------------------------------------
# 2. 이미지 로딩 및 전처리 함수
# -------------------------------------------------------
def load_video_frames(folder, modality='rgb', num_frames=18, target_size=(224,224)):
    """
    folder: 비디오 프레임들이 저장된 폴더 경로  
    modality: 'rgb'이면 .jpg, 'flow'이면 .png 파일 읽음  
    """
    if modality == 'rgb':
        ext = '*.jpg'
    elif modality == 'flow':
        ext = '*.png'
    else:
        raise ValueError("알 수 없는 modality입니다.")
    files = sorted(glob.glob(os.path.join(folder, ext)))
    files = files[:num_frames]
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ])
    frames = []
    for f in files:
        if modality == 'rgb':
            img = Image.open(f).convert('RGB')
            img = transform(img)
        else:
            img = Image.open(f).convert('L')
            img = img.convert('RGB')  # Optical Flow는 실제 데이터에 맞게 수정
            img = transform(img)
            img = img[:2, :, :]  # 앞의 2채널 사용
        frames.append(img)
    video = torch.stack(frames, dim=1)  # (C, T, H, W)
    video = video.unsqueeze(0)  # (1, C, T, H, W)
    return video

# -------------------------------------------------------
# 3. 폴더 순회 및 여러 비디오에 대해 피처 추출
# -------------------------------------------------------
def process_videos(modality, root_dir, extractor, device, num_frames=18, target_size=(224,224)):

    results = {}  # 결과 저장 딕셔너리: key = (modality, 클래스, 비디오명)
    for class_name in os.listdir(root_dir):
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        frames_folder = os.path.join(class_dir, "18frames")
        if not os.path.isdir(frames_folder):
            print(f"'18frames' 폴더가 {class_dir}에 없습니다.")
            continue
        for video_name in os.listdir(frames_folder):
            video_dir = os.path.join(frames_folder, video_name)
            if not os.path.isdir(video_dir):
                continue
            video = load_video_frames(video_dir, modality=modality, num_frames=num_frames, target_size=target_size).to(device)
            with torch.no_grad():
                features = extractor(video)
            key = (modality, class_name, video_name)
            results[key] = features.cpu().numpy()
            print(f"처리 완료: {modality}/{class_name}/{video_name}, 피처 shape: {features.shape}")
    return results

# -------------------------------------------------------
# 4. 메인 실행: 모델 생성 후 데이터 전체에 적용
# -------------------------------------------------------
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 모달리티별 모델 생성
    rgb_extractor = VideoFeatureExtractor(in_chans=3).to(device)
    flow_extractor = VideoFeatureExtractor(in_chans=2).to(device)
    
    # 데이터 root 디렉토리 설정 (경로 끝에 '\'를 사용하지 않도록 주의)
    root_rgb = r"C:\\Users\\swu\\Desktop\\data\\RGB\\training"
    root_flow = r"C:\\Users\\swu\\Desktop\\data\\OpticalFlow\\training"
    
    # 각 모달리티별로 비디오 순회하며 피처 추출
    rgb_results = process_videos('rgb', root_rgb, rgb_extractor, device)
    flow_results = process_videos('flow', root_flow, flow_extractor, device)
    
    # 필요에 따라 추출된 피처를 파일로 저장 (예: numpy 파일)
    np.save("rgb_features.npy", rgb_results)
    np.save("flow_features.npy", flow_results)
