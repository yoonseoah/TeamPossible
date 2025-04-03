


'''
from transformers import ASTModel, ASTFeatureExtractor

model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
feature_extractor = ASTFeatureExtractor.from_pretrained(model_name)
model = ASTModel.from_pretrained(model_name)

import torchaudio

# 오디오 파일 로드
audio_path = r"D:\Audio\training\adult+female+singing\18frames\2cEKxGB6-YM_35.wav"
waveform, sample_rate = torchaudio.load(audio_path)

# 입력 데이터 전처리
inputs = feature_extractor(waveform, sampling_rate=sample_rate, return_tensors="pt")

import torch

# 모델을 평가 모드로 설정
model.eval()

# 예측 수행
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# 가장 높은 확률의 클래스 예측
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class index:", predicted_class_idx)
'''