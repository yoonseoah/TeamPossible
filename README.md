# Missing Modality Generation via Video-Audio Semantic Mapping in Federated Learning for Action Recognition

### Research Keywords
**Multimodal, Video Action Recognition, Semantic Mapping, Federated Learning**

### Team Possible
이화여자대학교 캡스톤디자인과창업프로젝트  
팀 가능한  
김지은, 윤서아, 장은성  

### Problem Definition 
audio modality의 누락은 연합학습 성능 저하를 유발하며, 이는 행동 인식의 정확도를 감소시키는 원인입니다. 또한, 비디오 처리는 communication overhead를 크게 증가시키며, 이는 시간과 비용 측면에서 매우 비효율적입니다. 더불어, 데이터의 privacy 보호가 이루어지지 않을 경우, 모델 학습에 사용할 수 있는 데이터의 양이 크게 줄어듭니다. 선행 연구들에서는 Semantic Mapping을 위해 주로 BERT를 사용하고 있습니다. 하지만 BERT는 텍스트 기반 모델로 설계되었기에, video(image)에서 추출된 정보와의 상호작용에는 한계가 존재합니다. 이러한 문제들은 모두 모델 성능의 저하, 행동 인식의 정확도 감소로 이어집니다. 

### Solution
해당 레포지토리는 아래의 기술들을 사용해 구현한 LSTM 기반의 오디오 피처 생성기 코드를 포함하고 있습니다.

OpenCV (https://github.com/opencv/opencv)

NumPy (https://github.com/numpy/numpy)

PyTorch(VAE) (https://github.com/pytorch/examples/tree/main/vae)

Hugging Face (https://github.com/huggingface/transformers)

GPT-4 Vision API (https://platform.openai.com/)

PySyft (https://github.com/OpenMined/PySyft)

### Project Structure

```bash
AudioFeatureGeneration/
├── Audio-Feature-Generation/
│   ├── data/                         # 데이터 관련 디렉토리
│   │   ├── raw/                      # 원본 데이터 (비디오 파일)
│   │   ├── features/                 # Spatial & Temporal Feature 저장
│   │   ├── processed/                # 처리된 중간 결과 (RGB, Optical Flow 이미지)
│   │   ├── audio-features/           # 오디오 Feature 저장
│   │   ├── audio-processed/          # 오디오 전처리 파일 저장
│   │   ├── lstm_features/            # LSTM을 통해 생성된 비디오 Feature 저장 (x 값)
│   │   └── splited-data/             # Train/Val/Test 데이터셋 저장
│   │       ├── train_data.npy
│   │       ├── validation_data.npy
│   │       └── test_data.npy
│   │
│   ├── models/                       # 모델 가중치 및 체크포인트 파일
│   │   └── initial_model.pth         # 초기 학습된 모델 가중치
│   │
│   ├── results/                      # 테스트 및 분석 결과 저장
│   │   ├── classification_results.csv
│   │   └── reconstructed-audio/      # 재구성된 오디오 Feature 저장 (.npy 파일)
│   │
│   ├── scripts/                      # 프로젝트 주요 스크립트
│   │   ├── 001-preprocess.py         # RGB & Optical Flow 전처리 코드
│   │   ├── 002-feature-extraction.py # 영상 Feature 추출 코드
│   │   ├── 003-LSTM-extractor.py     # LSTM에 넣어 비디오 Feature(x) 추출
│   │   ├── 004-audio-preprocess.py   # 비디오에서 오디오 전처리 및 추출
│   │   ├── 005-audio-feature.py      # 오디오 Feature 추출 코드 (AENet 사용)
│   │   ├── 006-train.py              # 학습 코드
│   │   ├── 007-test.py               # 테스트 코드
│   │   └── 008-test-analysis.py      # 테스트 결과 분석 코드
│   │
│   ├── README.md                     # 프로젝트 설명 파일
│   └── requirements.txt              # 프로젝트 의존성 패키지 목록
│
├── myenv/                            # 가상 환경 디렉토리 (Git에 제외됨)
│   ├── Include/
│   ├── Lib/
│   ├── Scripts/
│   ├── share/
│   └── pyvenv.cfg
│
├── lstm_labels.csv                   # 비디오 파일 라벨 정보
├── no_audio_videos.csv               # 오디오 없는 비디오 리스트
└── .gitignore                        # 업로드 제외 항목 설정 파일
