# Missing Modality Generation via Video-Audio Semantic Mapping in Federated Learning for Action Recognition

### Research Keywords
**Multimodal, Video Action Recognition, Semantic Mapping, Federated Learning**

### Team Possible
이화여자대학교 캡스톤디자인과창업프로젝트  
팀 가능한  
김지은, 윤서아, 장은성  

### Problem Definition 

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
