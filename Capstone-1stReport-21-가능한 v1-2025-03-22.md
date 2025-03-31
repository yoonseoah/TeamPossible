<!-- Template for PROJECT REPORT of CapstoneDesign 2025-2H, initially written by khyoo -->
<!-- 본 파일은 2025년도 컴공 졸업프로젝트의 <1차보고서> 작성을 위한 기본 양식입니다. -->
<!-- 아래에 "*"..."*" 표시는 italic체로 출력하기 위해서 사용한 것입니다. -->
<!-- "내용"에 해당하는 부분을 지우고, 여러분 과제의 내용을 작성해 주세요. -->

# Team-Info
| (1) 과제명 | Generating Missing Auditory Modality via Semantic Mapping for Video Action Recognition
|:---  |---  |
| (2) 팀 번호 / 팀 이름 | 21-가능한 |
| (3) 팀 구성원 | **김지은** (2140010): 리더, *기본 데이터 전처리, caption-based dictionary 활용 아키텍처 설계 및 구현, transformer 활용 오디오 특징 추출* <br> **윤서아** (2168019): 팀원, *TSN 기반 feature extraction, BERT 활용 semantic mapping dictionary 설계* <br> **장은성** (2271052) : 팀원, *transformer 기반 프레임워크 중 video 전처리, feature 추출 부분 설계 및 구현*            |
| (4) 팀 지도교수 | 이형준 교수님 |
| (5) 과제 분류 | 연구 과제 |
| (6) 과제 키워드 | Generative AI, Video Action Recognition, Multimodal, Semantic Mapping  |
| (7) 과제 내용 요약 | 본 연구는 비디오 시퀀스를 기반으로 오디오 피처를 재구성하여, 결손된 오디오 데이터를 보완하고 이를 통해 모델의 정확도를 향상시키는 것을 목표로 한다. 이를 위해 비디오와 오디오 간의 의미적 정렬을 수행하는 개념인 Semantic Mapping을 차용하였다. 두 가지 아키텍처를 제안하는데, 공통적으로 비디오로부터 오디오 피처를 생성하지만, 검증 및 활용 방식에서 차이를 보인다. 첫 번째 아키텍처는 CNN과 LSTM을 활용하여 오디오 피처를 생성한 후, Semantic Dictionary를 이용해 의미적 유효성을 검증한다. 이 검증을 통과한 피처만 학습에 활용하는 방식이다. 두 번째 아키텍처는 Transformer를 활용해 대표 프레임을 추출하고, 캡션으로 변환한 뒤 이를 키로 사용하는 Semantic Dictionary을 만든다. 이 사전을 참조하여 오디오 피쳐를 생성해낸다. 두 아키텍처를 비교 분석함으로써, 의미 기반 피처 정렬 방식이 오디오 생성 및 영상 이해 성능에 미치는 영향을 규명하고자 한다. |

<br>

# Project-Summary
| 항목 | 내용 |
|:---  |---  |
| (1) 문제 정의 | 비디오 기반 행동 인식에서는 멀티모달(시각+청각) 정보를 활용한 학습이 정밀도를 높이는 데 필수적이다. 하지만 현실에서는 오디오가 손실되거나, 특정 모달리티만 라벨링된 데이터셋이 많아 멀티모달 학습의 효과가 제한된다. 특히 오디오 모달리티는 파일 손상, 녹음 품질 저하, 무관한 배경음 등으로 인해 활용이 어려운 경우가 많다. Target Customer는 멀티모달 행동 인식 시스템을 개발하려는 연구자, 기업, 산업계로, 오디오 결손 상황에서 학습 정확도를 유지할 수 있는 보완적 해결책이 필요하다. |
| (2) 기존연구와의 비교 | 기존 연구는 문제 해결을 위해 비디오의 시공간적 특징을 매핑해 오디오 피처를 재구성하는데, 이는 비디오 시퀀스와 의미적 정합성을 고려하지 못한다는 한계를 가진다. 또한, 비디오-오디오 의미적 매핑 사전을 구성해 비디오와 무관한 데이터를 드롭아웃하는 기법을 적용한 기존 연구는 여전히 오디오가 결손된 상황에 대한 극복이 어렵다. 본 과제는 이들 연구와 달리, 비디오-오디오 간의 **의미적 정합성(semantic consistency)**을 고려하여 오디오 피처를 생성하고, 정합성 검증을 통해 학습에 사용함으로써 더 높은 신뢰성과 정확도를 보장한다는 점에서 차별화된다. |
| (3) 제안 내용 | 본 프로젝트는 결손된 오디오 피처를 보완하고 행동 인식의 정확도를 향상시키기 위한 두 가지 아키텍처를 제안한다.<br> **① Semantic Validation Architecture**: Transformer 기반으로 비디오에서 피처를 추출한 후, 생성된 오디오 피처에 대해 액션 라벨 예측을 수행하고, 시맨틱 사전을 이용하여 정합성을 검증한 뒤, 정합성이 확인된 피처만 학습에 사용.<br> **② Caption-based Attention Mapping Architecture**: Transformer 기반으로 대표 RGB 프레임에서 자연어 캡션을 생성하고, 이를 시맨틱 사전의 키로 활용하여 비디오와 오디오 라벨 간의 의미 매핑을 attention 기반으로 정교화하여 정합성을 강화. |
| (4) 기대효과 및 의의 | - 오디오 결손 상황에서도 높은 정확도를 유지하는 행동 인식 모델 구현 가능<br> - 의미 기반 피처 생성 및 검증을 통해 멀티모달 학습의 신뢰성 제고<br> - 행동 인식 분야를 넘어 결손 오디오 복원, 의료 영상 등 다양한 멀티모달 응용 분야로 확장 가능<br> - AGI(Artificial General Intelligence)를 위한 인간 유사 인지 능력 구현에 기여 |
| (5) 주요 기능 리스트 | - **비디오 기반 오디오 피처 생성 모듈**: Transformer를 이용해 시공간 정보를 인코딩하고 오디오 피처 생성<br> - **시맨틱 사전 구축 및 활용 기능**: 비디오-오디오 간 의미적 일치 여부를 평가할 수 있도록 액션 라벨, 캡션 등을 기반으로 의미 사전 구성<br> - **오디오 피처 정합성 검증 기능**: 생성된 오디오 피처의 의미적 유효성을 판별하여, 학습에 사용 가능한 피처를 선별<br> - **자연어 캡셔닝 기반 매핑 강화 기능**: 대표 프레임을 텍스트로 표현하여 의미 정합성을 강화하고 매핑 정확도를 향상<br> - **모델 비교 평가 시스템**: 두 가지 제안 아키텍처의 학습 정확도, 정합성 판단 성능 등을 비교 분석하는 실험 기능 포함 |


<br>
 
# Project-Design & Implementation
| 항목 | 내용 |
|:---  |---  |
| (1) 요구사항 정의 | **[기능별 상세 요구사항]** <br> - 비디오의 시각적 정보와 시퀀스 정보로부터 오디오 피처를 생성한다. <br> - 생성된 오디오 피처의 의미적 정합성을 검증한다 <br> - 또는, 피처 자체에 정합성 부여를 위해 딕셔너리를 활용한다 <br> - 의미 정합성이 부여된 오디오 피처만 비디오 피처와 fuse하여 모델 학습에 사용한다. <br> - 두 아키텍처의 성능을 비교/분석해 최적화한다. <br> <br> **[설계 모델]** <br> - video feature extractor, audio feature extractor, audio feature generator, audio-video semantic dictionary, trainer, evaluator <br> - 위와 같이 모듈을 크게 6개로 나누어 설계했다. <br> <br> **[데이터 셋]** <br> - 비디오: Moments in Time(MiT) <br> - 오디오: AudioSet |
| (2) 전체 시스템 구성 | **[데이터 전처리]** <br> - 비디오는 먼저, 6fps의 RGB frame images로 전처리 하고, <br> - 추출된 프레임을 기반으로 Optical Flow images를 추출했다. <br> - 오디오는 AST의 feature extractor를 사용해 spectrogram으로 전처리했다. <br> <br> **[첫번째 아키텍처]** <br> ~~이미지 첨부하기~~ <br> - 이미지와 오디오 피처는 transformer를 사용해 임베딩으로 처리한다. <br> - LSTM을 기반으로 오디오 피처를 생성한다. <br> - 생성한 오디오 피처를 사전 학습된 AST classifier를 사용해 multi label prediction 한다. <br> - 예측된 라벨과 Semantic Dictionary를 비교해 의미 반영 정도를 계산한다. <br> - Threshold N을 지정해 N을 넘은 유사도를 가진 피처만을 '유의미한 피처'로 필터링해 학습에 활용한다. (이 때, N은 최적화 되어야 한다.) <br> - 이 때, Semantic Dictionary는 MiT의 action 라벨과 AudioSet의 오디오 라벨 간의 의미적 관계를 BERT로 매핑한 사전이다. <br> - 필터링된 오디오 피처와 비디오 피처를 fuse하여 최종 classification에 사용한다. <br> <br> **[두번째 아키텍처]** <br> ~~이미지 첨부하기~~ <br> - 이미지와 오디오 피처는 transformer를 사용해 임베딩으로 처리한다. <br> - 프레임 간 object의 이동량을 기준으로 영상 당 n개의 대표 이미지를 계산한다. <br> - n개의 대표 이미지에 대해 CLIP 모델을 사용해 캡션을 생성한다. <br> - N개의 캡션과 유사한 키 캡션을 Semantic Dictionary에서 찾는다. <br> - 키 캡션에 매핑된 오디오 라벨 임베딩을 참조한다. <br> - 이 때, Semantic Dictionary는 학습 비디오의 대표 이미지 캡션과 사전 학습된 AST classifier로 추출한 multi 오디오 라벨 간의 의미적 관계를 BERT로 매핑한 사전이다. <br> - 랜덤 노이즈를 초기 input으로 하는 GAN 구조로 오디오 피처를 생성한다. - 이 때, Dynamic Time Warping을 이용해 비디오와 오디오 피처의 시퀀스를 정합하게 한다. <br> - 원본 비디오의 오디오 피처를 ground truth로 하여 생성 네트워크를 학습시킨다. <br> - 생성된 오디오 피처와 비디오 피처를 fuse하여 최종 classification에 사용한다. |
| (3) 주요엔진 및 기능 설계 | **[공통 모듈]** <br> **Feature Extractor**: 전처리된 RGB images, Optical Flow images, Audio Spectrogram을 오디오 비디오 각각 사전 학습된 Transforemr를 이용해 고차원 임베딩으로 처리 (비디오는 ViT, 오디오는 AST 사용) <br> - **Feature Fusion&Classification**: 두 아키텍처의 네트워크 마지막에 공통으로 사용될 모듈로, 추출한 비디오 피처와 생성한 오디오 피처를 fuse하고, fuse된 피처를 input으로 최종 action label classification을 진행 <br> <br> **[첫번째 아키텍처 모듈]** <br> - **Semantic Dictionary**: MiT의 action 라벨과 AudioSet의 오디오 라벨 간의 의미적 관계를 BERT 임베딩을 통해 사전으로 구축 <br> - **Audio Feature Generator**: 비디오 임베딩을 입력으로 하여 LSTM 네트워크를 통해 오디오 피처 시퀀스를 생성 <br> - **Multi Label Predictor**: 사전학습 AST를 기반으로 생성한 피처에 대해 오디오 라벨 예측 <br> - **Feature Filter**: 생성된 피처에서 예측한 multi label과 semantic dictionary를 비교, threshold를 기준으로 생성 피처를 필터링 <br> <br> **[두번째 아키텍처 모듈]** <br> - **Representative Frame Extractor & CLIP-based Caption Generator**: 영상의 RGB frames에 대해 이동량이 큰 프레임 n개를 대표 이미지로 선정하고, CLIP 모델에 인풋으로 사용해 캡션을 생성 <br> - **Semantic Dictionary**: 학습 비디오의 대표 이미지 캡션과 AST 기반으로 추출된 오디오 라벨 간 의미적 관계를 BERT 임베딩을 통해 사전으로 구축 <br> - **Dictionary Referencing**: 생성된 캡션과 의미적으로 유사한 키 캡션을 사전에서 탐색, 해당 키 캡션과 연결된 오디오 라벨 임베딩을 참조 <br> - **Audio Feature Generator**: 랜덤 노이즈 + 딕셔너리에서 참조된 오디오 임베딩을 조건으로 하는 GAN 구조, 비디오 피처 시퀀스와 오디오 피처 시퀀스를 Dynamic Time Warping으로 정렬, 원본 오디오 피처를 ground truth로 하여 loss로 학습 |
| (4) 주요 기능의 구현 | **① Feature Filter(첫번째 아키텍처 모듈)** <br> - LSTM을 통해 생성된 오디오 피처는 AST 기반 Multi-label Predictor를 통해 예측된 오디오 라벨을 출력함 <br> - 해당 라벨은 BERT 임베딩된 Semantic Dictionary와 비교되어 의미 유사도(cosine similarity)가 계산됨 <br> - 이 때, Semantic Dictionary는 MiT action 라벨과 AudioSet 오디오 라벨의 BERT 기반 매핑으로 구축되어 있음 <br> - 예측된 오디오 라벨과 해당 비디오의 action label 간 의미 유사도가 **Threshold N** 이상인 경우에만 해당 오디오 피처를 정합한 것으로 간주 <br> 필터링된 오디오 피처만을 비디오 피처와 fuse하여 최종 classification에 사용해 정확도 향상 및 정합성 보장 <br> - Threshold N은 실험적으로 최적화되며, precision-recall tradeoff를 조절하는 핵심 하이퍼파라미터
 <br> **② Audio Feature Generator(두번째 아키텍처 모듈)** <br> - 대표 프레임에서 생성한 캡션과 Semantic Dictionary를 통해 의미적으로 매핑된 오디오 라벨 임베딩을 도출 <br> - 이를 조건으로 사용하여 랜덤 노이즈를 입력으로 하는 **Conditional GAN 구조**를 설계 <br> - Generator는 조건에 따라 의미 정합한 오디오 피처 시퀀스를 생성하며, Discriminator는 이를 평가 <br> - Dynamic Time Warping기법을 통해 비디오 피처 시퀀스와 생성된 오디오 피처 시퀀스 간 temporal alignment를 수행 <br> - Loss 구성: (1) Adversarial Loss (GAN 학습용), (2) Reconstruction Loss (원본 오디오 피처와의 유사도 기반) <br> - 이 복합 구조를 통해 비디오와 정렬되고 의미적으로 정합한 오디오 피처 시퀀스 생성 가능 |
| (5) 기타 | **[실험 환경]**  |

<br>
