SER System on Jetson Nano
===========================================
> 2025-1 IAP


## 프로젝트 개요
본 프로젝트에서는 Jetson Nano 보드 위에서 동작하는 실시간 음성 감정 인식(Speech Emotion Recognition; SER) 모델을 바탕으로 부정적인 감정을 포착하고 이를 완화하기 위한 솔루션을 제공합니다.

<br>

## 상세 설명
- **목표**: Jetson Nano 상에서 동작하는 실시간 SER 시스템 구현 및 감정 완화 솔루션 제공
- **감정 종류**: 전체 7개 (행복, 분노, 슬픔, 중립, 공포, 혐오, 놀람) / 상위 4개 (행복, 슬픔, 분노, 중립)
- **입력 데이터 특징**: Mel-spectrogram + Pitch(broadcast) + RMS(broadcast)로 이루어진 3채널 128*128 이미지
- **모델 구조**: ResNet34(Pretrained) + Custom Head
- **평가 지표**: UAR(Unweighted Average Recall), Macro F1
- **TensorRT 엔진**: FP32 / FP16

<br>

## 전체 구조 개요

<img width="800" height="241" alt="image" src="https://github.com/user-attachments/assets/889cc20f-1366-47a3-bae5-2e8be6c16e3c" />

<br><br>

## 입력 데이터 형식 및 모델 구조

### 1. 입력 데이터 형식

<img width="1000" height="326" alt="image" src="https://github.com/user-attachments/assets/2d840310-2abb-4974-aac7-a2fdbf957df7" />


| Category        | Parameters                                                                 |
|-----------------|----------------------------------------------------------------------------|
| **Audio**       | Sample rate = 16,000 Hz<br>Channel = mono                                  |
| **Framing**     | Frame length = 1024 samples (64 ms)<br>Hop length = 320 samples (20 ms)<br>Center = False<br>Segment = 128 frames (2.56 s) |
| **Mel-spectrogram** (torchaudio) | n_mels = 128<br>f_min = 50 Hz<br>f_max = 8000 Hz<br>power = 2.0 |
| **Pitch** (librosa.yin) | f_min = 50 Hz<br>f_max = 8000 Hz<br>Broadcast: 1D → 2D              |
| **RMS** (librosa.feature.rms) | Broadcast: 1D → 2D                                           |
| **최종 형식**  | 3채널 스택 (Mel/Pitch/RMS)<br>채널별 [0, 1] scaling 적용                  |

   
<br>

### 2. 모델 구조

**Backbone**: ResNet34 (Pretrained)

**Head**: Custom Head

<img width="600" height="346" alt="image" src="https://github.com/user-attachments/assets/b86772fa-a3f2-41ea-9746-66f7437ade75" />


<br><br>

## 결과 분석

### 1. 결과

| 지표          | 전체 7개 감정 | 주요 4개 감정 (happiness, angry, neutral, sadness) |
|---------------|---------------|---------------------------------------------------|
| **UAR (%)**   | 52.15         | 57.10                                             |
| **Macro-F1 (%)** | 53.56      | 60.05                                             |

<br><br>

### 2. TensorRT 변환에 따른 연산 시간 및 리소스 사용량 분석

- 전처리 및 추론 시간 변화

| Framework        | Preprocessing Time (ms) | Inference Time (ms) | PyTorch 대비 시간 감소율 |
|------------------|--------------------------|----------------------|---------------------------|
| **PyTorch**      | 37.141                  | 196.96              | -                         |
| **TensorRT (FP32)** | 55.989               | 29.98               | 63.3%                     |
| **TensorRT (FP16)** | 54.103               | 26.06               | 65.8%                     |

  
- 리소스 사용량 변화

| Framework        | RAM 사용량 (MB) | CPU 사용량 (%) |
|------------------|-----------------|----------------|
| **PyTorch**      | 2381            | 15.52          |
| **TensorRT (FP32)** | 3120         | 12.37          |
| **TensorRT (FP16)** | 2744         | 11.49          |


<br><br>

## Docker Image 사용 방법

이미지 파일 링크(구글 드라이브): https://drive.google.com/file/d/1FsbHj9kZLRWaHiiHyT2MLtQ0l1Q8g5F-/view?usp=drive_link 

해당 프로젝트는 실행을 위하여 USB 마이크 연결을 필요로 합니다.

### iap_team1 fianl project 코드 실행법

1. 먼저, 제공된 링크를 통해 "iap_team1_v3.tar" 파일을 다운로드해 주십시오. 해당 파일은 docker image 입니다.(5.1GB)

2. "iap_team1_v3.tar" 파일의 다운로드가 완료되었다면, 작업 영역으로 해당 파일을 옮기고 아래 코드를 순서대로 터미널에 입력하여 주십시오.

```
	a. docker load -i iap_team1_v3.tar

	b. docker images | grep iap_team1

	c. docker run --rm -it --device /dev/snd:/dev/snd --group-add audio -v "$(pwd)/test":/app/output iap_team1:v3

	d. cd /app

	e. python audio_test.py

	f. python main_resnet34.py (basic) // python main_resnet34_trt.py (fp32, tensorRT) // python main_resnet34_trt.py --fp16 (fp16, tensorRT)
```

3. 위 명령어에 대한 설명은 다음과 같습니다:

```
	a. "iap_team1_v3.tar"에 저장된 이미지 파일을 docker로 로드합니다.

	b. docker images가 정상적으로 로드되었는지 확인합니다. 이때, iap_team1:v3 이미지가 존재할 경우 정상적으로 로드 된 것입니다.

	c. 해당 이미지를 바탕으로 컨테이너를 호출합니다. 이때, USB 마이크 사용을 위하여 컨테이너에 디바이스를 부착하고, 출력 파일 확인을 위해 컨테이너 내부 output 폴더를 현재 작업영역/test 폴더에 연결시킵니다.

	d. 소스코드가 저장되어 있는 컨테이너 내부 폴더로 이동합니다.

	e. (중요!) 마이크가 몇번 device로 연결되어 있는지 확인합니다. 만약, USB 마이크가 11번 index가 아닐 경우, 'main_resnet34.py' 및 'main_resnet34_trt.py'의 'audio_capture' 함수에서 stream = p.open(format=pyaudio.paInt16, channels=1, rate=self.capture_rate, input=True, input_device_index=11, frames_per_buffer=self.chunk_size) 객체를 찾고, input_device_index 인자를 앞서 확인한 USB 마이크의 index로 변경하여 주십시오.

	f. 'main_resnet34.py'는 tensorRT가 적용되지 않은 코드, 'main_resnet34_trt.py'는 tensorRT가 적용된 최적화 코드이며 기본 엔진은 fp32입니다. '--fp16' parser를 통해 fp16 엔진을 선택할 수 있습니다.
```

4. 주의사항: 컨테이너 실행 후 처음으로 메인 코드를 실행할 시, 스레드 활성화 및 preprocessing 단계에서 약 1분 정도가 소모됩니다.

