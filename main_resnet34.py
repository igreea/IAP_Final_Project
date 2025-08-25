# -*- coding: utf-8 -*-
import numpy as np
np.complex = complex  # for compatibility with librosa
import librosa
import torch
import torch.nn as nn
import sounddevice as sd
import threading
from queue import Queue, Empty, Full
import time
import pyaudio
import datetime
from googleapiclient.discovery import build     # pip install google-api-python-client
import pandas as pd
import torchaudio
import torchsummary
import matplotlib.pyplot as plt
import torchvision.models as models
from scipy.signal import resample_poly
import os


# 1. 마이크 활성화
# 2. 모델 로드
# 3. 녹음 종료시까지 3번 loop 반복해 모델 출력 누적
        # 3-1. 마이크 통한 음성 입력                
        # 3-2. 음성 -> 멜스펙토그램 (1초 마다)       
        # 3-3. 멜스펙토 그램 모델 입력              
        # 3-4. 모델 출력 = [wav_id, chunk_index, prob_happiness, prob_angry, prob_disgust, prob_fear, prob_neutral, prob_sadness, prob_surprise] 결과 누적
# 4. 출력 정규화 후 majority 감정 결정 및 시각화
# 부정적인 형태의 감정(angry, disgust, feat, sadness, surprise)의 비율이 높은 경우 해당 감정을 완화시키는 유튜브 추천(API) 



class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or 1
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)
    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)
# 전체 구조를 fastai 구조와 똑같이 nn.Sequential로!
def build_fastai_sequential_resnet34_head7():
    backbone = models.resnet34(weights=None)  # pretrained 여부와 상관없음, state_dict만 쓸거면 None
    # ResNet34의 feature extractor 부분만 자름
    body = nn.Sequential(*list(backbone.children())[:-2])  # (0)번에 들어감

    # fastai-style custom head, (1)번에 들어감
    head = nn.Sequential(
        AdaptiveConcatPool2d(1),                 # (N,1024,1,1)
        nn.Flatten(),                            # (N,1024)
        nn.BatchNorm1d(1024),
        nn.Dropout(0.25),
        nn.Linear(1024, 512, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Dropout(0.5),
        nn.Linear(512, 7, bias=False)
    )

    # Sequential로 (0):body, (1):head
    return nn.Sequential(body, head)


class Speech_Emotion_Recognition:
    def __init__(
                self,
                model,
                ):
        
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.stop_event = threading.Event()

        # [audio_capture 메소드] 변수
        self.chunk_sec = 2.61
        self.capture_rate = 44100
        self.rate = 16000
        self.chunk_size = int( self.capture_rate * self.chunk_sec)
        self.queue_max_size = 1
        self.audio_q = Queue(maxsize = self.queue_max_size)   # 녹음 저장 큐
                                             
        # [mel_inference 메소드] 변수: 학습 데이터와 동일하게 설정 
        # 1. Mel Spectrogram
        self.n_fft = 1024
        self.hop_length = 320
        self.n_mels = 128
        self.power = 2.0
        self.center = False
        self.fmin = 50
        self.fmax = 8000

        # 2. Pitch(톤)
        
        # 3. RMS(세기) 
        
        self.wav_id = datetime.datetime.now().strftime('%Y%m%d_%H%M')     # '20240606_1532'
        self.result_list = []      # 결과 저장 리스트

        
    def audio_capture(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=self.capture_rate,
                        input=True,
			input_device_index=11,
                        frames_per_buffer=self.chunk_size)
        chunk_idx_cap = 1

        """ 마이크에서 2.56초 분량의 오디오를 읽어 큐에 넣기."""
        while not self.stop_event.is_set():
            data = stream.read(self.chunk_size, exception_on_overflow=False)
            audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0  # -1~1 정규화
            audio = resample_poly(audio, up=self.rate, down=self.capture_rate)
            self.audio_q.put((chunk_idx_cap, audio))
            chunk_idx_cap += 1
        stream.stop_stream()
        stream.close()
        p.terminate()    

    
    def mel_inference(self):
        """큐에서 오디오 청크를 받아 멜-스펙트로그램 계산 및 모델 추론"""
        # 해당 스레드는 음성 녹음 단위보다 짧아야 완전한 실시간성 유지됨
        # 사실 큐사이즈 늘리면 오래 걸려도 상관은 없음
        self.model.eval()
        while not self.stop_event.is_set():
            try:
                chunk_idx_mel, audio_samples = self.audio_q.get(timeout=0.1)  # 큐에서 1초 청크 꺼내기
                #print("data loaded")
            except Empty:
                continue
            #print("start preprocessing")
            #if (chunk_idx_mel % 5) == 1:
             #   plt.plot(audio_samples)
              #  plt.title(f"audio{chunk_idx_mel}wav")
               # plt.xlabel("Samples")
               # plt.ylabel("amp")
               # plt.grid(True)
               # plt.savefig(f"chunk_{chunk_idx_mel}.png")
               # plt.close()

            pretime = time.time()
            # 1. Mel Spectrogram (PyTorch tensor)
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                f_min=self.fmin,
                f_max=self.fmax,
                center = False,
                power = 2.0
            )

            # 2. AmplitudeToDB
            db_transform = torchaudio.transforms.AmplitudeToDB(stype = 'power',top_db=80.0)
            # torchaudio는 torch.tensor가 필요       
            audio_tensor = torch.from_numpy(audio_samples + 1e-10).float()
            mel_spec = mel_transform(audio_tensor)
            mel_db = db_transform(mel_spec)
            mel_db = mel_db - mel_db.max()  # 범위 (-80,0)
            #mel_db = mel_db.numpy()   # (n_mels, T) (128, 128)

            '''
            mel_mag = np.maximum(melspec, 1e-12)    
            mel_log = np.log(mel_mag + 1e-3)        # shape: n_mels, time_frames
            mel_log = mel_log.T 
            '''
                          
            # 2. Pitch (librosa.yin)
            pitches = librosa.yin(
                audio_samples,
                fmin=self.fmin,
                fmax=self.fmax,
                sr=self.rate,
                frame_length=self.n_fft,
                hop_length=self.hop_length,
                center=False
            )  # (T,)
            # NaN, 0 처리 (0이면 무음)
            pitches = np.nan_to_num(pitches)
            pitches = np.tile(pitches, (128, 1))  # (128, T)
            pitches = torch.from_numpy(pitches).float()
            #print("4", pitches.shape)

            # 3. RMS (librosa.rms)
            rms = librosa.feature.rms(
                y=audio_samples,
                frame_length=self.n_fft,
                hop_length=self.hop_length,
                center = False
            )[0]  # (T,)
            rms = np.clip(rms, 0, 0.6) # rms 크기 제한
            rms = np.tile(rms, (128, 1))  # (128, T)
            '''==================== 입력 정규화 ================'''
            #print("start normalizaiton")
            
            norm_data_1 = (mel_db + 80.0) / 80.0  # (-80, 0) --> [0, 1]로 정규화
            invalid_mask = (norm_data_1 < 0.0) | (norm_data_1 > 1.0)
            if invalid_mask.any():
                print("Out-of-range values:", norm_data_1[invalid_mask], mel_db.max(), mel_db.min())
                raise ValueError("Normalization failed for mel data")

            min_log_data_2, max_log_data_2 = torch.log10(torch.tensor(50.0)), torch.log10(torch.tensor(8000.0))
            log_data_2 = torch.log10(pitches)  # pitch log
            norm_data_2 = (log_data_2 - min_log_data_2) / (max_log_data_2 - min_log_data_2)  # min_max 정규화 적용[0, 1]로 정규화
            norm_data_2 = torch.clamp(norm_data_2, 0.0, 1.0)  # [0, 1]로 정규화

            norm_data_3 = rms / 0.6
            norm_data_3 = torch.from_numpy(norm_data_3).float()
           # print("5", norm_data_3.shape)
            norm_data_3 = torch.clamp(norm_data_3, 0.0, 1.0)  # [0, 1]로 정규화

            data_norm = torch.stack([norm_data_1, norm_data_2, norm_data_3])
            data_norm = torch.stack([norm_data_1, norm_data_2, norm_data_3]).unsqueeze(0)
            print(f"preprocessing : {(time.time()-pretime)*1000:.3f}ms")
            prev_time= time.time()
            with torch.no_grad():
                output = self.model(data_norm)  # 추론 수행 shape: (1, 7) --> 출력 logit형태  
                probs = torch.softmax(output, dim=1)[0].cpu().numpy()  # 확률로 변환
            cur_time = time.time()
            print(f"inference_time = {(cur_time - prev_time)*1000:.3f}ms")
            
            # 결과 저장 [wav_id, chunk_index, prob_happiness, ..., prob_surprise]
            wav_id = self.wav_id  # 현재 녹음 세션 ID (별도 관리)
            result_row = [wav_id, chunk_idx_mel] + probs.tolist()
            self.result_list.append(result_row)
            print(f"Chunk {chunk_idx_mel} 결과: {probs}")  # 실시간 추론 결과 출력 (옵션)
            

            
            
    def search_youtube_videos(self, query, max_results=3):
        API_KEY = "<YOUR_API_KEY>"   # 발급받은 키로 대체
        youtube = build("youtube", "v3", developerKey=API_KEY)
        search_response = youtube.search().list(
            q=query,
            part="snippet",
            type="video",
            maxResults=max_results
        ).execute()
        video_list = []
        for item in search_response["items"]:
            title = item["snippet"]["title"]
            url = f"https://www.youtube.com/watch?v={item['id']['videoId']}"
            video_list.append((title, url))
        return video_list
    
    # 감정별 확률 시각화 (mean_probs, emotions 변수는 기존 코드와 동일하게 사용)
    def plot_emotion_bar(self, mean_probs, emotions, wav_id=None):
        plt.figure(figsize=(8, 4))
        plt.bar(emotions, mean_probs, color='skyblue')
        plt.ylim(0, 1)
        plt.ylabel('Probability')
        plt.title("Emotion Probabilities")
        for i, v in enumerate(mean_probs):
            plt.text(i, v+0.02, f"{v:.2f}", ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig("./output/Visualization")
        plt.close()
                
    def summarize_and_recommend(self, threshold=0.3, topn=2):
        if len(self.result_list) == 0:
            print("녹음/추론 결과가 없습니다.")
            return
        
        emotions = ['happiness', 'angry', 'disgust', 
                   'fear', 'neutral', 'sadness', 'surprise']

        arr = np.array([r[2:] for r in self.result_list])  # (N, 7) N은 녹음시간 N sec
        
        mean_probs = arr.mean(axis=0)   # (7,)

        print("\n=== 감정별 평균 확률 ===")
        for emo, prob in zip(emotions, mean_probs):
            print(f"{emo:>8}: {prob:.3f}")
        
        negative = ['angry', 'disgust', 'fear', 'sadness', 'surprise']
        rec_keywords = {
            'angry': "화날 때 듣는 진정 음악",
            'disgust': "웃긴 동물 영상",
            'fear': "불안 해소 명상",
            'sadness': "기분 좋아지는 노래",
            'surprise': "편안한 마음 안정 영상"
        }

        # 부정 감정만 추출, 확률 높은 순 정렬
        neg_idx = [emotions.index(e) for e in negative]
        #print("neg_idx", neg_idx)
        neg_probs = [(emotions[i], mean_probs[i]) for i in neg_idx]
        #print("neg_probs", neg_probs)
        neg_probs_sorted = sorted(neg_probs, key=lambda x: x[1], reverse=True)
        #print("_sorted", neg_probs_sorted)


        print("\n--- 감정 완화 유튜브 추천 ---")
        count = 0
        for emo, prob in neg_probs_sorted:
            if prob > threshold and count < topn:
                print(f"\n[{emo.upper()}] 감정 완화 추천 검색어: '{rec_keywords[emo]}'")
                try:
                    videos = self.search_youtube_videos(rec_keywords[emo], max_results=3)
                    for i, (title, url) in enumerate(videos, 1):
                        print(f"  {i}. {title}\n     {url}")
                except Exception as e:
                    print("  (유튜브 API 검색 오류):", e)
                count += 1

        if count == 0:
            print("\n뚜렷한 부정 감정이 높지 않아 별도 추천이 없습니다 :)")
        self.plot_emotion_bar(mean_probs, emotions)

    def main(self):
        print("녹음 시작! (종료하려면 Enter 키를 누르세요)")
        t1 = threading.Thread(target=self.audio_capture)    # 녹음
        t2 = threading.Thread(target=self.mel_inference)    # 변환 + 추론
        
        t1.start()
        t2.start()
        # 사용자 입력 대기 (엔터키 입력시 녹음 종료)
        input("녹음을 끝내려면 Enter 키를 누르세요...")
        self.stop_event.set()

        t1.join()
        t2.join()
        print("녹음 종료 및 추론 완료.")

        # 결과 엑셀에 저장
        columns = ['wav_id', 'chunk_idx', 'prob_happiness', 'prob_angry', 'prob_disgust',
           'prob_fear', 'prob_neutral', 'prob_sadness', 'prob_surprise']
        # pandas DataFrame으로 변환
        df = pd.DataFrame(self.result_list, columns=columns)
        # 엑셀로 저장
        df.to_excel("./output/all_results.xlsx", index=False)
        print("결과 엑셀 저장 완료!")

        # --- 감정별 평균 출력 및 유튜브 추천 ---
        self.summarize_and_recommend(threshold=0.3, topn=2)

def get_x1(row):
   return torch.load(row['feat_path'])



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 모델 선언
    os.makedirs("./output", exist_ok=True)
    model = build_fastai_sequential_resnet34_head7()
    # pth 로드
    state_dict = torch.load("SER_MSElossflat.pth", map_location="cpu")
    model.load_state_dict(state_dict)  
    model.eval()

    #torchsummary.summary(model,(3,128,128))
    ser = Speech_Emotion_Recognition(model)
    ser.main()
