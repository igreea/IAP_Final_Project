import os
import torch
import torchaudio
import pandas as pd
import librosa
import numpy as np
from tqdm import tqdm

# ───────── 모델과 완전히 같은 파라미터 ─────────
SR, N_FFT, HOP = 16_000, 1024, 320 # 16kHz, 1024 FFT, 320 hop -> 50 frame / 1s
SEG_FRAMES    = 128                # -> 128 frames == 2.56 s
STRIDE_FRAMES = SEG_FRAMES + 32                 # segement 사이 빈 구간 길이 32 frame = 0.64초 간격
MEL_BINS      = 128
DISMISS = int(0.75 * SR) # 무시할 시간 프레임 수 (0.75초)
# SEG_SAMPLES: 한 세그먼트가 실제로 가지는 샘플 길이
#   = (SEG_FRAMES - 1) * HOP + N_FFT
#   이유: 첫 프레임은 N_FFT 샘플, 이후 프레임마다 HOP 샘플씩 더해지기 때문
SEG_SAMPLES = N_FFT + (SEG_FRAMES - 1) * HOP  # 41 664 = 1024 + 127 * 320

mel_tf = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR, n_fft=N_FFT, hop_length=HOP,
    win_length=N_FFT, n_mels=MEL_BINS,
    f_min=50, f_max=8000, power=2.0, center=False) # 16000Hz 기준 1초에 50프레임

def all_segments(wav):
    """파형(1,N) -> 세그먼트 리스트[(1,41 664)]"""
    total_frames = (wav.size(1) - N_FFT) // HOP + 1
    # 마지막 창 포함되도록 패딩
    need_pad = max(
        0,
        ( (total_frames - SEG_FRAMES) % STRIDE_FRAMES != 0 )
        * (STRIDE_FRAMES - (total_frames - SEG_FRAMES) % STRIDE_FRAMES)
    )
    if need_pad:
        wav = torch.nn.functional.pad(wav, (0, need_pad * HOP))
        total_frames += need_pad

    segs = []
    for start in range(0, total_frames - SEG_FRAMES + 1, STRIDE_FRAMES):
        s = start * HOP
        segs.append(wav[:, s : s + SEG_SAMPLES])
    if (need_pad*HOP > SR) and len(segs) > 1 : # 1초 이상 패딩된 경우
        segs = segs[:-1] # 마지막 세그먼트는 제거
    return segs

# ───────── 경로 지정 ─────────
ROOT    = "../tiny/emotion/set2"                   # WAV + CSV 위치
CSV_IN  = "merged_output.csv"
MELDIR  = "../tiny/emotion_mel/set2_3ch_aa"; os.makedirs(MELDIR, exist_ok=True)
CSV_OUT = os.path.join(MELDIR, "audio_segments_3ch.csv")

df_in = pd.read_csv(CSV_IN)
rows  = []

for wav_id, *prob in tqdm(
    df_in[["wav_id", *df_in.filter(like="prob_").columns]].values,
    desc="Mel-pitch-rms caching"
):
    wav, sr = torchaudio.load(os.path.join(ROOT, f"{wav_id}.wav"))
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    wav = wav[:, DISMISS:]  # 무시할 시간 프레임 제거
    for idx, seg in enumerate(all_segments(wav)):
        # 1) Mel-spectrogram (1, n_mels, frames)
        mel = mel_tf(seg + 1e-10)
    
        # # 2) Audio array for librosa (shape: SEG_SAMPLES,)
        audio_np = seg.squeeze(0).numpy()

        # # 3) Pitch (F0) 추출 -> 길이: SEG_FRAMES
        pitch = librosa.yin(
            audio_np,
            sr=SR,
            frame_length=N_FFT,
            hop_length=HOP,
            fmin=50,
            fmax=8000,
            center=False
        )  # shape (frames,)
        assert len(pitch) == SEG_FRAMES, f"pitch length {len(pitch)} != {SEG_FRAMES}"
        pitch = np.nan_to_num(pitch, nan=0.0)

        # # 4) RMS(음성 세기) 추출 -> shape (1, frames) -> flatten -> (frames,)
        rms = librosa.feature.rms(
            y=audio_np,
            frame_length=N_FFT,
            hop_length=HOP,
            center=False
        )[0]
        assert len(rms) == SEG_FRAMES, f"rms length {len(rms)} != {SEG_FRAMES}"

        assert not torch.isnan(mel).any(), "mel에 NaN이 있습니다"
        assert not np.isnan(pitch).any(), "pitch에 NaN이 있습니다"
        assert not np.isnan(rms).any(),   "rms에 NaN이 있습니다"

        # # 5) Tensor로 변환 및 브로드캐스트
        # mel: (1, n_mels, frames)
        # pitch_map: (1, n_mels, frames)
        # rms_map:   (1, n_mels, frames)
        pitch_t = torch.from_numpy(pitch).float()   # (frames,)
        pitch_map = pitch_t.unsqueeze(0).expand(MEL_BINS, SEG_FRAMES)
        pitch_map = pitch_map.unsqueeze(0)         # (1, n_mels, frames)

        rms_t = torch.from_numpy(rms).float()       # (frames,)
        rms_map = rms_t.unsqueeze(0).expand(MEL_BINS, SEG_FRAMES)
        rms_map = rms_map.unsqueeze(0)             # (1, n_mels, frames)

        # # 6) 3채널로 스택 -> (3, n_mels, frames)
        feat = torch.cat([mel, pitch_map, rms_map], dim=0)

        # 7) 저장
        out_name = f"{wav_id}_{idx}.pt"
        torch.save(feat, os.path.join(MELDIR, out_name))

        # CSV용 행 추가
        rows.append([out_name, *prob])

# 새 CSV 작성
cols = ["feat_path"] + list(df_in.filter(like="prob_").columns)
pd.DataFrame(rows, columns=cols).to_csv(CSV_OUT, index=False)
print(f"Saved {len(rows):,} segments -> {CSV_OUT}")
