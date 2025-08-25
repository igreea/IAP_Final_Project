import torch
import os
from tqdm import tqdm
import numpy as np
import torchaudio

# power 스펙트로그램(mel_tf 출력) → dB
db_tf = torchaudio.transforms.AmplitudeToDB( # 엡실론(1e-10) 내부 보정
    stype='power',   # mel_tf(power=2.0) 입력일 때 
    top_db=80.0      # dB 상한(옵션)
)

data_path = "../set2_3ch"
output_path = "../set2_3ch_norm"
sorted_files = sorted(os.listdir(data_path))
list_of_max = []
list_of_min = []
min_log_data_2, max_log_data_2 = torch.log10(torch.tensor(50.0)), torch.log10(torch.tensor(8000.0))
for data_name in tqdm(sorted_files):
    if not data_name.endswith(".pt"):
        continue
    data_path_full = os.path.join(data_path, data_name)
    data = torch.load(data_path_full)
    
    data_1, data_2, data_3 = data # mel, pitch, rms
    db_data_1 = db_tf(data_1)  # mel_tf(power=2.0) → dB
    db_data_1 = db_data_1 - db_data_1.max()  # dB 상한을 0으로 맞춤
    norm_data_1 = (db_data_1 + 80.0) / 80.0  # [0, 1]로 정규화
    invalid_mask = (norm_data_1 < 0.0) | (norm_data_1 > 1.0)
    if invalid_mask.any():
        print("Out-of-range values:", norm_data_1[invalid_mask], db_data_1.max(), db_data_1.min())
        raise ValueError("Normalization failed for mel data")




    log_data_2 = torch.log10(data_2)  # pitch log

    norm_data_2 = (log_data_2 - min_log_data_2) / (max_log_data_2 - min_log_data_2)  # [0, 1]로 정규화
    norm_data_2 = torch.clamp(norm_data_2, 0.0, 1.0)  # [0, 1]로 정규화

    norm_data_3 = data_3 / 0.6
    norm_data_3 = torch.clamp(norm_data_3, 0.0, 1.0)  # [0, 1]로 정규화

    data_norm = torch.stack([norm_data_1, norm_data_2, norm_data_3])
    torch.save(data_norm, os.path.join(output_path, data_name))
    assert not data_norm.isnan().any(), "Data contains NaN values"
    list_of_max.append((data_norm[0].max().item(), 
                        data_norm[1].max().item(), 
                        data_norm[2].max().item()))
    list_of_min.append((data_norm[0].min().item(),
                        data_norm[1].min().item(),
                        data_norm[2].min().item()))

print("Max value:", max(list_of_max))
print("Min value:", min(list_of_min))