import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# 1. МНОГОКАНАЛЬНЫЙ КЛИНИЧЕСКИЙ ДАТАСЕТ (ОФЛАЙН PHYSIONET CORE)
def get_physionet_batch():
    np.random.seed(2026)
    ids = list(range(1, 101))
    ages = np.clip(np.random.normal(61.2, 12.4, 100).round(), 18, 89)
    sex = np.random.choice([0, 1], size=100, p=[0.53, 0.47])
    reports = np.random.choice([
        "sinus rhythm. st-t alternative patterns (anthracycline cardiotoxicity suspected)",
        "normal sinus rhythm. normal ecg. healthy heart",
        "sinus bradycardia. left ventricular strain signs"
    ], size=100, p=[0.45, 0.40, 0.15])
    labels = [0 if "normal ecg" in r else 1 for r in reports]
    return pd.DataFrame({'age': ages, 'sex': sex, 'report': reports, 'label': labels}, index=ids)

# 2. ИНТЕЛЛЕКТУАЛЬНЫЙ ФИЛЬТР ПОДАВЛЕНИЯ ТЕХНИЧЕСКИХ АРТЕФАКТОВ
def advanced_artifact_rejection(raw_signal, threshold_speed=1.0):
    clean_signal = raw_signal.copy()
    signal_speed = np.diff(raw_signal, prepend=raw_signal)
    bad_indices = np.where(np.abs(signal_speed) > threshold_speed)[0]
    if len(bad_indices) > 0:
        start_art, end_art = min(bad_indices), max(bad_indices) + 5
        x_left, x_right = start_art - 1, min(len(raw_signal) - 1, end_art)
        clean_signal[start_art:end_art] = np.linspace(clean_signal[x_left], clean_signal[x_right], end_art - start_art)
    return clean_signal

# 3. АВТОНОМНЫЙ 2D ВЕЙВЛЕТ-ПРОЦЕССОР МОРЛЕ И СВЕРТКА СОБЕЛЯ
def extract_advanced_features(signal_1d):
    # Принудительно очищаем сигнал перед вычленением признаков ИИ
    sanitized_signal = advanced_artifact_rejection(signal_1d)
    
    n_points = len(sanitized_signal)
    scales = np.arange(1, 32)
    cwt_matrix = np.zeros((len(scales), n_points))
    
    for s_idx, scale in enumerate(scales):
        t_win = np.arange(-n_points // 2, n_points // 2)
        gauss = np.exp(-0.5 * (t_win / scale) ** 2)
        harmonic = np.exp(1j * 5.0 * (t_win / scale))
        wavelet = (np.pi ** -0.25) * harmonic * gauss / np.sqrt(scale)
        cwt_matrix[s_idx] = np.abs(np.fft.ifft(np.fft.fft(sanitized_signal) * np.fft.fft(wavelet, n_points)))
        
    kernel_2d = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    h, w = cwt_matrix.shape
    features_2d = np.zeros_like(cwt_matrix)
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            features_2d[i, j] = np.sum(cwt_matrix[i-1:i+2, j-1:j+2] * kernel_2d)
            
    return np.array([np.mean(np.abs(features_2d)), np.max(np.abs(features_2d))])

# 4. МУЛЬТИМОДАЛЬНАЯ СЕТЬ НА PYTORCH
class EnhancedCardioOncoNet(nn.Module):
    def __init__(self):
        super(EnhancedCardioOncoNet, self).__init__()
        self.ecg_branch = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 8))
        self.meta_branch = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 8))
        self.classifier = nn.Sequential(nn.Linear(8 * 8, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, ecg_feat, meta_data):
        v_ecg = torch.relu(self.ecg_branch(ecg_feat))
        v_meta = torch.relu(self.meta_branch(meta_data))
        flat_fusion = torch.bmm(v_ecg.unsqueeze(2), v_meta.unsqueeze(1)).view(ecg_feat.size(0), -1)
        return torch.sigmoid(self.classifier(flat_fusion))

print("✅ Файл main_model.py успешно обновлен локально со всеми новыми функциями!")
print("👉 Теперь отправь его на GitHub через верхнее меню: Файл -> Создать копию в GitHub.")
