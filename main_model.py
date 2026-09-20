"""
Педагогический прототип на чистом NumPy: пошаговая иллюстрация мультимодального
слияния (bilinear fusion) сигнала ЭКГ и клинических метаданных пациента.

ВНИМАНИЕ: все данные ниже синтетические (сгенерированы вручную для наглядности),
веса сети не обучены на реальных примерах. Скрипт демонстрирует ТОЛЬКО механику
вычислений — для рабочей версии на PyTorch см. advanced_model.py.
"""
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("⏳ Шаг 1: Подготовка синтетического набора данных пациентов...")

# Синтетические данные пяти условных пациентов (не реальные медицинские записи)
ids = list(range(1, 6))
sex_array = list(np.ones(5, dtype=int))
ages = [56.0, 19.0, 63.0, 45.0, 52.0]
labels = list([1, 0, 0, 1, 1]) # 1 - есть токсичность, 0 - здоров
reports = [
    'sinus rhythm. myocardial infarction signs (doxorubicin toxicity suspected)', 
    'normal ecg. sinus rhythm', 'sinus bradycardia. left ventricular hypertrophy',
    'st-t changes. subclinical cardiomyopathy symptoms', 'sinus tachycardia. acute cardiotoxicity patterns'
]

mock_data = {'ecg_id': ids, 'age': ages, 'sex': sex_array, 'report': reports, 'true_label': labels}
df = pd.DataFrame(mock_data).set_index('ecg_id')

# Шаг 2: Сигнал ЭКГ и Свертка (Ветка А)
fs = 500
time = np.linspace(0, 6, fs * 6)
def generate_clean_heartbeat(t):
    return (0.15 * np.exp(-((t - 0.2) / 0.03)**2) + -0.1 * np.exp(-((t - 0.3) / 0.01)**2) + 
            1.2 * np.exp(-((t - 0.32) / 0.015)**2) + -0.25 * np.exp(-((t - 0.35) / 0.015)**2) + 
            0.35 * np.exp(-((t - 0.55) / 0.05)**2))

ecg_signal = np.zeros_like(time)
for i in range(7):
    t_start = i * 0.8
    idx = (time >= t_start) & (time < t_start + 0.8)
    if np.any(idx): ecg_signal[idx] += generate_clean_heartbeat(time[idx] - t_start)

kernel = np.array([-0.1, -0.3, 0.8, 1.5, 0.8, -0.3, -0.1])
ecg_features = np.convolve(ecg_signal, kernel, mode='same')
v_ecg = np.array([np.mean(ecg_features), np.max(ecg_features)])

# Шаг 3: МедКарта через MLP + ReLU (Ветка Б)
x_meta = np.array([df.loc[1, 'age'], df.loc[1, 'sex']])
np.random.seed(42)
W_mlp = np.random.normal(size=(2, 2))
b_mlp = np.array([0.1, -0.2])
Z_mlp = np.dot(x_meta, W_mlp) + b_mlp
v_meta = np.maximum(0, Z_mlp)

# Шаг 4: Билинейное слияние (Bilinear Fusion)
V_fusion = np.outer(v_ecg, v_meta)

# Шаг 5: Выходной нейрон + сжатие в вероятность через сигмоиду
Z_out = np.sum(V_fusion * np.random.normal(size=(2, 2))) + 0.5
y_pred = 1 / (1 + np.exp(-Z_out))  # sigmoid(Z_out)

# Шаг 6: Расчет ошибки ИИ (BCE Loss)
y_true = df.loc[1, 'true_label']
# Клиппинг предотвращает log(0) -> -inf / NaN, если сигмоида насыщается
# (что легко происходит со случайными, необученными весами).
EPS = 1e-7
y_pred_clipped = np.clip(y_pred, EPS, 1 - EPS)
loss = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))

print("✅ Прямой проход (forward pass) завершён успешно!")
print(f"🎯 Расчётная вероятность (необученные случайные веса): {y_pred:.4f}")
print(f"🩺 Иллюстративный «риск» для Пациента №1: {y_pred * 100:.2f}%")
print(f"⚠️ Значение функции потерь (BCE Loss): {loss:.4f}")

# --- ОТРИСОВКА МЕДИЦИНСКОГО ЭКРАНА ПРОГНОЗА ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9))

# График 1: Подсветка опасных участков (Шаг 3)
normalized_features = (ecg_features - np.min(ecg_features)) / (np.max(ecg_features) - np.min(ecg_features))
danger_zones = normalized_features > 0.75
ax1.plot(time, ecg_signal, color='#1d3557', linewidth=2, label='Сигнал ЭКГ')
ax1.fill_between(time, -0.5, 1.5, where=danger_zones, color='#e63946', alpha=0.3, label='🚨 Зона аномалии (Обнаружен паттерн токсичности)')
ax1.set_title(f'Автоматический мониторинг ЭКГ ИИ (Пациент №1, {df.loc[1, "age"]} лет)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Вольтаж (мВ)')
ax1.set_ylim(-0.4, 1.4)
ax1.grid(True, linestyle=':', alpha=0.5)
ax1.legend(loc='upper right')

# График 2: Итоговый вердикт в процентах (Шаг 5 и 6)
ax2.barh(['Реальный диагноз (Полка 1)', 'Прогноз нашего ИИ (Полка 4)'], [y_true, y_pred], color=['darkcyan', 'crimson'], alpha=0.8)
ax2.set_xlim(0, 1.1)
ax2.set_title('Итоговый вердикт мультимодальной нейросети', fontsize=13, fontweight='bold')
ax2.set_xlabel('Вероятность кардиотоксичности (0.0 — Здоров, 1.0 — Максимальный риск)')
ax2.grid(axis='x', linestyle='--', alpha=0.5)
for i, v in enumerate([y_true, y_pred]):
    ax2.text(v + 0.02, i, f'{v*100:.2f}%', va='center', fontweight='bold', fontsize=11)

plt.tight_layout()

# В headless-окружениях (CI, серверы без дисплея) plt.show() не падает с
# исключением — он просто печатает предупреждение и ничего не показывает.
# Поэтому вместо try/except явно проверяем, интерактивен ли текущий backend.
_NON_INTERACTIVE_BACKENDS = {"agg", "pdf", "svg", "ps", "cairo", "template"}
if matplotlib.get_backend().lower() in _NON_INTERACTIVE_BACKENDS:
    output_path = "demo_forward_pass.png"
    plt.savefig(output_path, dpi=150)
    print(f"🖼️  Интерактивный дисплей недоступен — график сохранён в {output_path}")
else:
    plt.show()

print("\n✅ Демонстрация пайплайна завершена: все пять этапов синхронизированы и отработали.")
