import torch
import torch.nn as nn
import numpy as np

# 1. Пропишем упрощенную архитектуру твоей модели для экспорта графа
class BilinearMultimodalFusionNet(nn.Module):
    def __init__(self):
        super(BilinearMultimodalFusionNet, self).__init__()
        # Ветвь А: Обработка признаков ЭКГ спектрограмм после CWT (вход: 2D фичи)
        self.ecg_branch = nn.Sequential(
            nn.Linear(120, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        # Ветвь Б: Клинические метаданные (Возраст, Пол)
        self.meta_branch = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 32)
        )
        # Выходной классификатор после Билинейного слияния (размерность 32x32 = 1024)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, ecg_feats, meta_tensors):
        v_ecg = self.ecg_branch(ecg_feats)
        v_meta = self.meta_branch(meta_tensors)
        
        # Билинейное слияние тензоров: вычисление внешнего произведения (Outer Product)
        # Математика: Вектор (batch, 32, 1) x Вектор (batch, 1, 32) -> Матрица (batch, 32, 32)
        batch_size = v_ecg.size(0)
        bilinear_matrix = torch.bmm(v_ecg.unsqueeze(2), v_meta.unsqueeze(1))
        bilinear_vector = bilinear_matrix.view(batch_size, -1) # Сплющиваем в 1024
        
        return self.classifier(bilinear_vector)

# Инициализация модели и фейковый инференс для трассировки графа
model = BilinearMultimodalFusionNet()
model.eval()

dummy_ecg = torch.randn(1, 120)  # Пример 120 признаков CWT вейвлет-анализа
dummy_meta = torch.randn(1, 2)   # Пример метаданных (Возраст, Кодированный пол)

# 2. Экспорт базовой модели в стандартный ONNX
onnx_path = "model_base.onnx"
try:
    torch.onnx.export(
        model,
        (dummy_ecg, dummy_meta),
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['ecg_features', 'metadata_tensors'],
        output_names=['cardiotoxicity_probability'],
        dynamic_axes={'ecg_features': {0: 'batch_size'}, 'metadata_tensors': {0: 'batch_size'}, 'cardiotoxicity_probability': {0: 'batch_size'}}
    )
    print(f"✅ Базовая модель успешно экспортирована в {onnx_path}")
except Exception as exc:
    print(f"❌ Экспорт в ONNX не удался: {exc}")
    raise

# 3. Квантование до INT8 (динамическое сжатие весов для Edge CPU)
#
# NOTE: quantize_dynamic() delegates to onnx's shape-inference engine, which is
# known to be sensitive to the exact onnx / onnxruntime / torch version combo
# used to produce `onnx_path`. If quantization fails, the full-precision
# `model_base.onnx` above is still valid and usable — quantization is a
# best-effort size/latency optimization, not required for correctness.
try:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(
        model_input=onnx_path,
        model_output="model_quantized.onnx",
        weight_type=QuantType.QUInt8
    )
    print("🚀 INT8-квантование завершено! Файл 'model_quantized.onnx' готов для Edge-устройств/носимых мониторов.")
except ImportError:
    print("⚠️ Для квантования установи пакет: pip install onnxruntime")
except Exception as exc:
    print(f"⚠️ Квантование пропущено из-за несовместимости версий onnx/onnxruntime: {exc}")
    print(f"   Полноточная модель '{onnx_path}' уже экспортирована и пригодна к использованию.")
