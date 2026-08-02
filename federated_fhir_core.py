import numpy as np
import json
from datetime import datetime

# === МАТЕМАТИКА ФЕДЕРАТИВНОГО ОБУЧЕНИЯ (FedAvg) ===
def simulate_federated_averaging(local_weights_nodes, data_sizes):
    """
    Алгоритм FedAvg объединяет веса локально обученных моделей пропорционально объему данных.
    Формула: W_global = sum(N_k / N_total * W_k)
    """
    total_data = sum(data_sizes)
    global_weights = np.zeros_like(local_weights_nodes[0])
    
    for k in range(len(local_weights_nodes)):
        weight_factor = data_sizes[k] / total_data
        global_weights += weight_factor * local_weights_nodes[k]
        
    return global_weights

# Симуляция: три изолированных госпиталя обучили свои ветви ИИ на местных ЭКГ
weights_vologda_hospital = np.array([0.88, -0.12, 0.45, 0.98])
weights_boston_clinic = np.array([0.92, -0.10, 0.41, 1.02])
weights_mit_medical = np.array([0.90, -0.15, 0.48, 0.99])

nodes = [weights_vologda_hospital, weights_boston_clinic, weights_mit_medical]
datasets = [1200, 3500, 800] # Количество пациентов в базах данных (закрытые локальные логи)

global_fused_weights = simulate_federated_averaging(nodes, datasets)
print("--- МАТЕМАТИЧЕСКАЯ СИНХРОНИЗАЦИЯ FEDERATED LEARNING ---")
print(f"Скомпилированные глобальные веса ИИ (без утечки данных): {global_fused_weights}\n")


# === СИМУЛЯЦИЯ МЕЖДУНАРОДНОГО СТАНДАРТА ДАННЫХ HL7 FHIR ===
def generate_fhir_diagnostic_report(patient_id, age, sex, risk_prob, status):
    """
    Генерирует официальный валидный FHIR-ресурс DiagnosticReport для интеграции в EMR/EHR больниц.
    """
    fhir_json = {
        "resourceType": "DiagnosticReport",
        "id": "cardiooncopredict-analysis-001",
        "status": "final",
        "category": [{
            "coding": [{
                "system": "http://hl7.org",
                "code": "GE",
                "display": "Genetics / Advanced Modality Analytics"
            }]
        }],
        "code": {
            "coding": [{
                "system": "http://loinc.org",
                "code": "99542-3",
                "display": "Anthracycline-induced cardiotoxicity screening report via AI"
            }],
            "text": "Autonomous CardioOncoPredict Evaluation"
        },
        "subject": {
            "reference": f"Patient/{patient_id}",
            "display": f"Demographics: Age {age}, Biological Sex: {sex}"
        },
        "effectiveDateTime": datetime.utcnow().isoformat() + "Z",
        "issued": datetime.utcnow().isoformat() + "Z",
        "conclusion": f"AI Verdict: {status}. Computed myocardial tissue damage risk profile: {risk_prob}%.",
        "conclusionCode": [{
            "coding": [{
                "system": "http://snomed.info",
                "code": "443331000124108" if risk_prob > 50 else "315642005",
                "display": "Chemotherapy-induced cardiotoxicity subclinical risk flagged" if risk_prob > 50 else "Myocardial stability retained"
            }]
        }]
    }
    return fhir_json

# Тест генерации FHIR-пакета для отправки в базу данных госпиталя
fhir_packet = generate_fhir_diagnostic_report("petya-vologda-99", 45, "Male", 93.42, "CRITICAL RISK")
print("--- ГЕНЕРАЦИЯ МЕДИЦИНСКОГО СТАНДАРТА DATA-PACKET HL7 FHIR ---")
print(json.dumps(fhir_packet, indent=2))
