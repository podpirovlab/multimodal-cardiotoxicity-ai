"""HFA-ICOS baseline cardiovascular risk before anthracycline chemotherapy.

Lyon AR, Dent S, Stanway S, et al. Baseline cardiovascular risk assessment in cancer patients
scheduled to receive cardiotoxic cancer therapies: a position statement and new risk assessment
tools from the Cardio-Oncology Study Group of the Heart Failure Association of the European
Society of Cardiology in collaboration with the International Cardio-Oncology Society.
Eur J Heart Fail. 2020;22(11):1945-1960. doi:10.1002/ejhf.1920 (CC BY; Table 2).

The risk factors, their weights and levels of evidence are copied from Table 2; the rules are its
footnote: "Low risk = no risk factor OR one medium1 risk factor; Medium risk = medium risk factors
with a total of 2-4 points; High risk = medium risk factors with a total of >=5 points OR any high
risk factor; Very high risk = any very high risk factor."  A medium1 factor counts 1 point and a
medium2 factor 2 points, as the text's "a single medium2 risk factor ... with points totalling
2-4" implies.  The authors give the expected risk of future cardiotoxicity per level as expert
opinion, "a guide" still to be validated: low <2%, medium 2-9%, high 10-19%, very high >=20%.
The web tool runs the same table (assets/js/hfa-icos.js).  A clinical tool for professionals,
not a prediction for one person.
"""
from __future__ import annotations

from dataclasses import dataclass

# key, group, risk factor (as in Table 2), weight, level of evidence
FACTORS = (
    ("hf_cardiomyopathy", "Previous cardiovascular disease", "Heart failure or cardiomyopathy", "very_high", "B"),
    ("severe_valve_disease", "Previous cardiovascular disease", "Severe valvular heart disease", "high", "C"),
    ("mi_revascularisation", "Previous cardiovascular disease",
     "Myocardial infarction or previous coronary revascularisation (PCI or CABG)", "high", "C"),
    ("stable_angina", "Previous cardiovascular disease", "Stable angina", "high", "C"),
    ("lvef_below_50", "Previous cardiovascular disease", "Baseline LVEF <50%", "high", "B"),
    ("lvef_50_54", "Previous cardiovascular disease", "Borderline LVEF 50–54%", "medium2", "C"),
    ("troponin", "Cardiac biomarkers (where available)", "Elevated baseline troponin", "medium1", "C"),
    ("natriuretic_peptide", "Cardiac biomarkers (where available)", "Elevated baseline BNP or NT-proBNP", "medium1", "C"),
    ("age_80_plus", "Demographic and cardiovascular risk factors", "Age ≥80 years", "high", "B"),
    ("age_65_79", "Demographic and cardiovascular risk factors", "Age 65–79 years", "medium2", "B"),
    ("hypertension", "Demographic and cardiovascular risk factors", "Hypertension", "medium1", "B"),
    ("diabetes", "Demographic and cardiovascular risk factors", "Diabetes mellitus", "medium1", "C"),
    ("chronic_kidney_disease", "Demographic and cardiovascular risk factors", "Chronic kidney disease", "medium1", "C"),
    ("prior_anthracycline", "Previous cardiotoxic cancer treatment", "Previous anthracycline exposure", "high", "B"),
    ("prior_chest_radiotherapy", "Previous cardiotoxic cancer treatment",
     "Prior radiotherapy to left chest or mediastinum", "high", "C"),
    ("prior_other_chemotherapy", "Previous cardiotoxic cancer treatment",
     "Previous non-anthracycline-based chemotherapy", "medium1", "C"),
    ("smoking", "Lifestyle risk factors", "Current smoker or significant smoking history", "medium1", "C"),
    ("obesity", "Lifestyle risk factors", "Obesity (BMI >30 kg/m2)", "medium1", "C"),
)
# Table 2 footnotes: what counts as present
DEFINITIONS = {
    "troponin": "above the upper limit of normal for the local laboratory",
    "natriuretic_peptide": "above the upper limit of normal for the local laboratory",
    "hypertension": "systolic >140 mmHg or diastolic >90 mmHg, or on treatment",
    "diabetes": "HbA1c >7.0% (>53 mmol/mol), or on treatment",
    "chronic_kidney_disease": "eGFR <60 mL/min/1.73 m2",
}
EXCLUSIVE = (("age_80_plus", "age_65_79"), ("lvef_below_50", "lvef_50_54"))
POINTS = {"medium1": 1, "medium2": 2}
EXPECTED_RISK = {"low": "<2%", "medium": "2–9%", "high": "10–19%", "very_high": "≥20%"}   # expert opinion


@dataclass
class HFAICOSResult:
    level: str                # "low" | "medium" | "high" | "very_high"
    points: int               # medium1 + medium2 points
    deciding: list            # the factors that set the level
    expected_risk: str        # the authors' guide, expert opinion


def classify(present: set | list) -> HFAICOSResult:
    """Risk level from the keys of the risk factors that are present."""
    present = set(present)
    known = {f[0] for f in FACTORS}
    if present - known:
        raise ValueError(f"unknown risk factors: {sorted(present - known)}")
    for a, b in EXCLUSIVE:
        if a in present and b in present:
            raise ValueError(f"{a} and {b} cannot both be present")
    weight = {f[0]: f[3] for f in FACTORS}
    very_high = [k for k in present if weight[k] == "very_high"]
    high = [k for k in present if weight[k] == "high"]
    medium = [k for k in present if weight[k] in POINTS]
    points = sum(POINTS[weight[k]] for k in medium)
    if very_high:
        level, deciding = "very_high", very_high
    elif high:
        level, deciding = "high", high
    elif points >= 5:
        level, deciding = "high", medium
    elif points >= 2:
        level, deciding = "medium", medium
    else:
        level, deciding = "low", medium
    order = [f[0] for f in FACTORS]
    return HFAICOSResult(level=level, points=points, deciding=sorted(deciding, key=order.index),
                         expected_risk=EXPECTED_RISK[level])
