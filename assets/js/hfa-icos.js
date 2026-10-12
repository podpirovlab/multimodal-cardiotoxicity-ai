/* CardioOncoPredict — HFA-ICOS baseline cardiovascular risk before anthracycline chemotherapy
   (port of cardioonco/hfa_icos.py). Source: Lyon AR et al., Eur J Heart Fail 2020;22:1945-1960,
   doi:10.1002/ejhf.1920, Table 2 and its footnote (CC BY). A medium1 factor counts 1 point and a
   medium2 factor 2 points. A clinical tool for professionals, not a prediction for one person. */
(function (root) {
  "use strict";
  // key, weight, level of evidence, Table 2 wording, Russian translation
  const FACTORS = [
    ["hf_cardiomyopathy", "very_high", "B", "Heart failure or cardiomyopathy", "Сердечная недостаточность или кардиомиопатия"],
    ["severe_valve_disease", "high", "C", "Severe valvular heart disease", "Тяжёлый порок клапанов сердца"],
    ["mi_revascularisation", "high", "C", "Myocardial infarction or previous coronary revascularisation (PCI or CABG)",
      "Инфаркт миокарда или реваскуляризация коронарных артерий в прошлом (ЧКВ или АКШ)"],
    ["stable_angina", "high", "C", "Stable angina", "Стабильная стенокардия"],
    ["lvef_below_50", "high", "B", "Baseline LVEF <50%", "Исходная ФВ ЛЖ <50%"],
    ["lvef_50_54", "medium2", "C", "Borderline LVEF 50–54%", "Пограничная ФВ ЛЖ 50–54%"],
    ["troponin", "medium1", "C", "Elevated baseline troponin", "Повышенный исходный тропонин"],
    ["natriuretic_peptide", "medium1", "C", "Elevated baseline BNP or NT-proBNP", "Повышенный исходный BNP или NT-proBNP"],
    ["age_80_plus", "high", "B", "Age ≥80 years", "Возраст ≥80 лет"],
    ["age_65_79", "medium2", "B", "Age 65–79 years", "Возраст 65–79 лет"],
    ["hypertension", "medium1", "B", "Hypertension", "Артериальная гипертензия"],
    ["diabetes", "medium1", "C", "Diabetes mellitus", "Сахарный диабет"],
    ["chronic_kidney_disease", "medium1", "C", "Chronic kidney disease", "Хроническая болезнь почек"],
    ["prior_anthracycline", "high", "B", "Previous anthracycline exposure", "Антрациклины в прошлом"],
    ["prior_chest_radiotherapy", "high", "C", "Prior radiotherapy to left chest or mediastinum",
      "Лучевая терапия на левую половину грудной клетки или средостение в прошлом"],
    ["prior_other_chemotherapy", "medium1", "C", "Previous non-anthracycline-based chemotherapy", "Химиотерапия без антрациклинов в прошлом"],
    ["smoking", "medium1", "C", "Current smoker or significant smoking history", "Курит сейчас или долго курил(а)"],
    ["obesity", "medium1", "C", "Obesity (BMI >30 kg/m2)", "Ожирение (ИМТ >30 кг/м²)"],
  ];
  // the groups of Table 2, in order, with the keys they hold
  const GROUPS = [
    ["Previous cardiovascular disease", "Болезни сердца и сосудов в прошлом", 0, 6],
    ["Cardiac biomarkers (where available)", "Сердечные биомаркеры (если измерены)", 6, 8],
    ["Demographic and cardiovascular risk factors", "Возраст и факторы риска", 8, 13],
    ["Previous cardiotoxic cancer treatment", "Кардиотоксичное лечение рака в прошлом", 13, 16],
    ["Lifestyle risk factors", "Образ жизни", 16, 18],
  ];
  // Table 2 footnotes: what counts as present
  const DEFINITIONS = {
    troponin: ["above the upper limit of normal for the local laboratory", "выше верхней границы нормы местной лаборатории"],
    natriuretic_peptide: ["above the upper limit of normal for the local laboratory", "выше верхней границы нормы местной лаборатории"],
    hypertension: ["systolic >140 mmHg or diastolic >90 mmHg, or on treatment", "систолическое >140 или диастолическое >90 мм рт. ст., или лечение"],
    diabetes: ["HbA1c >7.0% (>53 mmol/mol), or on treatment", "HbA1c >7,0% (>53 ммоль/моль), или лечение"],
    chronic_kidney_disease: ["eGFR <60 mL/min/1.73 m2", "СКФ <60 мл/мин/1,73 м²"],
  };
  const EXCLUSIVE = [["age_80_plus", "age_65_79"], ["lvef_below_50", "lvef_50_54"]];
  const POINTS = { medium1: 1, medium2: 2 };
  const EXPECTED_RISK = { low: "<2%", medium: "2–9%", high: "10–19%", very_high: "≥20%" };   // expert opinion

  function classify(present) {
    const set = new Set(present), weight = Object.fromEntries(FACTORS.map((f) => [f[0], f[1]]));
    for (const k of set) if (!(k in weight)) throw new Error(`unknown risk factor: ${k}`);
    for (const [a, b] of EXCLUSIVE) if (set.has(a) && set.has(b)) throw new Error(`${a} and ${b} cannot both be present`);
    const order = FACTORS.map((f) => f[0]), keys = order.filter((k) => set.has(k));
    const veryHigh = keys.filter((k) => weight[k] === "very_high"), high = keys.filter((k) => weight[k] === "high");
    const medium = keys.filter((k) => weight[k] in POINTS), points = medium.reduce((s, k) => s + POINTS[weight[k]], 0);
    let level, deciding;
    if (veryHigh.length) [level, deciding] = ["very_high", veryHigh];
    else if (high.length) [level, deciding] = ["high", high];
    else if (points >= 5) [level, deciding] = ["high", medium];
    else if (points >= 2) [level, deciding] = ["medium", medium];
    else [level, deciding] = ["low", medium];
    return { level, points, deciding, expectedRisk: EXPECTED_RISK[level] };
  }

  root.CardioHFA = { FACTORS, GROUPS, DEFINITIONS, EXCLUSIVE, EXPECTED_RISK, classify };
  if (typeof module !== "undefined") module.exports = root.CardioHFA;
})(typeof window !== "undefined" ? window : globalThis);
