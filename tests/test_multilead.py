"""TWA in all leads together and the two-adjacent-precordial-leads rule (cardioonco.twa.analyze_leads)."""
import numpy as np
import pytest

from cardioonco.synth import SynthConfig, generate_12lead
from cardioonco.twa import PRECORDIAL, analyze_leads

NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def twelve(alt=0.0, hr=100, seed=5):
    _, X, r = generate_12lead(SynthConfig(fs=500, n_beats=160, heart_rate=hr, alternans_uv=alt, noise_uv=8, seed=seed))
    return X.T.copy(), r


def test_alternans_of_the_heart_is_found_in_adjacent_precordial_leads():
    X, _ = twelve(alt=15)
    res = analyze_leads(X, 500, NAMES)
    assert res.outcome == "positive" and res.reason == "adjacent_precordial"
    assert all(a in PRECORDIAL and PRECORDIAL.index(b) == PRECORDIAL.index(a) + 1 for a, b in res.positive_pairs)
    assert res.pair_estimate_uv > 1.9


def test_no_alternans_at_a_high_rate_is_negative():
    X, _ = twelve(alt=0, hr=108)
    res = analyze_leads(X, 500, NAMES)
    assert res.outcome == "negative" and res.positive_pairs == []


def test_alternans_in_one_lead_only_is_not_counted():
    """10 uV alternating in V4's ST-T alone (an electrode artefact, say) makes V4 positive,
    but one lead is not two adjacent precordial leads."""
    X, r = twelve(alt=0, hr=100)
    a, b = int(0.12 * 500), int(0.32 * 500)
    for k, rk in enumerate(r):
        if rk + b < X.shape[1]:
            X[9, rk + a:rk + b] += 0.010 * (-1) ** k
    res = analyze_leads(X, 500, NAMES)
    assert res.leads["V4"]["outcome"] == "positive"
    assert res.outcome == "indeterminate" and res.reason == "isolated"


def test_without_precordial_leads_the_rule_cannot_be_applied():
    X, _ = twelve(alt=15)
    res = analyze_leads(X[:2], 500, ["ECG1", "ECG2"])
    assert res.outcome == "indeterminate" and res.reason == "no_precordial"


def test_a_flat_limb_lead_is_reported_and_leaves_the_precordial_verdict_alone():
    X, _ = twelve(alt=0, hr=108)
    X[3] = 0.0                                        # aVR disconnected
    res = analyze_leads(X, 500, NAMES)
    assert set(res.leads) == set(NAMES)
    assert res.leads["aVR"]["outcome"] == "indeterminate"
    assert res.outcome == "negative"


def test_a_ten_second_ecg_is_not_analysable_rather_than_missing_leads():
    _, X, _ = generate_12lead(SynthConfig(fs=500, n_beats=12, heart_rate=72, noise_uv=8, seed=5))
    res = analyze_leads(X.T, 500, NAMES)
    assert res.outcome == "indeterminate" and res.reason == "not_analysable"
    assert all("error" in v for v in res.leads.values())
