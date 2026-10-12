"""HFA-ICOS anthracycline baseline risk (Lyon et al., Eur J Heart Fail 2020, Table 2)."""
import pytest

from cardioonco.hfa_icos import FACTORS, classify


@pytest.mark.parametrize("present,level,points", [
    (set(), "low", 0),
    ({"hypertension"}, "low", 1),                                               # one medium1
    ({"hypertension", "diabetes"}, "medium", 2),                                # two medium1
    ({"age_65_79"}, "medium", 2),                                               # a single medium2
    ({"age_65_79", "lvef_50_54"}, "medium", 4),
    ({"age_65_79", "hypertension", "diabetes", "smoking"}, "high", 5),          # medium factors, 5 points
    ({"prior_anthracycline"}, "high", 0),                                       # any high factor
    ({"age_80_plus", "obesity"}, "high", 1),
    ({"hf_cardiomyopathy", "hypertension"}, "very_high", 1),                    # any very high factor
])
def test_levels_follow_the_footnote_of_table_2(present, level, points):
    r = classify(present)
    assert (r.level, r.points) == (level, points)


def test_the_table_has_eighteen_factors_with_the_published_weights():
    weights = {f[0]: f[3] for f in FACTORS}
    assert len(FACTORS) == 18
    assert [k for k, w in weights.items() if w == "very_high"] == ["hf_cardiomyopathy"]
    assert sum(w == "high" for w in weights.values()) == 7
    assert sum(w == "medium2" for w in weights.values()) == 2
    assert sum(w == "medium1" for w in weights.values()) == 8


def test_contradictory_answers_are_refused():
    with pytest.raises(ValueError):
        classify({"age_80_plus", "age_65_79"})
    with pytest.raises(ValueError):
        classify({"lvef_below_50", "lvef_50_54"})
    with pytest.raises(ValueError):
        classify({"made_up"})


@pytest.mark.skipif(__import__("shutil").which("node") is None and not __import__("os").environ.get("CI"),
                    reason="node.js not installed")
def test_browser_gives_the_same_level_for_every_combination_of_factors():
    """All 2^18 combinations of answers: the same table, the same level and points (or the same refusal)."""
    import json
    import shutil
    import subprocess
    from pathlib import Path
    js_file = Path(__file__).resolve().parents[1] / "assets" / "js" / "hfa-icos.js"
    script = (f"const H=require({json.dumps(str(js_file))});const keys=H.FACTORS.map(f=>f[0]);const out=[];"
              "for(let m=0;m<(1<<keys.length);m++){const s=keys.filter((k,i)=>m&(1<<i));"
              "try{const r=H.classify(s);out.push(r.level[0]+r.points)}catch(e){out.push('x')}}"
              "console.log(JSON.stringify({table:H.FACTORS.map(f=>[f[0],f[1],f[2],f[3]]),out:out.join(',')}));")
    js = json.loads(subprocess.check_output([shutil.which("node"), "-e", script], text=True))
    assert js["table"] == [[k, w, e, label] for k, _, label, w, e in FACTORS]
    keys, out = [f[0] for f in FACTORS], []
    for m in range(1 << len(keys)):
        try:
            r = classify({k for i, k in enumerate(keys) if m >> i & 1})
            out.append(r.level[0] + str(r.points))
        except ValueError:
            out.append("x")
    assert js["out"] == ",".join(out)
