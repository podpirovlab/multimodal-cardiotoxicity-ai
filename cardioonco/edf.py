"""De-identify an EDF/EDF+ or BDF/BDF+ file without touching the signal (the same as anonymizeEDF
in assets/js/edf.js; the two give byte-identical files).

What is replaced, following the EDF+ specification (https://www.edfplus.info/specs/edfplus.html):
  * local patient identification  -> "X X X X"            (code, sex, birthdate, name unknown)
  * local recording identification -> "Startdate X X X X"  for EDF+/BDF+, "X" for plain EDF/BDF
  * startdate.starttime            -> 01.01.85 00.00.00    (the earliest date the field can hold;
                                                            it cannot be left empty)
  * annotation text in "EDF/BDF Annotations" signals is cleared; the empty time-keeping
    annotation that opens each data record is kept, so the file stays valid EDF+.
The waveform itself is not changed and can still be personal data: an ECG is a biometric signal.
"""
from __future__ import annotations

PATIENT_X = "X X X X"
RECORDING_X_PLUS = "Startdate X X X X"
RECORDING_X = "X"
DATE_X, TIME_X = "01.01.85", "00.00.00"


def _field(b: bytearray, off: int, n: int) -> str:
    return bytes(b[off:off + n]).decode("latin-1").strip()


def _put(b: bytearray, off: int, n: int, text: str) -> None:
    b[off:off + n] = text.ljust(n)[:n].encode("ascii")


def _count_texts(region: bytes) -> int:
    r"""Non-empty annotation texts in a run of TALs ("+onset[\x15dur]\x14text\x14...\x14\x00");
    the run may start inside a TAL, right after its onset, as after the time-keeping annotation."""
    n = 0
    for tal in region.split(b"\x00"):
        parts = tal.split(b"\x14")
        texts = parts[1:-1] if tal.startswith(b"+") or tal.startswith(b"-") else parts[:-1]
        n += sum(1 for t in texts if t)
    return n


def anonymize_edf(data: bytes) -> tuple[bytes, dict]:
    """Return the de-identified file and what was removed: {"patient", "recording", "startdate",
    "starttime", "annotations"} (the original header text and the number of annotation texts cleared)."""
    b = bytearray(data)
    bdf = b[0] == 0xFF and bytes(b[1:8]) == b"BIOSEMI"
    if not bdf and _field(b, 0, 8) != "0":
        raise ValueError("not an EDF/EDF+ or BDF file")
    plus = _field(b, 192, 44).startswith(("EDF+", "BDF+"))
    removed = {"patient": _field(b, 8, 80), "recording": _field(b, 88, 80),
               "startdate": _field(b, 168, 8), "starttime": _field(b, 176, 8), "annotations": 0}
    _put(b, 8, 80, PATIENT_X)
    _put(b, 88, 80, RECORDING_X_PLUS if plus else RECORDING_X)
    _put(b, 168, 8, DATE_X)
    _put(b, 176, 8, TIME_X)

    header = int(_field(b, 184, 8))
    ns = int(_field(b, 252, 4))
    labels = [_field(b, 256 + 16 * i, 16) for i in range(ns)]
    spr_off = 256 + ns * (16 + 80 + 8 + 8 + 8 + 8 + 8 + 80)
    spr = [int(_field(b, spr_off + 8 * i, 8)) for i in range(ns)]
    bps = 3 if bdf else 2
    record = sum(spr) * bps
    n_rec = (len(b) - header) // record if record else 0
    starts = [sum(spr[:i]) * bps for i in range(ns)]
    annot = [i for i, lab in enumerate(labels) if lab in ("EDF Annotations", "BDF Annotations")]
    for r in range(n_rec):
        for k, i in enumerate(annot):
            a = header + r * record + starts[i]
            z = a + spr[i] * bps
            keep = a
            if k == 0:                      # keep "+onset[\x15duration]\x14\x14", the time-keeping TAL
                first = b.find(b"\x14", a, z)
                second = b.find(b"\x14", first + 1, z) if first >= 0 else -1
                keep = second + 1 if second >= 0 else a
            removed["annotations"] += _count_texts(bytes(b[keep:z]))
            b[keep:z] = bytes(z - keep)
    return bytes(b), removed
