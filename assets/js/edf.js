/* File readers for the browser, no dependencies.
   parseEDF  European Data Format: EDF, EDF+ (continuous) and BDF/BDF+ (24-bit), per
             https://www.edfplus.info/specs/ — header, channel list, one channel decoded
             to physical values in mV.  EDF+D (discontinuous) is refused: beats on either
             side of a gap would be counted as neighbours and break the ABAB order.
   parseCSV  one numeric column of a CSV/TXT file, with a decimal comma when the
             columns are separated by ";" or tabs (the usual Russian/European export).
   leadFromLabel  the standard lead a channel label names, if any.
   anonymizeEDF   a de-identified copy of an EDF/BDF file (header fields and annotation text). */
(function (root) {
  "use strict";

  function ascii(bytes, off, len) {
    let s = "";
    for (let i = 0; i < len; i++) s += String.fromCharCode(bytes[off + i]);
    return s.trim();
  }

  // factor from the channel's physical dimension to mV; null if the unit is not a voltage we know
  function unitToMv(unit) {
    const u = (unit || "").trim().toLowerCase().replace(/[µμ]/g, "u");
    if (u === "uv") return 1e-3;
    if (u === "mv") return 1;
    if (u === "v") return 1e3;
    if (u === "nv") return 1e-6;
    return null;
  }

  function parseEDF(buf) {
    const bytes = new Uint8Array(buf);
    if (bytes.length < 256) throw new Error("File is too small to be an EDF file.");
    const bdf = bytes[0] === 0xff && ascii(bytes, 1, 7) === "BIOSEMI";
    if (!bdf && ascii(bytes, 0, 8) !== "0") throw new Error("Not an EDF/EDF+ or BDF file (bad version field).");
    const bytesPerSample = bdf ? 3 : 2;

    const nBytesHeader = parseInt(ascii(bytes, 184, 8), 10);
    const reserved = ascii(bytes, 192, 44);
    const nRecords = parseInt(ascii(bytes, 236, 8), 10);
    const recordDuration = parseFloat(ascii(bytes, 244, 8)) || 1;
    const ns = parseInt(ascii(bytes, 252, 4), 10);
    if (!ns || ns < 1 || !isFinite(nBytesHeader)) throw new Error("Could not read the EDF header (unexpected format).");
    if (/^(EDF|BDF)\+D/.test(reserved))
      throw Object.assign(new Error("This is a discontinuous recording (EDF+D): it has gaps in time, and alternans cannot be measured across a gap. Export one continuous segment."), { code: "discontinuous" });

    let p = 256;
    const field = (len) => { const vals = []; for (let i = 0; i < ns; i++) { vals.push(ascii(bytes, p, len)); p += len; } return vals; };
    const labels = field(16);
    field(80); // transducer type, unused
    const units = field(8);
    const physMin = field(8).map(parseFloat);
    const physMax = field(8).map(parseFloat);
    const digMin = field(8).map(parseFloat);
    const digMax = field(8).map(parseFloat);
    field(80); // prefiltering, unused
    const samplesPerRecord = field(8).map((v) => parseInt(v, 10));

    const signals = labels.map((label, i) => ({
      label,
      unit: units[i],
      unitToMv: unitToMv(units[i]),
      fs: samplesPerRecord[i] / recordDuration,
      physMin: physMin[i], physMax: physMax[i], digMin: digMin[i], digMax: digMax[i],
      samplesPerRecord: samplesPerRecord[i],
      annotations: /annotation/i.test(label),
    }));

    // byte offset, within one data record, where each signal's samples start
    const recordOffsets = [];
    let acc = 0;
    for (const s of signals) { recordOffsets.push(acc); acc += s.samplesPerRecord * bytesPerSample; }
    const recordBytes = acc;
    const dataStart = nBytesHeader;
    const available = Math.floor((bytes.length - dataStart) / recordBytes);
    const nRec = nRecords > 0 ? Math.min(nRecords, available) : available;

    // Physical values converted to mV. A channel whose unit is not a voltage we recognise
    // is read as if it were already in mV; signals[i].unitToMv === null tells the caller.
    // from/to (sample indices, to exclusive) read only part of the channel, so that a long
    // multi-lead recording never has to sit in memory whole.
    function getChannel(index, from, to) {
      const s = signals[index];
      if (!s || s.samplesPerRecord <= 0) throw new Error("Channel has no samples.");
      const scale = (s.digMax - s.digMin) !== 0 ? (s.physMax - s.physMin) / (s.digMax - s.digMin) : 1;
      const toMv = s.unitToMv === null ? 1 : s.unitToMv;
      const spr = s.samplesPerRecord, total = nRec * spr;
      const a = Math.max(0, from || 0), b = Math.min(total, to == null ? total : to);
      const out = new Float64Array(Math.max(0, b - a));
      for (let n = a; n < b; n++) {
        const r = Math.floor(n / spr);
        const off = dataStart + r * recordBytes + recordOffsets[index] + (n - r * spr) * bytesPerSample;
        let digital;
        if (bdf) {
          digital = bytes[off] | (bytes[off + 1] << 8) | (bytes[off + 2] << 16);
          if (digital & 0x800000) digital -= 0x1000000;       // sign-extend 24-bit
        } else {
          digital = (bytes[off] | (bytes[off + 1] << 8)) << 16 >> 16;
        }
        out[n - a] = (s.physMin + (digital - s.digMin) * scale) * toMv;
      }
      return out;
    }

    return { format: bdf ? "BDF" : "EDF", signals, nRecords: nRec, recordDuration, getChannel };
  }

  // De-identified copy of an EDF/EDF+ or BDF/BDF+ file, the signal untouched (port of
  // cardioonco/edf.py; byte-identical output). Patient field -> "X X X X", recording field ->
  // "Startdate X X X X" (EDF+/BDF+) or "X", start date and time -> 01.01.85 00.00.00, annotation
  // text cleared except the empty time-keeping annotation that opens each data record.
  function countTexts(region) {
    let n = 0, start = 0;
    for (let i = 0; i <= region.length; i++) {
      if (i < region.length && region[i] !== 0) continue;
      const tal = region.subarray(start, i), parts = [];
      let p0 = 0;
      for (let j = 0; j <= tal.length; j++) if (j === tal.length || tal[j] === 20) { parts.push(tal.length ? j - p0 : 0); p0 = j + 1; }
      const timed = tal.length && (tal[0] === 43 || tal[0] === 45);     // "+" or "-": the TAL starts with its onset
      const texts = timed ? parts.slice(1, -1) : parts.slice(0, -1);
      n += texts.filter((len) => len > 0).length;
      start = i + 1;
    }
    return n;
  }
  function anonymizeEDF(buf) {
    const b = new Uint8Array(buf.slice(0));
    const bdf = b[0] === 0xff && ascii(b, 1, 7) === "BIOSEMI";
    if (!bdf && ascii(b, 0, 8) !== "0") throw new Error("Not an EDF/EDF+ or BDF file (bad version field).");
    const plus = /^(EDF|BDF)\+/.test(ascii(b, 192, 44));
    const removed = { patient: ascii(b, 8, 80), recording: ascii(b, 88, 80), startdate: ascii(b, 168, 8),
                      starttime: ascii(b, 176, 8), annotations: 0 };
    const put = (off, n, text) => { const t = text.padEnd(n).slice(0, n); for (let i = 0; i < n; i++) b[off + i] = t.charCodeAt(i); };
    put(8, 80, "X X X X"); put(88, 80, plus ? "Startdate X X X X" : "X"); put(168, 8, "01.01.85"); put(176, 8, "00.00.00");
    const header = parseInt(ascii(b, 184, 8), 10), ns = parseInt(ascii(b, 252, 4), 10);
    const labels = Array.from({ length: ns }, (_, i) => ascii(b, 256 + 16 * i, 16));
    const sprOff = 256 + ns * (16 + 80 + 8 + 8 + 8 + 8 + 8 + 80);
    const spr = Array.from({ length: ns }, (_, i) => parseInt(ascii(b, sprOff + 8 * i, 8), 10));
    const bps = bdf ? 3 : 2, record = spr.reduce((x, y) => x + y, 0) * bps;
    const nRec = record ? Math.floor((b.length - header) / record) : 0;
    const starts = spr.map((_, i) => spr.slice(0, i).reduce((x, y) => x + y, 0) * bps);
    const annot = labels.map((l, i) => (l === "EDF Annotations" || l === "BDF Annotations" ? i : -1)).filter((i) => i >= 0);
    for (let r = 0; r < nRec; r++) {
      annot.forEach((i, k) => {
        const a = header + r * record + starts[i], z = a + spr[i] * bps;
        let keep = a;
        if (k === 0) {                       // keep "+onset[\x15duration]\x14\x14", the time-keeping TAL
          const first = b.subarray(a, z).indexOf(20);
          const second = first >= 0 ? b.subarray(a + first + 1, z).indexOf(20) : -1;
          keep = second >= 0 ? a + first + 1 + second + 1 : a;
        }
        removed.annotations += countTexts(b.subarray(keep, z));
        b.fill(0, keep, z);
      });
    }
    return { buffer: b.buffer, removed };
  }

  // One numeric column; header and other non-numeric lines are skipped.
  // "0,123;0,456" and "0,123<TAB>0,456" use a decimal comma; "0.123,0.456" uses a comma separator.
  function parseCSV(text, col = 0) {
    const out = [];
    for (const raw of text.split(/\r?\n/)) {
      const line = raw.trim();
      if (!line) continue;
      let fields;
      if (/[;\t]/.test(line)) fields = line.split(/\s*[;\t]\s*/).map((f) => f.replace(",", "."));
      else if (/^-?\d+,\d+(\s+-?\d+,\d+)*$/.test(line)) fields = line.split(/\s+/).map((f) => f.replace(",", "."));
      else fields = line.split(/[,\s]+/);
      const f = fields[col];
      if (f === undefined || !/^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/.test(f)) continue;
      out.push(parseFloat(f));
    }
    return Float64Array.from(out);
  }

  // Standard lead name from a channel label ("ECG II", "EKG V5", "Lead aVL", "v3") or "" if
  // the label does not name one of the twelve leads (e.g. "ECG1", "MLII", "Resp").
  function leadFromLabel(label) {
    const s = String(label || "").trim().replace(/^(ecg|ekg|экг|lead|отв\.?)[\s_:-]*/i, "");
    const m = s.match(/^(I{1,3}|aVR|aVL|aVF|V[1-6])$/i);
    if (!m) return "";
    const t = m[1].toUpperCase();
    return t.startsWith("AV") ? "a" + t.slice(1) : t;
  }

  root.CardioEDF = { parseEDF, parseCSV, unitToMv, leadFromLabel, anonymizeEDF };
  if (typeof module !== "undefined") module.exports = root.CardioEDF;
})(typeof window !== "undefined" ? window : globalThis);
