/* Minimal European Data Format (EDF/EDF+) reader, browser-side, no dependencies.
   Spec: https://www.edfplus.info/specs/edf.html
   Reads the header, exposes each signal's label/unit/sample rate, and decodes
   one chosen channel to a Float64Array of physical values (converted to mV). */
(function () {
  "use strict";

  function ascii(bytes, off, len) {
    let s = "";
    for (let i = 0; i < len; i++) s += String.fromCharCode(bytes[off + i]);
    return s.trim();
  }

  function parseEDF(buf) {
    const bytes = new Uint8Array(buf);
    if (bytes.length < 256) throw new Error("File is too small to be an EDF file.");
    const version = ascii(bytes, 0, 8);
    if (version !== "0") throw new Error("Not an EDF/EDF+ file (bad version field).");

    const nBytesHeader = parseInt(ascii(bytes, 184, 8), 10);
    const nRecords = parseInt(ascii(bytes, 236, 8), 10);
    const recordDuration = parseFloat(ascii(bytes, 244, 8)) || 1;
    const ns = parseInt(ascii(bytes, 252, 4), 10);
    if (!ns || ns < 1 || !isFinite(nBytesHeader)) throw new Error("Could not read the EDF header (unexpected format).");

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
    field(32); // reserved, unused

    const signals = labels.map((label, i) => ({
      label,
      unit: units[i],
      fs: samplesPerRecord[i] / recordDuration,
      physMin: physMin[i], physMax: physMax[i], digMin: digMin[i], digMax: digMax[i],
      samplesPerRecord: samplesPerRecord[i],
      annotations: /annotation/i.test(label),
    }));

    // byte offset, within one data record, where each signal's samples start
    const recordOffsets = [];
    let acc = 0;
    for (const s of signals) { recordOffsets.push(acc); acc += s.samplesPerRecord * 2; }
    const recordBytes = acc;
    const dataStart = nBytesHeader;
    const nRec = nRecords > 0 ? nRecords : Math.floor((bytes.length - dataStart) / recordBytes);

    function getChannel(index) {
      const s = signals[index];
      if (!s || s.samplesPerRecord <= 0) throw new Error("Channel has no samples.");
      const scale = (s.digMax - s.digMin) !== 0 ? (s.physMax - s.physMin) / (s.digMax - s.digMin) : 1;
      const out = new Float64Array(nRec * s.samplesPerRecord);
      const view = new DataView(buf);
      let o = 0;
      for (let r = 0; r < nRec; r++) {
        let off = dataStart + r * recordBytes + recordOffsets[index];
        for (let k = 0; k < s.samplesPerRecord; k++) {
          const digital = view.getInt16(off, true);
          out[o++] = s.physMin + (digital - s.digMin) * scale;
          off += 2;
        }
      }
      // normalise to mV, the unit the rest of the pipeline expects
      const u = (s.unit || "").toLowerCase().replace(/µ/g, "u");
      const toMv = u.includes("uv") ? (1 / 1000) : u === "v" ? 1000 : 1; // "mv" or unrecognised: assume already mV
      if (toMv !== 1) for (let i = 0; i < out.length; i++) out[i] *= toMv;
      return out;
    }

    return { signals, nRecords: nRec, recordDuration, getChannel };
  }

  window.CardioEDF = { parseEDF };
})();
