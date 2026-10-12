/* CardioOncoPredict — runs the analyses of CardioDSP off the page's main thread:
   analyzeTWA on one lead, analyzeLeads on all the standard leads, or the QT of their 10-s segments. */
importScripts("dsp.js");
self.onmessage = (e) => {
  const { id, kind, x, leads, names, segments, starts, fs } = e.data;
  try {
    const D = self.CardioDSP;
    const r = kind === "qt" ? D.measureQTOnSegments(segments, starts, fs)
      : kind === "leads" ? D.analyzeLeads(leads, names, fs) : D.analyzeTWA(x, fs);
    self.postMessage({ id, ok: true, r });
  } catch (err) {
    self.postMessage({ id, ok: false, err: { code: err.code, message: err.message, nBeats: err.nBeats, minBeats: err.minBeats } });
  }
};
