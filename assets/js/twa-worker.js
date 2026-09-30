/* CardioOncoPredict — runs CardioDSP.analyzeTWA off the page's main thread. */
importScripts("dsp.js");
self.onmessage = (e) => {
  const { id, x, fs } = e.data;
  try {
    self.postMessage({ id, ok: true, r: self.CardioDSP.analyzeTWA(x, fs) });
  } catch (err) {
    self.postMessage({ id, ok: false, err: { code: err.code, message: err.message, nBeats: err.nBeats, minBeats: err.minBeats } });
  }
};
