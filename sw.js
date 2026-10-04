/* CardioOncoPredict service worker: the tool keeps working offline after the first visit.
   Pages: network first, so a new version shows as soon as there is a connection.
   Other files: served from the cache and refreshed in the background.
   Nothing the user loads (their ECG files) ever passes through here: files are read locally. */
const VERSION = "cardioonco-1.1.1";
const CORE = [
  "./", "index.html", "ru.html", "manifest.webmanifest", "manifest-ru.webmanifest",
  "assets/css/site.css", "assets/css/fonts.css",
  "assets/js/dsp.js", "assets/js/edf.js", "assets/js/site.js", "assets/js/twa-worker.js",
  "assets/fonts/fraunces-var-latin.woff2", "assets/fonts/literata-var-cyrillic.woff2",
  "assets/fonts/pt-sans-400-latin.woff2", "assets/fonts/pt-sans-400-cyrillic.woff2",
  "assets/fonts/pt-sans-700-latin.woff2", "assets/fonts/pt-sans-700-cyrillic.woff2",
  "assets/fonts/pt-serif-400-latin.woff2", "assets/fonts/pt-serif-400-cyrillic.woff2",
  "assets/fonts/pt-serif-700-latin.woff2", "assets/fonts/pt-serif-700-cyrillic.woff2",
  "assets/fonts/pt-mono-400-latin.woff2", "assets/fonts/pt-mono-400-cyrillic.woff2",
  "assets/icons/icon-192.png", "assets/icons/icon-512.png",
];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(VERSION).then((c) => c.addAll(CORE)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys()
    .then((keys) => Promise.all(keys.filter((k) => k !== VERSION).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});

self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET" || new URL(req.url).origin !== self.location.origin) return;
  if (req.mode === "navigate") {
    e.respondWith(fetch(req)
      .then((res) => { const copy = res.clone(); caches.open(VERSION).then((c) => c.put(req, copy)); return res; })
      .catch(() => caches.match(req).then((hit) => hit || caches.match("index.html"))));
    return;
  }
  e.respondWith(caches.open(VERSION).then((c) => c.match(req).then((hit) => {
    const fresh = fetch(req).then((res) => { if (res.ok) c.put(req, res.clone()); return res; }).catch(() => hit);
    return hit || fresh;
  })));
});
