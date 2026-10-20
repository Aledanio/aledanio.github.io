// Service worker de Aledaño: funciona sin conexión una vez cargado.
// La página y los datos se piden primero a la red (para recibir días nuevos y
// cambios del juego) y, si no hay conexión, se sirven desde la caché.
// Los iconos y el logo salen directamente de la caché.

const CACHE = "aledanio-v1";
const BASICOS = [
  "./",
  "index.html",
  "logo.svg",
  "manifest.webmanifest",
  "icons/favicon.svg",
  "icons/favicon-32.png",
  "icons/favicon-16.png",
  "icons/favicon.ico",
  "icons/apple-touch-icon.png",
  "icons/icon-192.png",
  "icons/icon-512.png",
  "icons/icon-maskable-192.png",
  "icons/icon-maskable-512.png",
];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(BASICOS)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys()
      .then((claves) => Promise.all(claves.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

async function primeroRed(peticion) {
  const cache = await caches.open(CACHE);
  try {
    const respuesta = await fetch(peticion);
    if (respuesta.ok) cache.put(peticion, respuesta.clone());
    return respuesta;
  } catch (error) {
    const guardada = await cache.match(peticion, { ignoreSearch: peticion.mode === "navigate" });
    if (guardada) return guardada;
    throw error;
  }
}

async function primeroCache(peticion) {
  const guardada = await caches.match(peticion);
  if (guardada) return guardada;
  const respuesta = await fetch(peticion);
  if (respuesta.ok) (await caches.open(CACHE)).put(peticion, respuesta.clone());
  return respuesta;
}

self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.origin !== location.origin) return;
  const estatico = url.pathname.includes("/icons/") || url.pathname.endsWith("/logo.svg");
  e.respondWith(estatico ? primeroCache(e.request) : primeroRed(e.request));
});
