/**
 * Portadoc Browser Service Worker
 * Enables offline functionality and caches assets
 */

const CACHE_NAME = 'portadoc-v2';
const STATIC_ASSETS = [
  '/',
  '/index.html',
  '/manifest.json',
  '/icon-192.svg',
  '/icon-512.svg',
];

// Data files for sanitization and ranking
const DATA_ASSETS = [
  '/dictionaries/english.json',
  '/dictionaries/names.json',
  '/dictionaries/medical.json',
  '/dictionaries/custom.json',
  '/data/frequencies.json',
  '/data/bigrams.json',
  '/data/ocr_confusions.json',
];

// docTR model files (larger, cache separately)
const MODEL_ASSETS = [
  '/models/db_mobilenet_v2/model.json',
  '/models/crnn_mobilenet_v2/model.json',
];

// External resources to cache
const EXTERNAL_ASSETS = [
  // Tesseract.js core and worker
  'https://cdn.jsdelivr.net/npm/tesseract.js@5/dist/tesseract.min.js',
];

// Install event - cache static assets
self.addEventListener('install', (event) => {
  console.log('[SW] Installing service worker...');
  event.waitUntil(
    caches.open(CACHE_NAME).then(async (cache) => {
      console.log('[SW] Caching static assets');

      // Cache static assets (required)
      try {
        await cache.addAll(STATIC_ASSETS);
      } catch (err) {
        console.warn('[SW] Failed to cache static assets:', err);
      }

      // Cache data files (dictionaries, frequencies) - important for functionality
      try {
        await cache.addAll(DATA_ASSETS);
        console.log('[SW] Cached data assets');
      } catch (err) {
        console.warn('[SW] Failed to cache data assets:', err);
      }

      // Cache model files (optional, large files)
      try {
        await cache.addAll(MODEL_ASSETS);
        console.log('[SW] Cached model assets');
      } catch (err) {
        console.warn('[SW] Failed to cache model assets (expected in dev):', err);
      }
    })
  );
  // Take over immediately
  self.skipWaiting();
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
  console.log('[SW] Activating service worker...');
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames
          .filter((name) => name !== CACHE_NAME)
          .map((name) => {
            console.log('[SW] Deleting old cache:', name);
            return caches.delete(name);
          })
      );
    })
  );
  // Take control of all clients
  self.clients.claim();
});

// Fetch event - serve from cache, fallback to network
self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // Skip non-GET requests
  if (event.request.method !== 'GET') {
    return;
  }

  // Skip chrome-extension and other non-http(s) requests
  if (!url.protocol.startsWith('http')) {
    return;
  }

  // Network-first for API calls and dynamic content
  if (url.pathname.startsWith('/api/') || url.pathname.includes('.json')) {
    event.respondWith(networkFirst(event.request));
    return;
  }

  // Cache-first for static assets
  event.respondWith(cacheFirst(event.request));
});

// Cache-first strategy
async function cacheFirst(request) {
  const cachedResponse = await caches.match(request);
  if (cachedResponse) {
    return cachedResponse;
  }

  try {
    const networkResponse = await fetch(request);

    // Cache successful responses
    if (networkResponse.ok) {
      const cache = await caches.open(CACHE_NAME);
      cache.put(request, networkResponse.clone());
    }

    return networkResponse;
  } catch (error) {
    console.warn('[SW] Network request failed:', error);
    // Return offline page if available
    return caches.match('/offline.html');
  }
}

// Network-first strategy
async function networkFirst(request) {
  try {
    const networkResponse = await fetch(request);

    // Cache successful responses
    if (networkResponse.ok) {
      const cache = await caches.open(CACHE_NAME);
      cache.put(request, networkResponse.clone());
    }

    return networkResponse;
  } catch (error) {
    console.warn('[SW] Network request failed, trying cache:', error);
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
      return cachedResponse;
    }
    throw error;
  }
}

// Handle messages from the main thread
self.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }
});
