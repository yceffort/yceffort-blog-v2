const CACHE_VERSION = 'v4'
const STATIC_CACHE = `static-${CACHE_VERSION}`
const PAGES_CACHE = `pages-${CACHE_VERSION}`
const IMAGES_CACHE = `images-${CACHE_VERSION}`
const RSC_CACHE = `rsc-${CACHE_VERSION}`

const OFFLINE_URL = '/offline'
const PRECACHE_URLS = [OFFLINE_URL]
const ALL_CACHES = [STATIC_CACHE, PAGES_CACHE, IMAGES_CACHE, RSC_CACHE]

// 배포마다 ?dpl= 쿼리가 바뀌어 엔트리가 계속 쌓이므로 상한을 두고
// 오래된 것부터 정리한다 (Cache API의 keys()는 삽입 순서를 보장한다)
const MAX_ENTRIES = {
  [STATIC_CACHE]: 500,
}

const FONT_HOSTS = ['fonts.gstatic.com', 'cdn.jsdelivr.net']
const SKIP_HOSTS = [
  'google-analytics.com',
  'analytics.google.com',
  'googletagmanager.com',
  'vercel-insights.com',
  'va.vercel-scripts.com',
]

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches
      .open(PAGES_CACHE)
      .then((cache) => cache.addAll(PRECACHE_URLS))
      .then(() => self.skipWaiting()),
  )
})

self.addEventListener('activate', (event) => {
  event.waitUntil(
    Promise.all([
      // 내비게이션 요청을 워커 기동과 병렬로 출발시켜 콜드 기동이
      // TTFB에 직렬로 얹히지 않게 한다 (미지원 브라우저는 그냥 넘어간다)
      self.registration.navigationPreload?.enable(),
      caches
        .keys()
        .then((keys) =>
          Promise.all(
            keys
              .filter((key) => !ALL_CACHES.includes(key))
              .map((key) => caches.delete(key)),
          ),
        ),
    ]).then(() => self.clients.claim()),
  )
})

function isStaticAsset(url) {
  return url.pathname.startsWith('/_next/static/')
}

function isFontRequest(url) {
  return /\.woff2?$/.test(url.pathname)
}

function isImageRequest(url) {
  return (
    url.pathname === '/_next/image' ||
    // 코드 생성 썸네일 (?v= 파라미터가 캐시 버스팅을 담당한다)
    url.pathname.startsWith('/api/og/') ||
    /\.(?:png|jpe?g|gif|webp|avif|svg|ico)$/.test(url.pathname)
  )
}

// App Router의 소프트 내비게이션/프리페치는 ?_rsc= 쿼리가 붙은 fetch로 나간다
function isRSCRequest(request, url) {
  return url.searchParams.has('_rsc') || request.headers.get('rsc') === '1'
}

// 오프라인에서 srcset이 캐시에 없는 너비를 요청하면, 같은 원본의
// 캐시된 다른 변형이라도 찾아서 돌려준다
async function matchImageVariant(requestUrl) {
  const url = new URL(requestUrl)
  if (url.pathname !== '/_next/image') {
    return null
  }
  const source = url.searchParams.get('url')
  if (!source) {
    return null
  }
  const cache = await caches.open(IMAGES_CACHE)
  for (const request of await cache.keys()) {
    const cachedUrl = new URL(request.url)
    if (
      cachedUrl.pathname === '/_next/image' &&
      cachedUrl.searchParams.get('url') === source
    ) {
      return cache.match(request)
    }
  }
  return null
}

async function putWithTrim(cacheName, request, response) {
  const cache = await caches.open(cacheName)
  await cache.put(request, response)
  const max = MAX_ENTRIES[cacheName]
  const keys = await cache.keys()
  if (max && keys.length > max) {
    await Promise.all(
      keys.slice(0, keys.length - max).map((key) => cache.delete(key)),
    )
  }
}

async function cacheFirst(event, cacheName) {
  const cache = await caches.open(cacheName)
  const cached = await cache.match(event.request)
  if (cached) {
    return cached
  }

  try {
    const response = await fetch(event.request)
    if (response.ok) {
      event.waitUntil(putWithTrim(cacheName, event.request, response.clone()))
    }
    return response
  } catch {
    return new Response('', {status: 408})
  }
}

async function networkFirst(event, cacheName) {
  const {request} = event
  try {
    // 내비게이션이면 preload로 먼저 출발한 응답을 쓴다. 내비게이션이
    // 아니거나 preload 미지원이면 undefined로 resolve되어 fetch로 간다
    return (await event.preloadResponse) ?? (await fetch(request))
  } catch {
    const cache = await caches.open(cacheName)
    const cached = await cache.match(request)
    if (cached) {
      return cached
    }

    if (request.mode === 'navigate') {
      const pages = await caches.open(PAGES_CACHE)
      const offline = await pages.match(OFFLINE_URL)
      if (offline) {
        return offline
      }
    }
    return new Response('', {status: 408})
  }
}

async function handleNavigation(event) {
  return networkFirst(event, PAGES_CACHE)
}

async function handleImage(event) {
  const cache = await caches.open(IMAGES_CACHE)
  const cached = await cache.match(event.request)
  if (cached) {
    return cached
  }

  try {
    return await fetch(event.request)
  } catch {
    const fallback = await matchImageVariant(event.request.url)
    if (fallback) {
      return fallback
    }
    return new Response('', {status: 408})
  }
}

// 오프라인에서 캐시에 없는 RSC 요청에 503을 돌려주면 Next.js 라우터가
// MPA 내비게이션으로 폴백하고, 그 문서 요청은 networkFirst가
// 캐시된 HTML 또는 오프라인 페이지로 응답한다
async function handleRSC(event) {
  const {request} = event
  try {
    return await fetch(request)
  } catch {
    const cache = await caches.open(RSC_CACHE)
    const cached = await cache.match(request)
    if (cached) {
      return cached
    }
    return new Response('', {status: 503})
  }
}

self.addEventListener('fetch', (event) => {
  const {request} = event

  if (request.method !== 'GET') {
    return
  }

  const url = new URL(request.url)
  if (url.protocol !== 'https:' && url.protocol !== 'http:') {
    return
  }

  if (url.origin !== self.location.origin) {
    if (SKIP_HOSTS.some((host) => url.hostname.includes(host))) {
      return
    }
    if (FONT_HOSTS.includes(url.hostname)) {
      event.respondWith(cacheFirst(event, STATIC_CACHE))
      return
    }
    // 포스트 본문 이미지는 대부분 외부 도메인이라 함께 캐시한다
    if (request.destination === 'image') {
      event.respondWith(handleImage(event))
    }
    return
  }

  if (url.pathname.startsWith('/api/') && !isImageRequest(url)) {
    return
  }

  // Content-hashed static assets and fonts: cache forever
  if (isStaticAsset(url) || isFontRequest(url)) {
    event.respondWith(cacheFirst(event, STATIC_CACHE))
    return
  }

  // Page navigations: network-first, offline fallback
  if (request.mode === 'navigate') {
    event.respondWith(handleNavigation(event))
    return
  }

  if (isRSCRequest(request, url)) {
    event.respondWith(handleRSC(event))
    return
  }

  // Images (including the /_next/image optimizer): cache-first
  if (isImageRequest(url)) {
    event.respondWith(handleImage(event))
    return
  }

  // Other same-origin _next data: network-first
  if (url.pathname.startsWith('/_next/')) {
    event.respondWith(networkFirst(event, PAGES_CACHE))
  }
})

self.addEventListener('push', (event) => {
  if (!event.data) {
    return
  }
  let payload
  try {
    payload = event.data.json()
  } catch {
    return
  }
  event.waitUntil(
    self.registration.showNotification(payload.title || 'yceffort', {
      body: payload.body || '',
      icon: '/favicon/web-app-manifest-192x192.png',
      badge: '/favicon/favicon-96x96.png',
      data: {url: payload.url || '/'},
      // 같은 글의 알림이 중복으로 오면 하나로 병합한다
      tag: payload.url || 'yceffort',
    }),
  )
})

self.addEventListener('notificationclick', (event) => {
  event.notification.close()
  // 상대 URL은 일부 플랫폼(iOS PWA 등)의 openWindow에서 조용히 실패한다
  const url = new URL(event.notification.data?.url || '/', self.location.origin)
    .href
  // 알림 URL에는 utm 꼬리표가 붙으므로, 이미 열린 창을 찾을 때는 경로만 비교한다
  const samePage = (href) => new URL(href).pathname === new URL(url).pathname
  event.waitUntil(
    self.clients
      .matchAll({type: 'window', includeUncontrolled: true})
      .then((clientList) => {
        for (const client of clientList) {
          if (samePage(client.url) && 'focus' in client) {
            return client.focus()
          }
        }
        return self.clients.openWindow(url).catch(() => {
          // openWindow가 막히면 이미 떠 있는 창을 글로 이동시켜 재사용한다
          const client = clientList.find((c) => 'navigate' in c)
          return client?.navigate(url).then((c) => c?.focus())
        })
      }),
  )
})
