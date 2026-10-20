import {isPostPath} from './paths'

export type OfflineDeps = {
  caches: CacheStorage
  fetch: typeof fetch
  storage: Storage | null
  origin: string
  saveData: boolean
  now: () => number
}

const PAGES_CACHE = 'pages-v4'
const IMAGES_CACHE = 'images-v4'
const MAX_PAGES = 200
const MAX_IMAGES = 300
const MAX_IMAGES_PER_PAGE = 50

// HTML에서 오프라인 저장 대상 URL을 추출한다. 이미지는 img의 src/srcset과
// video의 poster, 문서는 같은 오리진 .html을 가리키는 iframe의 src다.
// 포스트 본문 이미지는 대부분 외부 도메인이므로 절대 URL도 포함한다.
// next/image의 srcset은 원본당 여러 너비를 포함하므로, 원본(url 파라미터)별로
// 1080에 가장 가까운 하나만 고른다
export function extractAssetUrls(html: string): {
  images: string[]
  documents: string[]
} {
  const found = new Set<string>()
  const documents = new Set<string>()
  const tagRe = /<(img|video|iframe)\b[^>]*>/gi
  // Next.js는 srcSet으로 렌더하므로 대소문자를 구분하면 srcset 후보를 통째로 놓친다
  const attrRes = {
    img: /(?:src|srcset)="([^"]+)"/gi,
    video: /poster="([^"]+)"/gi,
    iframe: /src="([^"]+)"/gi,
  }
  let tag: RegExpExecArray | null
  while ((tag = tagRe.exec(html))) {
    const name = tag[1].toLowerCase() as keyof typeof attrRes
    const attrRe = attrRes[name]
    let match: RegExpExecArray | null
    while ((match = attrRe.exec(tag[0]))) {
      for (const candidate of name === 'iframe'
        ? [match[1]]
        : match[1].split(',')) {
        const u = candidate.trim().split(' ')[0].replaceAll('&amp;', '&')
        if (name === 'iframe') {
          if (
            u.startsWith('/') &&
            !u.startsWith('//') &&
            u.split('?')[0].endsWith('.html')
          ) {
            documents.add(u)
          }
        } else if (
          u.startsWith('/_next/image?') ||
          u.startsWith('/api/og/') ||
          u.startsWith('https://') ||
          /^(?:\.\/|\/)[^\s"]+\.(?:png|jpe?g|gif|webp|avif|svg)$/.test(u)
        ) {
          found.add(u)
        }
      }
    }
    attrRe.lastIndex = 0
  }

  const bySource = new Map<string | null, {url: string; width: number}>()
  const plain: string[] = []
  for (const u of found) {
    if (!u.startsWith('/_next/image?')) {
      plain.push(u)
      continue
    }
    const params = new URLSearchParams(u.slice('/_next/image?'.length))
    const source = params.get('url')
    const width = Number(params.get('w')) || 0
    const current = bySource.get(source)
    if (!current || Math.abs(width - 1080) < Math.abs(current.width - 1080)) {
      bySource.set(source, {url: u, width})
    }
  }
  return {
    images: [...plain, ...[...bySource.values()].map((v) => v.url)],
    documents: [...documents],
  }
}

async function putWithTrim(
  cache: Cache,
  url: string,
  response: Response,
  max: number,
  keepUrl?: string,
) {
  await cache.put(url, response)
  const keys = (await cache.keys()).filter((key) => key.url !== keepUrl)
  if (keys.length > max) {
    await Promise.all(
      keys.slice(0, keys.length - max).map((key) => cache.delete(key)),
    )
  }
}

export type SavedSize = {bytes: number; external: number}

const NO_SIZE: SavedSize = {bytes: 0, external: 0}

const addSize = (a: SavedSize, b: SavedSize): SavedSize => ({
  bytes: a.bytes + b.bytes,
  external: a.external + b.external,
})

const sumSizes = (sizes: SavedSize[]) => sizes.reduce(addSize, NO_SIZE)

// 이미 캐시에 있는 자산도 이 글이 참조하므로 크기에 센다. 외부 이미지는 opaque
// 응답이라 바이트를 읽을 수 없어 장수로만 센다
async function saveImage(
  url: string,
  base: string,
  cache: Cache,
  deps: OfflineDeps,
): Promise<SavedSize> {
  try {
    const href = new URL(url, base).href
    let image = await cache.match(href)
    if (!image) {
      // 외부 이미지는 no-cors로 가져와 opaque 응답을 그대로 저장한다
      image = await deps.fetch(
        href,
        url.startsWith('https://') ? {mode: 'no-cors'} : undefined,
      )
      if (!image.ok && image.type !== 'opaque') {
        return NO_SIZE
      }
      await putWithTrim(cache, href, image.clone(), MAX_IMAGES)
    }
    return image.type === 'opaque'
      ? {bytes: 0, external: 1}
      : {bytes: (await image.blob()).size, external: 0}
  } catch {
    // 개별 자산 실패는 무시한다
    return NO_SIZE
  }
}

async function saveDocument(
  url: string,
  pageCache: Cache,
  imageCache: Cache,
  deps: OfflineDeps,
): Promise<SavedSize> {
  try {
    const href = new URL(url, deps.origin).href
    let response = await pageCache.match(href)
    if (!response) {
      response = await deps.fetch(href)
      if (!response.ok || response.redirected) {
        return NO_SIZE
      }
      await putWithTrim(
        pageCache,
        href,
        response.clone(),
        MAX_PAGES,
        new URL('/offline', deps.origin).href,
      )
    }
    const blob = await response.blob()
    // 문서 안의 ./ 상대 경로는 글이 아니라 문서 기준으로 해석해야 한다
    const {images} = extractAssetUrls(await blob.text())
    const sizes = await Promise.all(
      images.map((image) => saveImage(image, href, imageCache, deps)),
    )
    return addSize({bytes: blob.size, external: 0}, sumSizes(sizes))
  } catch {
    // 개별 자산 실패는 무시한다
    return NO_SIZE
  }
}

async function saveAssets(html: string, deps: OfflineDeps): Promise<SavedSize> {
  try {
    const {images, documents} = extractAssetUrls(html)
    const imageCache = await deps.caches.open(IMAGES_CACHE)
    const pageCache = await deps.caches.open(PAGES_CACHE)
    const targets = [
      ...documents.map((url) => ({url, isDocument: true})),
      ...images.map((url) => ({url, isDocument: false})),
    ].slice(0, MAX_IMAGES_PER_PAGE)
    return sumSizes(
      await Promise.all(
        targets.map(({url, isDocument}) =>
          isDocument
            ? saveDocument(url, pageCache, imageCache, deps)
            : saveImage(url, deps.origin, imageCache, deps),
        ),
      ),
    )
  } catch {
    // 본문을 읽지 못하면 자산 저장만 건너뛴다
    return NO_SIZE
  }
}

export async function savePage(
  path: string,
  options: {images: boolean},
  deps: OfflineDeps,
): Promise<SavedSize & {status: 'new' | 'updated' | 'failed'}> {
  const href = new URL(path, deps.origin).href
  try {
    const cache = await deps.caches.open(PAGES_CACHE)
    const existed = Boolean(await cache.match(href))
    const response = await deps.fetch(href)
    // 리다이렉트된 응답은 내비게이션 응답으로 쓰면 네트워크 오류가 되어
    // 오프라인에서 열리지 않는다
    if (!response.ok || response.redirected) {
      return {status: 'failed', ...NO_SIZE}
    }
    await putWithTrim(
      cache,
      href,
      response.clone(),
      MAX_PAGES,
      new URL('/offline', deps.origin).href,
    )
    const blob = await response.blob()
    const assets = options.images
      ? await saveAssets(await blob.text(), deps)
      : NO_SIZE
    return {
      status: existed ? 'updated' : 'new',
      bytes: blob.size + assets.bytes,
      external: assets.external,
    }
  } catch {
    return {status: 'failed', ...NO_SIZE}
  }
}

// 워커가 오프라인 내비게이션에서 이 응답을 돌려주면 브라우저가 리다이렉트를
// 재생해 캐시된 목적지로 간다
export async function saveRedirect(path: string, deps: OfflineDeps) {
  const href = new URL(path, deps.origin).href
  try {
    const response = await deps.fetch(href, {redirect: 'manual'})
    if (response.type === 'opaqueredirect') {
      const cache = await deps.caches.open(PAGES_CACHE)
      await cache.put(href, response)
    }
  } catch {
    // 리다이렉트 저장 실패는 무시한다
  }
}

export async function hasPage(path: string, deps: OfflineDeps) {
  const cache = await deps.caches.open(PAGES_CACHE)
  return Boolean(await cache.match(new URL(path, deps.origin).href))
}

export async function deletePage(path: string, deps: OfflineDeps) {
  const cache = await deps.caches.open(PAGES_CACHE)
  await cache.delete(new URL(path, deps.origin).href)
}

// React가 텍스트에서 이스케이프하는 다섯 가지만 푼다. &amp;를 마지막에 풀어야
// 원문의 &amp;lt; 같은 문자열이 두 번 풀리지 않는다
const decodeTitle = (title: string) =>
  title
    .replaceAll('&lt;', '<')
    .replaceAll('&gt;', '>')
    .replaceAll('&quot;', '"')
    .replaceAll('&#x27;', "'")
    .replaceAll('&amp;', '&')

async function savedPostKeys(cache: Cache) {
  return (await cache.keys()).filter((key) => {
    const url = new URL(key.url)
    return !url.search && isPostPath(url.pathname)
  })
}

export async function listSavedPosts(
  deps: OfflineDeps,
): Promise<Array<{path: string; title: string}>> {
  const cache = await deps.caches.open(PAGES_CACHE)
  const keys = (await savedPostKeys(cache)).toReversed()
  return Promise.all(
    keys.map(async (key) => {
      const path = new URL(key.url).pathname
      const html = (await (await cache.match(key))?.text()) ?? ''
      const title = /<title>([^<]*)<\/title>/.exec(html)?.[1] ?? path
      return {path, title: decodeTitle(title)}
    }),
  )
}

export async function clearSavedPosts(deps: OfflineDeps) {
  const cache = await deps.caches.open(PAGES_CACHE)
  const keys = [
    ...(await savedPostKeys(cache)),
    ...(await cache.keys()).filter((key) =>
      new URL(key.url).pathname.endsWith('.html'),
    ),
  ]
  await Promise.all(keys.map((key) => cache.delete(key)))
  await deps.caches.delete(IMAGES_CACHE)
}
