import {
  clearSavedPosts,
  deletePage,
  hasPage,
  saveRedirect,
  savePage,
  type OfflineDeps,
} from './cache'
import {readMeta, writeMeta, type SavedMeta} from './meta'
import {homePathOf, isPostPath, localeOfPath, savedPathOf} from './paths'

export function browserDeps(): OfflineDeps {
  let storage: Storage | null = null
  try {
    storage = window.localStorage
  } catch {
    // 사이트 데이터를 막아 두면 localStorage 접근 자체가 SecurityError를 던진다
  }
  return {
    caches: window.caches,
    fetch: window.fetch.bind(window),
    storage,
    origin: window.location.origin,
    saveData:
      (navigator as Navigator & {connection?: {saveData?: boolean}}).connection
        ?.saveData === true,
    now: () => Date.now(),
  }
}

async function fetchRecent(
  locale: 'ko' | 'en',
  deps: OfflineDeps,
): Promise<Array<{path: string}> | null> {
  try {
    const response = await deps.fetch(
      new URL(
        locale === 'en' ? '/api/recent-posts/en' : '/api/recent-posts',
        deps.origin,
      ).href,
    )
    if (!response.ok) {
      return null
    }
    const {posts} = (await response.json()) as {posts: Array<{path: string}>}
    return posts
  } catch {
    return null
  }
}

export async function recordRead(pathname: string, deps: OfflineDeps) {
  if (!isPostPath(pathname)) {
    return false
  }
  const result = await savePage(pathname, {images: true}, deps)
  const meta = readMeta(deps.storage)
  writeMeta(deps.storage, {
    ...meta,
    locale: localeOfPath(pathname),
    auto: meta.auto.filter((path) => path !== pathname),
    sizes:
      result.status === 'failed'
        ? meta.sizes
        : {
            ...meta.sizes,
            [pathname]: {bytes: result.bytes, external: result.external},
          },
  })
  return result.status === 'new'
}

export async function syncRecent(deps: OfflineDeps) {
  if (deps.saveData) {
    return 0
  }
  const meta = readMeta(deps.storage)
  let target: string[] = []
  if (meta.count > 0) {
    const list = await fetchRecent(meta.locale, deps)
    if (!list) {
      return 0
    }
    target = list
      .slice(0, meta.count)
      .map((post) => post.path)
      .filter((path) => !meta.excluded.includes(path))
  }

  const removed = meta.auto.filter((p) => !target.includes(p))
  for (const path of removed) {
    await deletePage(path, deps)
  }

  const added: string[] = []
  const measured: SavedMeta['sizes'] = {}
  for (const path of target) {
    // 크기를 재기 전에 저장된 글은 한 번 다시 받아 잰다
    if ((await hasPage(path, deps)) && meta.sizes[path]) {
      continue
    }
    const result = await savePage(path, {images: meta.images}, deps)
    if (result.status !== 'failed') {
      measured[path] = {bytes: result.bytes, external: result.external}
    }
    if (result.status === 'new') {
      added.push(path)
    }
  }

  // 받는 동안 읽음 처리나 ✕로 바뀐 기록을 덮어쓰지 않도록 다시 읽는다
  const latest = readMeta(deps.storage)
  writeMeta(deps.storage, {
    ...latest,
    auto: target.filter(
      (path) => latest.auto.includes(path) || added.includes(path),
    ),
    sizes: {
      ...Object.fromEntries(
        Object.entries(latest.sizes).filter(
          ([path]) => !removed.includes(path),
        ),
      ),
      ...measured,
    },
    lastSync: deps.now(),
  })
  return added.length
}

export async function syncOnLaunch(
  pathname: string,
  deps: OfflineDeps,
): Promise<{savedCurrent: boolean; announce: number | null}> {
  const locale = localeOfPath(pathname)
  writeMeta(deps.storage, {...readMeta(deps.storage), locale})
  const savedCurrent = await recordRead(pathname, deps)
  await savePage(homePathOf(locale), {images: true}, deps)
  if (locale === 'en') {
    await saveRedirect('/', deps)
  }
  await savePage(savedPathOf(locale), {images: false}, deps)

  const received = await syncRecent(deps)
  const meta = readMeta(deps.storage)
  if (received === 0 || meta.announced) {
    return {savedCurrent, announce: null}
  }
  writeMeta(deps.storage, {...meta, announced: true})
  return {savedCurrent, announce: received}
}

export async function deleteSaved(path: string, deps: OfflineDeps) {
  await deletePage(path, deps)
  const meta = readMeta(deps.storage)
  writeMeta(deps.storage, {
    ...meta,
    auto: meta.auto.filter((p) => p !== path),
    sizes: Object.fromEntries(
      Object.entries(meta.sizes).filter(([p]) => p !== path),
    ),
    excluded: meta.excluded.includes(path)
      ? meta.excluded
      : [...meta.excluded, path],
  })
}

export async function clearAllSaved(deps: OfflineDeps) {
  await clearSavedPosts(deps)
  writeMeta(deps.storage, {
    ...readMeta(deps.storage),
    auto: [],
    excluded: [],
    sizes: {},
  })
}

export function updateSettings(
  patch: Partial<Pick<SavedMeta, 'count' | 'images' | 'locale'>>,
  deps: OfflineDeps,
): SavedMeta {
  const meta = {...readMeta(deps.storage), ...patch}
  writeMeta(deps.storage, meta)
  return meta
}
