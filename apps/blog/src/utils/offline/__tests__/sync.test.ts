import {beforeEach, describe, expect, it} from 'vitest'

import {listSavedPosts, savePage} from '../cache'
import {readMeta, writeMeta} from '../meta'
import {
  clearAllSaved,
  deleteSaved,
  recordRead,
  syncOnLaunch,
  syncRecent,
  updateSettings,
} from '../sync'
import {createDeps} from './fakeCaches'

const ORIGIN = 'https://yceffort.kr'
const KO_RECENT = ['a', 'b', 'c', 'd', 'e', 'f'].map((s) => `/2026/10/${s}`)
const EN_RECENT = ['a-en', 'b-en', 'c-en', 'd-en', 'e-en', 'f-en'].map(
  (s) => `/en/2026/10/${s}`,
)

const recentResponse = (paths: string[]) =>
  Response.json({posts: paths.map((path) => ({path, title: path}))})

const route = (url: URL) => {
  if (url.pathname === '/api/recent-posts') {
    return recentResponse(KO_RECENT)
  }
  if (url.pathname === '/api/recent-posts/en') {
    return recentResponse(EN_RECENT)
  }
  return new Response(
    `<title>${url.pathname}</title><img src="/images${url.pathname}.png">`,
  )
}

// route가 돌려주는 글 HTML과 같은 문자열로 기대값을 만든다
const postBytes = (path: string) =>
  new Blob([`<title>${path}</title><img src="/images${path}.png">`]).size +
  new Blob([
    `<title>/images${path}.png</title><img src="/images/images${path}.png.png">`,
  ]).size

describe('sync', () => {
  let deps: ReturnType<typeof createDeps>

  const isCached = async (path: string) =>
    Boolean(await (await deps.caches.open('pages-v4')).match(ORIGIN + path))

  beforeEach(() => {
    deps = createDeps(route)
  })

  it('saves the newest count posts as auto', async () => {
    expect(await syncRecent(deps)).toBe(5)
    expect(readMeta(deps.storage).auto).toEqual([
      '/2026/10/a',
      '/2026/10/b',
      '/2026/10/c',
      '/2026/10/d',
      '/2026/10/e',
    ])
  })

  it('does not tag an already read post as auto', async () => {
    await recordRead('/2026/10/a', deps)
    expect(await syncRecent(deps)).toBe(4)
    expect(readMeta(deps.storage).auto).not.toContain('/2026/10/a')
  })

  it('removes auto entries pushed out of the newest count', async () => {
    await savePage('/2026/09/z', {images: false}, deps)
    writeMeta(deps.storage, {...readMeta(deps.storage), auto: ['/2026/09/z']})

    await syncRecent(deps)

    expect(await isCached('/2026/09/z')).toBe(false)
    expect(readMeta(deps.storage).auto).not.toContain('/2026/09/z')
  })

  it('keeps read entries pushed out of the newest count', async () => {
    await recordRead('/2026/09/z', deps)
    await syncRecent(deps)
    expect(await isCached('/2026/09/z')).toBe(true)
  })

  it('removes all auto entries when count is 0 without fetching the list', async () => {
    await syncRecent(deps)
    updateSettings({count: 0}, deps)
    deps.fetch.mockClear()

    expect(await syncRecent(deps)).toBe(0)
    expect(deps.fetch).not.toHaveBeenCalled()
    expect(readMeta(deps.storage).auto).toEqual([])
    for (const path of KO_RECENT.slice(0, 5)) {
      expect(await isCached(path)).toBe(false)
    }
  })

  it('keeps auto entries when the recent list fetch fails', async () => {
    await syncRecent(deps)
    const auto = readMeta(deps.storage).auto

    for (const fail of [
      async () => {
        throw new TypeError('Failed to fetch')
      },
      async () => new Response('', {status: 503}),
      async () => new Response('<html>'),
    ]) {
      deps.fetch.mockImplementation(fail)
      expect(await syncRecent(deps)).toBe(0)
      expect(readMeta(deps.storage).auto).toEqual(auto)
      for (const path of auto) {
        expect(await isCached(path)).toBe(true)
      }
    }
  })

  it('does nothing for auto entries when saveData is on', async () => {
    await syncRecent(deps)
    const auto = readMeta(deps.storage).auto
    updateSettings({count: 0}, deps)
    deps.saveData = true
    deps.fetch.mockClear()

    expect(await syncRecent(deps)).toBe(0)
    expect(deps.fetch).not.toHaveBeenCalled()
    expect(readMeta(deps.storage).auto).toEqual(auto)
    for (const path of auto) {
      expect(await isCached(path)).toBe(true)
    }
  })

  it('removes auto entries of the previous locale after a locale switch', async () => {
    await syncRecent(deps)
    const koAuto = readMeta(deps.storage).auto

    await syncOnLaunch('/en/2026/10/a-en', deps)

    expect(deps.fetch).toHaveBeenCalledWith(`${ORIGIN}/api/recent-posts/en`)
    for (const path of koAuto) {
      expect(await isCached(path)).toBe(false)
    }
    expect(readMeta(deps.storage).auto).toEqual(EN_RECENT.slice(1, 5))
  })

  it('turns an auto post into read when opened', async () => {
    await syncRecent(deps)

    expect(await recordRead('/2026/10/a', deps)).toBe(false)
    expect(readMeta(deps.storage).auto).not.toContain('/2026/10/a')
    expect(await isCached('/2026/10/a')).toBe(true)
  })

  it('skips excluded posts', async () => {
    await syncRecent(deps)
    await deleteSaved('/2026/10/a', deps)

    expect(await syncRecent(deps)).toBe(0)
    expect(await isCached('/2026/10/a')).toBe(false)
    expect(readMeta(deps.storage).auto).not.toContain('/2026/10/a')
  })

  it('clearAllSaved resets exclusions', async () => {
    await syncRecent(deps)
    await deleteSaved('/2026/10/a', deps)
    await recordRead('/2026/09/z', deps)
    const images = async () => (await deps.caches.open('images-v4')).keys()
    expect(await images()).not.toHaveLength(0)

    await clearAllSaved(deps)

    expect(await images()).toHaveLength(0)
    expect(await listSavedPosts(deps)).toEqual([])
    expect(await syncRecent(deps)).toBe(5)
    expect(readMeta(deps.storage).auto).toEqual(KO_RECENT.slice(0, 5))
  })

  describe('sizes', () => {
    it('records a size for each auto-saved path', async () => {
      await syncRecent(deps)
      const {sizes} = readMeta(deps.storage)
      expect(Object.keys(sizes)).toEqual(KO_RECENT.slice(0, 5))
      for (const path of KO_RECENT.slice(0, 5)) {
        expect(sizes[path]).toEqual({bytes: postBytes(path), external: 0})
      }
    })

    it('records the size of a read post', async () => {
      await recordRead('/2026/10/a', deps)
      expect(readMeta(deps.storage).sizes['/2026/10/a']).toEqual({
        bytes: postBytes('/2026/10/a'),
        external: 0,
      })
    })

    it('removes the size on deleteSaved', async () => {
      await syncRecent(deps)
      await deleteSaved('/2026/10/a', deps)
      expect(readMeta(deps.storage).sizes).not.toHaveProperty('/2026/10/a')
      expect(readMeta(deps.storage).sizes).toHaveProperty('/2026/10/b')
    })

    it('resets sizes on clearAllSaved', async () => {
      await syncRecent(deps)
      await clearAllSaved(deps)
      expect(readMeta(deps.storage).sizes).toEqual({})
    })

    it('re-measures an auto post saved before sizes existed', async () => {
      await savePage('/2026/10/a', {images: false}, deps)
      writeMeta(deps.storage, {...readMeta(deps.storage), auto: ['/2026/10/a']})
      await syncRecent(deps)
      expect(readMeta(deps.storage).sizes['/2026/10/a']).toEqual({
        bytes: postBytes('/2026/10/a'),
        external: 0,
      })
      expect(readMeta(deps.storage).auto).toContain('/2026/10/a')
    })

    // 옛 워커가 pages-v4에 남긴 글처럼 메타 없이 캐시에만 있는 글은
    // 다시 받지 않고 캐시에 든 만큼만 잰다
    it('measures a cached post outside the recent list from the cache alone', async () => {
      await savePage('/2026/09/z', {images: true}, deps)
      await savePage('/2026/09/y', {images: false}, deps)
      deps.fetch.mockClear()

      await syncRecent(deps)

      const {sizes, auto} = readMeta(deps.storage)
      expect(sizes['/2026/09/z']).toEqual({
        bytes: postBytes('/2026/09/z'),
        external: 0,
      })
      expect(sizes['/2026/09/y']).toEqual({
        bytes: new Blob([
          '<title>/2026/09/y</title><img src="/images/2026/09/y.png">',
        ]).size,
        external: 0,
      })
      expect(deps.fetch).not.toHaveBeenCalledWith(`${ORIGIN}/2026/09/z`)
      expect(deps.fetch).not.toHaveBeenCalledWith(`${ORIGIN}/2026/09/y`)
      expect(deps.fetch).not.toHaveBeenCalledWith(
        `${ORIGIN}/images/2026/09/y.png`,
      )
      expect(auto).not.toContain('/2026/09/z')
      expect(await isCached('/2026/09/z')).toBe(true)
    })

    it('drops the size of an auto post that left the target list', async () => {
      await savePage('/2026/09/z', {images: false}, deps)
      writeMeta(deps.storage, {
        ...readMeta(deps.storage),
        auto: ['/2026/09/z'],
        sizes: {'/2026/09/z': {bytes: 10, external: 0}},
      })
      await syncRecent(deps)
      expect(readMeta(deps.storage).sizes).not.toHaveProperty('/2026/09/z')
    })
  })

  it('announces only the first auto save', async () => {
    expect(await syncOnLaunch('/', deps)).toEqual({
      savedCurrent: false,
      announce: 5,
    })

    updateSettings({count: 10}, deps)
    expect(await syncOnLaunch('/', deps)).toEqual({
      savedCurrent: false,
      announce: null,
    })
    expect(readMeta(deps.storage).auto).toContain('/2026/10/f')
  })

  it('saves the root redirect on launch in en and a plain root in ko', async () => {
    const redirectRoute = (url: URL) =>
      url.pathname === '/'
        ? Object.defineProperty(new Response(null, {status: 200}), 'type', {
            value: 'opaqueredirect',
          })
        : route(url)

    deps = createDeps(redirectRoute)
    await syncOnLaunch('/en/2026/10/a-en', deps)
    expect(await isCached('/')).toBe(true)
    expect(await isCached('/en')).toBe(true)

    deps = createDeps(route)
    await syncOnLaunch('/', deps)
    expect(await isCached('/')).toBe(true)
    expect(deps.fetch).not.toHaveBeenCalledWith(`${ORIGIN}/`, {
      redirect: 'manual',
    })
  })

  it('records the locale of the read post', async () => {
    await recordRead('/en/2026/10/a', deps)
    expect(readMeta(deps.storage).locale).toBe('en')
  })

  it('saves home and saved page on launch but does not list them', async () => {
    await syncOnLaunch('/', deps)

    expect(await isCached('/')).toBe(true)
    expect(await isCached('/saved')).toBe(true)
    expect((await listSavedPosts(deps)).map((post) => post.path)).toEqual(
      KO_RECENT.slice(0, 5).toReversed(),
    )
  })
})
