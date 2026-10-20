import {beforeEach, describe, expect, it} from 'vitest'

import {
  clearSavedPosts,
  extractAssetUrls,
  hasPage,
  listSavedPosts,
  savePage,
} from '../cache'
import {createDeps} from './fakeCaches'

const ORIGIN = 'https://yceffort.kr'

const page = (title: string, body = '') =>
  new Response(
    `<html><head><title>${title}</title></head><body>${body}</body></html>`,
  )

const bytesOf = (text: string) => new Blob([text]).size
const postHtml = (body: string) =>
  `<html><head><title>a</title></head><body>${body}</body></html>`

const cachedUrls = async (deps: ReturnType<typeof createDeps>, name: string) =>
  (await (await deps.caches.open(name)).keys()).map((key) => key.url)

const EMBEDS = `<iframe src="/demos/number-flow/animation-lab.html?scene=delta" title="x" loading="lazy" sandbox="allow-scripts"></iframe>
<iframe src="/demos/coldpath/a.html" width="100%"></iframe>
<iframe src="https://www.youtube.com/embed/X9eRLElSW1c"></iframe>
<iframe src="//jsfiddle.net/yceffort/n23uLwak/embedded/"></iframe>
<video src="/2021/08/queue-priority.mp4" poster="/2021/08/poster.jpg" controls></video>`

const DEMO = `${ORIGIN}/demos/number-flow/animation-lab.html?scene=delta`

describe('extractAssetUrls', () => {
  it('extracts one srcSet variant per source, closest to 1080', () => {
    const html =
      '<img srcSet="/_next/image?url=%2Fa.png&amp;w=640&amp;q=75 640w, /_next/image?url=%2Fa.png&amp;w=1080&amp;q=75 1080w" src="/_next/image?url=%2Fa.png&amp;w=3840&amp;q=75">'
    expect(extractAssetUrls(html).images).toEqual([
      '/_next/image?url=%2Fa.png&w=1080&q=75',
    ])
  })

  it('extracts document-relative images', () => {
    expect(extractAssetUrls('<img src="./banner-image.png">').images).toEqual([
      './banner-image.png',
    ])
  })

  it('extracts same-origin html iframes and video posters only', () => {
    const {images, documents} = extractAssetUrls(EMBEDS)
    expect(documents).toEqual([
      '/demos/number-flow/animation-lab.html?scene=delta',
      '/demos/coldpath/a.html',
    ])
    expect(images).toContain('/2021/08/poster.jpg')
    const all = JSON.stringify([images, documents])
    expect(all).not.toContain('.mp4')
    expect(all).not.toContain('youtube')
    expect(all).not.toContain('jsfiddle')
  })
})

describe('savePage', () => {
  let deps: ReturnType<typeof createDeps>

  beforeEach(() => {
    deps = createDeps((url) => page(url.pathname))
  })

  it('stores a post as new, then updated', async () => {
    expect((await savePage('/2026/10/a', {images: false}, deps)).status).toBe(
      'new',
    )
    expect((await savePage('/2026/10/a', {images: false}, deps)).status).toBe(
      'updated',
    )
    expect(
      await (await deps.caches.open('pages-v4')).match(`${ORIGIN}/2026/10/a`),
    ).toBeDefined()
  })

  it('reports whether a page is cached', async () => {
    expect(await hasPage('/2026/10/a', deps)).toBe(false)
    await savePage('/2026/10/a', {images: false}, deps)
    expect(await hasPage('/2026/10/a', deps)).toBe(true)
  })

  it('does not store redirected responses', async () => {
    deps.fetch.mockImplementation(async () => {
      const response = page('English home')
      Object.defineProperty(response, 'redirected', {value: true})
      return response
    })
    expect((await savePage('/', {images: false}, deps)).status).toBe('failed')
    expect(await cachedUrls(deps, 'pages-v4')).toEqual([])
  })

  it('does not store non-ok responses', async () => {
    deps.fetch.mockImplementation(
      async () => new Response('not found', {status: 404}),
    )
    expect(await savePage('/2026/10/missing', {images: false}, deps)).toEqual({
      status: 'failed',
      bytes: 0,
      external: 0,
    })
    expect(await cachedUrls(deps, 'pages-v4')).toEqual([])
  })

  describe('size', () => {
    it('counts the post html and its images', async () => {
      const body = '<img src="/images/a.png">'
      const png = 'x'.repeat(1234)
      deps = createDeps((url) =>
        url.pathname.endsWith('.png')
          ? new Response(png)
          : new Response(postHtml(body)),
      )
      expect(await savePage('/2026/10/a', {images: true}, deps)).toEqual({
        status: 'new',
        bytes: bytesOf(postHtml(body)) + 1234,
        external: 0,
      })
    })

    it('counts documents and the images they reference', async () => {
      const post = postHtml(
        '<iframe src="/demos/2026/08/banner-motion.html"></iframe>',
      )
      const doc = '<html><img src="./banner-image.png"></html>'
      const png = 'p'.repeat(777)
      deps = createDeps((url) =>
        url.pathname.endsWith('.html')
          ? new Response(doc)
          : url.pathname.endsWith('.png')
            ? new Response(png)
            : new Response(post),
      )
      const result = await savePage('/2026/10/a', {images: true}, deps)
      expect(result.bytes).toBe(bytesOf(post) + bytesOf(doc) + 777)
      expect(await cachedUrls(deps, 'images-v4')).toEqual([
        `${ORIGIN}/demos/2026/08/banner-image.png`,
      ])
    })

    it('counts an image already cached by another post', async () => {
      const body = '<img src="/images/shared.png">'
      deps = createDeps((url) =>
        url.pathname.endsWith('.png')
          ? new Response('s'.repeat(500))
          : new Response(postHtml(body)),
      )
      await savePage('/2026/10/a', {images: true}, deps)
      const result = await savePage('/2026/10/b', {images: true}, deps)
      expect(result.bytes).toBe(bytesOf(postHtml(body)) + 500)
      expect(
        deps.fetch.mock.calls.filter(
          ([input]) => new Request(input).url === `${ORIGIN}/images/shared.png`,
        ),
      ).toHaveLength(1)
    })

    it('counts opaque external images separately', async () => {
      const body = '<img src="https://cdn.example.com/b.png">'
      deps = createDeps((url) =>
        url.hostname === 'cdn.example.com'
          ? Object.defineProperty(new Response('hidden'), 'type', {
              value: 'opaque',
            })
          : new Response(postHtml(body)),
      )
      expect(await savePage('/2026/10/a', {images: true}, deps)).toEqual({
        status: 'new',
        bytes: bytesOf(postHtml(body)),
        external: 1,
      })
    })
  })

  it('saves images only when images is true', async () => {
    deps = createDeps((url) =>
      url.pathname.endsWith('.png')
        ? new Response('png')
        : page(
            'a',
            '<img src="/images/a.png"><img src="https://cdn.example.com/b.png">',
          ),
    )

    await savePage('/2026/10/a', {images: false}, deps)
    expect(await cachedUrls(deps, 'images-v4')).toEqual([])

    await savePage('/2026/10/a', {images: true}, deps)
    expect((await cachedUrls(deps, 'images-v4')).toSorted()).toEqual([
      'https://cdn.example.com/b.png',
      `${ORIGIN}/images/a.png`,
    ])
    expect(deps.fetch).toHaveBeenCalledWith('https://cdn.example.com/b.png', {
      mode: 'no-cors',
    })
  })

  it('saves iframe demos and video posters with images', async () => {
    deps = createDeps((url) =>
      url.pathname.endsWith('.html')
        ? new Response('<html>demo</html>')
        : url.pathname.endsWith('.jpg')
          ? new Response('jpg')
          : page('a', EMBEDS),
    )

    expect((await savePage('/2026/10/a', {images: true}, deps)).status).toBe(
      'new',
    )

    const pages = await cachedUrls(deps, 'pages-v4')
    expect(pages).toContain(`${ORIGIN}/2026/10/a`)
    expect(pages).toContain(DEMO)
    expect(await cachedUrls(deps, 'images-v4')).toEqual([
      `${ORIGIN}/2021/08/poster.jpg`,
    ])
    expect(deps.fetch).not.toHaveBeenCalledWith(
      `${ORIGIN}/2021/08/queue-priority.mp4`,
      undefined,
    )
  })

  it('saves images referenced by a document relative to the document', async () => {
    deps = createDeps((url) =>
      url.pathname.endsWith('.html')
        ? new Response('<html><img src="./banner-image.png"></html>')
        : url.pathname.endsWith('.png')
          ? new Response('png')
          : page(
              'a',
              '<iframe src="/demos/2026/08/banner-motion.html"></iframe>',
            ),
    )

    await savePage('/2026/10/a', {images: true}, deps)

    expect(await cachedUrls(deps, 'pages-v4')).toContain(
      `${ORIGIN}/demos/2026/08/banner-motion.html`,
    )
    expect(await cachedUrls(deps, 'images-v4')).toEqual([
      `${ORIGIN}/demos/2026/08/banner-image.png`,
    ])
  })

  it('keeps /offline when trimming pages over 200', async () => {
    const pages = await deps.caches.open('pages-v4')
    await pages.put(`${ORIGIN}/offline`, page('offline'))
    for (let i = 0; i < 200; i++) {
      await pages.put(`${ORIGIN}/2026/01/p${i}`, page(`p${i}`))
    }

    await savePage('/2026/10/a', {images: false}, deps)

    const urls = await cachedUrls(deps, 'pages-v4')
    expect(urls).toHaveLength(201)
    expect(urls).toContain(`${ORIGIN}/offline`)
    expect(urls).not.toContain(`${ORIGIN}/2026/01/p0`)
    expect(urls).toContain(`${ORIGIN}/2026/01/p1`)
    expect(urls).toContain(`${ORIGIN}/2026/10/a`)
  })
})

describe('listSavedPosts', () => {
  let deps: ReturnType<typeof createDeps>

  beforeEach(() => {
    deps = createDeps((url) => page(url.pathname))
  })

  it('lists legacy entries as read with the first title', async () => {
    const pages = await deps.caches.open('pages-v4')
    await pages.put(
      new Request(`${ORIGIN}/2026/09/old`),
      new Response(
        '<html><head><title>옛 글</title></head><body><svg><title>Mail icon</title></svg></body></html>',
      ),
    )
    expect(await listSavedPosts(deps)).toEqual([
      {path: '/2026/09/old', title: '옛 글'},
    ])
  })

  it('decodes the entities React escapes in titles', async () => {
    const pages = await deps.caches.open('pages-v4')
    await pages.put(
      `${ORIGIN}/2026/10/a`,
      new Response('<title>It&#x27;s &lt;a&gt; &amp; &quot;b&quot;</title>'),
    )
    expect(await listSavedPosts(deps)).toEqual([
      {path: '/2026/10/a', title: 'It\'s <a> & "b"'},
    ])
  })

  it('lists only post paths, newest saved first', async () => {
    const pages = await deps.caches.open('pages-v4')
    for (const path of [
      '/',
      '/en',
      '/offline',
      '/saved',
      '/pages/2',
      '/tags/react',
      '/2026/10/a?x=1',
      '/2026/10/a',
      '/2026/10/b',
    ]) {
      await pages.put(`${ORIGIN}${path}`, page(path))
    }
    expect((await listSavedPosts(deps)).map((post) => post.path)).toEqual([
      '/2026/10/b',
      '/2026/10/a',
    ])
  })
})

describe('clearSavedPosts', () => {
  it('removes posts and html documents but keeps /offline', async () => {
    const deps = createDeps((url) =>
      url.pathname.endsWith('.html') || url.pathname.endsWith('.jpg')
        ? new Response('asset')
        : page('a', EMBEDS),
    )
    const pages = await deps.caches.open('pages-v4')
    await pages.put(`${ORIGIN}/offline`, page('offline'))
    await savePage('/2026/10/a', {images: true}, deps)
    expect(await cachedUrls(deps, 'pages-v4')).toContain(DEMO)

    await clearSavedPosts(deps)

    expect(await cachedUrls(deps, 'pages-v4')).toEqual([`${ORIGIN}/offline`])
    expect(await cachedUrls(deps, 'images-v4')).toEqual([])
  })

  it('lists only posts even when html documents are cached', async () => {
    const deps = createDeps((url) =>
      url.pathname.endsWith('.html') ? new Response('demo') : page('a', EMBEDS),
    )
    await savePage('/2026/10/a', {images: true}, deps)
    expect(await cachedUrls(deps, 'pages-v4')).toContain(DEMO)
    expect((await listSavedPosts(deps)).map((post) => post.path)).toEqual([
      '/2026/10/a',
    ])
  })
})
