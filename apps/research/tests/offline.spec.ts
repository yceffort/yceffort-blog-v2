import {mkdtemp, rm} from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'

import {chromium, expect, test} from '@playwright/test'
import type {Page} from '@playwright/test'

const slug = 'suspense-error-boundary-deep-dive'
const title = 'Suspense & ErrorBoundary 딥다이브: 던지는 컴포넌트의 세계'
const base = 'http://localhost:3011'

async function saveFromViewer(page: Page) {
  await page.goto(`${base}/slides/${slug}`, {waitUntil: 'domcontentloaded'})
  await page
    .locator('.marp-slides')
    .click({button: 'right', position: {x: 350, y: 250}})
  await page
    .getByRole('button', {name: `${slug} 오프라인 저장`, exact: true})
    .click()
  await expect(page.locator('.offline-toast')).toContainText(
    `저장 완료: ${title}`,
  )
}

test('selected decks survive a restart offline with animations, notes, timer and presenter sync', async () => {
  const profile = await mkdtemp(
    path.join(os.tmpdir(), 'research-offline-test-'),
  )
  let context = await chromium.launchPersistentContext(profile, {
    headless: true,
    reducedMotion: 'no-preference',
  })
  try {
    context.setDefaultTimeout(15000)
    context.setDefaultNavigationTimeout(15000)
    const response = await context.request.get(
      `${base}/api/slides/${slug}/offline`,
    )
    const {notes}: {notes: string[]} = await response.json()
    const noteIndex = notes.findIndex((note) => note.trim())
    const nextNoteIndex = notes.findIndex(
      (note, index) => index > noteIndex && note.trim(),
    )
    expect(noteIndex).toBeGreaterThanOrEqual(0)
    expect(nextNoteIndex).toBeGreaterThan(noteIndex)
    let page = await context.newPage()
    await page.goto(`${base}/slides/promise-deep-dive`, {
      waitUntil: 'domcontentloaded',
    })
    await saveFromViewer(page)
    await page.goto(`${base}/offline`, {waitUntil: 'domcontentloaded'})
    await expect(page.locator('.offline-deck')).toHaveCount(1)
    await expect(page.locator('.offline-deck h2')).toHaveText(title)
    await context.close()

    context = await chromium.launchPersistentContext(profile, {
      headless: true,
      reducedMotion: 'no-preference',
    })
    context.setDefaultTimeout(15000)
    context.setDefaultNavigationTimeout(15000)
    await context.setOffline(true)
    page = await context.newPage()
    const errors: string[] = []
    page.on('pageerror', (error) => errors.push(error.message))
    // A direct path must work on the first request after a disconnected restart.
    await page.goto(`${base}/offline/${slug}#34`, {
      waitUntil: 'domcontentloaded',
    })
    await expect(page).toHaveURL(
      new RegExp(`/offline/${slug}\\?snapshot=research-deck-v1-[a-f0-9-]+#34$`),
    )
    await expect(
      page.locator('.swiper-slide-active .mermaid svg'),
    ).toBeVisible()
    await page.reload({waitUntil: 'domcontentloaded'})
    await expect(page.locator('.marp-slides')).toBeVisible()
    await page.goto(`${base}/`, {waitUntil: 'domcontentloaded'})
    await expect(page).toHaveURL(`${base}/offline`)
    await expect(page.locator('.offline-deck')).toHaveCount(1)
    await page.getByRole('link', {name: '슬라이드 열기', exact: true}).click()
    await expect(page.locator('.marp-slides')).toBeVisible()
    // Page 34 was never opened while online. Mermaid loads this diagram's code lazily.
    await page.goto(`${base}/slides/${slug}#34`, {
      waitUntil: 'domcontentloaded',
    })
    await expect(page).toHaveURL(
      new RegExp(`/offline/${slug}\\?snapshot=research-deck-v1-[a-f0-9-]+#34$`),
    )
    await expect(
      page.locator('.swiper-slide-active .mermaid svg'),
    ).toBeVisible()
    // Exercise the real animation engine, including effects not used online.
    for (const transition of ['slide', 'fade', 'zoom', 'glide']) {
      await context.addCookies([
        {name: 'tw-transition', value: transition, url: base},
      ])
      await page.reload({waitUntil: 'domcontentloaded'})
      const slides = page.locator('.marp-slides')
      await expect(slides).toHaveAttribute('data-transition', transition)
      await slides.evaluate((element) => {
        element.addEventListener('transitionrun', (event) => {
          const target = event.target
          if (
            target instanceof HTMLElement &&
            target.matches('.swiper-wrapper, .swiper-slide')
          ) {
            element.setAttribute(
              'data-tested-animation',
              (event as TransitionEvent).propertyName,
            )
          }
        })
      })
      const previousPage = Number(new URL(page.url()).hash.slice(1))
      await page.keyboard.press('ArrowRight')
      await expect(page).toHaveURL(new RegExp(`#${previousPage + 1}$`))
      await expect(slides).toHaveAttribute(
        'data-tested-animation',
        /^(transform|opacity)$/,
      )
    }
    await page.goto(`${base}/offline/${slug}#${noteIndex + 1}`, {
      waitUntil: 'domcontentloaded',
    })
    await expect(page.locator('.marp-slides')).toBeVisible()
    const presenterPromise = context.waitForEvent('page')
    await page.keyboard.press('p')
    const presenter = await presenterPromise
    await expect(presenter).toHaveURL(
      new RegExp(
        `/offline/${slug}/presenter\\?snapshot=research-deck-v1-[a-f0-9-]+#${noteIndex + 1}$`,
      ),
    )
    presenter.on('pageerror', (error) => errors.push(error.message))
    const noteBody = presenter.locator('.marp-presenter-notes-content').last()
    await expect(noteBody).toHaveText(notes[noteIndex])
    await expect(
      presenter.locator('.marp-presenter-slide section'),
    ).toHaveCount(2)
    await presenter.getByRole('button', {name: '시작', exact: true}).click()
    await expect(presenter.getByText('00:01', {exact: true})).toBeVisible()
    await presenter.getByRole('button', {name: '일시정지', exact: true}).click()
    for (let index = noteIndex; index < nextNoteIndex; index++) {
      await presenter.getByRole('button', {name: '다음 ▶', exact: true}).click()
    }
    await expect(page).toHaveURL(new RegExp(`#${nextNoteIndex + 1}$`))
    await expect(noteBody).toHaveText(notes[nextNoteIndex])
    await page.keyboard.press('ArrowLeft')
    await expect(noteBody).toHaveText(notes[nextNoteIndex - 1] || '노트 없음')
    await presenter.getByRole('button', {name: '리셋', exact: true}).click()
    await expect(presenter.getByText('00:00', {exact: true})).toBeVisible()
    await presenter.reload({waitUntil: 'domcontentloaded'})
    await expect(noteBody).toHaveText(notes[nextNoteIndex - 1] || '노트 없음')
    await presenter.goto(`${base}/slides/${slug}/presenter#${nextNoteIndex}`, {
      waitUntil: 'domcontentloaded',
    })
    await expect(presenter).toHaveURL(
      new RegExp(
        `/offline/${slug}/presenter\\?snapshot=research-deck-v1-[a-f0-9-]+#${nextNoteIndex}$`,
      ),
    )
    await expect(noteBody).toHaveText(notes[nextNoteIndex - 1] || '노트 없음')
    await presenter.close()
    await page.keyboard.press('End')
    await expect(page).toHaveURL(/#76$/)
    await page.goto(`${base}/offline`, {waitUntil: 'domcontentloaded'})
    await page.getByRole('link', {name: '슬라이드 열기', exact: true}).click()
    await expect(page).toHaveURL(/#76$/)
    await expect(page.locator('.marp-slides')).toBeVisible()
    await page.goto(`${base}/slides/promise-deep-dive`, {
      waitUntil: 'domcontentloaded',
    })
    await expect(page.locator('.offline-error')).toContainText(
      '저장되어 있지 않습니다',
    )
    await page.goto(`${base}/offline`, {waitUntil: 'domcontentloaded'})
    await page.getByRole('button', {name: '삭제', exact: true}).click()
    await expect(page.getByText('아직 저장한 자료가 없습니다')).toBeVisible()
    await page.reload()
    await expect(page.getByText('아직 저장한 자료가 없습니다')).toBeVisible()
    expect(errors).toEqual([])
  } finally {
    await context.close()
    await rm(profile, {recursive: true, force: true})
  }
})

test('saved routes start without waiting for a stalled network', async ({
  page,
  context,
}) => {
  await saveFromViewer(page)
  // Venue Wi-Fi without internet: the browser stays online but nothing answers.
  let networkRequests = 0
  await context.route(
    (url) => url.pathname === '/offline' || url.pathname === `/offline/${slug}`,
    () => {
      networkRequests++
    },
  )
  const viewer = await context.newPage()
  const start = Date.now()
  await viewer.goto(`${base}/offline/${slug}`, {
    waitUntil: 'domcontentloaded',
    timeout: 5000,
  })
  await expect(viewer.locator('.marp-slides')).toBeVisible({timeout: 5000})
  expect(Date.now() - start).toBeLessThan(5000)
  await viewer.goto('/offline', {waitUntil: 'domcontentloaded', timeout: 5000})
  await expect(viewer.locator('.offline-deck')).toHaveCount(1)
  expect(networkRequests).toBe(0)
})

test('saved routes ignore new HTML whose new scripts cannot be downloaded', async ({
  page,
  context,
  request,
}) => {
  await saveFromViewer(page)
  const html = await (await request.get('/offline')).text()
  let changed = 0
  const nextHtml = html.replace(
    /\/_next\/static\/chunks\/[^"\\<>\s]+\.js/g,
    () => `/_next/static/chunks/unavailable-deployment-${changed++}.js`,
  )
  expect(changed).toBeGreaterThan(0)
  let documents = 0
  let scripts = 0
  await context.route(`**/offline/${slug}`, (route) => {
    documents++
    return route.fulfill({body: nextHtml, contentType: 'text/html'})
  })
  await context.route('**/unavailable-deployment-*.js', (route) => {
    scripts++
    return route.abort('internetdisconnected')
  })
  const viewer = await context.newPage()
  await viewer.goto(`/offline/${slug}`, {waitUntil: 'domcontentloaded'})
  await expect(viewer.locator('.marp-slides')).toBeVisible()
  expect(documents).toBe(0)
  expect(scripts).toBe(0)
  await context.setOffline(true)
  await viewer.reload({waitUntil: 'domcontentloaded'})
  await expect(viewer.locator('.marp-slides')).toBeVisible()
})

test('offline paths resolve on the server before any deck has been saved', async ({
  page,
  request,
}) => {
  for (const suffix of ['', '/presenter']) {
    const response = await request.get(`/offline/${slug}${suffix}`)
    expect(response.status()).toBe(200)
  }
  await page.goto(`/offline/${slug}`, {waitUntil: 'domcontentloaded'})
  await expect(page.locator('.offline-error')).toContainText(
    '저장되어 있지 않습니다',
  )
  await expect(page).toHaveURL(`${base}/offline/${slug}`)
})

test('failed asset or runtime updates preserve the previous complete download', async ({
  page,
  context,
  request,
}) => {
  const response = await request.get(`/api/slides/${slug}/offline`)
  const original = await response.json()
  const manifest = await (await request.get('/offline-runtime.json')).json()
  // Chromium's request interception aborts favicon.ico requests. The cold-start
  // test uses the real manifest; fault-injection tests omit this optional icon.
  await context.route('**/offline-runtime.json', (route) =>
    route.fulfill({
      json: {
        ...manifest,
        assets: manifest.assets.filter(
          (asset: {url: string}) => !asset.url.endsWith('.ico'),
        ),
      },
    }),
  )
  const payload = {...original, html: [...original.html]}
  // Exercise SVG/background dependencies in a never-visited slide.
  payload.html[60] +=
    '<svg><image href="/profile.png" /></svg><img alt="offline fixture" src="/profile.png"><div style="background-image:url(/favicon/favicon.svg)"></div>'
  await context.route(`**/api/slides/${slug}/offline`, (route) =>
    route.fulfill({json: payload}),
  )
  await saveFromViewer(page)
  await page.goto('/offline', {waitUntil: 'domcontentloaded'})
  payload.title = 'updated deck'
  await context.route('**/profile.png', (route) =>
    route.fulfill({status: 503, body: 'unavailable'}),
  )
  await page
    .getByRole('button', {
      name: `${slug} 오프라인 저장본 업데이트`,
      exact: true,
    })
    .click()
  await expect(page.locator('.offline-toast')).toContainText(
    '내려받지 못했습니다',
  )
  await expect(page.locator('.offline-deck h2')).toHaveText(title)
  await context.unroute('**/profile.png')
  const badManifest = {
    ...manifest,
    revision: 'abcdef',
    assets: [{url: manifest.shell, sha256: 'invalid'}],
  }
  await context.route('**/offline-runtime.json', (route) =>
    route.fulfill({json: badManifest}),
  )
  await page
    .getByRole('button', {
      name: `${slug} 오프라인 저장본 업데이트`,
      exact: true,
    })
    .click()
  await expect(page.locator('.offline-toast')).toContainText(
    '사이트가 업데이트되었습니다',
  )
  await context.setOffline(true)
  await page.reload()
  await expect(page.locator('.offline-deck h2')).toHaveText(title)
  await page.getByRole('link', {name: '슬라이드 열기', exact: true}).click()
  await expect(page.locator('.marp-slides')).toBeVisible()
  await page.goto(`/offline/${slug}#61`, {waitUntil: 'domcontentloaded'})
  const image = page.getByAltText('offline fixture')
  await expect(image).toHaveAttribute('src', /\/offline-assets\//)
  await expect
    .poll(() =>
      image.evaluate(
        (element: HTMLImageElement) =>
          element.complete && element.naturalWidth > 0,
      ),
    )
    .toBe(true)
})

test('a cancelled download is never listed as saved', async ({
  page,
  context,
}) => {
  let release: (() => void) | undefined
  const held = new Promise<void>((resolve) => {
    release = resolve
  })
  await context.route('**/offline-runtime.json', async (route) => {
    await held
    await route.abort()
  })
  await page.goto(`/slides/${slug}`, {waitUntil: 'domcontentloaded'})
  await page
    .locator('.marp-slides')
    .click({button: 'right', position: {x: 350, y: 250}})
  await page
    .getByRole('button', {name: `${slug} 오프라인 저장`, exact: true})
    .click()
  await page
    .getByRole('button', {name: `${slug} 다운로드 취소`, exact: true})
    .click()
  release?.()
  await expect(page.locator('.offline-toast')).toContainText(
    '다운로드를 취소했습니다',
  )
  await page.goto('/offline', {waitUntil: 'domcontentloaded'})
  await expect(page.getByText('아직 저장한 자료가 없습니다')).toBeVisible()
})

test('unpublished decks are not served in production', async ({request}) => {
  const draftSlug = 'example'
  for (const url of [
    `/slides/${draftSlug}`,
    `/slides/${draftSlug}/presenter`,
    `/slides/${draftSlug}.md`,
    `/api/slides/${draftSlug}/raw`,
    `/api/slides/${draftSlug}/offline`,
  ]) {
    expect((await request.get(url)).status(), url).toBe(404)
  }
})

test('download API rejects invalid and missing slugs', async ({request}) => {
  expect((await request.get('/api/slides/invalid.slug/offline')).status()).toBe(
    404,
  )
  expect(
    (await request.get('/api/slides/does-not-exist/offline')).status(),
  ).toBe(404)
})

test('a narrow home card downloads only the selected deck', async ({
  page,
}, testInfo) => {
  await page.setViewportSize({width: 390, height: 844})
  await page.goto('/', {waitUntil: 'domcontentloaded'})
  const card = page.locator('article').first()
  const selectedTitle = await card.locator('h3').innerText()
  await card.getByRole('button', {name: /오프라인 저장$/}).click()
  await expect(page.locator('.offline-toast')).toContainText(
    `저장 완료: ${selectedTitle}`,
  )
  await page.goto('/offline', {waitUntil: 'domcontentloaded'})
  await expect(page.locator('.offline-deck')).toHaveCount(1)
  await expect(page.locator('.offline-deck h2')).toHaveText(selectedTitle)
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true)
  await page.screenshot({
    path: testInfo.outputPath('mobile-library.png'),
    fullPage: true,
  })
})
