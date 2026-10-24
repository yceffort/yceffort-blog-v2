import {expect, test} from '@playwright/test'
import type {APIRequestContext, Page} from '@playwright/test'

import type {OfflineDeck, SavedDeck} from '../src/lib/offline/types'

async function openNotes(page: Page, request: APIRequestContext) {
  const original = (await (
    await request.get('/api/slides/suspense-error-boundary-deep-dive/offline')
  ).json()) as OfflineDeck
  const longNote = Array.from(
    {length: 32},
    (_, index) =>
      `${index + 1}. 발표 중 읽어야 할 설명입니다. 아래에도 노트가 이어집니다.`,
  ).join('\n\n')
  const deck: SavedDeck = {
    ...original,
    slug: 'presenter-notes-test',
    html: original.html.slice(0, 4),
    notes: [longNote, '짧은 노트입니다.', '', longNote],
    revision: 'notes-fixture',
    savedAt: Date.now(),
    bytes: 0,
    assetCache: 'research-deck-v1-notes-fixture',
    assets: [],
  }
  await page.goto('/offline', {waitUntil: 'domcontentloaded'})
  await page.evaluate(async (saved) => {
    await caches.open(saved.assetCache)
    await new Promise<void>((resolve, reject) => {
      const open = indexedDB.open('research-offline', 1)
      open.addEventListener('upgradeneeded', () => {
        open.result.createObjectStore('decks', {keyPath: 'slug'})
      })
      open.addEventListener('error', () => reject(open.error))
      open.addEventListener('success', () => {
        const db = open.result
        const tx = db.transaction('decks', 'readwrite')
        tx.objectStore('decks').put(saved)
        tx.addEventListener('complete', () => {
          db.close()
          resolve()
        })
        tx.addEventListener('error', () => {
          db.close()
          reject(tx.error)
        })
      })
    })
  }, deck)
  await page.goto(`/offline/${deck.slug}/presenter`, {
    waitUntil: 'domcontentloaded',
  })
}

for (const viewport of [
  {name: 'desktop', width: 1200, height: 800, touch: false},
  {name: 'phone', width: 390, height: 844, touch: true},
  {name: 'landscape phone', width: 844, height: 390, touch: true},
]) {
  test.describe(viewport.name, () => {
    test.use({
      viewport: {width: viewport.width, height: viewport.height},
      hasTouch: viewport.touch,
    })

    test('notes show a scrollbar only when they overflow, remaining content and reset on slide change', async ({
      page,
      request,
    }, testInfo) => {
      await openNotes(page, request)
      const panel = page.getByRole('region', {name: '발표자 노트', exact: true})
      const content = page.getByRole('region', {name: '발표자 노트 본문'})
      // 넘치는 글이 없으면 트랙이 숨겨져 접근성 트리에서도 빠지므로 includeHidden 으로 찾는다.
      const scrollbar = page.getByRole('scrollbar', {
        name: '발표자 노트 스크롤',
        includeHidden: true,
      })
      const more = page.getByRole('button', {name: '↓ 아래에 내용 더 있음'})
      await expect(scrollbar).toBeVisible()
      await expect(scrollbar).toHaveAttribute('aria-disabled', 'false')
      await expect(more).toBeVisible()
      const trackBounds = (await scrollbar.boundingBox())!
      const thumbBounds = (await scrollbar.locator('div').boundingBox())!
      expect(thumbBounds.height).toBeLessThan(trackBounds.height)
      await page.screenshot({path: testInfo.outputPath('notes-with-more.png')})

      if (viewport.touch) await more.tap()
      else await more.click()
      await expect
        .poll(() => content.evaluate((node) => node.scrollTop))
        .toBeGreaterThan(0)
      await expect(page.getByText('1 / 4', {exact: true})).toBeVisible()

      await scrollbar.focus()
      await page.keyboard.press('End')
      await expect(scrollbar).toHaveAttribute('aria-valuenow', '100')
      await expect(more).toHaveCount(0)
      await expect(panel.getByText('✓ 끝까지 읽음')).toBeVisible()
      await expect(page.getByText('1 / 4', {exact: true})).toBeVisible()
      await page.screenshot({path: testInfo.outputPath('notes-at-end.png')})
      await page.keyboard.press('Home')
      await expect(scrollbar).toHaveAttribute('aria-valuenow', '0')
      await expect(more).toBeVisible()

      // Dragging the visible thumb and clicking/tapping the track both scroll.
      const thumb = (await scrollbar.locator('div').boundingBox())!
      await page.mouse.move(
        thumb.x + thumb.width / 2,
        thumb.y + thumb.height / 2,
      )
      await page.mouse.down()
      await page.mouse.move(
        thumb.x + thumb.width / 2,
        trackBounds.y + trackBounds.height - 1,
      )
      await page.mouse.up()
      await expect(scrollbar).toHaveAttribute('aria-valuenow', '100')
      const position = {x: trackBounds.width / 2, y: 1}
      if (viewport.touch) await scrollbar.tap({position})
      else await scrollbar.click({position})
      await expect(scrollbar).toHaveAttribute('aria-valuenow', '0')

      // Native reading keys on the text must not change slides.
      await content.focus()
      await page.keyboard.press('Space')
      await expect
        .poll(() => content.evaluate((node) => node.scrollTop))
        .toBeGreaterThan(0)
      await expect(page.getByText('1 / 4', {exact: true})).toBeVisible()
      await page.getByRole('button', {name: '다음 ▶', exact: true}).click()
      await expect(content).toHaveText('짧은 노트입니다.')
      await expect(scrollbar).toHaveAttribute('aria-disabled', 'true')
      await expect(scrollbar).toBeHidden()
      await expect(more).toHaveCount(0)
      await expect(panel.getByText('노트 전체 표시 중')).toBeVisible()
      await page.getByRole('button', {name: '다음 ▶', exact: true}).click()
      await expect(content).toHaveText('노트 없음')
      await expect(scrollbar).toBeHidden()
      await expect(more).toHaveCount(0)
      await page.getByRole('button', {name: '다음 ▶', exact: true}).click()
      await expect(more).toBeVisible()
      await expect(scrollbar).toBeVisible()
      await expect(scrollbar).toHaveAttribute('aria-valuenow', '0')
      await expect
        .poll(() => content.evaluate((node) => node.scrollTop))
        .toBe(0)
      await page.setViewportSize({width: 700, height: 500})
      await expect(more).toBeVisible()
      const panelBounds = (await panel.boundingBox())!
      expect(panelBounds.y + panelBounds.height).toBeLessThan(500)
    })

    test('dragging the top edge of the notes resizes them across slide changes', async ({
      page,
      request,
    }) => {
      await openNotes(page, request)
      const panel = page.getByRole('region', {name: '발표자 노트', exact: true})
      const content = page.getByRole('region', {name: '발표자 노트 본문'})
      const handle = page.getByRole('separator', {
        name: '발표자 노트 높이 조절',
      })
      const preview = page.locator('.marp-presenter-slide').first()
      const next = page.getByRole('button', {name: '다음 ▶', exact: true})
      const height = async () => (await panel.boundingBox())!.height
      const drag = async (toY: number) => {
        const grip = (await handle.boundingBox())!
        const x = grip.x + grip.width / 2
        await page.mouse.move(x, grip.y + grip.height / 2)
        await page.mouse.down()
        await page.mouse.move(x, toY, {steps: 4})
        await page.mouse.up()
      }

      const before = (await panel.boundingBox())!
      const previewBefore = (await preview.boundingBox())!.height
      const grip = (await handle.boundingBox())!
      await drag(grip.y + grip.height / 2 - 30)
      const taller = (await panel.boundingBox())!
      expect(taller.height).toBeGreaterThan(before.height)
      if (viewport.name === 'desktop')
        expect(Math.abs(taller.height - before.height - 30)).toBeLessThan(2)
      // The notes grow upward, the controls below stay put.
      expect(
        Math.abs(taller.y + taller.height - before.y - before.height),
      ).toBeLessThan(2)
      expect((await preview.boundingBox())!.height).toBeLessThan(previewBefore)

      // Dragging past the top still leaves a slide preview and the controls on screen.
      await drag(1)
      const largest = await height()
      expect((await preview.boundingBox())!.height).toBeGreaterThan(20)
      const nextBounds = (await next.boundingBox())!
      expect(nextBounds.y + nextBounds.height).toBeLessThanOrEqual(
        viewport.height,
      )

      await next.click()
      await expect(content).toHaveText('짧은 노트입니다.')
      expect(Math.abs((await height()) - largest)).toBeLessThan(2)

      await drag(viewport.height - 1)
      expect(await height()).toBeLessThan(before.height)
      await expect(content).toBeVisible()
    })
  })
}
