/**
 * 덱의 첫 장을 찍어 OG 썸네일(public/thumbnails/{slug}.webp, 1200x630)로 저장한다.
 * 렌더는 앱과 같은 src/lib/marp.ts를 쓴다. 16:9 장을 1.91:1로 맞추느라 위아래를 24px씩 잘라 낸다.
 *
 * Usage:
 *   node scripts/generate-cover-thumbnail.mjs <deck-file-path>...
 *   node scripts/generate-cover-thumbnail.mjs --all
 */
import {
  createReadStream,
  existsSync,
  readdirSync,
  readFileSync,
  statSync,
} from 'node:fs'
import {createServer} from 'node:http'
import {registerHooks} from 'node:module'
import {basename, extname, resolve} from 'node:path'

import {chromium} from '@playwright/test'
import sharp from 'sharp'

// marp.ts는 확장자 없이 상대 import한다. Node가 찾을 수 있게 .ts를 붙여 본다
registerHooks({
  resolve(specifier, context, next) {
    try {
      return next(specifier, context)
    } catch (err) {
      if (!specifier.startsWith('.') || extname(specifier)) throw err
      return next(`${specifier}.ts`, context)
    }
  },
})
const {generateRenderedMarp} = await import('../src/lib/marp.ts')

const APP_ROOT = resolve(import.meta.dirname, '..')
const DECK_DIR = resolve(APP_ROOT, 'research')
const PUBLIC_DIR = resolve(APP_ROOT, 'public')
const THUMB_DIR = resolve(PUBLIC_DIR, 'thumbnails')

const args = process.argv.slice(2)
const files = args.includes('--all')
  ? readdirSync(DECK_DIR)
      .filter((f) => f.endsWith('.md'))
      .map((f) => resolve(DECK_DIR, f))
  : args.map((p) => resolve(p))
if (files.length === 0) {
  console.error(
    'Usage: node scripts/generate-cover-thumbnail.mjs <deck-file-path>... | --all',
  )
  process.exit(1)
}

// 덱 안의 /fonts/..., /feconf-2026/... 같은 절대 경로를 앱과 같이 public에서 찾는다
const TYPES = {
  '.woff2': 'font/woff2',
  '.woff': 'font/woff',
  '.ttf': 'font/ttf',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.webp': 'image/webp',
  '.gif': 'image/gif',
}
const server = createServer((req, res) => {
  const file = resolve(
    PUBLIC_DIR,
    `.${decodeURIComponent(req.url.split('?')[0])}`,
  )
  if (
    !file.startsWith(PUBLIC_DIR) ||
    !existsSync(file) ||
    statSync(file).isDirectory()
  ) {
    res.writeHead(404).end()
    return
  }
  res.writeHead(200, {
    'content-type': TYPES[extname(file)] ?? 'application/octet-stream',
  })
  createReadStream(file).pipe(res)
}).listen(0)
const origin = `http://localhost:${server.address().port}`

const browser = await chromium.launch()
const page = await browser.newPage({
  viewport: {width: 1280, height: 720},
  deviceScaleFactor: 2,
})
try {
  for (const file of files) {
    const slug = basename(file, '.md')
    const {html, css, fonts} = await generateRenderedMarp(
      readFileSync(file, 'utf8'),
    )
    await page.goto(`${origin}/`)
    await page.setContent(
      `<!doctype html><style>${fonts.join('\n')} body{margin:0}</style><span id="host"></span>`,
    )
    // Marp.tsx와 같이 shadow root에 테마 css와 여백 규칙을 넣는다
    await page.evaluate(
      async ({slide, theme}) => {
        const root = document
          .getElementById('host')
          .attachShadow({mode: 'open'})
        root.innerHTML = `<style>${theme}
:host > [data-marpit-svg] { display: block; width: 1280px; height: 720px; }
section { padding: var(--marp-slide-padding, 30px 40px) !important; }</style>${slide}`
        await Promise.all(
          [...root.querySelectorAll('img')].map((img) =>
            img.decode().catch(() => {}),
          ),
        )
        await document.fonts.ready
      },
      {slide: html[0], theme: css},
    )
    const png = await page.screenshot({
      clip: {x: 0, y: 0, width: 1280, height: 720},
    })
    await sharp(png)
      .resize(1200, 675)
      .extract({left: 0, top: 22, width: 1200, height: 630})
      .webp({quality: 86})
      .toFile(resolve(THUMB_DIR, `${slug}.webp`))
    console.log(`done  ${slug}`)
  }
} finally {
  await browser.close()
  server.close()
}
