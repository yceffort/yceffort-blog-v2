/**
 * unDraw 일러스트로 썸네일을 만드는 공통 엔진. apps/blog의
 * scripts/generate-undraw-thumbnail.mjs가 대상 파일, slug 규칙, 키 위치만 넘겨 쓴다.
 *
 * 1) Claude가 제목/설명을 보고 unDraw 카탈로그(사람 없는 그림만)에서 이름 하나를 고른다
 *    → frontmatter `art.undraw`에 기록(있으면 그대로 쓴다. 손으로 바꾸고 다시 돌리면 됨)
 * 2) 그 SVG를 frontmatter `art.hue` 색으로 칠해 {thumbDir}/{slug}.webp 로 저장
 *
 * 고를 대상이 없으면 API를 부르지 않으므로 키 없이도 렌더링만 할 수 있다.
 */
import {
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  writeFileSync,
} from 'node:fs'
import {dirname, resolve} from 'node:path'

import Anthropic from '@anthropic-ai/sdk'
import sharp from 'sharp'

const SVG_DIR = resolve(
  dirname(new URL(import.meta.url).pathname),
  '../node_modules/undraw-svg/svgs',
)
const WIDTH = 1200
const HEIGHT = 630
const BATCH = 40

const ACCENT = {
  warm: '#d97706',
  rose: '#be185d',
  violet: '#6d28d9',
  blue: '#1d4ed8',
  cyan: '#0e7490',
  green: '#047857',
  slate: '#334155',
}
const PAPER = {
  warm: '#fbf6ee',
  rose: '#fbf1f4',
  violet: '#f4f1fb',
  blue: '#eff4fb',
  cyan: '#edf7f8',
  green: '#eef8f2',
  slate: '#f3f4f6',
}
const HUES = Object.keys(ACCENT)

// 사람 피부색 fill이 있는 그림과 이름에 사람이 드러나는 그림은 뺀다
const SKIN =
  /fill="#(a0616a|ffb6b6|ffb8b8|9f616a|ffb7b7|feb8b8|ffb9b9|e8a3a3|9e616a|fbbebe|be6f72|a06e6e|d0b0a0|f2a7a7|ed9da0|fed2b1|ffb5b5|ffd7ba)"/i
const PEOPLE_NAME =
  /walk|people|person|team|together|\bman\b|woman|girl|boy|friend|couple|family|selfie|yoga|fitness|danc|jogging|love|wedding|portrait|profile|avatar|hiring|interview|meditat|relax|sleep|hang/i

// 키가 들어 있는 첫 파일에서 읽는다. 파일만 있고 키가 없으면 다음 후보로 넘어간다
function loadEnv(envPaths, key) {
  for (const path of envPaths.filter((p) => existsSync(p))) {
    const match = readFileSync(path, 'utf-8').match(
      new RegExp(`^${key}=(.+)$`, 'm'),
    )
    if (match) {
      return match[1].trim()
    }
  }
  throw new Error(`${key} not found in ${envPaths.join(', ')}`)
}

function loadCatalog() {
  return readdirSync(SVG_DIR)
    .filter((f) => f.endsWith('.svg'))
    .map((f) => f.replace(/\.svg$/, ''))
    .filter(
      (name) =>
        !PEOPLE_NAME.test(name) &&
        !SKIN.test(readFileSync(resolve(SVG_DIR, `${name}.svg`), 'utf-8')),
    )
}

function hashCode(str) {
  let h = 0
  for (const ch of str) {
    h = (h * 31 + ch.charCodeAt(0)) | 0
  }
  return Math.abs(h)
}

function readTarget(path, slugOf) {
  const content = readFileSync(path, 'utf-8')
  const match = content.match(/^---\n([\s\S]*?)\n---/)
  if (!match) {
    throw new Error(`frontmatter not found: ${path}`)
  }
  const fm = match[1]
  const field = (key) =>
    (fm.match(new RegExp(`^${key}: (.*)$`, 'm')) || [])[1]
      ?.replace(/^'|'$/g, '')
      .replace(/''/g, "'") ?? ''
  const artField = (key) =>
    (fm.match(new RegExp(`^  ${key}: (.*)$`, 'm')) || [])[1]
  const slug = slugOf(path)
  return {
    path,
    slug,
    title: field('title').replace(/<\/?em>/g, ''),
    description: field('description'),
    hue: HUES.includes(artField('hue'))
      ? artField('hue')
      : HUES[hashCode(slug) % HUES.length],
    undraw: artField('undraw'),
  }
}

function writeUndrawToFrontmatter(path, name) {
  const content = readFileSync(path, 'utf-8')
  const match = content.match(/^---\n([\s\S]*?)\n---/)
  if (!match) {
    return
  }
  let fm = match[1].replace(/^  undraw: .*\n?/m, '')
  if (/^art:\s*$/m.test(fm)) {
    fm = fm.replace(/^art:\s*$/m, `art:\n  undraw: ${name}`)
  } else {
    fm = `${fm}\nart:\n  undraw: ${name}`
  }
  writeFileSync(path, content.replace(match[0], `---\n${fm}\n---`))
}

async function pickNames(claude, targets, catalog, used, {subject, noun}) {
  const res = await claude.messages.create({
    model: 'claude-sonnet-5',
    max_tokens: 16000,
    messages: [
      {
        role: 'user',
        content: `아래 ${subject} 각각에 가장 어울리는 unDraw 일러스트를 카탈로그에서 하나씩 고른다. 사물 위주 카탈로그다. ${noun}의 핵심 사물이나 행위(서버, 컨테이너, 브라우저, 코드 창, 파일, 캐시, 네트워크, 번들, 애니메이션, 차트, 책, 열쇠 등)에 가장 가까운 이름을 고른다. 이 목록 안에서는 같은 이름을 두 ${noun}에 쓰지 않는다. "이미 쓰인 이름"은 가능하면 피하되 훨씬 더 어울리면 써도 된다. 카탈로그에 없는 이름은 절대 쓰지 않는다.
JSON 배열만 출력: [{"slug": "...", "pick": "<catalog name>"}]

## 카탈로그
${catalog.join(', ')}

## 이미 쓰인 이름
${used.length ? used.join(', ') : '(없음)'}

## ${noun}
${targets.map((t) => `- slug: ${t.slug}\n  title: ${t.title}\n  description: ${t.description.slice(0, 200)}`).join('\n')}`,
      },
    ],
  })
  const text = res.content.map((c) => c.text ?? '').join('')
  const picks = JSON.parse(
    text.slice(text.indexOf('['), text.lastIndexOf(']') + 1),
  )
  const map = new Map(picks.map((p) => [p.slug, p.pick]))
  for (const t of targets) {
    if (!catalog.includes(map.get(t.slug))) {
      throw new Error(`pick not in catalog: ${t.slug} → ${map.get(t.slug)}`)
    }
  }
  return map
}

async function render(target, thumbDir) {
  const svg = readFileSync(
    resolve(SVG_DIR, `${target.undraw}.svg`),
    'utf-8',
  ).replace(/<svg /, `<svg color="${ACCENT[target.hue]}" `)
  const illo = await sharp(Buffer.from(svg))
    .resize({width: 920, height: 540, fit: 'inside'})
    .png()
    .toBuffer()
  const meta = await sharp(illo).metadata()
  const outPath = resolve(thumbDir, `${target.slug}.webp`)
  mkdirSync(dirname(outPath), {recursive: true})
  await sharp({
    create: {
      width: WIDTH,
      height: HEIGHT,
      channels: 3,
      background: PAPER[target.hue],
    },
  })
    .composite([
      {
        input: illo,
        left: Math.round((WIDTH - meta.width) / 2),
        top: Math.round((HEIGHT - meta.height) / 2),
      },
    ])
    .webp({quality: 85})
    .toFile(outPath)
}

// translationsOf: 고른 이름을 같이 기록할 번역본 경로, wording: 프롬프트에 쓸 대상 이름
export async function generateUndrawThumbnails({
  files,
  force,
  usage,
  envPaths,
  thumbDir,
  slugOf,
  translationsOf = () => [],
  wording,
}) {
  if (files.length === 0) {
    console.error(usage)
    process.exit(1)
  }

  const catalog = loadCatalog()
  // 앱은 webp를 png보다 먼저 쓰므로, 실사 png를 직접 넣은 대상은 webp를 만들지 않는다
  const targets = files
    .map((path) => readTarget(path, slugOf))
    .filter((t) => !existsSync(resolve(thumbDir, `${t.slug}.png`)))
    .toSorted((a, b) => b.slug.localeCompare(a.slug))
  const used = new Set(
    targets.filter((t) => !force && t.undraw).map((t) => t.undraw),
  )
  const pending = targets.filter((t) => force || !t.undraw)
  console.log(
    `catalog ${catalog.length}, targets ${targets.length}, to pick ${pending.length}`,
  )

  if (pending.length > 0) {
    const claude = new Anthropic({
      apiKey: loadEnv(envPaths, 'ANTHROPIC_API_KEY'),
      defaultHeaders: {
        'anthropic-workspace-id': loadEnv(envPaths, 'ANTHROPIC_WORKSPACE_ID'),
      },
    })
    for (let i = 0; i < pending.length; i += BATCH) {
      const batch = pending.slice(i, i + BATCH)
      const picks = await pickNames(claude, batch, catalog, [...used], wording)
      for (const target of batch) {
        target.undraw = picks.get(target.slug)
        used.add(target.undraw)
        for (const path of [
          target.path,
          ...translationsOf(target.path).filter((p) => existsSync(p)),
        ]) {
          writeUndrawToFrontmatter(path, target.undraw)
        }
      }
      console.log(
        `picked ${Math.min(i + BATCH, pending.length)}/${pending.length}`,
      )
    }
  }

  for (const target of targets) {
    await render(target, thumbDir)
    console.log(`done  ${target.slug}  [${target.hue}]  ${target.undraw}`)
  }
}
