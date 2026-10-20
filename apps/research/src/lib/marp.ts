import {Marp as MarpCore} from '@marp-team/marp-core'
import markdownItCjkFriendly from 'markdown-it-cjk-friendly'
import postcss from 'postcss'
import type {Result as PostCSSResult, AtRule} from 'postcss'
import postcssImportUrl from 'postcss-import-url'

import {parsePresenterNotes} from './parsePresenterNotes'
import {feconfTheme} from './themes/feconf'
import {inkTheme} from './themes/ink'
import {kakaoTheme} from './themes/kakao'
import {midnightTheme} from './themes/midnight'
import {yceffortTheme} from './themes/yceffort'

const postcssStripFontFace = Object.assign(
  () => ({
    postcssPlugin: 'marp-strip-font-face',
    AtRule: (
      rule: AtRule,
      {result}: {result: PostCSSResult & {fonts?: AtRule[]}},
    ) => {
      if (rule.name === 'font-face') {
        if (!result.fonts) {
          result.fonts = []
        }
        result.fonts.push(rule)
        rule.remove()
      }
    },
  }),
  {postcss: true as const},
)

interface RenderedMarp {
  html: string[]
  css: string
  fonts: string[]
  notes: string[]
}

const marpCache = new Map<string, Promise<RenderedMarp>>()

export function generateRenderedMarp(markdown: string): Promise<RenderedMarp> {
  const cached = marpCache.get(markdown)
  if (cached) {
    return cached
  }
  const pending = renderMarp(markdown)
  marpCache.set(markdown, pending)
  return pending
}

async function renderMarp(markdown: string): Promise<RenderedMarp> {
  const marp = new MarpCore({
    container: false,
    script: false,
    printable: false,
  })

  // 한글처럼 띄어쓰기 없이 구두점과 붙는 텍스트에서 **강조**가 리터럴로 노출되는
  // CommonMark flanking 규칙 문제를 구제한다. blog의 remark-cjk-friendly와 같은 프로젝트다.
  marp.use(markdownItCjkFriendly)

  // 본문 링크는 상대 경로와 앵커를 포함해 새 탭에서 연다.
  marp.use((md) => {
    md.renderer.rules.link_open = (
      tokens: {attrSet: (name: string, value: string) => void}[],
      idx: number,
      options: unknown,
      _env: unknown,
      slf: {
        renderToken: (tokens: unknown, idx: number, options: unknown) => string
      },
    ) => {
      tokens[idx].attrSet('target', '_blank')
      tokens[idx].attrSet('rel', 'noopener noreferrer')
      return slf.renderToken(tokens, idx, options)
    }
  })

  // mermaid 펜스는 서버에서 파싱하지 않고 placeholder만 남긴다.
  // 실제 렌더는 클라이언트(Marp.tsx)가 textContent를 읽어 수행한다.
  marp.use((md) => {
    const defaultFence = md.renderer.rules.fence
    md.renderer.rules.fence = (
      tokens: {info: string; content: string}[],
      idx: number,
      options: unknown,
      env: unknown,
      slf: unknown,
    ) => {
      const token = tokens[idx]
      if (token.info.trim() === 'mermaid') {
        return `<div class="mermaid">${md.utils.escapeHtml(token.content.trim())}</div>`
      }
      return defaultFence(tokens, idx, options, env, slf)
    }
  })

  marp.themeSet.add(yceffortTheme)
  marp.themeSet.add(midnightTheme)
  marp.themeSet.add(kakaoTheme)
  marp.themeSet.add(feconfTheme)
  marp.themeSet.add(inkTheme)

  const {html, css} = marp.render(markdown, {htmlAsArray: true})

  const result = await postcss()
    .use(postcssImportUrl({resolveUrls: true}))
    .use(postcssStripFontFace)
    .process(css, {from: undefined})

  const typedResult = result as PostCSSResult & {fonts?: AtRule[]}
  const fonts: string[] = (typedResult.fonts || []).map((font) =>
    font.toString(),
  )
  const notes = parsePresenterNotes(markdown)

  return {
    html,
    css: result.css,
    fonts,
    notes,
  }
}
