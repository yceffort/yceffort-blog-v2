import type {Element, ElementContent, Properties, Root, RootContent} from 'hast'
import {toJsxRuntime} from 'hast-util-to-jsx-runtime'
import type {ComponentProps} from 'react'
import {Fragment, jsx, jsxs} from 'react/jsx-runtime'
import remarkCjkFriendly from 'remark-cjk-friendly'
import remarkGfm from 'remark-gfm'
import remarkParse from 'remark-parse'
import remarkRehype from 'remark-rehype'
import {unified} from 'unified'

const processor = unified()
  .use(remarkParse)
  .use(remarkGfm)
  .use(remarkCjkFriendly)
  .use(remarkRehype)

// h2 제목이 절의 종류를 정한다. 그 밖의 h2는 주간의 흐름 절이다.
const SECTION_KINDS: Record<string, string> = {
  '오늘 하나만 본다면': 'pick',
  '읽을 만한 것': 'reads',
  '짧게 알아두기': 'brief',
  '수집 상태': 'status',
  '이번 주': 'overview',
}

function textOf(node: RootContent): string {
  if (node.type === 'text') {
    return node.value
  }
  if ('children' in node) {
    return node.children.map(textOf).join('')
  }
  return ''
}

// 태그를 포함한 교차 타입으로 좁혀야 거짓 분기에서 Element 전체가 빠지지 않는다.
function isElement<T extends string>(
  node: RootContent,
  tagName: T,
): node is Element & {tagName: T} {
  return node.type === 'element' && node.tagName === tagName
}

function wrap(tagName: string, properties: Properties): Element {
  return {type: 'element', tagName, properties, children: []}
}

// 항목 안의 문단에 역할을 붙인다. 첫 문단이 출처 표기(코드로 시작)면 meta,
// 그다음 문단이 훅, 요점 목록 뒤의 문단이 "왜 볼 만한가"다.
function labelParagraphs(article: Element) {
  let seenList = false
  let paragraphs = 0
  for (const child of article.children) {
    if (isElement(child, 'ul')) {
      seenList = true
      continue
    }
    if (!isElement(child, 'p')) {
      continue
    }
    const first = child.children[0]
    const className = seenList
      ? 'news-why'
      : paragraphs === 0 &&
          first?.type === 'element' &&
          first.tagName === 'code'
        ? 'news-meta'
        : paragraphs <= 1
          ? 'news-hook'
          : ''
    paragraphs += 1
    if (className) {
      child.properties = {...child.properties, className: [className]}
    }
  }
}

// h2마다 section, h3마다 article로 감싸 CSS가 절의 종류별로 꾸밀 수 있게 한다.
function sectionize(root: Root): Root {
  const out: ElementContent[] = []
  let section: Element | null = null
  let article: Element | null = null

  for (const node of root.children) {
    if (node.type === 'doctype') {
      continue
    }
    if (isElement(node, 'h2')) {
      const kind = SECTION_KINDS[textOf(node).trim()] ?? 'thread'
      section = wrap('section', {
        className: ['news-section'],
        'data-kind': kind,
      })
      section.children.push(node)
      article = null
      out.push(section)
      continue
    }
    if (isElement(node, 'h3')) {
      article = wrap('article', {className: ['news-item']})
      article.children.push(node)
      ;(section ?? {children: out}).children.push(article)
      continue
    }
    ;(article ?? section ?? {children: out}).children.push(node)
  }

  for (const node of out) {
    if (isElement(node, 'section')) {
      for (const child of node.children) {
        if (isElement(child, 'article')) {
          labelParagraphs(child)
        }
      }
    }
  }
  return {type: 'root', children: out}
}

function Anchor({href, children, ...rest}: ComponentProps<'a'>) {
  const external = typeof href === 'string' && /^https?:/.test(href)
  return (
    <a
      href={href}
      target={external ? '_blank' : undefined}
      rel={external ? 'noopener noreferrer' : undefined}
      {...rest}
    >
      {children}
    </a>
  )
}

export function renderIssueBody(body: string) {
  const tree = processor.runSync(processor.parse(body))
  return toJsxRuntime(sectionize(tree), {
    Fragment,
    jsx,
    jsxs,
    components: {a: Anchor},
  })
}
