import {readdirSync, readFileSync} from 'node:fs'
import {join} from 'node:path'

import frontMatter from 'front-matter'

export type IssueKind = 'daily' | 'weekly'

export interface Issue {
  kind: IssueKind
  slug: string
  date: string
  count: number
  period: {start: string; end: string} | null
  // "오늘 하나만 본다면"의 제목. 목록에서 호를 대표하는 한 줄로 쓴다.
  lead: string
  body: string
}

interface Attributes {
  date?: string | Date
  count?: number
  period?: string
}

const CONTENT_DIR = join(process.cwd(), 'content')

function toDateString(value: string | Date | undefined, fallback: string) {
  if (value instanceof Date) {
    return value.toISOString().slice(0, 10)
  }
  return value ?? fallback
}

function parsePeriod(value: string | undefined) {
  if (!value) {
    return null
  }
  const [start, end] = value.split('~').map((part) => part.trim())
  return start && end ? {start, end} : null
}

export function getIssue(kind: IssueKind, slug: string): Issue | null {
  let raw: string
  try {
    raw = readFileSync(join(CONTENT_DIR, kind, `${slug}.md`), 'utf8')
  } catch {
    return null
  }
  const {attributes, body} = frontMatter<Attributes>(raw)
  const period = parsePeriod(attributes.period)
  return {
    kind,
    slug,
    date: toDateString(attributes.date, period?.end ?? slug),
    count: attributes.count ?? 0,
    period,
    lead: body.match(/^### (.+)$/m)?.[1]?.trim() ?? '',
    body,
  }
}

export function listIssues(kind: IssueKind): Issue[] {
  return readdirSync(join(CONTENT_DIR, kind))
    .filter((name) => name.endsWith('.md'))
    .map((name) => name.slice(0, -'.md'.length))
    .toSorted()
    .toReversed()
    .map((slug) => getIssue(kind, slug))
    .filter((issue): issue is Issue => issue !== null)
}

export function getLatest(kind: IssueKind): Issue | null {
  return listIssues(kind)[0] ?? null
}

// 목록은 최신순이므로 다음 호가 앞에 온다.
export function getNeighbors(kind: IssueKind, slug: string) {
  const issues = listIssues(kind)
  const index = issues.findIndex((issue) => issue.slug === slug)
  return {
    next: index > 0 ? issues[index - 1] : null,
    prev: index >= 0 && index < issues.length - 1 ? issues[index + 1] : null,
  }
}
