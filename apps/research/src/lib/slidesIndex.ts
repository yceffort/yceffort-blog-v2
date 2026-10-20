import fs from 'fs'
import path from 'path'

import matter from 'gray-matter'
import {cacheLife, cacheTag} from 'next/cache'
import {cache} from 'react'

import {isTransitionType} from '@/components/MarpSlides.constants'
import type {TransitionType} from '@/components/MarpSlides.constants'

import {generateRenderedMarp} from './marp'

export interface SlideIndexEntry {
  slug: string
  title: string
  description?: string
  tags?: string[]
  date?: string
  published: boolean
  featured: boolean
  post?: string
  transition?: TransitionType
  markdown: string
}

const RESEARCH_DIR = path.join(process.cwd(), 'research')

// YAML의 unquoted date는 gray-matter가 Date 객체로 파싱하므로,
// String()으로 감싸면 "Wed Aug 05 ..." 형태가 되어 localeCompare 정렬이 깨진다.
// 항상 yyyy-MM-dd로 정규화한다.
function normalizeDate(value: unknown): string | undefined {
  if (!value) {
    return undefined
  }
  const date = new Date(value as string | Date)
  if (Number.isNaN(date.getTime())) {
    return undefined
  }
  return date.toISOString().slice(0, 10)
}

function toEntry(slug: string, markdown: string): SlideIndexEntry {
  const {data} = matter(markdown)
  return {
    slug,
    title: data.title ? String(data.title) : slug,
    description: data.description ? String(data.description) : undefined,
    tags: Array.isArray(data.tags) ? (data.tags as string[]) : undefined,
    date: normalizeDate(data.date),
    published: data.published !== false,
    featured: data.featured === true,
    post: typeof data.post === 'string' ? data.post : undefined,
    transition: isTransitionType(data.transition) ? data.transition : undefined,
    markdown,
  }
}

export const getAllSlides = cache(
  function getAllSlidesImpl(): SlideIndexEntry[] {
    return fs
      .readdirSync(RESEARCH_DIR)
      .filter((f) => f.endsWith('.md'))
      .map((file) =>
        toEntry(
          file.replace(/\.md$/, ''),
          fs.readFileSync(path.join(RESEARCH_DIR, file), 'utf-8'),
        ),
      )
      .toSorted((a, b) => (b.date ?? '').localeCompare(a.date ?? ''))
  },
)

// 목록, 피드, 사이트맵, llms에 싣는 덱
export function getPublishedSlides() {
  return getAllSlides().filter((slide) => slide.published)
}

// published: false 덱은 프로덕션에서 직접 접근(뷰어, 발표자, raw, offline)도 막는다. dev에서는 초안 확인용으로 연다
function isAccessible(slide: SlideIndexEntry) {
  return process.env.NODE_ENV !== 'production' || slide.published
}

export const getSlideBySlug = cache(function getSlideBySlugImpl(
  slug: string,
): SlideIndexEntry | null {
  const filePath = path.join(RESEARCH_DIR, `${slug}.md`)
  if (!fs.existsSync(filePath)) {
    return null
  }
  const entry = toEntry(slug, fs.readFileSync(filePath, 'utf-8'))
  return isAccessible(entry) ? entry : null
})

export function getSlideStaticParams() {
  return getAllSlides()
    .filter(isAccessible)
    .map(({slug}) => ({slug}))
}

const THUMB_DIR = path.join(process.cwd(), 'public/thumbnails')

// scripts/generate-cover-thumbnail.mjs 가 만든 webp 가 있으면 그 경로. 없으면 /api/og 폴백
// 재생성해도 파일명이 같아 캐시가 안 깨지므로 수정 시각을 버전으로 붙인다
function resolveThumbnail(slug: string): string | undefined {
  const file = path.join(THUMB_DIR, `${slug}.webp`)
  if (!fs.existsSync(file)) {
    return undefined
  }
  const version = Math.floor(fs.statSync(file).mtimeMs / 1000).toString(36)
  return `/thumbnails/${slug}.webp?v=${version}`
}

// 뷰어와 발표자 화면이 같은 캐시 항목을 공유한다
export async function getRenderedSlide(slug: string) {
  'use cache'
  cacheLife('max')
  cacheTag(`slide:${slug}`)

  const entry = getSlideBySlug(slug)
  if (!entry) {
    return null
  }
  const {markdown, ...meta} = entry
  return {
    ...meta,
    ...(await generateRenderedMarp(markdown)),
    thumbnail: resolveThumbnail(slug),
  }
}
