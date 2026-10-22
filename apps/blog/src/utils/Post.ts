import fs from 'fs'

import frontMatter from 'front-matter'
import {sync} from 'glob'
import {cache} from 'react'
import readingTime from 'reading-time'

import {
  POPULAR_POSTS_COUNT,
  PRERENDER_POSTS_COUNT,
  RECENT_POSTS_COUNT,
} from '@/constants'

import relatedBySlug from '../../data/related-posts.json'
import type {ArtSpec, FrontMatter, Post, TagWithCount} from '../type'
import {getPopularPostSlugs} from './analytics'
import {POST_ROOT, isLocaleFile, pathToSlug} from './postPaths'
import type {Locale} from './postPaths'

const THUMB_DIR = `${process.cwd()}/public/thumbnails`

// 코드 생성 썸네일(api/og/art) 캐시 무효화용 버전. 아트 디자인이 바뀌면 올린다.
const ART_VERSION = 6

// public/thumbnails 에 생성 일러스트(webp)나 실사(png)가 있으면 그 경로, 없으면 frontmatter art 로 코드 아트 URL
// 둘 다 없는 옛 글은 썸네일 없이 둔다
// 재생성해도 파일명이 같아 캐시가 안 깨지므로 수정 시각을 버전으로 붙인다
export function resolveThumbnail(
  slug: string,
  art?: ArtSpec,
): string | undefined {
  const ext = ['webp', 'png'].find((e) =>
    fs.existsSync(`${THUMB_DIR}/${slug}.${e}`),
  )
  if (!ext) {
    return art ? buildArtThumbnail(slug, art) : undefined
  }
  const version = Math.floor(
    fs.statSync(`${THUMB_DIR}/${slug}.${ext}`).mtimeMs / 1000,
  ).toString(36)
  return `/thumbnails/${slug}.${ext}?v=${version}`
}

export function buildArtThumbnail(seed: string, art?: ArtSpec): string {
  const params = new URLSearchParams({v: String(ART_VERSION), slug: seed})
  if (art?.layout) {
    params.set('layout', art.layout)
  }
  if (art?.hue) {
    params.set('hue', art.hue)
  }
  if (art?.tone) {
    params.set('tone', art.tone)
  }
  return `/api/og/art?${params.toString()}`
}

export type {Locale}

// cache()는 요청 단위 중복 제거라 빌드 워커가 페이지를 여러 장 그릴 때 페이지마다 글 전체를
// 다시 읽는다. 빌드와 요청 시점에는 글이 바뀌지 않으므로 프로세스 수명 동안 재사용한다.
// 프로덕션 런타임(서버 인스턴스)에서도 같은 Map 이 살아 있어 전체 글을 메모리에 들고 있는다.
// 개발 모드는 초안 수정이 곧바로 반영되어야 하므로 채우지 않는다.
const allPostsCache = new Map<Locale, Post[]>()

export const getAllPosts = cache(async function getAllPostsImpl(
  locale: Locale = 'ko',
): Promise<Post[]> {
  const cached = allPostsCache.get(locale)
  if (cached) {
    return cached
  }

  const files = sync(`${POST_ROOT}/**/*.md*`).toReversed()

  const posts = files
    .filter((f) => isLocaleFile(f, locale))
    .reduce<Post[]>((prev, path) => {
      const file = fs.readFileSync(path, {encoding: 'utf8'})
      const {attributes, body} = frontMatter<FrontMatter>(file)
      const fm: FrontMatter = attributes
      const {tags: fmTags, published, date, updated} = fm

      const slug = pathToSlug(path)

      const isDev = process.env.NODE_ENV !== 'production'
      if (published || isDev) {
        const tags: string[] = (fmTags || []).map((tag: string) => tag.trim())
        const stats = readingTime(body, {wordsPerMinute: 250})

        const thumbnail = resolveThumbnail(slug, fm.art)
        const createdAt = new Date(date).toISOString().substring(0, 19)
        const updatedAt =
          updated && new Date(updated).toISOString().substring(0, 19)

        const result: Post = {
          frontMatter: {
            ...fm,
            tags,
            date: createdAt,
            // 발행한 날 고친 것은 수정으로 보여주지 않는다
            updated:
              updatedAt && updatedAt.slice(0, 10) > createdAt.slice(0, 10)
                ? updatedAt
                : undefined,
            thumbnail,
          },
          body,
          fields: {
            slug,
          },
          path,
          readingTime: Math.max(1, Math.ceil(stats.minutes)),
        }
        prev.push(result)
      }
      return prev
    }, [])
    .toSorted((a, b) => {
      if (a.frontMatter.date < b.frontMatter.date) {
        return 1
      }
      if (a.frontMatter.date > b.frontMatter.date) {
        return -1
      }
      return 0
    })

  if (process.env.NODE_ENV === 'production') {
    allPostsCache.set(locale, posts)
  }

  return posts
})

export const findPostByYearAndSlug = cache(
  async function findPostByYearAndSlugImpl(
    year: string,
    slug: string[],
    locale: Locale = 'ko',
  ) {
    const slugs = [year, ...slug].join('/')
    const posts = await getAllPosts(locale)
    return posts.find((p) => p?.fields?.slug === slugs)
  },
)

export const getAllTagsFromPosts = cache(async function getAllTagsFromPostsImpl(
  locale: Locale = 'ko',
): Promise<TagWithCount[]> {
  const posts = await getAllPosts(locale)
  const tagCountMap = new Map<string, number>()

  for (const post of posts) {
    for (const tag of post.frontMatter.tags) {
      tagCountMap.set(tag, (tagCountMap.get(tag) ?? 0) + 1)
    }
  }

  return Array.from(tagCountMap.entries())
    .map(([tag, count]) => ({tag, count}))
    .toSorted((a, b) => b.count - a.count)
})

export const getSeriesPosts = cache(async function getSeriesPostsImpl(
  seriesName: string,
  locale: Locale = 'ko',
): Promise<Post[]> {
  const posts = await getAllPosts(locale)
  return posts
    .filter((post) => post.frontMatter.series === seriesName)
    .toSorted(
      (a, b) =>
        (a.frontMatter.seriesOrder ?? 0) - (b.frontMatter.seriesOrder ?? 0),
    )
})

export async function getRelatedPosts(
  slug: string,
  tags: string[],
  locale: Locale = 'ko',
  excludeSeries?: string,
  limit = 4,
): Promise<Post[]> {
  const posts = await getAllPosts(locale)
  const tagSet = new Set(tags)

  // scripts/generate-related-posts.mjs 가 Jev로 고른 목록. 아직 없는 새 글은 태그 겹침으로 채운다.
  const picked = (
    (relatedBySlug[locale] as Record<string, {slug: string}[]>)[slug] ?? []
  )
    .map((r) => posts.find((p) => p.fields.slug === r.slug))
    .filter(
      (p): p is Post =>
        p != null &&
        (excludeSeries == null || p.frontMatter.series !== excludeSeries),
    )

  const byTags = posts
    .filter(
      (p) =>
        p.fields.slug !== slug &&
        (excludeSeries == null || p.frontMatter.series !== excludeSeries),
    )
    .map((post) => ({
      post,
      score: post.frontMatter.tags.reduce(
        (n, t) => (tagSet.has(t) ? n + 1 : n),
        0,
      ),
    }))
    .filter(({score}) => score > 0)
    .toSorted((a, b) =>
      b.score !== a.score
        ? b.score - a.score
        : a.post.frontMatter.date < b.post.frontMatter.date
          ? 1
          : -1,
    )
    .map(({post}) => post)

  return [...new Set([...picked, ...byTags])].slice(0, limit)
}

export const getFeaturedPosts = cache(async function getFeaturedPostsImpl(
  locale: Locale = 'ko',
): Promise<{popular: Post[]; recent: Post[]}> {
  const allPosts = await getAllPosts(locale)
  // 최근 글을 먼저 확정해야 새 글이 인기에 올라도 홈 맨 위 Recent에서 빠지지 않는다
  const recent = allPosts.slice(0, RECENT_POSTS_COUNT)
  const shown = new Set(recent.map((p) => p.fields.slug))
  const popularCount = POPULAR_POSTS_COUNT
  // 번역이 없거나 featured: false이거나 최근 글과 겹치는 글이 걸러지므로 후보를 넉넉히 가져온다
  const popularSlugs = await getPopularPostSlugs(
    POPULAR_POSTS_COUNT * 3 + RECENT_POSTS_COUNT,
  )

  const popular = popularSlugs
    .map((slug) => allPosts.find((p) => p.fields.slug === slug))
    .filter(
      (p): p is Post =>
        p != null &&
        p.frontMatter.featured !== false &&
        !shown.has(p.fields.slug),
    )
    .slice(0, popularCount)

  if (popular.length < popularCount) {
    const slugSet = new Set([...shown, ...popular.map((p) => p.fields.slug)])
    for (const p of allPosts) {
      if (popular.length >= popularCount) {
        break
      }
      if (!slugSet.has(p.fields.slug)) {
        popular.push(p)
        slugSet.add(p.fields.slug)
      }
    }
  }

  return {popular, recent}
})

// 빌드 시 사전 생성할 슬러그. 검색 롱테일 유입이 콜드 함수 렌더(TTFB 상승)를
// 밟지 않도록, 홈 노출용(getFeaturedPosts)보다 넓게 최근 1년 인기 상위를 포함한다.
export const getPrerenderSlugs = cache(async function getPrerenderSlugsImpl(
  locale: Locale = 'ko',
): Promise<string[]> {
  const allPosts = await getAllPosts(locale)
  const existing = new Set(allPosts.map((p) => p.fields.slug))
  const popularSlugs = await getPopularPostSlugs(PRERENDER_POSTS_COUNT, 365)

  const slugs = new Set(popularSlugs.filter((slug) => existing.has(slug)))
  for (const p of allPosts.slice(0, RECENT_POSTS_COUNT)) {
    slugs.add(p.fields.slug)
  }
  return [...slugs]
})
