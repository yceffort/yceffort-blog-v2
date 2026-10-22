import fs from 'fs'
import path from 'path'

import frontMatter from 'front-matter'
import {sync} from 'glob'
import {cache} from 'react'

import type {Series} from '../type'
import {getAllPosts, getSeriesPosts} from './Post'
import type {Locale} from './postPaths'

const SERIES_ROOT = path.join(process.cwd(), 'series')

interface SeriesFrontMatter {
  name: string
  title?: string
  description: string
}

function latestPostDate(series: Series): string {
  return series.posts.reduce(
    (latest, post) =>
      post.frontMatter.date > latest ? post.frontMatter.date : latest,
    '',
  )
}

function readSeriesFile(filePath: string) {
  return frontMatter<SeriesFrontMatter>(
    fs.readFileSync(filePath, {encoding: 'utf8'}),
  )
}

export const getAllSeries = cache(async function getAllSeriesImpl(
  locale: Locale = 'ko',
): Promise<Series[]> {
  const files = sync(`${SERIES_ROOT}/*.md`).filter((f) => !f.endsWith('.en.md'))
  const enPosts = locale === 'en' ? await getAllPosts('en') : []

  const series = await Promise.all(
    files.map(async (filePath): Promise<Series> => {
      const ko = readSeriesFile(filePath)
      const koPosts = await getSeriesPosts(ko.attributes.name)
      // 영문본의 series 값은 글마다 표기가 달라 이름으로 묶지 않고, 한국어 원문 글의 번역본을 같은 순서로 모은다
      const enPath = filePath.replace(/\.md$/, '.en.md')
      const readmePath =
        locale === 'en' && fs.existsSync(enPath) ? enPath : filePath
      const {attributes, body} =
        readmePath === filePath ? ko : readSeriesFile(readmePath)
      return {
        slug: path.basename(filePath, '.md'),
        name: attributes.name,
        title: attributes.title ?? attributes.name,
        description: attributes.description,
        body,
        path: readmePath,
        posts:
          locale === 'en'
            ? koPosts.flatMap(
                (post) =>
                  enPosts.find((p) => p.fields.slug === post.fields.slug) ?? [],
              )
            : koPosts,
      }
    }),
  )

  return series
    .filter((s) => s.posts.length > 0)
    .toSorted((a, b) => (latestPostDate(a) < latestPostDate(b) ? 1 : -1))
})

export const findSeriesBySlug = cache(async function findSeriesBySlugImpl(
  slug: string,
  locale: Locale = 'ko',
): Promise<Series | undefined> {
  const series = await getAllSeries(locale)
  return series.find((s) => s.slug === slug)
})

export const findSeriesByPostSlug = cache(
  async function findSeriesByPostSlugImpl(
    postSlug: string,
    locale: Locale = 'ko',
  ): Promise<Series | undefined> {
    const series = await getAllSeries(locale)
    return series.find((s) =>
      s.posts.some((post) => post.fields.slug === postSlug),
    )
  },
)
