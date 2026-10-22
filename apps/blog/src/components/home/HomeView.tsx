import * as stylex from '@stylexjs/stylex'
import {cacheLife, cacheTag} from 'next/cache'
import Link from 'next/link'
import {connection} from 'next/server'
import {Suspense} from 'react'

import FeaturedPost from '@/components/home/FeaturedPost'
import Hero from '@/components/home/HeroE'
import * as recentStyles from '@/components/home/recent.styles'
import RecentRow from '@/components/home/RecentRow'
import * as ambientStyles from '@/components/layout/ambient.styles'
import * as sectionStyles from '@/components/layout/section.styles'
import SeriesCard from '@/components/series/SeriesCard'
import {HOME_SERIES_COUNT} from '@/constants'
import {
  resolveThumbnail,
  getAllPosts,
  getAllTagsFromPosts,
  getFeaturedPosts,
} from '@/utils/Post'
import type {Locale} from '@/utils/Post'
import {getAllSeries} from '@/utils/Series'

const LABELS = {
  ko: {pathPrefix: ''},
  en: {pathPrefix: '/en'},
} as const

const sx = stylex.create({
  section: {
    '@layer utilities': {
      display: 'grid',
      // 시리즈 카드 4장이 어느 폭에서도 빈자리 없이 떨어지도록 1열, 2열, 4열만 쓴다
      gridTemplateColumns: {
        default: 'repeat(1, minmax(0, 1fr))',
        '@media (width >= 48rem)': 'repeat(2, minmax(0, 1fr))',
        '@media (width >= 80rem)': 'repeat(4, minmax(0, 1fr))',
      },
      gap: 'calc(var(--spacing) * 8)',
    },
  },
})
async function getCachedHomeData(locale: Locale) {
  'use cache'

  cacheLife('hours')
  cacheTag(`home:${locale}`)
  return getHomeData(locale)
}
async function getHomeData(locale: Locale) {
  const [{popular: posts, recent}, allPosts, tags, series] = await Promise.all([
    getFeaturedPosts(locale),
    getAllPosts(locale),
    getAllTagsFromPosts(locale),
    getAllSeries(locale),
  ])
  // 대표 글은 썸네일이 있는 가장 새 글. 공지처럼 썸네일이 없는 글은 줄 목록으로만 보여준다
  const featured = recent.find((p) => p.frontMatter.thumbnail) ?? null
  const recentPosts = recent.filter((p) => p !== featured)
  // 커버가 없는 시리즈는 카드를 그리지 않는다
  const seriesCards = series
    .flatMap((s) => {
      const thumbnail = resolveThumbnail(`series/${s.slug}`)
      return thumbnail ? [{series: s, thumbnail}] : []
    })
    .slice(0, HOME_SERIES_COUNT)
  const postCount = allPosts.length
  const tagCount = tags.length
  const currentYear = new Date().getFullYear()
  const earliestYear = allPosts
    .map((p) => new Date(p.frontMatter.date).getFullYear())
    .reduce((a, b) => Math.min(a, b), currentYear)
  const yearsWriting = Math.max(1, currentYear - earliestYear + 1)
  return {
    featured,
    posts,
    recentPosts,
    seriesCards,
    seriesCount: series.length,
    postCount,
    tagCount,
    yearsWriting,
  }
}
export default function HomeView({locale}: {locale: Locale}) {
  return (
    <Suspense>
      <HomeContent locale={locale} />
    </Suspense>
  )
}
async function HomeContent({locale}: {locale: Locale}) {
  if (process.env.NODE_ENV !== 'production') {
    await connection()
  }
  const homeData =
    process.env.NODE_ENV === 'production'
      ? await getCachedHomeData(locale)
      : await getHomeData(locale)
  const {
    featured,
    posts,
    recentPosts,
    seriesCards,
    seriesCount,
    postCount,
    tagCount,
    yearsWriting,
  } = homeData
  const {pathPrefix} = LABELS[locale]
  return (
    <div className={`page-view ${ambientStyles.page_view}`}>
      <Hero
        postCount={postCount}
        tagCount={tagCount}
        yearsWriting={yearsWriting}
      />

      {featured && (
        <>
          <div className={`sec-head ${sectionStyles.sec_head}`}>
            <div>
              <span className={`sec-count ${sectionStyles.sec_count}`}>
                01 ITEMS
              </span>
              <h2 className={sectionStyles.element_h2}>Latest</h2>
            </div>
            <div className={`line ${sectionStyles.line}`} />
            <div className={`hint ${sectionStyles.hint}`}>just published</div>
          </div>
          <FeaturedPost post={featured} pathPrefix={pathPrefix} />
        </>
      )}

      {posts.length > 0 && (
        <>
          <div className={`sec-head ${sectionStyles.sec_head}`}>
            <div>
              <span className={`sec-count ${sectionStyles.sec_count}`}>
                {String(posts.length).padStart(2, '0')} ITEMS
              </span>
              <h2 className={sectionStyles.element_h2}>
                {'Popular '}
                <em className={sectionStyles.element_em}>this season</em>
              </h2>
            </div>
            <div className={`line ${sectionStyles.line}`} />
            <div className={`hint ${sectionStyles.hint}`}>30 days</div>
          </div>
          <section className={`rec-list ${recentStyles.rec_list}`}>
            {posts.map((post, i) => (
              <RecentRow
                key={post.fields.slug}
                post={post}
                index={i}
                variant="thumb"
                pathPrefix={pathPrefix}
              />
            ))}
          </section>
        </>
      )}

      {recentPosts.length > 0 && (
        <>
          <div className={`sec-head ${sectionStyles.sec_head}`}>
            <div>
              <span className={`sec-count ${sectionStyles.sec_count}`}>
                {String(recentPosts.length).padStart(2, '0')} ITEMS
              </span>
              <h2 className={sectionStyles.element_h2}>Recent</h2>
            </div>
            <div className={`line ${sectionStyles.line}`} />
            <div className={`hint ${sectionStyles.hint}`}>latest writing</div>
          </div>
          <section className={`rec-list ${recentStyles.rec_list}`}>
            {recentPosts.map((post, i) => (
              <RecentRow
                key={post.fields.slug}
                post={post}
                index={i}
                variant="thumb"
                pathPrefix={pathPrefix}
              />
            ))}
          </section>
        </>
      )}

      {seriesCards.length > 0 && (
        <>
          <div className={`sec-head ${sectionStyles.sec_head}`}>
            <div>
              <span className={`sec-count ${sectionStyles.sec_count}`}>
                {String(seriesCount).padStart(2, '0')} ITEMS
              </span>
              <h2 className={sectionStyles.element_h2}>
                {'Series '}
                <em className={sectionStyles.element_em}>one thread</em>
              </h2>
            </div>
            <div className={`line ${sectionStyles.line}`} />
            <div className={`hint ${sectionStyles.hint}`}>
              <Link href={`${pathPrefix}/series`}>view all →</Link>
            </div>
          </div>
          <section className={stylex.props(sx.section).className}>
            {seriesCards.map(({series, thumbnail}) => (
              <SeriesCard
                key={series.slug}
                series={series}
                thumbnail={thumbnail}
                pathPrefix={pathPrefix}
              />
            ))}
          </section>
        </>
      )}
    </div>
  )
}
