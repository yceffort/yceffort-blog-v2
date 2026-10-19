import * as stylex from '@stylexjs/stylex'
import {cacheLife, cacheTag} from 'next/cache'
import Link from 'next/link'
import {connection} from 'next/server'
import {Suspense} from 'react'

import Hero from '@/components/home/HeroE'
import * as recentStyles from '@/components/home/recent.styles'
import RecentRow from '@/components/home/RecentRow'
import * as ambientStyles from '@/components/layout/ambient.styles'
import * as sectionStyles from '@/components/layout/section.styles'
import PostCard from '@/components/post/PostCard'
import PopularSeriesCard from '@/components/series/PopularSeriesCard'
import SeriesRow from '@/components/series/SeriesRow'
import {HOME_RECENT_CARD_COUNT, HOME_SERIES_COUNT} from '@/constants'
import {
  resolveThumbnail,
  getAllPosts,
  getAllTagsFromPosts,
  getFeaturedPosts,
} from '@/utils/Post'
import type {Locale} from '@/utils/Post'
import {getAllSeries, getPopularSeries} from '@/utils/Series'

const LABELS = {
  ko: {pathPrefix: '', popularBadge: '인기 포스트'},
  en: {pathPrefix: '/en', popularBadge: 'Popular post'},
} as const

const sx = stylex.create({
  section: {
    '@layer utilities': {
      display: 'grid',
      gridTemplateColumns: {
        default: 'repeat(1, minmax(0, 1fr))',
        '@media (width >= 48rem)': 'repeat(2, minmax(0, 1fr))',
        '@media (width >= 64rem)': 'repeat(3, minmax(0, 1fr))',
      },
      gap: 'calc(var(--spacing) * 8)',
    },
  },
  rows: {
    '@layer utilities': {
      marginTop: 'calc(var(--spacing) * 8)',
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
  const topSeries = await getPopularSeries(locale)
  // 썸네일이 없으면 시리즈 카드를 그리지 않으므로 인기 글 자리도 비워 두지 않는다
  const popularSeries =
    topSeries && resolveThumbnail(`series/${topSeries.slug}`) ? topSeries : null
  const [{popular: posts, recent: recentPosts}, allPosts, tags, series] =
    await Promise.all([
      getFeaturedPosts(locale, popularSeries ? 1 : 0),
      getAllPosts(locale),
      getAllTagsFromPosts(locale),
      getAllSeries(locale),
    ])
  const postCount = allPosts.length
  const tagCount = tags.length
  const currentYear = new Date().getFullYear()
  const earliestYear = allPosts
    .map((p) => new Date(p.frontMatter.date).getFullYear())
    .reduce((a, b) => Math.min(a, b), currentYear)
  const yearsWriting = Math.max(1, currentYear - earliestYear + 1)
  return {
    posts,
    recentPosts,
    series,
    popularSeries,
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
    posts,
    recentPosts,
    series,
    popularSeries,
    postCount,
    tagCount,
    yearsWriting,
  } = homeData
  const {pathPrefix, popularBadge} = LABELS[locale]
  const popularSeriesThumbnail =
    popularSeries && resolveThumbnail(`series/${popularSeries.slug}`)
  return (
    <div className={`page-view ${ambientStyles.page_view}`}>
      <Hero
        postCount={postCount}
        tagCount={tagCount}
        yearsWriting={yearsWriting}
      />

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
          <section className={stylex.props(sx.section).className}>
            {recentPosts.slice(0, HOME_RECENT_CARD_COUNT).map((post) => (
              <PostCard
                key={post.fields.slug}
                post={post}
                pathPrefix={pathPrefix}
                priority
              />
            ))}
          </section>
          <section
            className={`rec-list ${recentStyles.rec_list} ${stylex.props(sx.rows).className}`}
          >
            {recentPosts.slice(HOME_RECENT_CARD_COUNT).map((post, i) => (
              <RecentRow
                key={post.fields.slug}
                post={post}
                index={HOME_RECENT_CARD_COUNT + i}
                pathPrefix={pathPrefix}
              />
            ))}
          </section>
        </>
      )}

      <div className={`sec-head ${sectionStyles.sec_head}`}>
        <div>
          <span className={`sec-count ${sectionStyles.sec_count}`}>
            {String(posts.length + (popularSeries ? 1 : 0)).padStart(2, '0')}{' '}
            ITEMS
          </span>
          <h2 className={sectionStyles.element_h2}>
            {'Popular '}
            <em className={sectionStyles.element_em}>this season</em>
          </h2>
        </div>
        <div className={`line ${sectionStyles.line}`} />
        <div className={`hint ${sectionStyles.hint}`}>hover · tilt · open</div>
      </div>
      <section className={stylex.props(sx.section).className}>
        {posts.map((post) => (
          <PostCard
            key={post.fields.slug}
            post={post}
            pathPrefix={pathPrefix}
            badge={popularBadge}
          />
        ))}
        {popularSeries && popularSeriesThumbnail && (
          <PopularSeriesCard
            series={popularSeries}
            thumbnail={popularSeriesThumbnail}
            pathPrefix={pathPrefix}
          />
        )}
      </section>

      {series.length > 0 && (
        <>
          <div className={`sec-head ${sectionStyles.sec_head}`}>
            <div>
              <span className={`sec-count ${sectionStyles.sec_count}`}>
                {String(series.length).padStart(2, '0')} ITEMS
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
          <section className={`rec-list ${recentStyles.rec_list}`}>
            {series.slice(0, HOME_SERIES_COUNT).map((s, i) => (
              <SeriesRow
                key={s.slug}
                series={s}
                index={i}
                pathPrefix={pathPrefix}
              />
            ))}
          </section>
        </>
      )}
    </div>
  )
}
