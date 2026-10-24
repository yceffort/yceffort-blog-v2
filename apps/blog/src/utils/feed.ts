import {stripTitleEmphasis} from '@yceffort/shared/utils'
import {NextResponse} from 'next/server'

import {SiteConfig} from '@/config'
import {getAllPosts} from '@/utils/Post'
import {toInstant} from '@/utils/postDate'
import {pathPrefixOf, type Locale} from '@/utils/postPaths'

export async function buildFeedResponse(locale: Locale) {
  const allPosts = await getAllPosts(locale)
  const siteUrl = `${SiteConfig.url}${pathPrefixOf(locale)}`
  // 인자 없는 new Date()는 요청 시점 IO로 간주돼 라우트가 동적이 되고, 런타임에
  // posts를 읽으려다 500이 난다. 최신 글 날짜(빌드 타임 고정값)를 쓴다.
  const lastBuildDate = (
    allPosts[0] ? toInstant(allPosts[0].frontMatter.date) : new Date(0)
  ).toUTCString()

  const feedXml = `<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">
  <channel>
    <title>${SiteConfig.title}${locale === 'en' ? ' (English)' : ''}</title>
    <link>${siteUrl}</link>
    <atom:link href="${siteUrl}/feed.xml" rel="self" type="application/rss+xml" />
    <description>${SiteConfig.subtitle}</description>
    <language>${locale === 'en' ? 'en' : 'ko-KR'}</language>
    <lastBuildDate>${lastBuildDate}</lastBuildDate>
    ${allPosts
      .map((post) => {
        const postDate = toInstant(post.frontMatter.date)
        const postUrl = `${siteUrl}/${post.fields.slug}`
        const postDescription =
          post.frontMatter.description || post.body.substring(0, 150) + '...'

        return `
        <item>
          <title><![CDATA[${stripTitleEmphasis(post.frontMatter.title)}]]></title>
          <link>${postUrl}?utm_source=rss&amp;utm_medium=feed</link>
          <guid>${postUrl}</guid>
          <pubDate>${postDate.toUTCString()}</pubDate>
          <description><![CDATA[${postDescription}]]></description>
          ${post.frontMatter.tags
            .map((tag) => `<category><![CDATA[${tag}]]></category>`)
            .join('')}
        </item>
      `
      })
      .join('')}
  </channel>
</rss>`

  return new NextResponse(feedXml, {
    headers: {
      'Content-Type': 'application/xml',
    },
  })
}
