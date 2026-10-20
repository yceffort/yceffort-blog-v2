import {stripTitleEmphasis} from '@yceffort/shared/utils'
import {NextResponse} from 'next/server'

import {getAllPosts} from '@/utils/Post'
import {pathPrefixOf, type Locale} from '@/utils/postPaths'

const RECENT_POST_COUNT = 20

type RecentPost = {path: string; title: string}

export async function buildRecentPostsResponse(locale: Locale) {
  const allPosts = await getAllPosts(locale)
  const prefix = pathPrefixOf(locale)

  const posts: RecentPost[] = allPosts
    .slice(0, RECENT_POST_COUNT)
    .map((post) => ({
      path: `${prefix}/${post.fields.slug}`,
      title: stripTitleEmphasis(post.frontMatter.title),
    }))

  return NextResponse.json({posts})
}
