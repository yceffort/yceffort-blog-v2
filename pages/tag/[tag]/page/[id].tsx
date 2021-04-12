import { GetStaticPaths, GetStaticProps } from 'next'
import { useRouter } from 'next/router'
import React from 'react'

import Feed from '#components/Feed'
import Layout from '#components/Layout'
import Page from '#components/Page'
import Pagination from '#components/Pagination'
import Sidebar from '#components/Sidebar/Sidebar'
import config from '#src/config'
import { DEFAULT_NUMBER_OF_POSTS } from '#commons/const'
import { Post } from '#commons/types'
import { getAllPosts, getAllTagsFromPosts } from '#utils/Markdown'

export default function Tag({
  posts,
  tag,
  pageNo,
  hasNextPage,
}: {
  posts: Array<Post>
  tag: string
  pageNo: string
  hasNextPage: boolean
}) {
  const page = parseInt(pageNo)

  const router = useRouter()

  if (router.isFallback) {
    return <div>Loading...</div>
  }

  return (
    <Layout
      title={`Tag - ${tag}`}
      description={config.subtitle}
      url={`https://yceffort.kr/tag/${tag}/page/${pageNo}`}
    >
      <Sidebar />
      <Page title={tag}>
        <Feed posts={posts} />
        <Pagination
          prevPagePath={page === 1 ? '/' : `/tag/${tag}/page/${page - 1}`}
          nextPagePath={`/tag/${tag}/page/${page + 1}`}
          hasPrevPage={page > 1}
          hasNextPage={hasNextPage}
        />
      </Page>
    </Layout>
  )
}

export const getStaticPaths: GetStaticPaths = async () => {
  const allTags = await getAllTagsFromPosts()
  const posts = await getAllPosts()

  const paths: any[] = []
  allTags.forEach(({ tag }) => {
    const tagsCount: number = posts.filter((post) =>
      post.frontmatter.tags.find((t) => t === tag),
    ).length

    ;[
      ...new Array(Math.round(tagsCount / DEFAULT_NUMBER_OF_POSTS)).keys(),
    ].forEach((i) => {
      paths.push({ params: { tag, id: `${i + 1}` } })
    })
  })

  return {
    paths,
    fallback: 'blocking',
  }
}

export const getStaticProps: GetStaticProps = async ({ params }) => {
  const allPosts = await getAllPosts()
  let resultPosts: Array<Post> = []
  let tag = 'javascript'
  let pageNo = 1
  let hasNextPage = true

  if (params) {
    tag = (params.tag as string) || 'javascript'
    pageNo = (params.id || 1) as number

    const postsWithTag = allPosts.filter((post) =>
      post.frontmatter.tags.find((t) => t === tag),
    )

    const startIndex = (pageNo - 1) * DEFAULT_NUMBER_OF_POSTS
    const endIndex = startIndex + DEFAULT_NUMBER_OF_POSTS

    resultPosts = postsWithTag.slice(startIndex, endIndex)

    hasNextPage =
      Math.ceil(postsWithTag.length / DEFAULT_NUMBER_OF_POSTS) > pageNo
  }

  return {
    props: {
      posts: resultPosts.map((post) => ({ ...post, path: '' })),
      tag,
      pageNo,
      hasNextPage,
    },
  }
}