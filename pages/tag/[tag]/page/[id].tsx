import { GetStaticPaths, GetStaticProps } from 'next'
import React from 'react'

import Feed from '../../../../src/components/Feed'
import Layout from '../../../../src/components/Layout'
import Page from '../../../../src/components/Page'
import Pagination from '../../../../src/components/Pagination'
import Sidebar from '../../../../src/components/Sidebar/Sidebar'
import config from '../../../../src/config'
import { DEFAULT_NUMBER_OF_POSTS } from '../../../../src/types/const'
import { Post } from '../../../../src/types/types'
import {
  getAllPosts,
  getAllTagsFromPosts,
} from '../../../../src/utils/FrontMatters'

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
  return (
    <Layout title={`Tag - ${tag}`} description={config.subtitle}>
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
  allTags.forEach((tag) => {
    const tagsCount: number = posts.filter((post) =>
      post.frontmatter.tags.some((t) => t === tag),
    ).length

    ;[
      ...new Array(Math.round(tagsCount / DEFAULT_NUMBER_OF_POSTS)).keys(),
    ].forEach((i) => {
      paths.push({ params: { tag, id: `${i + 1}` } })
    })
  })

  return {
    paths,
    fallback: false,
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
      post.frontmatter.tags.some((t) => t === tag),
    )

    const startIndex = (pageNo - 1) * DEFAULT_NUMBER_OF_POSTS
    const endIndex = startIndex + DEFAULT_NUMBER_OF_POSTS

    resultPosts = postsWithTag.slice(startIndex, endIndex)

    hasNextPage =
      Math.floor(postsWithTag.length / DEFAULT_NUMBER_OF_POSTS) > pageNo
  }

  return {
    props: { posts: resultPosts, tag, pageNo, hasNextPage },
  }
}
