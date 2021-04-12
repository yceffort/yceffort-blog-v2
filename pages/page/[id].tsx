import React from 'react'
import { GetStaticPaths, GetStaticProps } from 'next'

import { DEFAULT_NUMBER_OF_POSTS } from '#commons/const'
import { getAllPosts } from '#utils/Markdown'
import { Post } from '#commons/types'
import Feed from '#components/Feed'
import Layout from '#components/Layout'
import config from '#src/config'
import Sidebar from '#components/Sidebar/Sidebar'
import Page from '#components/Page'
import Pagination from '#components/Pagination'

export default function IndexPage({
  pageNo,
  posts,
  hasNextPage,
}: {
  pageNo: string
  posts: Array<Post>
  hasNextPage: boolean
}) {
  const page = parseInt(pageNo)
  return (
    <Layout
      title={`Page No ${page}`}
      description={config.title}
      url={`https://yceffort/page/${page}`}
    >
      <Sidebar />
      <Page>
        <Feed posts={posts} />
        <Pagination
          prevPagePath={page === 1 ? '/' : `/page/${page - 1}`}
          nextPagePath={`/page/${page + 1}`}
          hasPrevPage={true}
          hasNextPage={hasNextPage}
        />
      </Page>
    </Layout>
  )
}

export const getStaticPaths: GetStaticPaths = async () => {
  const posts = await getAllPosts()

  const paths = [
    ...new Array(Math.round(posts.length / DEFAULT_NUMBER_OF_POSTS)).keys(),
  ].map((i) => ({ params: { id: `${i + 1}` } }))

  return {
    paths,
    fallback: 'blocking',
  }
}

export const getStaticProps: GetStaticProps = async ({ params }) => {
  const allPosts = await getAllPosts()
  let posts: Array<Post> = allPosts.slice(0, DEFAULT_NUMBER_OF_POSTS)
  let pageNo = 1
  let hasNextPage = true

  if (params && params.id && typeof params.id === 'string') {
    pageNo = parseInt(params.id)

    posts = allPosts.slice(
      pageNo * DEFAULT_NUMBER_OF_POSTS,
      pageNo * DEFAULT_NUMBER_OF_POSTS + DEFAULT_NUMBER_OF_POSTS,
    )

    hasNextPage = Math.floor(allPosts.length / DEFAULT_NUMBER_OF_POSTS) > pageNo
  }

  return {
    props: {
      posts: posts.map((post) => ({ ...post, path: '' })),
      pageNo,
      hasNextPage,
    },
  }
}