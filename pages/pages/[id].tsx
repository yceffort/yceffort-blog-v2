import React from 'react'
import { GetStaticPaths, GetStaticProps } from 'next'

import { Post } from '#commons/types'
import { getAllPosts } from '#utils/posts'
import { DEFAULT_NUMBER_OF_POSTS } from '#commons/const'
import SiteConfig from '#src/config'
import { PageSeo } from '#components/SEO'
import ListLayout from '#components/layouts/List'

export default function Blog({
  posts,
  pageNo,
  hasNextPage,
}: {
  posts: Array<Post>
  pageNo: number
  hasNextPage: boolean
}) {
  return (
    <>
      <PageSeo
        title={`Posts (${pageNo})`}
        description={SiteConfig.subtitle}
        url={`${SiteConfig.url}`}
      />
      <ListLayout
        posts={posts}
        pageNo={pageNo}
        hasNextPage={hasNextPage}
        title={`Page ${pageNo}`}
      />
    </>
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

interface PageInterface {
  [key: string]: string | undefined
  id: string
}

export const getStaticProps: GetStaticProps = async ({ params }) => {
  const { id } = params as PageInterface
  const allPosts = await getAllPosts()
  const pageNo = parseInt(id)

  const startIndex = (pageNo - 1) * DEFAULT_NUMBER_OF_POSTS
  const endIndex = startIndex + DEFAULT_NUMBER_OF_POSTS

  const posts = allPosts.slice(startIndex, endIndex)

  const hasNextPage =
    Math.floor(allPosts.length / DEFAULT_NUMBER_OF_POSTS) > pageNo

  return {
    props: {
      posts: posts.map((post) => ({ ...post, path: '' })),
      pageNo,
      hasNextPage,
    },
  }
}
