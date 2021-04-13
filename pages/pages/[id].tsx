import React from 'react'
import { GetStaticPaths, GetStaticProps } from 'next'

import { Post } from '#commons/types'
import { getAllPosts } from '#utils/Markdown'
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
        title={`Blog - ${SiteConfig.author.name}`}
        description={SiteConfig.subtitle}
        url={`${SiteConfig.url}/blog`}
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

export const getStaticProps: GetStaticProps = async ({ params }) => {
  const allPosts = await getAllPosts()
  let posts: Array<Post> = []
  let pageNo = 1
  let hasNextPage = true

  if (params && params.id && typeof params.id === 'string') {
    pageNo = parseInt(params.id)

    posts = allPosts.slice(
      (pageNo - 1) * DEFAULT_NUMBER_OF_POSTS,
      (pageNo - 1) * DEFAULT_NUMBER_OF_POSTS + DEFAULT_NUMBER_OF_POSTS,
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
