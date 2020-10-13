import React from 'react'
import { GetStaticPaths, GetStaticProps } from 'next'

import { getAllPosts } from '../../src/utils/frontMatters'
import { DEFAULT_NUMBER_OF_POSTS } from '../../src/types/const'
import { Post } from '../../src/types/types'
import Feed from '../../src/components/Feed/Feed'

export default function Page({
  pageNo,
  posts,
}: {
  pageNo: number
  posts: Array<Post>
}) {
  return (
    <>
      <>페이지 {pageNo}</>
      <Feed posts={posts} />
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
    fallback: false,
  }
}

export const getStaticProps: GetStaticProps = async ({ params }) => {
  const allPosts = await getAllPosts()
  let posts: Array<Post> = allPosts.slice(0, DEFAULT_NUMBER_OF_POSTS)
  let pageNo = 1

  if (params && params.id && typeof params.id === 'string') {
    pageNo = parseInt(params.id)

    posts = allPosts.slice(
      pageNo * DEFAULT_NUMBER_OF_POSTS,
      pageNo * DEFAULT_NUMBER_OF_POSTS + DEFAULT_NUMBER_OF_POSTS,
    )
  }

  return {
    props: { posts, pageNo },
  }
}
