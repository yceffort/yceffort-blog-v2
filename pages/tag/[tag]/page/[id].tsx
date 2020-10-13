import { GetStaticPaths, GetStaticProps } from 'next'
import React from 'react'

import Feed from '../../../../src/components/Feed/Feed'
import { DEFAULT_NUMBER_OF_POSTS } from '../../../../src/types/const'
import { Post } from '../../../../src/types/types'
import { getAllPosts } from '../../../../src/utils/frontMatters'

export default function Tag({
  posts,
  tag,
  pageNo,
}: {
  posts: Array<Post>
  tag: string
  pageNo: number
}) {
  return (
    <>
      태그 {tag} 페이지 {pageNo}
      <Feed posts={posts} />
    </>
  )
}

export const getStaticPaths: GetStaticPaths = async () => {
  let allTags: string[] = []
  const posts = await getAllPosts()

  posts.forEach(({ frontmatter: { tags } }) => {
    tags.forEach((tag) => {
      allTags.push(tag.trim())
    })
  })

  allTags = [...new Set(allTags)]

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
  let posts: Array<Post> = allPosts
  let tagName = 'javascript'

  if (params) {
    tagName = (params.tag as string) || 'javascript'
    const pageNo = (params.id || 1) as number

    posts = posts.filter((post) =>
      post.frontmatter.tags.some((tag) => tag === tagName),
    )

    posts = allPosts.slice(
      pageNo * DEFAULT_NUMBER_OF_POSTS,
      pageNo * DEFAULT_NUMBER_OF_POSTS + DEFAULT_NUMBER_OF_POSTS,
    )
  }

  return {
    props: { posts, tag: tagName, pageNo: 1 },
  }
}
