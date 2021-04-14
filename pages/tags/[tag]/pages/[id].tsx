import { GetStaticPaths, GetStaticProps } from 'next'
import React from 'react'

import { SiteConfig } from '#src/config'
import { DEFAULT_NUMBER_OF_POSTS } from '#commons/const'
import { Post } from '#commons/types'
import { getAllPosts, getAllTagsFromPosts } from '#utils/posts'
import { PageSeo } from '#components/SEO'
import ListLayout from '#components/layouts/List'

export default function Tag({
  posts,
  tag,
  pageNo,
  hasNextPage,
}: {
  tag: string
  posts: Array<Post>
  pageNo: number
  hasNextPage: boolean
}) {
  const title = `${
    tag[0].toUpperCase() + tag.split(' ').join('-').slice(1)
  } ${pageNo}`
  return (
    <>
      <PageSeo
        title={`${tag} (${pageNo})`}
        description={`${tag} tags - ${SiteConfig.title}`}
        url={`${SiteConfig.url}/tags/${tag}`}
      />
      <ListLayout
        posts={posts}
        title={title}
        pageNo={pageNo}
        hasNextPage={hasNextPage}
        nextPath={`/tags/${tag}/pages/${pageNo + 1}`}
        prevPath={`/tags/${tag}/pages/${pageNo - 1}`}
      />
    </>
  )
}

export const getStaticPaths: GetStaticPaths = async () => {
  const allTags = await getAllTagsFromPosts()
  const posts = await getAllPosts()

  const paths: any[] = []
  allTags.forEach(({ tag }) => {
    const tagsCount: number = posts.filter((post) =>
      post.frontMatter.tags.find((t) => t === tag),
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

interface PageInterface {
  [key: string]: string | undefined
  tag: string
  id: string
}

export const getStaticProps: GetStaticProps = async ({ params }) => {
  const allPosts = await getAllPosts()
  const { tag = 'javascript', id = '1' } = params as PageInterface
  const pageNo = parseInt(id)

  const postsWithTag = allPosts.filter((post) =>
    post.frontMatter.tags.find((t) => t === tag),
  )

  const startIndex = (pageNo - 1) * DEFAULT_NUMBER_OF_POSTS
  const endIndex = startIndex + DEFAULT_NUMBER_OF_POSTS

  const resultPosts = postsWithTag.slice(startIndex, endIndex)

  const hasNextPage =
    Math.ceil(postsWithTag.length / DEFAULT_NUMBER_OF_POSTS) > pageNo

  return {
    props: {
      posts: resultPosts.map((post) => ({ ...post, path: '' })),
      tag,
      pageNo,
      hasNextPage,
    },
  }
}
