import { GetStaticPaths, GetStaticProps } from 'next'
import React from 'react'

import Layout from '../../../../src/components/Layout/Layout'
import PostRenderer from '../../../../src/components/Post/Post'
import { Post } from '../../../../src/types/types'
import { getAllPosts } from '../../../../src/utils/frontMatters'

export default function PostPage({ post }: { post: Post }) {
  const {
    frontmatter: { title, description },
  } = post
  return (
    <Layout title={title} description={description || config.subtitle}>
      <PostRenderer post={post} />
    </Layout>
  )
}

export const getStaticPaths: GetStaticPaths = async () => {
  const allPosts = await getAllPosts()
  const paths = allPosts.reduce((prev, { fields: { slug } }) => {
    const [year, month, day, title] = `${slug.replace('.md', '')}`.split('/')
    if (title) {
      prev.push({ params: { year, month, day, title } })
    }
    return prev
  }, [] as Array<{ params: { year: string; month: string; day: string; title: string } }>)

  return {
    paths,
    fallback: true,
  }
}
export const getStaticProps: GetStaticProps = async ({ params }) => {
  let post: Post | undefined
  if (params) {
    const { year, month, day, title } = params
    const slug = [year, month, day, title].join('/')
    const posts = await getAllPosts()
    post = posts.find(({ fields: { slug: postSlug } }) => postSlug === slug)

    if (!post) {
      post = posts.find(({ fields: { slug: postSlug } }) => postSlug === slug)
    }
  }

  return {
    props: { post },
  }
}
