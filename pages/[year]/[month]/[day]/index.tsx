import { GetStaticPaths, GetStaticProps } from 'next'
import React from 'react'
import DefaultErrorPage from 'next/error'
import qs from 'query-string'

import { Post } from '#types/types'
import { getAllPosts, parseMarkdownToHTML } from '#utils/Markdown'
import Layout from '#components/Layout'
import PostRenderer from '#components/Post/Post'

export default function PostPage({
  post,
  thumbnailUrl,
}: {
  post?: Post
  thumbnailUrl: string
}) {
  return post ? (
    <Layout
      title={post.frontmatter.title}
      description={post.frontmatter.description}
      url={`https://yceffort.kr/${post.fields.slug}`}
      socialImage={thumbnailUrl}
    >
      <PostRenderer post={post} />
    </Layout>
  ) : (
    <DefaultErrorPage statusCode={404} />
  )
}

export const getStaticPaths: GetStaticPaths = async () => {
  const allPosts = await getAllPosts()
  const paths = allPosts.reduce((prev, { fields: { slug } }) => {
    const splits = `${slug.replace('.md', '')}`.split('/')
    if (splits.length === 3) {
      const [year, month, title] = splits
      prev.push({ params: { year, month, day: title } })
    }
    return prev
  }, [] as Array<{ params: { year: string; month: string; day: string } }>)

  return {
    paths,
    fallback: 'blocking',
  }
}
export const getStaticProps: GetStaticProps = async ({ params }) => {
  let post: Post | undefined
  let thumbnailUrl = ''

  if (params) {
    const { year, month, day: title } = params
    const slug = [year, month, title].join('/')
    const posts = await getAllPosts()
    post = posts.find(({ fields: { slug: postSlug } }) => postSlug === slug)

    if (post) {
      post.parsedBody = await parseMarkdownToHTML(post.body)

      const thumbnailHost = `https://us-central1-yceffort.cloudfunctions.net/screenshot`

      const queryString = qs.stringify({
        tags: post.frontmatter.tags.map((tag) => tag.trim()).join(','),
        title: post.frontmatter.title,
        url: `https://yceffort.kr/${post.fields.slug}`,
        slug: post.fields.slug,
      })

      thumbnailUrl = `${thumbnailHost}?${queryString}`
    }
  }

  return {
    props: {
      post: post ? { ...post, path: '' } : null,
      thumbnailUrl: process.env.NODE_ENV === 'production' ? thumbnailUrl : '',
    },
  }
}
