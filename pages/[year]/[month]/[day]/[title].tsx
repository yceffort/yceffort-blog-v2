import { GetStaticPaths, GetStaticProps } from 'next'
import React from 'react'
import DefaultErrorPage from 'next/error'
import qs from 'query-string'
import { MdxRemote } from 'next-mdx-remote/types'

import { getAllPosts, parseMarkdownToMDX } from '#utils/Markdown'
import Layout from '#components/Layout'
import PostRenderer from '#components/Post/Post'
import config from '#src/config'
import { Post } from '#commons/types'

export default function PostPage({
  post,
  thumbnailUrl,
  mdx,
}: {
  post?: Post
  mdx?: MdxRemote.Source
  thumbnailUrl: string
}) {
  return post && mdx ? (
    <Layout
      title={post.frontmatter.title}
      description={post.frontmatter.description || config.subtitle}
      url={`https://yceffort.kr/${post.fields.slug}`}
      socialImage={thumbnailUrl}
    >
      <PostRenderer post={post} mdx={mdx} />
    </Layout>
  ) : (
    <DefaultErrorPage statusCode={404} />
  )
}

export const getStaticPaths: GetStaticPaths = async () => {
  const allPosts = await getAllPosts()
  const paths = allPosts.reduce((prev, { fields: { slug } }) => {
    const splits = `${slug.replace('.md', '')}`.split('/')
    if (splits.length === 4) {
      const [year, month, day, title] = splits
      prev.push({ params: { year, month, day, title } })
    }
    return prev
  }, [] as Array<{ params: { year: string; month: string; day: string; title: string } }>)

  return {
    paths,
    fallback: 'blocking',
  }
}
export const getStaticProps: GetStaticProps = async ({ params }) => {
  let post: Post | undefined
  let thumbnailUrl = ''
  let mdx

  if (params) {
    const { year, month, day, title } = params
    const slug = [year, month, day, title].join('/')
    const posts = await getAllPosts()
    post = posts.find(({ fields: { slug: postSlug } }) => postSlug === slug)

    if (post) {
      const thumbnailHost = `https://us-central1-yceffort.cloudfunctions.net/screenshot`

      const queryString = qs.stringify({
        tags: post.frontmatter.tags.map((tag) => tag.trim()).join(','),
        title: post.frontmatter.title,
        url: `https://yceffort.kr/${post.fields.slug}`,
        slug: post.fields.slug,
      })

      thumbnailUrl = `${thumbnailHost}?${queryString}`
      mdx = await parseMarkdownToMDX(post.body)
    }
  }

  return {
    props: {
      post: post ? { ...post, path: '' } : null,
      thumbnailUrl: process.env.NODE_ENV === 'production' ? thumbnailUrl : '',
      mdx,
    },
  }
}
