import { GetStaticPaths, GetStaticProps } from 'next'
import React from 'react'
import qs from 'query-string'
import { MdxRemote } from 'next-mdx-remote/types'
import hydrate from 'next-mdx-remote/hydrate'

import { Post } from '#commons/types'
import { parseMarkdownToMDX } from '#utils/Markdown'
import PostLayout from '#components/layouts/Post'
import MDXComponents from '#components/MDXComponents'
import { getAllPosts } from '#utils/posts'

export default function PostPage({
  post,
  mdx,
}: {
  post: Post
  mdx: MdxRemote.Source
  thumbnailUrl: string
}) {
  return (
    mdx &&
    post && (
      <PostLayout frontMatter={post?.frontmatter} slug={post.fields.slug}>
        {hydrate(mdx, {
          components: MDXComponents,
        })}
      </PostLayout>
    )
  )
}

export const getStaticPaths: GetStaticPaths = async () => {
  const allPosts = await getAllPosts()
  const paths = allPosts.reduce((prev, { fields: { slug } }) => {
    const [year, ...slugs] = `${slug.replace('.md', '')}`.split('/')

    prev.push({ params: { year, slugs } })
    return prev
  }, [] as Array<{ params: { year: string; slugs: string[] } }>)

  return {
    paths,
    fallback: 'blocking',
  }
}
export const getStaticProps: GetStaticProps = async ({ params }) => {
  let post: Post | undefined
  let thumbnailUrl = ''
  let mdx = null

  if (params) {
    const { year, slugs } = params
    const slug = [year, ...(slugs as string[])].join('/')
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
