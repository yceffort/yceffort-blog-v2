'use client'

import Head from 'next/head'
import { GetStaticPaths, GetStaticProps } from 'next'
import { MDXRemote, MDXRemoteSerializeResult } from 'next-mdx-remote'
import { Suspense } from 'react'

import { Post } from '#src/type'
import { parseMarkdownToMdx } from '#utils/Markdown'
import PostLayout from '#components/layouts/Post'
import MdxComponents from '#components/MDXComponents'
import { getAllPosts } from '#utils/Post'
import Meta from '#components/Meta'

export default function PostPage({
  post,
  mdx,
}: {
  post: Post
  mdx: MDXRemoteSerializeResult
}) {
  return (
    <>
      <Head>
        <Meta
          name="description"
          content={post.frontMatter.description || "yceffort's blog"}
        />
      </Head>
      <PostLayout frontMatter={post.frontMatter} slug={post.fields.slug}>
        <Suspense fallback={null}>
          <MDXRemote {...mdx} components={MdxComponents} />
        </Suspense>
      </PostLayout>
    </>
  )
}

export const getStaticPaths: GetStaticPaths = async () => {
  const allPosts = await getAllPosts()
  const paths: Array<{
    params: { year: string; slugs: string[] }
  }> = allPosts.reduce<Array<{ params: { year: string; slugs: string[] } }>>(
    (prev, { fields: { slug } }) => {
      const [year, ...slugs] = `${slug.replace('.md', '')}`.split('/')

      prev.push({ params: { year, slugs } })
      return prev
    },
    [],
  )

  return {
    paths,
    fallback: 'blocking',
  }
}

interface SlugInterface {
  [key: string]: string | string[] | undefined
  year: string
  slugs: string[]
}

export const getStaticProps: GetStaticProps = async ({ params }) => {
  const { year, slugs } = params as SlugInterface

  const slug = [year, ...(slugs as string[])].join('/')
  const posts = await getAllPosts()
  const post = posts.find((p) => p?.fields?.slug === slug)
  if (post) {
    const source = await parseMarkdownToMdx(post.body, post.path)

    return {
      props: {
        post,
        mdx: source,
      },
    }
  }
  return {
    notFound: true,
  }
}
