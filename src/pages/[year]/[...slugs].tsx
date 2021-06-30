import { GetStaticPaths, GetStaticProps } from 'next'
import { MDXRemote, MDXRemoteSerializeResult } from 'next-mdx-remote'

import { Post } from '#commons/types'
import { parseMarkdownToMDX } from '#utils/Markdown'
import PostLayout from '#components/layouts/Post'
import MDXComponents from '#components/MDXComponents'
import { getAllPosts } from '#utils/Post'

export default function PostPage({
  post,
  mdx,
}: {
  post: Post
  mdx: MDXRemoteSerializeResult
}) {
  return (
    <PostLayout frontMatter={post.frontMatter} slug={post.fields.slug}>
      <MDXRemote {...mdx} components={MDXComponents} />
    </PostLayout>
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
    const source = await parseMarkdownToMDX(post.body, post.path)

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
