import { GetStaticPaths, GetStaticProps } from 'next'
import { ParsedUrlQuery } from 'node:querystring'

import { getAllPosts } from '#utils/Post'
import { parseMarkdownToMDX } from '#utils/Markdown'

export const getPostStaticPaths = (
  year: string,
): GetStaticPaths => async () => {
  const posts = (await getAllPosts()).filter(({ fields: { slug } }) =>
    slug.startsWith(year),
  )

  const paths: Array<{
    params: { slugs: string[] }
  }> = posts.reduce<Array<{ params: { slugs: string[] } }>>(
    (prev, { fields: { slug } }) => {
      const [...slugs] = `${slug.replace('.md', '')}`.split('/')

      prev.push({ params: { slugs } })
      return prev
    },
    [],
  )

  return {
    paths,
    fallback: 'blocking',
  }
}

interface SlugInterface extends ParsedUrlQuery {
  [key: string]: string | string[] | undefined
  slugs: string[]
}

export const getPostStaticProps = (year: string): GetStaticProps => async ({
  params,
}) => {
  const { slugs } = params as SlugInterface

  const slug = [...(slugs as string[])].join('/')
  const posts = (await getAllPosts()).filter(({ fields: { slug } }) =>
    slug.startsWith(year),
  )
  const post = posts.find(
    ({ fields: { slug: postSlug } }) => postSlug === `${slug}`,
  )

  return {
    props: {
      post,
      mdx: post && (await parseMarkdownToMDX(post.body, post.path)),
    },
  }
}
