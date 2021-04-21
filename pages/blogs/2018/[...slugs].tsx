import React from 'react'
import { MdxRemote } from 'next-mdx-remote/types'
import hydrate from 'next-mdx-remote/hydrate'

import { Post } from '#commons/types'
import PostLayout from '#components/layouts/Post'
import MDXComponents from '#components/MDXComponents'
import { getPostStaticPaths, getPostStaticProps } from '#utils/Post'

export default function PostPage({
  post,
  mdx,
}: {
  post: Post
  mdx: MdxRemote.Source
}) {
  return (
    <PostLayout frontMatter={post?.frontMatter} slug={post.fields.slug}>
      {hydrate(mdx, {
        components: MDXComponents,
      })}
    </PostLayout>
  )
}

export const getStaticPaths = getPostStaticPaths('2018')

export const getStaticProps = getPostStaticProps('2018')
