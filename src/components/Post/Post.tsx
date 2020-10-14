import React from 'react'
import Link from 'next/link'
import Head from 'next/head'
import styled from 'styled-components'

import { Post } from '../../types/types'
import Tags from './components/Tags'
import Author from './components/Author'
import Meta from './components/meta'
import Content from './components/Content'

const PostFooter = styled.div`
  max-width: 40rem;
  margin: 0 auto;
  padding: 0 0.9375rem;

  @media screen and (min-width: 960px) {
    padding: 0;
  }
`

const PostHomeButton = styled.a`
  display: block;
  max-width: 5.625rem;
  height: 2.1875rem;
  padding: 0 1.5rem;
  line-height: 2.1875rem;
  text-align: center;
  color: #222;
  border: 1px solid #e6e6e6;
  border-radius: 1.25rem;
  font-size: 1rem;
  font-weight: 400;
  margin-left: auto;
  margin-right: auto;
  margin-top: 1.625rem;

  &:focus,
  &:hover {
    color: #5d93ff;
  }

  @media screen and (min-width: 960px) {
    position: fixed;
    max-width: auto;
    margin: 0;
    top: 30px;
    left: 30px;
  }
`

export default function PostRenderer({ post }: { post: Post }) {
  const {
    parsedBody,
    frontmatter: { tags, date, title },
  } = post

  return (
    <div>
      <Head>
        <link
          rel="stylesheet"
          href="https://cdn.jsdelivr.net/npm/katex@0.12.0/dist/katex.min.css"
          integrity="sha384-AfEj0r4/OFrOo5t7NnNe46zW/tFgW6x/bCJG8FqQCEo3+Aro6EYUG4+cU+KJWu/X"
          crossOrigin="anonymous"
        />
      </Head>
      <Link href="/" passHref>
        <PostHomeButton>All Posts</PostHomeButton>
      </Link>

      <div>
        {parsedBody ? <Content title={title} body={parsedBody} /> : null}
      </div>

      <PostFooter>
        <Meta dateTime={date} />
        <Tags tags={tags} />
        <Author />
      </PostFooter>
    </div>
  )
}
