import React from 'react'
import Link from 'next/link'
import styled from 'styled-components'

import { Post } from '../../types/types'
import Tags from './Tags/Tags'
import Author from './Author/Author'
import Meta from './Meta/meta'
import Content from './Content/Content'

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
    body,
    frontmatter: { tags, date, title },
  } = post

  return (
    <div>
      <Link href="/" passHref>
        <PostHomeButton>All Posts</PostHomeButton>
      </Link>

      <div>
        <Content title={title} body={body} />
      </div>

      <PostFooter>
        <Meta dateTime={date} />
        <Tags tags={tags} />
        <Author />
      </PostFooter>
    </div>
  )
}
