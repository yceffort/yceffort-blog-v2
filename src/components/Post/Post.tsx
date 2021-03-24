import React from 'react'
import Link from 'next/link'
import Head from 'next/head'
import styled from 'styled-components'
import { Fab, Action } from 'react-tiny-fab'
import { BiPlus, BiArrowToTop } from 'react-icons/bi'
import { ImGithub } from 'react-icons/im'

import Author from '#components/Post/components/Author'
import Meta from '#components/Post/components/meta'
import Content from '#components/Post/components/Content'
import Tags from '#components/Post/components/Tags'
import { Post } from '#types/types'

import 'react-tiny-fab/dist/styles.css'

const PostFooter = styled.div`
  max-width: 59.0625rem;
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

  const scrollToTop = () => {
    const c = document.documentElement.scrollTop || document.body.scrollTop
    if (c > 0) {
      window.requestAnimationFrame(scrollToTop)
      window.scrollTo(0, c - c / 8)
    }
  }

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

      <Fab
        icon={<BiPlus size="2em" />}
        event="click"
        mainButtonStyles={{ backgroundColor: '#00b7ff' }}
        style={{ bottom: 0, right: 0 }}
      >
        <Action
          text="create discussion"
          onClick={() =>
            window.open(
              `https://github.com/yceffort/yceffort-blog-v2/issues/new?labels=%F0%9F%92%AC%20Discussion&title=[Discussion]&assignees=yceffort&body=https://yceffort.kr/${post.fields.slug}`,
            )
          }
          style={{ backgroundColor: 'rgb(77, 139, 198)' }}
        >
          <ImGithub size="2em" />
        </Action>
        <Action
          text="TOP"
          onClick={scrollToTop}
          style={{ backgroundColor: 'rgb(77, 139, 198)' }}
        >
          <BiArrowToTop size="2em" />
        </Action>
      </Fab>
    </div>
  )
}
