// @flow strict
import { format, toDate } from 'date-fns'
import Link from 'next/link'
import React from 'react'
import styled from 'styled-components'

import { Post } from '#commons/types'

const FeedItem = styled.div`
  margin-bottom: 2.03125rem;

  &:last-child {
    margin-bottom: 0.5px;
  }
`

const FeedItemMeta = styled.div`
  font-size: 0.875rem;
  color: #222;
  font-weight: 600;
  text-transform: uppercase;
`

const FeedItemMetaTime = styled.time`
  font-size: 0.875rem;
  color: #222;
  font-weight: 600;
  text-transform: uppercase;
`

const FeedItemMetaDivider = styled.span`
  margin: 0 0.3125rem;
`

const FeedItemMetaCategoryLink = styled.a`
  font-size: 0.875rem;
  color: #f7a046;
  font-weight: 600;
  text-transform: uppercase;

  &:hover,
  &:focus {
    color: #5d93ff;
  }
`

const FeedItemTitle = styled.h2`
  font-size: 1.6875rem;
  line-height: 2.4375rem;
  margin-top: 0;
  margin-bottom: 0.8125rem;
`

const FeedItemTitleLink = styled.a`
  color: #222;
  &:hover,
  &:focus {
    color: #222;
    border-bottom: 1px solid #222;
  }
`

const FeedItemDescription = styled.p`
  font-size: 1rem;
  line-height: 1.625rem;
  margin-bottom: 1.21875rem;
  word-break: break-all;
`

const FeedItemReadMore = styled.a`
  font-size: 1rem;
  color: #5d93ff;

  &:hover,
  &:focus {
    color: #5d93ff;
    border-bottom: 1px solid #5d93ff;
  }
`

const Feed = ({ posts }: { posts?: Array<Post> }) => (
  <>
    {posts && posts.length ? (
      posts.map((post, index) => {
        const {
          frontmatter: { date, title, description, tags },
          fields: { slug },
        } = post
        return (
          <FeedItem key={index}>
            <FeedItemMeta>
              <FeedItemMetaTime>
                {format(toDate(date), 'MMMM d yyyy')}
              </FeedItemMetaTime>
              <FeedItemMetaDivider />

              {tags.map((tag, index) => (
                <Link passHref href={`/tag/${tag}/page/1`} key={index}>
                  <FeedItemMetaCategoryLink>{tag} </FeedItemMetaCategoryLink>
                </Link>
              ))}
            </FeedItemMeta>
            <FeedItemTitle>
              <Link passHref href={`/${slug}`}>
                <FeedItemTitleLink>{title}</FeedItemTitleLink>
              </Link>
            </FeedItemTitle>
            <FeedItemDescription>{description}</FeedItemDescription>
            <Link passHref href={`/${slug}`}>
              <FeedItemReadMore>Read</FeedItemReadMore>
            </Link>
          </FeedItem>
        )
      })
    ) : (
      <>there are no posts</>
    )}
  </>
)

export default Feed
