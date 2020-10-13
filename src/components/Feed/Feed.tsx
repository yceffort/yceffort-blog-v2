// @flow strict
import React from 'react'
import styled from 'styled-components'

import type { Edges } from '../../types'

type Props = {
  edges: Edges
}

const FeedItem = styled.div`
  margin-bottom: 1.25px;

  &:last-child {
    margin-bottom: 0.5px;
  }
`

const Feed = ({ edges }: Props) => (
  <div className="feed">
    {edges.map((edge) => (
      <FeedItem key={edge.node.fields.slug}>
        <div className="feed__item-meta">
          <span>10 11 2019</span>
          {/* <time
            className='feed__item-meta-time'
            dateTime={moment(edge.node.frontmatter.date).format('MMMM D, YYYY')}
          >
            {moment(edge.node.frontmatter.date).format('MMMM YYYY')}
          </time> */}
          <span className="divider" />
          <span className="feed__item-meta-category">
            <a className={'feed__item-meta-category-link'}>카테고리영역</a>
          </span>
        </div>
        <h2 className="feed__item-title">
          <a className="feed__item-title-link">제목 영역</a>
        </h2>
        <p className="feed__item-description">
          {edge.node.frontmatter.description}
        </p>
        <a className="feed__item-readmore">읽기 영dur</a>
      </FeedItem>
    ))}
  </div>
)

export default Feed
