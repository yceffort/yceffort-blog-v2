import Link from 'next/link'
import React from 'react'
import styled from 'styled-components'

const TagsContainer = styled.div`
  margin-bottom: 0.8125rem;
`

const TagsList = styled.ul`
  list-style: none;
  margin: 0 -0.625rem;
  padding: 0;
`

const TagItem = styled.li`
  display: inline-block;
  margin: 0.625rem 0.3125rem;
`

const TagItemLink = styled.a`
  display: inline-block;
  height: 2.1875rem;
  padding: 0 1.5rem;
  line-height: 2.1875rem;
  border: 1px solid #e6e6e6;
  text-decoration: none;
  border-radius: 1.25rem;
  color: #222;

  &:focus,
  &::hover {
    color: #5d93ff;
  }
`

export default function Tags({ tags }: { tags: string[] }) {
  return (
    <TagsContainer>
      <TagsList>
        {tags.map((tag) => (
          <TagItem key={tag}>
            <Link href={`/tag/${tag}/page/1`} passHref>
              <TagItemLink>{tag}</TagItemLink>
            </Link>
          </TagItem>
        ))}
      </TagsList>
    </TagsContainer>
  )
}
