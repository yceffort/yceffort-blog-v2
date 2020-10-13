import React, { ReactNode, useEffect, useRef } from 'react'
import styled from 'styled-components'

const PageDiv = styled.div`
  margin-bottom: 3.25rem;

  @media screen and (min-width: 685px) {
    width: calc(58.275% - 0.78125rem);

    &:nth-child(1n) {
      float: left;
      margin-right: 1.875rem;
      clear: none;
    }

    &:last-child {
      margin-right: 0;
    }

    &:nth-child(12n) {
      margin-right: 0;
      float: right;
    }

    &:nth-child(12n + 1) {
      clear: both;
    }
  }

  @media screen and (min-width: 960px) {
    width: calc(66.6% - 0.625rem);

    &:nth-child(1n) {
      float: left;
      margin-right: 1.875rem;
      clear: none;
    }

    &:last-child {
      margin-right: 0;
    }

    &:nth-child(3n) {
      margin-right: 0;
      float: right;
    }

    &:nth-child(3n + 1) {
      clear: both;
    }
  }
`

const PageInner = styled.div`
  padding: 1.5625rem 1.25rem;

  @media screen and (min-width: 685px) {
    padding: 1.875rem 1.25rem;
  }

  @media screen and (min-width: 960px) {
    padding: 2.5rem 2.1875rem;
  }
`

const PageTitle = styled.h1`
  font-size: 2.5rem;
  font-weight: 600;
  line-height: 3.25rem;
  margin-top: 0;
  margin-bottom: 2.35625rem;
`

const PageBody = styled.div`
  font-size: 1rem;
  line-height: 1.625rem;
  margin: 0 0 1.625rem;
`

export default function Page({
  title,
  children,
}: {
  title?: string
  children: ReactNode
}) {
  const pageRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (pageRef && pageRef.current) {
      pageRef.current.scrollIntoView()
    }
  }, [])

  return (
    <PageDiv ref={pageRef}>
      <PageInner>
        {title ? <PageTitle>{title}</PageTitle> : null}
        <PageBody>{children}</PageBody>
      </PageInner>
    </PageDiv>
  )
}
