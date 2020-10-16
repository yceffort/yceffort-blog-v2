import React, { ReactNode } from 'react'
import Head from 'next/head'
import styled from 'styled-components'

import config from '../config'

const LayoutDiv = styled.div`
  max-width: 66.875rem;
  margin-left: auto;
  margin-right: auto;

  &:before {
    content: '';
    display: table;
  }

  &:after {
    content: '';
    display: table;
    clear: both;
  }
`

const Layout = ({
  children,
  title,
  description,
  socialImage,
  url,
}: {
  children: ReactNode
  title: string
  description?: string
  socialImage?: string
  url: string
}) => {
  const {
    author: { photo },
    subtitle,
  } = config
  const layoutMetaImageUrl = `https://yceffort.kr/${socialImage || photo}`
  const layoutDescription = description || subtitle

  return (
    <LayoutDiv>
      <Head>
        <html lang="kr" />
        <title>{title}</title>
        <meta name="description" content={layoutDescription} />
        <meta property="og:site_name" content={title} />
        <meta property="og:title" content={title} />
        <meta property="og:url" content={url} />
        <meta property="og:image" content={layoutMetaImageUrl} />
        <meta property="og:description" content={layoutDescription} />
        <meta name="twitter:title" content={title} />
        <meta name="twitter:description" content={layoutDescription} />
        <meta name="twitter:image" content={layoutMetaImageUrl} />
        <meta name="twitter:card" content="summary" />
      </Head>
      {children}
    </LayoutDiv>
  )
}

export default Layout
