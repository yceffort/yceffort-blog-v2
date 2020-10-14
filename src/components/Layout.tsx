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
}: {
  children: ReactNode
  title: string
  description?: string
  socialImage?: string
}) => {
  const {
    author: { photo },
  } = config
  const metaImageUrl = socialImage || photo

  return (
    <LayoutDiv>
      <Head>
        <html lang="kr" />
        <title>{title}</title>
        <meta name="description" content={description} />
        <meta property="og:site_name" content={title} />
        <meta property="og:image" content={metaImageUrl} />
        <meta name="twitter:card" content="summary" />
        <meta name="twitter:title" content={title} />
        <meta name="twitter:description" content={description} />
        <meta name="twitter:image" content={metaImageUrl} />
      </Head>
      {children}
    </LayoutDiv>
  )
}

export default Layout
