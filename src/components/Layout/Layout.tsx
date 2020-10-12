import React, { ReactNode } from 'react'
import Head from 'next/head'

import config from '../../../config'

type Props = {
  children: ReactNode,
  title: string,
  description?: string,
  socialImage?: string,
}

const Layout = ({ children, title, description, socialImage }: Props) => {
  const { author, url } = config
  const metaImage = socialImage != null ? socialImage : author.photo
  const metaImageUrl = url

  return (
    <div>
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
    </div>
  )
}

export default Layout
