import { GetServerSideProps } from 'next'
import React from 'react'

import Screenshot from '#components/screenshot'

// ${hostUrl}/generate-screenshot?title=hello&tags=react,javascript&categories=vscode&imageSrc=https://cdn.pixabay.com/photo/2015/04/23/22/00/tree-736885__340.jpg&imageCredit=sexy&url=https://yceffort.kr/2020/03/nextjs-02-data-fetching#4-getserversideprops
export default function GenerateScreenshot({
  title,
  tags,
  url,
  imageSrc,
  imageCredit,
}: {
  title: string
  tags: string
  url: string
  imageSrc: string
  imageCredit: string
}) {
  return (
    <Screenshot
      title={title}
      tags={tags.split(',')}
      url={url}
      imageSrc={imageSrc}
      imageCredit={imageCredit}
    />
  )
}

export const getServerSideProps: GetServerSideProps = async (context) => {
  const {
    title,
    tags,
    url,
    imageSrc = '',
    imageCredit = '',
    slug,
  } = context.query

  let engTitle = ''

  if (slug) {
    const splitSlug = (slug as string).split('/')
    const tempTitle = splitSlug[splitSlug.length - 1].replace(/-/gi, ' ')
    engTitle = tempTitle.charAt(0).toUpperCase() + tempTitle.slice(1)
  }

  return {
    props: { title: engTitle || title, tags, url, imageSrc, imageCredit },
  }
}
