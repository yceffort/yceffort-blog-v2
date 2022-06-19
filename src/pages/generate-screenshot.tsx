import type { GetServerSideProps } from 'next'

import Screenshot from '#components/screenshot'

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

export const getServerSideProps: GetServerSideProps = async ({ query }) => {
  const { title, tags, url, imageSrc = '', imageCredit = '', slug } = query

  if (!slug && typeof slug !== 'string') {
    return {
      notFound: true,
    }
  }

  const splitSlug = (slug as string).split('/')
  const tempTitle = splitSlug[splitSlug.length - 1].replace(/-/gi, ' ')
  const engTitle = tempTitle.charAt(0).toUpperCase() + tempTitle.slice(1)

  return {
    props: { title: engTitle || title, tags, url, imageSrc, imageCredit },
  }
}
