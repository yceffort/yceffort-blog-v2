import qs from 'query-string'

export function getThumbnailURL({
  tags,
  title,
  path = '',
  slug,
}: {
  tags: string[]
  title: string
  path?: string
  slug: string
}): string {
  // if (process.env.NODE_ENV !== 'production') {
  //   return ''
  // }

  const thumbnailHost = `https://us-central1-yceffort.cloudfunctions.net/screenshot`

  const queryString = qs.stringify(
    {
      tags: (tags || []).join(','),
      title,
      url: `https://yceffort.kr/${path}`,
      slug,
    },
    { skipEmptyString: true, skipNull: true },
  )

  return `${thumbnailHost}?${queryString}`
}
