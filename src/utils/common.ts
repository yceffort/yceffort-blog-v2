import qs from 'query-string'

export function getThumbnailUrl({
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
  const thumbnailHost =
    (process.env.NODE_ENV === 'production'
      ? `https://asia-northeast3-yceffort.cloudfunctions.net`
      : 'http://localhost:5000/yceffort/asia-northeast3') + '/screenshot'

  const queryString = qs.stringify(
    {
      tags: (tags || []).join(','),
      title,
      url: `https://yceffort.kr/${path}`,
      slug,
    },
    {skipEmptyString: true, skipNull: true},
  )

  return `${thumbnailHost}?${queryString}`
}
