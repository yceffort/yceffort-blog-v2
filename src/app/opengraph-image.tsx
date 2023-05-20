import { ImageResponse } from 'next/server'

import { SiteConfig } from '#src/config'
import OpenGraphComponent, { OpenGraphImageSize } from '#components/OpenGraph'

export const runtime = 'edge'

export const alt = SiteConfig.author.name
export const size = OpenGraphImageSize

export const contentType = 'image/png'

export default function OpenGraphImage() {
  return new ImageResponse(
    (
      <OpenGraphComponent
        title="Welcome to yceffort's blog"
        url="https://yceffort.kr"
        tags={['blog', 'frontend']}
      />
    ),
    { ...size },
  )
}
