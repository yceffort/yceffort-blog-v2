import {ImageResponse} from 'next/og'
import sharp from 'sharp'

import {SiteConfig} from '@/config'
import {loadOgFont, OG_FONT_URL} from '@/utils/ogFont'
import {unblockSvgLoader} from '@/utils/ogSharpUnblock'
export async function GET(request: Request) {
  try {
    unblockSvgLoader()
    const {searchParams} = new URL(request.url)
    const title = searchParams.get('title')
    const description = searchParams.get('description')
    const tagsParam = searchParams.get('tags')
    const pathParam = searchParams.get('path')
    const type = searchParams.get('type')
    const size = searchParams.get('size') // 'large' for mobile cover

    const isLarge = size === 'large'
    const fontSize = {
      title: isLarge ? 80 : 64,
      description: isLarge ? 36 : 28,
      tags: isLarge ? 28 : 22,
      address: isLarge ? 32 : 26,
    }
    const descMaxLength = isLarge ? 60 : 80
    if (!title) {
      return new Response('Missing title', {
        status: 400,
      })
    }
    const urlParam = searchParams.get('url')
    const thumbnailParam = searchParams.get('thumbnail')
    const address =
      urlParam || (pathParam ? `${SiteConfig.url}${pathParam}` : SiteConfig.url)
    const tags = tagsParam ? tagsParam.split(',') : []

    const protocol = request.headers.get('x-forwarded-proto') || 'http'
    const host = request.headers.get('host')
    const baseUrl = `${protocol}://${host}`
    const hasThumbnail = !!thumbnailParam
    const imageUrl = new URL(
      thumbnailParam ||
        `/${type === 'page' ? 'og-background-page.jpg' : 'og-background.jpg'}`,
      baseUrl,
    )
    // thumbnail은 이 사이트의 경로만 받는다. 이어 붙이면 `.evil.com/x.png`로 호스트가 바뀐다
    if (imageUrl.origin !== new URL(baseUrl).origin) {
      return new Response('Invalid thumbnail', {status: 400})
    }
    const imageRes = await fetch(imageUrl)
    if (!imageRes.ok) {
      throw new Error(`Failed to fetch image: ${imageUrl}`)
    }
    const imageBuffer = Buffer.from(await imageRes.arrayBuffer())
    const imageMime = hasThumbnail ? 'image/png' : 'image/jpeg'
    // satori 는 WebP 를 그리지 못하고 빈 이미지로 넘어간다. 썸네일은 PNG 로 바꿔 넣는다
    const imageBase64 = (
      hasThumbnail ? await sharp(imageBuffer).png().toBuffer() : imageBuffer
    ).toString('base64')

    const fontData = await loadOgFont()
    if (!fontData) {
      throw new Error(`Failed to fetch font: ${OG_FONT_URL}`)
    }
    return new ImageResponse(
      <div
        style={{
          display: 'flex',
          height: '100%',
          width: '100%',
          position: 'relative',
          backgroundColor: 'black',
          fontFamily: '"NanumGothic"',
        }}
      >
        <img
          alt={title}
          src={`data:${imageMime};base64,${imageBase64}`}
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            objectFit: 'cover',
            filter: hasThumbnail ? 'blur(1px)' : 'blur(3px)',
            transform: 'scale(1.02)',
          }}
        />

        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            width: '100%',
            height: '100%',
            alignItems: 'flex-start',
            justifyContent: 'center',
            position: 'relative',
            padding: '50px 70px',
            ...(hasThumbnail
              ? {
                  backgroundImage:
                    'linear-gradient(to right, rgba(0,0,0,0.8) 0%, rgba(0,0,0,0.5) 55%, rgba(0,0,0,0.25) 100%)',
                }
              : {
                  backgroundColor: 'rgba(0, 0, 0, 0.4)',
                }),
          }}
        >
          <div
            style={{
              display: 'flex',
              fontSize: fontSize.title,
              fontWeight: 700,
              lineHeight: 1.2,
              color: '#fff',
              textAlign: 'left',
              wordBreak: 'keep-all',
              letterSpacing: '-0.03em',
              marginBottom: isLarge ? 12 : 8,
              justifyContent: 'flex-start',
              textShadow: '2px 2px 8px rgba(0,0,0,0.8)',
              width: '100%',
            }}
          >
            {title}
          </div>

          {description ? (
            <div
              style={{
                display: 'flex',
                fontSize: fontSize.description,
                fontWeight: 400,
                color: '#eee',
                lineHeight: 1.4,
                textAlign: 'left',
                wordBreak: 'keep-all',
                letterSpacing: '-0.02em',
                marginBottom: isLarge ? 20 : 16,
                maxWidth: '90%',
                justifyContent: 'flex-start',
                textShadow: '1px 1px 6px rgba(0,0,0,0.7)',
                width: '100%',
              }}
            >
              {description.length > descMaxLength
                ? description.slice(0, descMaxLength) + '...'
                : description}
            </div>
          ) : null}

          {tags.length > 0 ? (
            <div
              style={{
                display: 'flex',
                flexDirection: 'row',
                flexWrap: 'wrap',
                justifyContent: 'flex-start',
                gap: '10px',
                marginBottom: 20,
                maxWidth: '90%',
                width: '100%',
              }}
            >
              {tags.map((tag, index) => (
                <div
                  key={index}
                  style={{
                    display: 'flex',
                    backgroundColor: 'rgba(255, 255, 255, 0.2)',
                    borderRadius: isLarge ? '18px' : '15px',
                    padding: isLarge ? '10px 18px' : '8px 15px',
                    fontSize: fontSize.tags,
                    color: '#fff',
                    textShadow: '1px 1px 4px rgba(0,0,0,0.6)',
                    fontWeight: 400,
                    whiteSpace: 'nowrap',
                  }}
                >
                  #{tag}
                </div>
              ))}
            </div>
          ) : null}

          <div
            style={{
              display: 'flex',
              fontSize: fontSize.address,
              fontWeight: 700,
              color: '#fff',
              marginTop: isLarge ? 12 : 10,
              paddingTop: isLarge ? 20 : 16,
              width: '100%',
              justifyContent: 'flex-start',
              textShadow: '1px 1px 6px rgba(0,0,0,0.7)',
              borderTop: '2px solid rgba(255,255,255,0.3)',
            }}
          >
            {address}
          </div>
        </div>
      </div>,
      {
        width: 1200,
        height: 630,
        fonts: [
          {
            name: 'NanumGothic',
            data: fontData,
            style: 'normal',
            weight: 700,
          },
        ],
        headers: {
          'Cache-Control': 'public, max-age=31536000, immutable',
        },
      },
    )
  } catch (e: unknown) {
    return new Response(
      `Failed to generate the image: ${e instanceof Error ? e.message : 'Unknown error'}`,
      {
        status: 500,
      },
    )
  }
}
