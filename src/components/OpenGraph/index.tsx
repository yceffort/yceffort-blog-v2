export const OpenGraphImageSize = {
  width: 1200,
  height: 1546 / (2850 / 1200),
}

const textOptions = {
  color: 'white',
  fontSize: '80px',
  fontWeight: 'bold',
  textShadow: '2px 2px 4px rgba(0, 0, 0, 0.5)',
  lineHeight: '100%',
} as const

export default function OpenGraphComponent({
  title,
  tags,
  url,
  imageSrc,
}: {
  title: string
  tags: string[]
  url: string
  imageSrc?: string
}) {
  return (
    <>
      <img
        alt={title}
        src={imageSrc || 'https://yceffort.kr/default-image.png'}
        style={{
          width: `${OpenGraphImageSize.width}px`,
          height: `${OpenGraphImageSize.height}px`,
          objectFit: 'cover', // 이미지를 커버하도록 설정
          filter: 'brightness(80%)', // 파란색 필터 적용
          backgroundColor: 'rgba(0, 0, 255, 0.2)',
        }}
      />
      <span
        style={{
          position: 'absolute',
          top: '25%',
          left: '80px',
          width: `${OpenGraphImageSize.width - 160}px`,
          wordWrap: 'break-word',
          wordBreak: 'break-all',
          ...textOptions,
        }}
      >
        {title}
      </span>
      <span
        style={{
          position: 'absolute',
          top: '65%',
          left: '80px',
          color: 'white',
          fontSize: '40px',
          fontWeight: 'bold',
          textShadow: '2px 2px 4px rgba(0, 0, 0, 0.5)',
          textAlign: 'center', // 텍스트 가운데 정렬
          lineHeight: '100%', // 텍스트 세로 정렬
        }}
      >
        {tags.map((tag) => `#${tag}`).join(' ')}
      </span>
      <span
        style={{
          position: 'absolute',
          top: '75%',
          left: '80px',
          color: 'white',
          fontSize: '40px',
          fontWeight: 'bold',
          textShadow: '2px 2px 4px rgba(0, 0, 0, 0.5)',
          textAlign: 'center', // 텍스트 가운데 정렬
          lineHeight: '100%', // 텍스트 세로 정렬
        }}
      >
        {url}
      </span>
    </>
  )
}
