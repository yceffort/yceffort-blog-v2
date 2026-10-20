import * as stylex from '@stylexjs/stylex'

const sx = stylex.create({
  figure: {
    '@layer utilities': {
      marginBlock: 'calc(var(--spacing) * 6)',
    },
  },
  iframe: {
    '@layer utilities': {
      width: '100%',
      borderRadius: 'var(--radius-lg)',
      borderStyle: 'var(--blog-border-style)',
      borderWidth: '1px',
      borderColor: {
        default: 'oklch(92% 0.004 286.32)',
        ':is(.dark *)': 'oklch(37% 0.013 285.805)',
      },
      backgroundColor: 'var(--color-white)',
    },
  },
  figcaption: {
    '@layer utilities': {
      marginTop: 'calc(var(--spacing) * 2)',
      textAlign: 'center',
      fontSize: 'var(--text-sm)',
      lineHeight: 'var(--blog-leading, var(--text-sm--line-height))',
      color: {
        default: 'oklch(55.2% 0.016 285.938)',
        ':is(.dark *)': 'oklch(70.5% 0.015 286.067)',
      },
    },
  },
})
interface LiveDemoProps {
  /** public/ 아래에 둔 자급자족 HTML 데모 경로 (예: /demos/2026/08/banner-motion.html) */
  src: string
  title: string
  height?: number
}

/**
 * 본문 안에서 바로 돌아가는 HTML 데모. public/demos/ 아래의 자급자족 HTML을
 * sandbox iframe으로 렌더링한다.
 */
export default function LiveDemo({src, title, height = 480}: LiveDemoProps) {
  return (
    <figure className={stylex.props(sx.figure).className}>
      <iframe
        src={src}
        title={title}
        loading="lazy"
        // allow-same-origin이 없으면 서비스 워커가 iframe 문서 요청을 가로채지
        // 못해 설치된 PWA의 오프라인 저장분이 열리지 않는다. 데모는 전부 public/
        // 아래 자급자족 파일이라 격리를 풀어도 사이트 스크립트와 같은 신뢰 범위다
        // oxlint-disable-next-line react/iframe-missing-sandbox
        sandbox="allow-scripts allow-same-origin"
        style={{
          height,
        }}
        className={stylex.props(sx.iframe).className}
      />
      <figcaption className={stylex.props(sx.figcaption).className}>
        {title} ·{' '}
        <a href={src} target="_blank" rel="noreferrer">
          새 탭에서 열기 ↗
        </a>
      </figcaption>
    </figure>
  )
}
