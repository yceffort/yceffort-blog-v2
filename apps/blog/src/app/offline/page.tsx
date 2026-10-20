import * as stylex from '@stylexjs/stylex'
import type {Metadata} from 'next'

import * as ambientStyles from '@/components/layout/ambient.styles'

const sx = stylex.create({
  div: {
    '@layer utilities': {
      display: 'flex',
      minHeight: '50vh',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      textAlign: 'center',
    },
  },
  svg: {
    '@layer utilities': {
      marginBottom: 'calc(var(--spacing) * 6)',
      color: {
        default: 'oklch(70.5% 0.015 286.067)',
        ':is(.dark *)': 'oklch(55.2% 0.016 285.938)',
      },
    },
  },
  h1: {
    '@layer utilities': {
      marginBottom: 'calc(var(--spacing) * 3)',
      fontSize: 'var(--text-2xl)',
      lineHeight: 'var(--blog-leading, var(--text-2xl--line-height))',
      fontWeight: 'var(--font-weight-bold)',
      color: {
        default: 'oklch(21% 0.006 285.885)',
        ':is(.dark *)': 'oklch(96.7% 0.001 286.375)',
      },
    },
  },
  p: {
    '@layer utilities': {
      marginBottom: 'calc(var(--spacing) * 6)',
      maxWidth: 'var(--container-md)',
      color: {
        default: 'oklch(44.2% 0.017 285.786)',
        ':is(.dark *)': 'oklch(70.5% 0.015 286.067)',
      },
    },
  },
  a: {
    '@layer utilities': {
      borderRadius: 'var(--radius-lg)',
      backgroundColor: {
        default: 'oklch(21% 0.006 285.885)',
        '@media (hover: hover)': {
          default: null,
          ':hover': 'oklch(27.4% 0.006 286.033)',
          ':is(.dark *):hover': 'oklch(92% 0.004 286.32)',
        },
        ':is(.dark *)': 'oklch(96.7% 0.001 286.375)',
      },
      paddingInline: 'calc(var(--spacing) * 5)',
      paddingBlock: 'calc(var(--spacing) * 2.5)',
      fontSize: 'var(--text-sm)',
      lineHeight: 'var(--blog-leading, var(--text-sm--line-height))',
      fontWeight: 'var(--font-weight-medium)',
      color: {
        default: 'var(--color-white)',
        ':is(.dark *)': 'oklch(21% 0.006 285.885)',
      },
      transitionProperty:
        'color, background-color, border-color, outline-color, text-decoration-color, fill, stroke, --blog-gradient-from, --blog-gradient-via, --blog-gradient-to',
      transitionTimingFunction:
        'var(--blog-ease, var(--default-transition-timing-function))',
      transitionDuration:
        'var(--blog-duration, var(--default-transition-duration))',
    },
  },
})
export const metadata: Metadata = {
  title: 'Offline - yceffort',
}
export default function OfflinePage() {
  return (
    <div className={`page-view ${ambientStyles.page_view}`}>
      <div className={stylex.props(sx.div).className}>
        <svg
          width="64"
          height="64"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
          className={stylex.props(sx.svg).className}
        >
          <path d="M1 1l22 22M16.72 11.06A10.94 10.94 0 0119 12.55M5 12.55a10.94 10.94 0 015.17-2.39M10.71 5.05A16 16 0 0122.56 9M1.42 9a15.91 15.91 0 014.7-2.88M8.53 16.11a6 6 0 016.95 0M12 20h.01" />
        </svg>
        <h1 className={stylex.props(sx.h1).className}>오프라인 상태입니다</h1>
        <p className={stylex.props(sx.p).className}>
          인터넷 연결이 끊어졌습니다. 블로그를 앱으로 설치하면 읽은 글과 최근
          글을 오프라인에서도 읽을 수 있습니다.
        </p>
        {/* oxlint-disable-next-line next/no-html-link-for-pages -- 오프라인 폴백에서는 라우터 없이 전체 리로드가 필요하다 */}
        <a href="/" className={stylex.props(sx.a).className}>
          홈으로 돌아가기
        </a>
      </div>
    </div>
  )
}
