'use client'

import {useTheme} from 'next-themes'
import {useSyncExternalStore} from 'react'

import {icon_btn, icon_svg_18} from '@/components/LayoutWrapper.styles'

function subscribeNoop() {
  return () => {}
}

export default function ThemeToggle() {
  const {resolvedTheme, setTheme} = useTheme()
  // 서버는 테마를 모르므로 hydration 전에는 아이콘을 고정한다.
  const mounted = useSyncExternalStore(
    subscribeNoop,
    () => true,
    () => false,
  )

  const dark = mounted && resolvedTheme === 'dark'

  return (
    <button
      type="button"
      className={icon_btn}
      aria-label={dark ? '밝은 테마로' : '어두운 테마로'}
      onClick={() => setTheme(dark ? 'light' : 'dark')}
    >
      {dark ? (
        <svg
          className={icon_svg_18}
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.8"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden="true"
        >
          <circle cx="12" cy="12" r="4" />
          <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41" />
        </svg>
      ) : (
        <svg
          className={icon_svg_18}
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.8"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden="true"
        >
          <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
        </svg>
      )}
    </button>
  )
}
