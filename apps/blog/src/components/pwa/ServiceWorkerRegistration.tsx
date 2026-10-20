'use client'

import * as stylex from '@stylexjs/stylex'
import {usePathname} from 'next/navigation'
import {useEffect, useRef, useState} from 'react'

import {useIsStandalone} from '@/hooks/useIsStandalone'
import {useLocale} from '@/hooks/useLocale'
import {browserDeps, recordRead, syncOnLaunch} from '@/utils/offline/sync'

const sx = stylex.create({
  output: {
    '@layer utilities': {
      '--blog-translate-y': '0px',
      translate: 'var(--blog-translate-x) var(--blog-translate-y)',
      opacity: '100%',
    },
  },
  output2: {
    '@layer utilities': {
      '--blog-translate-y': 'calc(var(--spacing) * 3)',
      translate: 'var(--blog-translate-x) var(--blog-translate-y)',
      opacity: '0%',
    },
  },
  output3: {
    '@layer utilities': {
      position: 'fixed',
      bottom: 'calc(var(--spacing) * 6)',
      left: 'calc(1 / 2 * 100%)',
      zIndex: '50',
      '--blog-translate-x': 'calc(calc(1 / 2 * 100%) * -1)',
      translate: 'var(--blog-translate-x) var(--blog-translate-y)',
      borderRadius: 'calc(infinity * 1px)',
      backgroundColor: {
        default: 'oklch(21% 0.006 285.885)',
        ':is(.dark *)': 'oklch(96.7% 0.001 286.375)',
      },
      paddingInline: 'calc(var(--spacing) * 4)',
      paddingBlock: 'calc(var(--spacing) * 2)',
      fontSize: 'var(--text-sm)',
      lineHeight: 'var(--blog-leading, var(--text-sm--line-height))',
      fontWeight: 'var(--font-weight-medium)',
      whiteSpace: 'nowrap',
      color: {
        default: 'var(--color-white)',
        ':is(.dark *)': 'oklch(21% 0.006 285.885)',
      },
      '--blog-shadow':
        '0 10px 15px -3px var(--blog-shadow-color, rgb(0 0 0 / 0.1)), 0 4px 6px -4px var(--blog-shadow-color, rgb(0 0 0 / 0.1))',
      boxShadow:
        'var(--blog-inset-shadow), var(--blog-inset-ring-shadow), var(--blog-ring-offset-shadow), var(--blog-ring-shadow), var(--blog-shadow)',
      transitionProperty: 'opacity,translate',
      transitionTimingFunction:
        'var(--blog-ease, var(--default-transition-timing-function))',
      transitionDuration: '300ms',
      '--blog-duration': '300ms',
    },
  },
})
const EXIT_DURATION = 300
export function ServiceWorkerRegistration() {
  const [message, setMessage] = useState<string | null>(null)
  const [visible, setVisible] = useState(false)
  const timers = useRef<ReturnType<typeof setTimeout>[]>([])
  const launched = useRef(false)
  const lastPath = useRef<string | null>(null)
  const standalone = useIsStandalone()
  const pathname = usePathname()
  const {locale} = useLocale()
  const isEn = locale === 'en'
  useEffect(() => {
    if ('serviceWorker' in navigator) {
      void navigator.serviceWorker.register('/sw.js')
    }
    return () => {
      for (const timer of timers.current) {
        clearTimeout(timer)
      }
      timers.current = []
    }
  }, [])
  useEffect(() => {
    if (!standalone) {
      return
    }
    const show = (text: string) => {
      for (const timer of timers.current) {
        clearTimeout(timer)
      }
      setMessage(text)
      setVisible(false)
      timers.current = [
        // 마운트 직후 한 프레임 뒤에 표시해야 등장 트랜지션이 걸린다
        setTimeout(() => setVisible(true), 30),
        setTimeout(() => {
          setVisible(false)
          timers.current.push(setTimeout(() => setMessage(null), EXIT_DURATION))
        }, 2500),
      ]
    }
    const readText = isEn ? '✓ Saved for offline' : '✓ 오프라인에 저장됨'
    const deps = browserDeps()
    if (!launched.current) {
      launched.current = true
      lastPath.current = pathname
      void (async () => {
        try {
          const {savedCurrent, announce} = await syncOnLaunch(
            window.location.pathname,
            deps,
          )
          if (typeof announce === 'number') {
            show(
              isEn
                ? `✓ Saved ${announce} recent posts for offline`
                : `✓ 최근 글 ${announce}편을 오프라인에 저장했습니다`,
            )
          } else if (savedCurrent) {
            show(readText)
          }
        } catch {}
      })()
      return
    }
    if (pathname === lastPath.current) {
      return
    }
    lastPath.current = pathname
    void (async () => {
      try {
        if (await recordRead(pathname, deps)) {
          show(readText)
        }
      } catch {}
    })()
  }, [standalone, pathname, isEn])
  if (!message) {
    return null
  }
  return (
    <output
      className={
        stylex.props(sx.output3, visible ? sx.output : sx.output2).className
      }
    >
      {message}
    </output>
  )
}
