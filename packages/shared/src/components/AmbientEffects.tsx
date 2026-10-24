'use client'

import {useEffect} from 'react'

export default function AmbientEffects({
  classNames,
}: {
  classNames?: {animBg?: string; grain?: string; cursorGlow?: string}
}) {
  useEffect(() => {
    if (typeof window === 'undefined') {
      return undefined
    }
    const media = window.matchMedia('(prefers-reduced-motion: reduce)')
    if (media.matches) {
      return undefined
    }
    const glow = document.getElementById('cursor-glow')
    if (!glow) {
      return undefined
    }
    let raf = 0
    let nextX = window.innerWidth / 2
    let nextY = window.innerHeight / 2
    const onMove = (e: PointerEvent) => {
      nextX = e.clientX
      nextY = e.clientY
      if (!raf) {
        raf = window.requestAnimationFrame(() => {
          // left/top 이동은 layout shift 로 집계돼 CLS 에 쌓인다. transform 이동은 집계에서 제외된다.
          glow.style.transform = `translate3d(${nextX}px, ${nextY}px, 0) translate(-50%, -50%)`
          raf = 0
        })
      }
    }
    window.addEventListener('pointermove', onMove, {
      passive: true,
    })
    return () => {
      window.removeEventListener('pointermove', onMove)
      if (raf) {
        window.cancelAnimationFrame(raf)
      }
    }
  }, [])
  return (
    <>
      <div id="anim-bg" aria-hidden="true" className={classNames?.animBg} />
      <div id="grain" aria-hidden="true" className={classNames?.grain} />
      <div
        id="cursor-glow"
        aria-hidden="true"
        className={classNames?.cursorGlow}
      />
    </>
  )
}
