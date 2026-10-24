'use client'

import {useCallback, useEffect, useId, useRef, useState} from 'react'
import type {KeyboardEvent, PointerEvent} from 'react'

import * as styles from './PresenterView.styles'

export const DEFAULT_NOTES_FONT_SIZE = 16
export const MIN_NOTES_FONT_SIZE = 12
export const MAX_NOTES_FONT_SIZE = 40
const NOTES_FONT_SIZE_STEP = 2

export function PresenterNotes({
  note,
  height,
  fontSize,
  onFontSizeChange,
}: {
  note: string
  height?: number
  fontSize: number
  onFontSizeChange: (delta: number) => void
}) {
  const viewportId = useId()
  const viewportRef = useRef<HTMLDivElement>(null)
  const contentRef = useRef<HTMLDivElement>(null)
  const trackRef = useRef<HTMLDivElement>(null)
  const thumbRef = useRef<HTMLDivElement>(null)
  const dragOffset = useRef<number | null>(null)
  const [scroll, setScroll] = useState({
    overflow: false,
    more: false,
    percent: 0,
  })

  const measure = useCallback(() => {
    const viewport = viewportRef.current!
    const height = trackRef.current!.clientHeight
    const max = Math.max(0, viewport.scrollHeight - viewport.clientHeight)
    const thumb = Math.min(
      height,
      Math.max(
        24,
        (height * viewport.clientHeight) / (viewport.scrollHeight || 1),
      ),
    )
    return {viewport, max, thumb, travel: height - thumb}
  }, [])

  const syncScroll = useCallback(() => {
    // 언마운트 중에는 ref 가 effect cleanup 보다 먼저 비워져서, 그 사이에 온 콜백은 건너뛴다.
    if (!viewportRef.current || !trackRef.current || !thumbRef.current) return
    const {viewport, max, thumb, travel} = measure()
    const position = Math.min(max, Math.max(0, viewport.scrollTop))
    const ratio = max > 0 ? position / max : 0
    // Moving the thumb should not rerender the slide previews or timer.
    thumbRef.current.style.height = `${thumb}px`
    thumbRef.current.style.transform = `translateY(${ratio * travel}px)`
    const next = {
      overflow: max > 1,
      more: max - position > 1,
      percent: Math.round(ratio * 100),
    }
    setScroll((previous) =>
      previous.overflow === next.overflow &&
      previous.more === next.more &&
      previous.percent === next.percent
        ? previous
        : next,
    )
  }, [measure])

  useEffect(() => {
    const viewport = viewportRef.current!
    viewport.scrollTop = 0
    const observer = new ResizeObserver(syncScroll)
    observer.observe(viewport)
    observer.observe(contentRef.current!)
    observer.observe(trackRef.current!)
    viewport.addEventListener('scroll', syncScroll, {passive: true})
    return () => {
      observer.disconnect()
      viewport.removeEventListener('scroll', syncScroll)
    }
  }, [syncScroll])

  const moveThumb = (clientY: number, offset: number) => {
    const {viewport, max, travel} = measure()
    if (travel <= 0) return
    const y = clientY - trackRef.current!.getBoundingClientRect().top - offset
    viewport.scrollTop = (Math.max(0, Math.min(travel, y)) / travel) * max
  }

  const startDrag = (event: PointerEvent<HTMLDivElement>) => {
    if (event.button !== 0 || !scroll.overflow) return
    event.preventDefault()
    event.currentTarget.focus()
    event.currentTarget.setPointerCapture(event.pointerId)
    const {thumb} = measure()
    dragOffset.current =
      event.target === thumbRef.current
        ? event.clientY - thumbRef.current.getBoundingClientRect().top
        : thumb / 2
    moveThumb(event.clientY, dragOffset.current)
  }

  const handleScrollKey = (event: KeyboardEvent<HTMLDivElement>) => {
    const {viewport, max} = measure()
    const page = viewport.clientHeight * 0.8
    let target = viewport.scrollTop
    switch (event.key) {
      case 'ArrowUp':
        target -= 40
        break
      case 'ArrowDown':
        target += 40
        break
      case 'PageUp':
        target -= page
        break
      case 'PageDown':
        target += page
        break
      case ' ':
        target += event.shiftKey ? -page : page
        break
      case 'Home':
        target = 0
        break
      case 'End':
        target = max
        break
      default:
        return
    }
    event.preventDefault()
    viewport.scrollTop = target
  }

  return (
    <section
      className={`marp-presenter-notes ${styles.notesPanel}`}
      aria-label="발표자 노트"
      style={height === undefined ? undefined : {height, maxHeight: 'none'}}
    >
      <div className={styles.notesHeader}>
        <div className={styles.notesLabel}>발표자 노트</div>
        <div className={styles.notesFontControls}>
          <button
            type="button"
            className={styles.notesFontButton}
            onClick={() => onFontSizeChange(-NOTES_FONT_SIZE_STEP)}
            disabled={fontSize <= MIN_NOTES_FONT_SIZE}
            aria-label="글자 작게"
          >
            A-
          </button>
          <span className={styles.notesFontSize}>{fontSize}px</span>
          <button
            type="button"
            className={styles.notesFontButton}
            onClick={() => onFontSizeChange(NOTES_FONT_SIZE_STEP)}
            disabled={fontSize >= MAX_NOTES_FONT_SIZE}
            aria-label="글자 크게"
          >
            A+
          </button>
        </div>
      </div>
      <div className={styles.notesBody}>
        <section
          id={viewportId}
          ref={viewportRef}
          className={`marp-presenter-notes-viewport ${styles.notesViewport}`}
          aria-label="발표자 노트 본문"
          // Scrollable text needs keyboard focus for native PageDown/Space scrolling.
          // eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex
          tabIndex={0}
        >
          <div
            ref={contentRef}
            className={`marp-presenter-notes-content ${styles.notesContent}`}
            style={{fontSize}}
          >
            {note || <span className={styles.noNotes}>노트 없음</span>}
          </div>
        </section>
        <div
          ref={trackRef}
          className={
            scroll.overflow ? styles.notesTrack : styles.notesTrackIdle
          }
          role="scrollbar"
          aria-label="발표자 노트 스크롤"
          aria-controls={viewportId}
          aria-orientation="vertical"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={scroll.percent}
          aria-disabled={!scroll.overflow}
          tabIndex={scroll.overflow ? 0 : -1}
          onKeyDown={handleScrollKey}
          onPointerDown={startDrag}
          onPointerMove={(event) => {
            if (dragOffset.current !== null)
              moveThumb(event.clientY, dragOffset.current)
          }}
          onPointerUp={(event) => {
            dragOffset.current = null
            if (event.currentTarget.hasPointerCapture(event.pointerId))
              event.currentTarget.releasePointerCapture(event.pointerId)
          }}
          onLostPointerCapture={() => {
            dragOffset.current = null
          }}
          onPointerCancel={() => {
            dragOffset.current = null
          }}
        >
          <div ref={thumbRef} className={styles.notesThumb} />
        </div>
      </div>
      <div className={styles.notesStatus}>
        {scroll.more ? (
          <button
            type="button"
            className={styles.notesMore}
            onClick={() => {
              const viewport = viewportRef.current!
              viewport.scrollTop += viewport.clientHeight * 0.8
            }}
          >
            ↓ 아래에 내용 더 있음
          </button>
        ) : (
          <span>
            {!note
              ? '이 슬라이드에는 노트가 없습니다'
              : scroll.overflow
                ? '✓ 끝까지 읽음'
                : '노트 전체 표시 중'}
          </span>
        )}
      </div>
    </section>
  )
}
