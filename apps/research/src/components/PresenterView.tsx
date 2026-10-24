'use client'

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
} from 'react'
import type {PointerEvent} from 'react'

import {useBroadcastChannel} from '@/hooks/useBroadcastChannel'
import {useTimer} from '@/hooks/useTimer'
import {getSlideGroups} from '@/lib/slideNavigation'

import {Marp} from './Marp'
import {
  DEFAULT_NOTES_FONT_SIZE,
  MAX_NOTES_FONT_SIZE,
  MIN_NOTES_FONT_SIZE,
  PresenterNotes,
} from './PresenterNotes'
import * as styles from './PresenterView.styles'

// Resizing the notes always leaves this much room for them and the slide previews.
const MIN_NOTES_HEIGHT = 120
const MIN_SLIDES_HEIGHT = 96
// The notes font size is a persisted preference, read as an external store so
// the server render and the first client render agree on the default.
const NOTES_FONT_SIZE_KEY = 'research-presenter-notes-font-size'
const notesFontSizeListeners = new Set<() => void>()

function subscribeNotesFontSize(listener: () => void) {
  notesFontSizeListeners.add(listener)
  return () => notesFontSizeListeners.delete(listener)
}

function readNotesFontSize() {
  try {
    const size = Number(localStorage.getItem(NOTES_FONT_SIZE_KEY))
    return size >= MIN_NOTES_FONT_SIZE && size <= MAX_NOTES_FONT_SIZE
      ? size
      : DEFAULT_NOTES_FONT_SIZE
  } catch {
    return DEFAULT_NOTES_FONT_SIZE
  }
}

function writeNotesFontSize(size: number) {
  try {
    localStorage.setItem(NOTES_FONT_SIZE_KEY, String(size))
  } catch {
    // A disabled/full preference store must not interrupt a presentation.
  }
  notesFontSizeListeners.forEach((listener) => listener())
}

interface PresenterViewProps {
  dataHtml: string
  dataCss: string
  dataFonts: string
  dataNotes: string
  slug: string
  channelName?: string
}

export function PresenterView({
  dataHtml,
  dataCss,
  dataFonts,
  dataNotes,
  slug,
  channelName = `marp-slides-${slug}`,
}: PresenterViewProps) {
  const html = useMemo(() => {
    try {
      return JSON.parse(dataHtml) as string[]
    } catch {
      return []
    }
  }, [dataHtml])

  const fonts = useMemo(() => {
    try {
      return JSON.parse(dataFonts) as string[]
    } catch {
      return []
    }
  }, [dataFonts])

  const notes = useMemo(() => {
    try {
      return JSON.parse(dataNotes) as string[]
    } catch {
      return []
    }
  }, [dataNotes])

  const css = dataCss
  const slideGroups = useMemo(() => getSlideGroups(html), [html])
  const [{activeIndex, showHiddenSlides}, setNavigation] = useState(() => ({
    activeIndex: slideGroups.visible[0] ?? 0,
    showHiddenSlides: slideGroups.visible.length === 0,
  }))
  const slideIndices = showHiddenSlides ? slideGroups.all : slideGroups.visible
  const activePosition = slideIndices.indexOf(activeIndex)
  const nextIndex = slideIndices[activePosition + 1]
  const {elapsedTime, isRunning, toggle, reset} = useTimer()
  const slidesRef = useRef<HTMLDivElement>(null)
  const notesResize = useRef<{y: number; height: number; max: number} | null>(
    null,
  )
  const [notesHeight, setNotesHeight] = useState<number>()
  const notesFontSize = useSyncExternalStore(
    subscribeNotesFontSize,
    readNotesFontSize,
    () => DEFAULT_NOTES_FONT_SIZE,
  )

  const changeNotesFontSize = (delta: number) => {
    writeNotesFontSize(
      Math.min(
        MAX_NOTES_FONT_SIZE,
        Math.max(MIN_NOTES_FONT_SIZE, notesFontSize + delta),
      ),
    )
  }

  const {sendSlideChange, requestSync} = useBroadcastChannel(channelName, {
    onSlideChange: (index, includeHidden) => {
      if (Number.isInteger(index) && index >= 0 && index < html.length) {
        setNavigation({
          activeIndex: index,
          showHiddenSlides: includeHidden || slideGroups.hidden.includes(index),
        })
      }
    },
  })

  useEffect(() => {
    requestSync()
  }, [requestSync])

  const goToPrev = useCallback(() => {
    const newIndex = slideIndices[activePosition - 1]
    if (newIndex !== undefined) {
      setNavigation({activeIndex: newIndex, showHiddenSlides})
      sendSlideChange(newIndex, 'presenter', showHiddenSlides)
    }
  }, [activePosition, slideIndices, showHiddenSlides, sendSlideChange])

  const goToNext = useCallback(() => {
    if (nextIndex !== undefined) {
      setNavigation({activeIndex: nextIndex, showHiddenSlides})
      sendSlideChange(nextIndex, 'presenter', showHiddenSlides)
    }
  }, [nextIndex, showHiddenSlides, sendSlideChange])

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Let focused notes and their scrollbar handle reading keys themselves.
      if (
        e.defaultPrevented ||
        (e.target instanceof Element &&
          e.target.closest('.marp-presenter-notes'))
      )
        return
      switch (e.key) {
        case 'ArrowLeft':
          goToPrev()
          break
        case 'ArrowRight':
        case ' ':
          e.preventDefault()
          goToNext()
          break
        case 'Home':
          if (slideIndices.length > 0) {
            setNavigation({activeIndex: slideIndices[0], showHiddenSlides})
            sendSlideChange(slideIndices[0], 'presenter', showHiddenSlides)
          }
          break
        case 'End':
          if (slideIndices.length > 0) {
            const lastIndex = slideIndices[slideIndices.length - 1]
            setNavigation({activeIndex: lastIndex, showHiddenSlides})
            sendSlideChange(lastIndex, 'presenter', showHiddenSlides)
          }
          break
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [goToPrev, goToNext, slideIndices, showHiddenSlides, sendSlideChange])

  const startNotesResize = (event: PointerEvent<HTMLHRElement>) => {
    if (event.button !== 0) return
    event.preventDefault()
    event.currentTarget.setPointerCapture(event.pointerId)
    const height =
      event.currentTarget.nextElementSibling!.getBoundingClientRect().height
    const slides = slidesRef.current!.getBoundingClientRect().height
    notesResize.current = {
      y: event.clientY,
      height,
      max: Math.max(height, height + slides - MIN_SLIDES_HEIGHT),
    }
  }

  const moveNotesResize = (event: PointerEvent<HTMLHRElement>) => {
    const start = notesResize.current
    if (!start) return
    const height = start.height + start.y - event.clientY
    setNotesHeight(
      Math.round(Math.min(start.max, Math.max(MIN_NOTES_HEIGHT, height))),
    )
  }

  const endNotesResize = () => {
    notesResize.current = null
  }

  const marpRenderData = useMemo(() => ({html, css, fonts}), [html, css, fonts])
  const currentNote = notes[activeIndex] || ''
  const hasNextSlide = nextIndex !== undefined

  if (html.length === 0) {
    return (
      <div className={styles.presenterView}>
        <div style={{padding: 24}}>슬라이드를 로드할 수 없습니다.</div>
      </div>
    )
  }

  return (
    <div className={styles.presenterView}>
      <div className={styles.timerBar}>
        <span className={styles.timer}>{elapsedTime}</span>
        <button
          className={isRunning ? styles.timerButtonRunning : styles.timerButton}
          onClick={toggle}
        >
          {isRunning ? '일시정지' : '시작'}
        </button>
        <button className={styles.timerButton} onClick={reset}>
          리셋
        </button>
      </div>

      <div ref={slidesRef} className={styles.slidesContainer}>
        <div className={styles.slideWrapper}>
          <div className={styles.slideLabel}>
            현재 슬라이드
            {slideGroups.hidden.includes(activeIndex) ? ' · 숨김' : ''}
          </div>
          <div className={`marp-presenter-slide ${styles.slideContent}`}>
            <Marp
              rendered={marpRenderData}
              page={activeIndex + 1}
              border={false}
              fit="contain"
              className={`marp-presenter-container ${styles.marpContainer}`}
            />
          </div>
        </div>

        <div className={styles.slideWrapper}>
          <div className={styles.slideLabelNext}>다음 슬라이드</div>
          <div className={`marp-presenter-slide ${styles.slideContent}`}>
            {hasNextSlide ? (
              <Marp
                rendered={marpRenderData}
                page={nextIndex + 1}
                border={false}
                fit="contain"
                className={`marp-presenter-container ${styles.marpContainer}`}
              />
            ) : (
              <div className={styles.noNextSlide}>마지막 슬라이드</div>
            )}
          </div>
        </div>
      </div>

      <hr
        className={styles.notesResizer}
        aria-label="발표자 노트 높이 조절"
        onPointerDown={startNotesResize}
        onPointerMove={moveNotesResize}
        onPointerUp={endNotesResize}
        onLostPointerCapture={endNotesResize}
      />

      <PresenterNotes
        key={activeIndex}
        note={currentNote}
        height={notesHeight}
        fontSize={notesFontSize}
        onFontSizeChange={changeNotesFontSize}
      />

      <div className={styles.controlBar}>
        <div className={styles.navigation}>
          <button
            className={styles.navButton}
            onClick={goToPrev}
            disabled={activePosition === 0}
          >
            ◀ 이전
          </button>
          <span className={styles.pageIndicator}>
            {activePosition + 1} / {slideIndices.length}
            {showHiddenSlides && slideGroups.hidden.length > 0
              ? ' · 숨김 포함'
              : ''}
          </span>
          <button
            className={styles.navButton}
            onClick={goToNext}
            disabled={!hasNextSlide}
          >
            다음 ▶
          </button>
        </div>
        <div className={styles.keyHints}>
          <span>← → 슬라이드 이동</span>
          <span>Space 다음 슬라이드</span>
          <span>Home / End 처음 / 끝</span>
        </div>
      </div>
    </div>
  )
}
