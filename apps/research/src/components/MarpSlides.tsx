'use client'

import * as stylex from '@stylexjs/stylex'
import dynamic from 'next/dynamic'
import {useCallback, useEffect, useMemo, useRef, useState} from 'react'
import type {
  MouseEvent as ReactMouseEvent,
  PointerEvent as ReactPointerEvent,
} from 'react'
import type {Swiper as SwiperClass} from 'swiper'
import 'swiper/css'
import 'swiper/css/effect-creative'
import 'swiper/css/effect-fade'
import {EffectCreative, EffectFade, Virtual} from 'swiper/modules'
import {Swiper, SwiperSlide} from 'swiper/react'

import {useBroadcastChannel} from '@/hooks/useBroadcastChannel'
import {useDrawing} from '@/hooks/useDrawing'
import {useLaserPointer} from '@/hooks/useLaserPointer'
import {offlineHref} from '@/lib/offline/client'
import {getSlideGroups} from '@/lib/slideNavigation'

import {Marp} from './Marp'
import {MarpContextMenu} from './MarpContextMenu'
import type {ContextMenuActions} from './MarpContextMenu'
import {MarpDrawingLayer} from './MarpDrawingLayer'
import {MarpHelpModal} from './MarpHelpModal'
import {MarpOverview} from './MarpOverview'
import {MarpSearchModal} from './MarpSearchModal'
import {readTransition} from './MarpSlides.constants'
import type {TransitionType} from './MarpSlides.constants'
import * as styles from './MarpSlides.styles'
import {styles as sx} from './MarpSlides.styles'

// qrcode.react는 QR을 열 때만 필요하다
const MarpQrModal = dynamic(
  () => import('./MarpQrModal').then((mod) => mod.MarpQrModal),
  {ssr: false},
)

// 터치 롱프레스로 컨텍스트 메뉴를 여는 기준. 이동 허용치는 스와이프와 구분하기 위한 값이다
const LONG_PRESS_MS = 500
const LONG_PRESS_MOVE_TOLERANCE = 10

// QR로 들어온 방문을 GA4 세션 소스(qr)로 가르기 위한 캠페인 파라미터. 쿼리는 해시 앞에 와야 한다.
function qrSlideUrl(slug: string, index: number) {
  const params = new URLSearchParams({
    utm_source: 'qr',
    utm_medium: 'slides',
    utm_campaign: slug,
  })
  return `${window.location.origin}/slides/${slug}?${params}#${index + 1}`
}

const INTERACTIVE_TARGET =
  'a[href], button, input, select, textarea, [contenteditable]'

function isInteractiveEvent(event: Event) {
  return event
    .composedPath()
    .some((node) => node instanceof Element && node.matches(INTERACTIVE_TARGET))
}

const NO_SLIDES: number[] = []

// #3 같은 해시를 0부터 시작하는 슬라이드 번호로 바꾼다. 범위 밖이면 null
function readHashIndex(length: number): number | null {
  const pageNum = parseInt(window.location.hash.slice(1), 10)
  return pageNum >= 1 && pageNum <= length ? pageNum - 1 : null
}

function parseJsonArray(data: string, label: string): string[] {
  try {
    return JSON.parse(data) as string[]
  } catch (error) {
    // eslint-disable-next-line no-console
    console.error(`Failed to parse ${label} data:`, error)
    return []
  }
}

interface MarpSlidesProps {
  dataHtml: string
  dataCss: string
  dataFonts: string
  slug: string
  offline?: boolean
  channelName?: string
  presenterHref?: string
  onPageChange?: (page: number) => void
  postUrl?: string
  // 덱 frontmatter의 transition. 지정되면 뷰어 설정(cookie)보다 우선한다
  defaultTransition?: TransitionType
}

export function MarpSlides({
  dataHtml,
  dataCss,
  dataFonts,
  slug,
  offline = false,
  channelName = `marp-slides-${slug}`,
  presenterHref,
  onPageChange,
  postUrl,
  defaultTransition,
}: MarpSlidesProps) {
  const presenterUrl =
    presenterHref ??
    (offline ? offlineHref(slug, true) : `/slides/${slug}/presenter`)

  const html = useMemo(() => parseJsonArray(dataHtml, 'HTML'), [dataHtml])
  const fonts = useMemo(() => parseJsonArray(dataFonts, 'fonts'), [dataFonts])
  const css = dataCss
  const slideGroups = useMemo(() => getSlideGroups(html), [html])

  // 상태 관리 — SSR과 동일한 초기값(0)으로 시작하여 hydration mismatch 방지
  const [{activeIndex, showHiddenSlides}, setNavigation] = useState(() => ({
    activeIndex: slideGroups.visible[0] ?? 0,
    showHiddenSlides: slideGroups.visible.length === 0,
  }))
  const slideIndices = showHiddenSlides ? slideGroups.all : slideGroups.visible
  const activePosition = Math.max(0, slideIndices.indexOf(activeIndex))
  const multiple = slideIndices.length > 1
  const [isBottomHovered, setIsBottomHovered] = useState(false)
  const [isOverviewOpen, setIsOverviewOpen] = useState(false)
  const [contextMenu, setContextMenu] = useState<{x: number; y: number} | null>(
    null,
  )
  const [isHelpOpen, setIsHelpOpen] = useState(false)
  const [qrUrl, setQrUrl] = useState<string | null>(null)
  const [isPrinting, setIsPrinting] = useState(false)
  const [transition, setTransition] = useState<TransitionType>('slide')
  const [isSearchOpen, setIsSearchOpen] = useState(false)
  const [isLaserMode, setIsLaserMode] = useState(false)
  const [isDrawingMode, setIsDrawingMode] = useState(false)
  // 오버레이가 떠 있으면 키보드와 휠로 슬라이드를 넘기지 않는다
  const isOverlayOpen =
    isOverviewOpen ||
    isSearchOpen ||
    isHelpOpen ||
    qrUrl !== null ||
    contextMenu !== null
  const swiperRef = useRef<SwiperClass | null>(null)
  const containerRef = useRef<HTMLDivElement | null>(null)
  // 터치 롱프레스도 우클릭과 같은 메뉴를 연다
  const longPressTimerRef = useRef<NodeJS.Timeout | null>(null)
  const longPressStartRef = useRef<{x: number; y: number} | null>(null)
  // 롱프레스로 연 직후 따라오는 click은 메뉴를 바로 닫으므로 다음 pointerdown까지 무시한다
  const longPressFiredRef = useRef(false)
  const wheelTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const activeIndexRef = useRef(activeIndex)
  const showHiddenSlidesRef = useRef(showHiddenSlides)
  const lastVisibleIndexRef = useRef(slideGroups.visible[0] ?? 0)

  const applyNavigation = useCallback(
    (index: number, includeHidden = showHiddenSlidesRef.current) => {
      if (!Number.isInteger(index) || index < 0 || index >= html.length) {
        return null
      }
      const isHidden = slideGroups.hidden.includes(index)
      const nextShowHidden = includeHidden || isHidden
      if (
        index === activeIndexRef.current &&
        nextShowHidden === showHiddenSlidesRef.current
      ) {
        return null
      }
      const next = {activeIndex: index, showHiddenSlides: nextShowHidden}
      activeIndexRef.current = index
      showHiddenSlidesRef.current = nextShowHidden
      if (!isHidden) {
        lastVisibleIndexRef.current = index
      }
      setNavigation(next)
      onPageChange?.(index + 1)
      const newHash = `#${index + 1}`
      if (window.location.hash !== newHash) {
        window.location.hash = newHash
      }
      window.gtag?.('event', 'slide_view', {
        slide_number: index + 1,
        page_path: window.location.pathname,
      })
      return next
    },
    [html.length, slideGroups.hidden, onPageChange],
  )

  const laserRef = useLaserPointer(isLaserMode)
  const drawing = useDrawing(isDrawingMode, activeIndex)

  const {sendSlideChange} = useBroadcastChannel(channelName, {
    onSlideChange: applyNavigation,
    onSyncRequest: () => ({
      index: activeIndexRef.current,
      showHiddenSlides: showHiddenSlidesRef.current,
    }),
  })

  const navigateTo = useCallback(
    (index: number, includeHidden = showHiddenSlidesRef.current) => {
      const next = applyNavigation(index, includeHidden)
      if (next) {
        sendSlideChange(next.activeIndex, 'audience', next.showHiddenSlides)
      }
    },
    [applyNavigation, sendSlideChange],
  )

  const slideUrl = (index: number) =>
    `${window.location.origin}/slides/${slug}#${index + 1}`
  const goHome = useCallback(() => {
    window.location.href = offline ? '/offline' : '/'
  }, [offline])
  const openPresenter = useCallback(() => {
    window.open(presenterUrl, 'presenter', 'width=1200,height=800')
  }, [presenterUrl])

  // 원본 슬라이드 번호와 숨김 장을 제외한 Swiper 위치를 구분해 동기화한다.
  useEffect(() => {
    const swiper = swiperRef.current
    if (swiper && !swiper.destroyed && swiper.activeIndex !== activePosition) {
      swiper.slideTo(activePosition)
    }
  }, [activePosition, showHiddenSlides])

  // 클라이언트에서만 실행되는 초기화 - hash에서 초기 슬라이드 동기화
  useEffect(() => {
    const initialIndex = readHashIndex(html.length)
    if (initialIndex) {
      navigateTo(initialIndex)
    }

    setTransition(defaultTransition ?? readTransition())
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const handleActiveIndexChange = useCallback(
    (instance: SwiperClass) => {
      const newIndex = slideIndices[instance.activeIndex]
      if (newIndex !== undefined) {
        navigateTo(newIndex)
      }
    },
    [slideIndices, navigateTo],
  )

  const handleSwiper = useCallback((instance: SwiperClass) => {
    swiperRef.current = instance
  }, [])

  // 키보드 네비게이션
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // 입력 요소에 포커스가 있으면 단축키 무시
      const target = e.target as HTMLElement | null
      if (
        target &&
        (target.tagName === 'INPUT' ||
          target.tagName === 'TEXTAREA' ||
          target.isContentEditable)
      ) {
        return
      }

      // 물리 키 기준(e.code)으로 비교해 한영 전환/Caps Lock 상태와 무관하게 동작

      // 도움말 토글 (? 키)
      if (e.key === '?' || (e.code === 'Slash' && e.shiftKey)) {
        setIsHelpOpen((prev) => !prev)
        return
      }

      // 검색 열기 (/ 또는 Cmd/Ctrl+F)
      if (
        (e.code === 'Slash' && !e.shiftKey) ||
        ((e.metaKey || e.ctrlKey) && e.code === 'KeyF')
      ) {
        e.preventDefault()
        setIsSearchOpen(true)
        return
      }

      // 레이저 포인터 토글 (L 키)
      if (e.code === 'KeyL') {
        setIsLaserMode((prev) => !prev)
        return
      }

      // 드로잉 모드 토글 (D 키)
      if (e.code === 'KeyD') {
        setIsDrawingMode((prev) => !prev)
        return
      }

      // QR 코드 토글 (Q 키)
      if (e.code === 'KeyQ') {
        setQrUrl((prev) =>
          prev ? null : qrSlideUrl(slug, activeIndexRef.current),
        )
        return
      }

      // 오버뷰 토글 (G 키)
      if (e.code === 'KeyG') {
        if (multiple) {
          setIsOverviewOpen((prev) => !prev)
        }
        return
      }

      // 발표자 모드 열기 (P 키)
      if (e.code === 'KeyP') {
        openPresenter()
        return
      }

      // ESC로 검색/QR/도움말/오버뷰 닫기
      if (e.key === 'Escape') {
        if (isSearchOpen) {
          setIsSearchOpen(false)
          return
        }
        if (qrUrl) {
          setQrUrl(null)
          return
        }
        if (isHelpOpen) {
          setIsHelpOpen(false)
          return
        }
        if (isOverviewOpen) {
          setIsOverviewOpen(false)
          return
        }
      }

      if (isOverlayOpen || !multiple) {
        return
      }

      switch (e.key) {
        case 'ArrowLeft':
          swiperRef.current?.slidePrev()
          break
        case 'ArrowRight':
          swiperRef.current?.slideNext()
          break
        case 'Home':
          swiperRef.current?.slideTo(0)
          break
        case 'End':
          swiperRef.current?.slideTo(slideIndices.length - 1)
          break
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [
    multiple,
    slideIndices.length,
    isOverviewOpen,
    isHelpOpen,
    qrUrl,
    isSearchOpen,
    isOverlayOpen,
    slug,
    openPresenter,
  ])

  // 휠 네비게이션 (300ms 디바운스)
  const handleWheel = (e: React.WheelEvent<HTMLDivElement>) => {
    if (!multiple || isOverlayOpen || wheelTimeoutRef.current) {
      return
    }
    if (e.deltaY > 0) {
      swiperRef.current?.slideNext()
    } else if (e.deltaY < 0) {
      swiperRef.current?.slidePrev()
    }
    wheelTimeoutRef.current = setTimeout(() => {
      wheelTimeoutRef.current = null
    }, 300)
  }

  // 클릭 네비게이션 (좌우/상하 10% 영역) (memoized)
  const handleSlideClick = useCallback(
    (e: ReactMouseEvent<HTMLDivElement>) => {
      // Shadow DOM 안의 링크 클릭은 슬라이드 이동 영역 클릭으로 처리하지 않는다.
      if (isInteractiveEvent(e.nativeEvent)) {
        return
      }

      // 롱프레스로 메뉴를 연 손가락의 click이다. 이동 동작으로 이어지면 안 된다
      if (longPressFiredRef.current) {
        return
      }

      const rect = e.currentTarget.getBoundingClientRect()
      const xPercent = ((e.clientX - rect.left) / rect.width) * 100
      const yPercent = ((e.clientY - rect.top) / rect.height) * 100

      // 상단 10% 영역 클릭 - 첫 슬라이드로
      if (yPercent <= 10) {
        if (multiple) {
          swiperRef.current?.slideTo(0)
        }
      }
      // 하단 10% 영역 클릭 - 루트 페이지로
      else if (yPercent >= 90) {
        goHome()
      }
      // 좌측 10% 영역 클릭 - 이전 슬라이드
      else if (xPercent <= 10 && multiple) {
        swiperRef.current?.slidePrev()
      }
      // 우측 10% 영역 클릭 - 다음 슬라이드
      else if (xPercent >= 90 && multiple) {
        swiperRef.current?.slideNext()
      }
      // 중앙 영역은 아무 동작 없음
    },
    [multiple, goHome],
  )

  // 해시 변경 감지
  useEffect(() => {
    const handleHashChange = () => {
      const newIndex = readHashIndex(html.length)
      if (newIndex !== null && newIndex !== activeIndexRef.current) {
        navigateTo(newIndex)
      }
    }

    window.addEventListener('hashchange', handleHashChange)
    return () => window.removeEventListener('hashchange', handleHashChange)
  }, [html.length, navigateTo])

  const clearNavigationHover = useCallback(() => {
    if (containerRef.current) delete containerRef.current.dataset.navigationEdge
    setIsBottomHovered(false)
  }, [])

  const handleContextMenu = useCallback(
    (e: ReactMouseEvent<HTMLDivElement>) => {
      if (isInteractiveEvent(e.nativeEvent)) return
      e.preventDefault()
      setContextMenu({x: e.clientX, y: e.clientY})
    },
    [],
  )

  const cancelLongPress = useCallback(() => {
    if (longPressTimerRef.current) {
      clearTimeout(longPressTimerRef.current)
      longPressTimerRef.current = null
    }
    longPressStartRef.current = null
  }, [])

  const handlePointerDown = useCallback(
    (e: ReactPointerEvent<HTMLDivElement>) => {
      longPressFiredRef.current = false
      cancelLongPress()

      // 슬라이드 위에서만 연다. 모달, 드로잉 캔버스, 메뉴 자신은 Swiper 밖에 있다
      if (
        e.pointerType === 'mouse' ||
        isInteractiveEvent(e.nativeEvent) ||
        !(e.target as HTMLElement).closest('.swiper')
      ) {
        return
      }

      const {clientX, clientY} = e
      longPressStartRef.current = {x: clientX, y: clientY}
      longPressTimerRef.current = setTimeout(() => {
        longPressTimerRef.current = null
        longPressStartRef.current = null
        longPressFiredRef.current = true
        setContextMenu({x: clientX, y: clientY})
      }, LONG_PRESS_MS)
    },
    [cancelLongPress],
  )

  const handlePointerMove = useCallback(
    (e: ReactPointerEvent<HTMLDivElement>) => {
      if (e.pointerType === 'mouse') {
        const slide = (e.target as Element).closest('[data-slide-surface]')
        if (slide && !isInteractiveEvent(e.nativeEvent)) {
          const rect = slide.getBoundingClientRect()
          const x = (e.clientX - rect.left) / rect.width
          const y = (e.clientY - rect.top) / rect.height
          const edge =
            y <= 0.1
              ? 'top'
              : y >= 0.9
                ? 'bottom'
                : x <= 0.1
                  ? 'left'
                  : x >= 0.9
                    ? 'right'
                    : ''
          e.currentTarget.dataset.navigationEdge = edge
          setIsBottomHovered(edge === 'bottom')
        } else {
          clearNavigationHover()
        }
      }
      const start = longPressStartRef.current
      if (!start) {
        return
      }
      if (
        Math.abs(e.clientX - start.x) > LONG_PRESS_MOVE_TOLERANCE ||
        Math.abs(e.clientY - start.y) > LONG_PRESS_MOVE_TOLERANCE
      ) {
        cancelLongPress()
      }
    },
    [cancelLongPress, clearNavigationHover],
  )

  useEffect(() => cancelLongPress, [cancelLongPress])

  // 컨텍스트 메뉴 외부 클릭 시 닫기
  const isContextMenuOpen = contextMenu !== null
  useEffect(() => {
    if (!isContextMenuOpen) {
      return undefined
    }
    const handleClickOutside = () => {
      if (!longPressFiredRef.current) {
        setContextMenu(null)
      }
    }
    document.addEventListener('click', handleClickOutside)
    return () => document.removeEventListener('click', handleClickOutside)
  }, [isContextMenuOpen])

  useEffect(() => {
    if (!isPrinting) {
      return undefined
    }
    const handleAfterPrint = () => setIsPrinting(false)
    window.addEventListener('afterprint', handleAfterPrint)
    // 인쇄 전용 슬라이드들이 마운트되도록 한 프레임 대기 후 인쇄
    const timer = window.setTimeout(() => {
      window.print()
    }, 250)
    return () => {
      window.clearTimeout(timer)
      window.removeEventListener('afterprint', handleAfterPrint)
    }
  }, [isPrinting])

  const marpRenderData = useMemo(() => ({html, css, fonts}), [html, css, fonts])

  if (html.length === 0) {
    return (
      <div className={styles.errorMessage}>슬라이드를 로드할 수 없습니다.</div>
    )
  }

  const menuActions: ContextMenuActions = {
    prev: () => swiperRef.current?.slidePrev(),
    next: () => swiperRef.current?.slideNext(),
    first: () => swiperRef.current?.slideTo(0),
    last: () => swiperRef.current?.slideTo(slideIndices.length - 1),
    goTo: (index) => navigateTo(index),
    toggleHidden: () => {
      if (showHiddenSlides) {
        navigateTo(lastVisibleIndexRef.current, false)
      } else if (slideGroups.hidden.length > 0) {
        navigateTo(slideGroups.hidden[0], true)
      }
    },
    overview: () => setIsOverviewOpen(true),
    presenter: openPresenter,
    fullscreen: () => {
      if (document.fullscreenElement) {
        void document.exitFullscreen()
      } else {
        void containerRef.current?.requestFullscreen()
      }
    },
    copyLink: () => void navigator.clipboard.writeText(slideUrl(activeIndex)),
    qr: () => setQrUrl(qrSlideUrl(slug, activeIndex)),
    print: () => setIsPrinting(true),
    toggleLaser: () => setIsLaserMode((prev) => !prev),
    toggleDrawing: () => setIsDrawingMode((prev) => !prev),
    help: () => setIsHelpOpen(true),
    home: goHome,
  }

  return (
    <div
      ref={containerRef}
      className={`marp-slides ${
        stylex.props(
          sx.marpSlides,
          multiple && sx.multiple,
          isLaserMode && sx.laserMode,
        ).className ?? ''
      }`}
      data-transition={transition}
      data-laser={isLaserMode}
      data-printing={isPrinting}
      onContextMenu={handleContextMenu}
      onWheel={handleWheel}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerLeave={clearNavigationHover}
      onPointerUp={cancelLongPress}
      onPointerCancel={cancelLongPress}
    >
      <Swiper
        key={`${transition}-${showHiddenSlides}`}
        // 전환 효과 변경으로 재생성되어도 해시에서 복원한 현재 위치를 유지한다.
        initialSlide={activePosition}
        modules={[Virtual, EffectFade, EffectCreative]}
        virtual={{enabled: multiple, addSlidesBefore: 1, addSlidesAfter: 1}}
        enabled={multiple}
        allowTouchMove={multiple}
        noSwipingSelector={`.swiper-no-swiping, ${INTERACTIVE_TARGET}`}
        speed={transition === 'none' ? 0 : transition === 'glide' ? 600 : 350}
        effect={
          transition === 'fade'
            ? 'fade'
            : transition === 'zoom' || transition === 'glide'
              ? 'creative'
              : 'slide'
        }
        fadeEffect={{crossFade: true}}
        creativeEffect={
          transition === 'glide'
            ? {
                // 나가는 장은 살짝 물러나며 흐려지고, 들어오는 장이 그 위로 미끄러져 덮는다
                prev: {opacity: 0.35, scale: 0.97, translate: ['-20%', 0, 0]},
                next: {translate: ['100%', 0, 0]},
              }
            : {
                prev: {opacity: 0, scale: 0.7, translate: [0, 0, -200]},
                next: {opacity: 0, scale: 1.3, translate: [0, 0, 200]},
              }
        }
        onActiveIndexChange={handleActiveIndexChange}
        onSwiper={handleSwiper}
        // 접근성 개선
        a11y={{
          enabled: true,
          prevSlideMessage: '이전 슬라이드',
          nextSlideMessage: '다음 슬라이드',
          firstSlideMessage: '첫 번째 슬라이드',
          lastSlideMessage: '마지막 슬라이드',
          paginationBulletMessage: '슬라이드 {{index}}로 이동',
        }}
      >
        {slideIndices.map((i, position) => (
          <SwiperSlide key={i} virtualIndex={position}>
            <div
              className={styles.marpSlide}
              data-slide-surface
              {...(multiple
                ? {
                    onClick: handleSlideClick,
                    role: 'button',
                    tabIndex: 0,
                    'aria-label': `슬라이드 ${i + 1}${slideGroups.hidden.includes(i) ? ' (숨김)' : ''}`,
                  }
                : {})}
            >
              <Marp border rendered={marpRenderData} page={i + 1} />

              {/* 클릭 가능 영역 시각적 표시 (hover 시) */}
              {multiple && (
                <>
                  <div
                    className={styles.clickAreaLeft}
                    data-navigation-area="left"
                    aria-hidden="true"
                  />
                  <div
                    className={styles.clickAreaRight}
                    data-navigation-area="right"
                    aria-hidden="true"
                  />
                </>
              )}

              {/* 상단 영역 - 첫 슬라이드로 */}
              <div
                className={styles.clickAreaTop}
                data-navigation-area="top"
                aria-hidden="true"
              />

              {/* 하단 영역 - 루트 페이지로 */}
              <div
                className={styles.clickAreaBottom}
                data-navigation-area="bottom"
                aria-hidden="true"
              />
            </div>
          </SwiperSlide>
        ))}
      </Swiper>

      {multiple && (
        <div className={styles.progressBar}>
          <div
            className={styles.progressBarFill}
            style={{
              width: `${((activePosition + 1) / slideIndices.length) * 100}%`,
            }}
          />
        </div>
      )}

      {multiple && (
        <div
          className={
            isBottomHovered ? styles.pageIndicatorVisible : styles.pageIndicator
          }
        >
          {activePosition + 1} / {slideIndices.length}
        </div>
      )}

      {multiple && isOverviewOpen && (
        <MarpOverview
          rendered={marpRenderData}
          slideIndices={slideIndices}
          hiddenIndices={slideGroups.hidden}
          activeIndex={activeIndex}
          onSelect={(index) => {
            navigateTo(index)
            setIsOverviewOpen(false)
          }}
          onClose={() => setIsOverviewOpen(false)}
        />
      )}

      {isLaserMode && (
        <div ref={laserRef} className={styles.laserDot} aria-hidden="true" />
      )}

      {isDrawingMode && (
        <MarpDrawingLayer
          drawing={drawing}
          onClose={() => setIsDrawingMode(false)}
        />
      )}

      {/* 인쇄(PDF) 전용 컨테이너 - 모든 슬라이드를 페이지 단위로 렌더링 */}
      {isPrinting && (
        <div
          className={`marp-print-container ${styles.printContainer}`}
          aria-hidden="true"
        >
          {html.map((_, i) => (
            <div key={i} className="marp-print-slide">
              <Marp border={false} rendered={marpRenderData} page={i + 1} />
            </div>
          ))}
        </div>
      )}

      {isSearchOpen && (
        <MarpSearchModal
          html={html}
          excluded={showHiddenSlides ? NO_SLIDES : slideGroups.hidden}
          onSelect={(index) => {
            navigateTo(index)
            setIsSearchOpen(false)
          }}
          onClose={() => setIsSearchOpen(false)}
        />
      )}

      {qrUrl && <MarpQrModal qrUrl={qrUrl} onClose={() => setQrUrl(null)} />}

      {isHelpOpen && <MarpHelpModal onClose={() => setIsHelpOpen(false)} />}

      {contextMenu && (
        <MarpContextMenu
          x={contextMenu.x}
          y={contextMenu.y}
          position={activePosition}
          total={slideIndices.length}
          slideCount={html.length}
          multiple={multiple}
          hiddenCount={slideGroups.hidden.length}
          showHiddenSlides={showHiddenSlides}
          isLaserMode={isLaserMode}
          isDrawingMode={isDrawingMode}
          offline={offline}
          slug={slug}
          postUrl={postUrl}
          actions={menuActions}
          onClose={() => setContextMenu(null)}
        />
      )}
    </div>
  )
}
