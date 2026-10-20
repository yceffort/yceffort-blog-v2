'use client'

import * as stylex from '@stylexjs/stylex'
import Image from 'next/image'
import Link from 'next/link'
import {usePathname} from 'next/navigation'
import {
  useEffect,
  useRef,
  useState,
  type ReactNode,
  type RefObject,
} from 'react'

import * as headerStyles from '@/components/layout/header.styles'
import {SiteConfig} from '@/config'
import {useIsStandalone} from '@/hooks/useIsStandalone'
import {useLocale} from '@/hooks/useLocale'
import profile from '@/public/profile.jpeg'
import {skipLink} from '@/styles/accessibility.styles'

import SiteSearch from '../search/SiteSearch'
import TweaksPanel from '../settings/TweaksPanel'
import Footer from './Footer'
import LanguageSwitch from './LanguageSwitch'
import MobileNav from './MobileNav'
import ScrollTop from './ScrollTop'
import SectionContainer from './SectionContainer'
const sx = stylex.create({
  link: {
    '@layer utilities': {
      display: 'flex',
      alignItems: 'center',
      gap: 'calc(var(--spacing) * 3)',
    },
  },
  image: {
    '@layer utilities': {
      height: 'calc(var(--spacing) * 10)',
      width: 'calc(var(--spacing) * 10)',
      borderRadius: 'calc(infinity * 1px)',
    },
  },
  nav: {
    '@layer utilities': {
      display: {
        default: 'none',
        '@media (width >= 40rem)': 'flex',
      },
      alignItems: 'center',
    },
  },
  svg: {
    '@layer utilities': {
      height: '18px',
      width: '18px',
    },
  },
  containerClass: {
    '@layer utilities': {
      maxWidth: {
        default: null,
        '@media (width >= 80rem)': 'var(--container-5xl)',
      },
    },
  },
  container: {
    '@layer utilities': {
      maxWidth: {
        default: null,
        '@media (width >= 80rem)': 'var(--container-7xl)',
      },
    },
  },
  container2: {
    '@layer utilities': {
      maxWidth: 'var(--container-6xl)',
    },
  },
  main: {
    '@layer utilities': {
      minHeight: 'calc(100vh - 260px)',
      paddingTop: 'calc(var(--spacing) * 6)',
    },
  },
})
function HeaderLogo() {
  const {pathPrefix} = useLocale()
  return (
    <Link
      href={pathPrefix || '/'}
      aria-label="yceffort's blog"
      className={stylex.props(sx.link).className}
    >
      <div className={`logo-ring ${headerStyles.logo_ring}`} aria-hidden="true">
        <span className={headerStyles.logo_ring_span}>
          <Image
            src={profile}
            alt=""
            width={40}
            height={40}
            className={stylex.props(sx.image).className}
            priority
          />
        </span>
      </div>
      <div className={`logo-name ${headerStyles.logo_name}`}>yceffort</div>
    </Link>
  )
}
function HeaderNav() {
  const {pathPrefix} = useLocale()
  const pathname = usePathname() ?? '/'
  const standalone = useIsStandalone()
  const baseMenu = pathPrefix
    ? SiteConfig.menu.map((link) =>
        // 영문판이 있는 메뉴만 /en으로 보낸다
        ['/pages/1', '/about', '/series', '/tags'].includes(link.path)
          ? {
              ...link,
              path: `${pathPrefix}${link.path}`,
            }
          : link,
      )
    : SiteConfig.menu
  const menu = standalone
    ? [...baseMenu, {label: 'Saved', path: `${pathPrefix}/saved`}]
    : baseMenu
  const isActive = (path: string) => {
    if (path.startsWith('http')) {
      return false
    }
    if (path.endsWith('/pages/1')) {
      const base = path.replace(/\/pages\/1$/, '/pages')
      return pathname === path || pathname.startsWith(`${base}/`)
    }
    return pathname === path || pathname.startsWith(`${path}/`)
  }
  return (
    <nav
      className={`nav-pills ${headerStyles.nav_pills} ${stylex.props(sx.nav).className}`}
      aria-label="main"
    >
      {menu.map((link) => {
        const external = link.path.startsWith('http')
        const active = !external && isActive(link.path)
        const common = {
          className: 'nav-link ' + headerStyles.nav_link,
          'data-active': active ? 'true' : 'false',
          'data-external': external ? 'true' : 'false',
        } as const
        const labelClassName = active
          ? `nav-link-label ${headerStyles.nav_link_label_active}`
          : 'nav-link-label'
        if (external) {
          return (
            <a
              key={link.label}
              href={link.path}
              target="_blank"
              rel="noopener noreferrer"
              {...common}
              className={common.className}
            >
              <span className={labelClassName}>{link.label}</span>
              <svg
                className={`nav-link-ext ${headerStyles.nav_link_ext}`}
                width="11"
                height="11"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.2"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden="true"
              >
                <path d="M7 17L17 7" />
                <path d="M8 7h9v9" />
              </svg>
            </a>
          )
        }
        return (
          <Link
            key={link.label}
            href={link.path}
            {...common}
            className={common.className}
          >
            <span className={labelClassName}>{link.label}</span>
          </Link>
        )
      })}
    </nav>
  )
}

// 모바일에서 아래로 스크롤하면 헤더를 숨기고, 위로 스크롤하거나 화면을 탭하면 다시 보인다
const HIDE_AFTER = 80
const DIRECTION_THRESHOLD = 8
function useHideHeaderOnScroll(headerRef: RefObject<HTMLElement | null>) {
  useEffect(() => {
    const header = headerRef.current
    if (!header) {
      return undefined
    }
    let lastY = window.scrollY
    let hidden = false
    const setHidden = (next: boolean) => {
      if (hidden === next) {
        return
      }
      hidden = next
      header.dataset.hidden = next ? 'true' : 'false'
    }
    const onScroll = () => {
      const y = window.scrollY
      const delta = y - lastY
      if (y <= 0) {
        setHidden(false)
        lastY = y
        return
      }
      if (Math.abs(delta) < DIRECTION_THRESHOLD) {
        return
      }
      setHidden(delta > 0 && y > HIDE_AFTER)
      lastY = y
    }
    const onClick = () => {
      setHidden(false)
    }
    window.addEventListener('scroll', onScroll, {
      passive: true,
    })
    window.addEventListener('click', onClick)
    return () => {
      window.removeEventListener('scroll', onScroll)
      window.removeEventListener('click', onClick)
    }
  }, [headerRef])
}
function Header({enSlugs}: {enSlugs: string[]}) {
  const [tweaksOpen, setTweaksOpen] = useState(false)
  const headerRef = useRef<HTMLElement>(null)
  useHideHeaderOnScroll(headerRef)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setTweaksOpen(false)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])
  return (
    <>
      <header
        ref={headerRef}
        className={`site-header ${headerStyles.site_header}`}
      >
        <div className={`site-header-inner ${headerStyles.site_header_inner}`}>
          <HeaderLogo />
          <div className={`header-right ${headerStyles.header_right}`}>
            <HeaderNav />
            <span
              className={`header-sep ${headerStyles.header_sep}`}
              aria-hidden="true"
            />
            <div className={`header-icons ${headerStyles.header_icons}`}>
              <SiteSearch />
              <button
                type="button"
                className={`icon-btn ${headerStyles.icon_btn}`}
                aria-label="Tweaks"
                aria-expanded={tweaksOpen}
                onClick={() => setTweaksOpen((v) => !v)}
              >
                <svg
                  width="18"
                  height="18"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <circle cx="12" cy="12" r="3" />
                  <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
                </svg>
              </button>
              <a
                href="https://github.com/yceffort"
                target="_blank"
                rel="noopener noreferrer"
                className={`icon-btn ${headerStyles.icon_btn}`}
                aria-label="GitHub"
              >
                <svg
                  viewBox="0 0 24 24"
                  className={stylex.props(sx.svg).className}
                  fill="currentColor"
                >
                  <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z" />
                </svg>
              </a>
              <LanguageSwitch enSlugs={enSlugs} />
            </div>
            <MobileNav />
          </div>
        </div>
      </header>
      <TweaksPanel open={tweaksOpen} onClose={() => setTweaksOpen(false)} />
    </>
  )
}
const LayoutWrapper = ({
  children,
  enSlugs,
}: {
  children: ReactNode
  enSlugs: string[]
}) => {
  const pathname = usePathname()
  const isEn = useLocale().locale === 'en'
  // 영문 경로도 같은 폭을 쓰도록 /en 접두사를 떼고 고른다
  const path = pathname === '/en' ? '/' : pathname?.replace(/^\/en(?=\/.)/, '')
  const containerClass = stylex.props(
    path === '/about'
      ? sx.container2
      : path === '/' ||
          path === '/series' ||
          path?.startsWith('/pages') ||
          path?.startsWith('/tags')
        ? sx.container
        : sx.containerClass,
  ).className
  return (
    <>
      <a href="#main" className={`skip-link ${skipLink}`}>
        {isEn ? 'Skip to content' : '본문으로 건너뛰기'}
      </a>
      <Header enSlugs={enSlugs} />
      <SectionContainer className={containerClass}>
        <main id="main" className={stylex.props(sx.main).className}>
          {children}
        </main>
        <Footer />
      </SectionContainer>
      <ScrollTop />
    </>
  )
}
export default LayoutWrapper
