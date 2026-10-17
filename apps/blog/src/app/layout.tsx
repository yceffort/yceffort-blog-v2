import * as stylex from '@stylexjs/stylex'
import {
  AmbientEffects,
  BotTracker,
  GoogleAnalyticsPageViewTracker,
  GoogleAnalyticsScripts,
  OutboundLinkTracker,
  Providers,
} from '@yceffort/shared/components'
import {THEME_COOKIE_SCRIPT} from '@yceffort/shared/utils'
import type {Metadata} from 'next'

import '@/styles/stylex.css'
import '@/styles/tokens.stylex'

import {Suspense, type ReactNode} from 'react'

import {CwvTelemetry} from '@/components/analytics/CwvTelemetry'
import {GoogleAnalyticsWebVitalsTracker} from '@/components/analytics/GoogleAnalyticsWebVitalsTracker'
import {InternalNavTracker} from '@/components/analytics/InternalNavTracker'
import * as ambientStyles from '@/components/layout/ambient.styles'
import LayoutWrapper from '@/components/layout/LayoutWrapper'
import NavigationDirection from '@/components/layout/NavigationDirection'
import PullToRefresh from '@/components/pwa/PullToRefresh'
import {ServiceWorkerRegistration} from '@/components/pwa/ServiceWorkerRegistration'
import {SiteConfig} from '@/config'
import {buildOgImageUrl} from '@/utils/og'
import {getAllPosts} from '@/utils/Post'
const sx = stylex.create({
  body: {
    '@layer utilities': {
      WebkitFontSmoothing: 'antialiased',
      MozOsxFontSmoothing: 'grayscale',
    },
  },
})
export const metadata: Metadata = {
  title: SiteConfig.title,
  description: SiteConfig.url,
  authors: [
    {
      name: SiteConfig.author.name,
    },
  ],
  referrer: 'origin-when-cross-origin',
  creator: SiteConfig.author.name,
  publisher: SiteConfig.author.name,
  metadataBase: new URL('https://yceffort.kr'),
  formatDetection: {
    email: false,
    address: false,
    telephone: false,
  },
  openGraph: {
    title: SiteConfig.title,
    description: 'Frontend-focused full stack engineer',
    url: 'https://yceffort.kr',
    siteName: SiteConfig.title,
    images: [
      {
        url: buildOgImageUrl({
          title: SiteConfig.title,
          description: 'Frontend-focused full stack engineer',
          type: 'page',
        }),
        width: 1200,
        height: 630,
        alt: SiteConfig.title,
      },
    ],
    locale: 'ko_KR',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: SiteConfig.title,
    description: 'Frontend-focused full stack engineer',
    images: [
      buildOgImageUrl({
        title: SiteConfig.title,
        description: 'Frontend-focused full stack engineer',
        type: 'page',
      }),
    ],
  },
  icons: {
    icon: '/favicon/apple-touch-icon.png',
    shortcut: '/favicon/apple-touch-icon.png',
    apple: '/favicon/apple-touch-icon.png',
    other: {
      rel: '/favicon/apple-icon-precomposed',
      url: '/favicon/apple-icon-precomposed.png',
    },
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
    },
  },
}
const GA_MEASUREMENT_ID = SiteConfig.googleAnalyticsId
export default async function Layout({children}: {children: ReactNode}) {
  const enSlugs = (await getAllPosts('en')).map((post) => post.fields.slug)
  return (
    <>
      <html lang="ko" data-scroll-behavior="smooth" suppressHydrationWarning>
        <head>
          <script
            dangerouslySetInnerHTML={{
              __html: THEME_COOKIE_SCRIPT,
            }}
          />
          <link
            rel="alternate"
            type="application/rss+xml"
            title="RSS Feed"
            href="/feed.xml"
          />
          <link
            rel="icon"
            type="image/png"
            href="/favicon/favicon-96x96.png"
            sizes="96x96"
          />
          <link rel="icon" type="image/svg+xml" href="/favicon/favicon.svg" />
          <link rel="shortcut icon" href="/favicon/favicon.ico" />
          <link
            rel="apple-touch-icon"
            sizes="180x180"
            href="/favicon/apple-touch-icon.png"
          />
          <link rel="manifest" href="/favicon/site.webmanifest" />
          <meta name="theme-color" content="#ffffff" />
          <meta name="mobile-web-app-capable" content="yes" />
        </head>
        <body className={stylex.props(sx.body).className}>
          <Suspense fallback={null}>
            <NavigationDirection />
          </Suspense>
          <AmbientEffects
            classNames={{
              animBg: ambientStyles.effect_anim_bg,
              grain: ambientStyles.effect_grain,
              cursorGlow: ambientStyles.effect_cursor_glow,
            }}
          />
          <PullToRefresh />
          <Providers>
            <Suspense fallback={null}>
              <LayoutWrapper enSlugs={enSlugs}>{children}</LayoutWrapper>
            </Suspense>
          </Providers>
          {GA_MEASUREMENT_ID && process.env.NODE_ENV === 'production' && (
            <>
              <GoogleAnalyticsScripts
                measurementId={GA_MEASUREMENT_ID}
                siteUrl={SiteConfig.url}
              />
              <Suspense fallback={null}>
                <GoogleAnalyticsPageViewTracker />
              </Suspense>
              <OutboundLinkTracker />
              <InternalNavTracker />
            </>
          )}
          {process.env.NODE_ENV === 'production' && (
            <>
              <GoogleAnalyticsWebVitalsTracker />
              <CwvTelemetry />
              <BotTracker />
              <ServiceWorkerRegistration />
            </>
          )}
        </body>
      </html>
    </>
  )
}
