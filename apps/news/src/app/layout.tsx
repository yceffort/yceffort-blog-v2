import '@/styles/stylex.css'
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
import {Suspense, type ReactNode} from 'react'

import LayoutWrapper from '@/components/LayoutWrapper'
import {SiteConfig} from '@/config'

export const metadata: Metadata = {
  title: {
    default: SiteConfig.title,
    template: `%s | ${SiteConfig.title}`,
  },
  description: SiteConfig.description,
  authors: [{name: SiteConfig.author.name}],
  referrer: 'origin-when-cross-origin',
  creator: SiteConfig.author.name,
  publisher: SiteConfig.author.name,
  metadataBase: new URL(SiteConfig.url),
  openGraph: {
    title: SiteConfig.title,
    description: SiteConfig.description,
    url: SiteConfig.url,
    siteName: SiteConfig.title,
    locale: 'ko_KR',
    type: 'website',
  },
  formatDetection: {
    email: false,
    address: false,
    telephone: false,
  },
  icons: {
    icon: '/favicon/apple-touch-icon.png',
    shortcut: '/favicon/apple-touch-icon.png',
    apple: '/favicon/apple-touch-icon.png',
  },
  robots: {
    index: true,
    follow: true,
  },
}

export default function Layout({children}: {children: ReactNode}) {
  return (
    <html lang="ko" suppressHydrationWarning>
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: THEME_COOKIE_SCRIPT,
          }}
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
      </head>
      <body>
        <AmbientEffects />
        <Providers>
          <LayoutWrapper>{children}</LayoutWrapper>
        </Providers>
        {process.env.NODE_ENV === 'production' && (
          <>
            <GoogleAnalyticsScripts
              measurementId={SiteConfig.googleAnalyticsId}
              siteUrl={SiteConfig.url}
            />
            <Suspense fallback={null}>
              <GoogleAnalyticsPageViewTracker />
              <BotTracker />
            </Suspense>
            <OutboundLinkTracker />
          </>
        )}
      </body>
    </html>
  )
}
