import './tailwind.css'

import Script from 'next/script'

import {Analytics} from '@vercel/analytics/react'

import type {Metadata} from 'next'
import type {ReactNode} from 'react'

import LayoutWrapper from '#components/LayoutWrapper'
import {Providers} from '#components/Provider'
import {SiteConfig} from '#src/config'

export const metadata: Metadata = {
  title: SiteConfig.title,
  description: SiteConfig.url,
  authors: [{name: SiteConfig.author.name}],
  referrer: 'origin-when-cross-origin',
  creator: SiteConfig.author.name,
  publisher: SiteConfig.author.name,
  metadataBase: new URL('https://yceffort.kr'),
  formatDetection: {
    email: false,
    address: false,
    telephone: false,
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

export default function Layout({children}: {children: ReactNode}) {
  return (
    <>
      <html lang="kr" suppressHydrationWarning>
        <body className="bg-white text-black antialiased dark:bg-gray-900 dark:text-white">
          <Providers>
            <LayoutWrapper>{children}</LayoutWrapper>
          </Providers>
          <Script
            src={`https://www.googletagmanager.com/gtag/js?id=${SiteConfig.googleAnalyticsId}`}
            strategy="afterInteractive"
          />
          <Script id="google-analytics" strategy="afterInteractive">
            {`
          window.dataLayer = window.dataLayer || [];
          function gtag(){dataLayer.push(arguments);}
          gtag('js', new Date());

          gtag('config', '${SiteConfig.googleAnalyticsId}');
        `}
          </Script>
          <Analytics />
        </body>
      </html>
    </>
  )
}
