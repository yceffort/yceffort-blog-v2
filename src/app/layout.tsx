import './tailwind.css'

import type { Metadata } from 'next'
import { ReactNode } from 'react'
import Script from 'next/script'

import { SiteConfig } from '#src/config'
import LayoutWrapper from '#components/LayoutWrapper'
import { Providers } from '#components/Provider'

export const metadata: Metadata = {
  title: SiteConfig.title,
  description: SiteConfig.url,
  authors: [{ name: SiteConfig.author.name }],
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
    icon: '/favicon/apple-icon.png',
    shortcut: '/favicon/apple-icon.png',
    apple: '/favicon/apple-icon.png',
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
  viewport: {
    width: 'device-width',
    initialScale: 1,
  },
}

export default function Layout({ children }: { children: ReactNode }) {
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
          function gtag(){window.dataLayer.push(arguments);}
          gtag('js', new Date());

          gtag('config', '${SiteConfig.googleAnalyticsId}');
        `}
          </Script>
        </body>
      </html>
    </>
  )
}
