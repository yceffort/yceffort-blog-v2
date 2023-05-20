import '../tailwind.css'

import type { Metadata } from 'next'
import { ReactNode } from 'react'

import { SiteConfig } from '#src/config'
import LayoutWrapper from '#components/LayoutWrapper'

export const metadata: Metadata = {
  title: SiteConfig.title,
  description: SiteConfig.url,
  authors: [{ name: SiteConfig.author.name }],
  referrer: 'origin-when-cross-origin',
  creator: SiteConfig.author.name,
  publisher: SiteConfig.author.name,
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
      <html lang="kr">
        <body>
          <LayoutWrapper>{children}</LayoutWrapper>
        </body>
      </html>
    </>
  )
}
