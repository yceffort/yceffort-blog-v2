import 'src/tailwind.css'
import App from 'next/app'
import Head from 'next/head'
import Router from 'next/router'
import { ThemeProvider } from 'next-themes'
import { DefaultSeo } from 'next-seo'
import Script from 'next/script'
import { getAnalytics } from "firebase/analytics";

import { SiteConfig as config } from '#src/config'
import { SEO } from '#components/SEO'
import LayoutWrapper from '#components/LayoutWrapper'
import firebaseApp from "#src/lib/firebase";

class MyApp extends App {
  public componentDidMount() {
    getAnalytics(firebaseApp)

    window.history.scrollRestoration = 'auto'

    const cachedScrollPositions: Array<[number, number]> = []
    let shouldScrollRestore: null | { x: number; y: number }

    Router.events.on('routeChangeStart', () => {
      cachedScrollPositions.push([window.scrollX, window.scrollY])
    })

    Router.events.on('routeChangeComplete', () => {
      if (shouldScrollRestore) {
        const { x, y } = shouldScrollRestore
        window.scrollTo(x, y)
        shouldScrollRestore = null
      }
      window.history.scrollRestoration = 'auto'
    })

    Router.beforePopState(() => {
      if (cachedScrollPositions.length > 0) {
        const scrolledPosition = cachedScrollPositions.pop()
        if (scrolledPosition) {
          shouldScrollRestore = {
            x: scrolledPosition[0],
            y: scrolledPosition[1],
          }
        }
      }
      window.history.scrollRestoration = 'manual'
      return true
    })
  }

  public render() {
    const { Component, pageProps } = this.props

    return (
      <ThemeProvider attribute="class">
        <Head>
          <meta content="width=device-width, initial-scale=1" name="viewport" />
        </Head>
        <DefaultSeo {...SEO} />
        <LayoutWrapper>
          <Component {...pageProps} />
        </LayoutWrapper>
        <Script
          async
          src={`https://www.googletagmanager.com/gtag/js?id=${config.googleAnalyticsId}`}
        />
        <Script
          id="gtag"
          dangerouslySetInnerHTML={{
            __html: `
                window.dataLayer = window.dataLayer || [];
                function gtag(){dataLayer.push(arguments);}
                gtag('js', new Date());

                gtag('config', '${config.googleAnalyticsId}', {
                  page_path: window.location.pathname,
                });
              `,
          }}
        />
      </ThemeProvider>
    )
  }
}

export default MyApp
