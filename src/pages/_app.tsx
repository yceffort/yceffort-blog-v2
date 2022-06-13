import 'src/tailwind.css'
import App from 'next/app'
import Head from 'next/head'
// eslint-disable-next-line import/no-named-as-default
import Router from 'next/router'
import { ThemeProvider } from 'next-themes'
import { DefaultSeo } from 'next-seo'
import Script from 'next/script'

import { SiteConfig as config } from '#src/config'
import { SEO } from '#components/SEO'
import LayoutWrapper from '#components/LayoutWrapper'

class MyApp extends App {
  public componentDidMount() {
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
        {process.env.NODE_ENV === 'production' && (
          <>
            <Script src="https://www.gstatic.com/firebasejs/8.1.1/firebase-app.js" />
            <Script src="https://www.gstatic.com/firebasejs/8.1.1/firebase-analytics.js" />
            <Script
              id="firebase"
              strategy="lazyOnload"
              dangerouslySetInnerHTML={{
                __html: `
                  var firebaseConfig = {
                    apiKey: "AIzaSyDXDGGUots5JHk39kfGGV5ueRd09Ot3f50",
                    authDomain: "yceffort.firebaseapp.com",
                    databaseURL: "https://yceffort.firebaseio.com",
                    projectId: "yceffort",
                    storageBucket: "yceffort.appspot.com",
                    messagingSenderId: "754165146494",
                    appId: "1:754165146494:web:41d36183a76fb998f4892f",
                    measurementId: "G-PEKGCL9BKE"
                  };
                  if ((firebase.apps || []).length === 0) {
                    firebase.initializeApp(firebaseConfig);
                  }
                  firebase.analytics();
                `,
              }}
            />
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
          </>
        )}
      </ThemeProvider>
    )
  }
}

export default MyApp
