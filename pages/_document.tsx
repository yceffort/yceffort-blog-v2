import React from 'react'
import Document, {
  Html,
  Head,
  Main,
  NextScript,
  DocumentContext,
} from 'next/document'
import { ServerStyleSheet } from 'styled-components'

import config from '../src/config'

interface Props {
  styleTags: any
}

export default class MyDocument extends Document<Props> {
  static async getInitialProps({ req, renderPage }: DocumentContext) {
    const sheet = new ServerStyleSheet()
    const page = renderPage((App: any) => (props: any) =>
      sheet.collectStyles(<App {...props} />),
    )
    const styleTags = sheet.getStyleElement()

    let userAgent
    if (req && req.headers) {
      userAgent = req.headers['user-agent']
    }

    // const initialProps = await Document.getInitialProps(context);
    return {
      ...page,
      styleTags,
      isPublic: !userAgent || !userAgent.match(/(iOS|Android)/i),
    }
  }

  render() {
    const {
      props: { styleTags },
    } = this

    return (
      <Html lang="ko">
        <Head>
          {styleTags}
          <meta httpEquiv="x-ua-compatible" content="ie=edge" />
          <meta property="og:type" content="blog" />
          <meta property="og:locale" content="ko_KR" />

          <link
            rel="apple-touch-icon"
            sizes="57x57"
            href="/favicon/apple-icon-57x57.png"
          />
          <link
            rel="apple-touch-icon"
            sizes="60x60"
            href="/favicon/apple-icon-60x60.png"
          />
          <link
            rel="apple-touch-icon"
            sizes="72x72"
            href="/favicon/apple-icon-72x72.png"
          />
          <link
            rel="apple-touch-icon"
            sizes="76x76"
            href="/favicon/apple-icon-76x76.png"
          />
          <link
            rel="apple-touch-icon"
            sizes="114x114"
            href="/favicon/apple-icon-114x114.png"
          />
          <link
            rel="apple-touch-icon"
            sizes="120x120"
            href="/favicon/apple-icon-120x120.png"
          />
          <link
            rel="apple-touch-icon"
            sizes="144x144"
            href="/favicon/apple-icon-144x144.png"
          />
          <link
            rel="apple-touch-icon"
            sizes="152x152"
            href="/favicon/apple-icon-152x152.png"
          />
          <link
            rel="apple-touch-icon"
            sizes="180x180"
            href="/favicon/apple-icon-180x180.png"
          />
          <link
            rel="icon"
            type="image/png"
            sizes="192x192"
            href="/favicon/android-icon-192x192.png"
          />
          <link
            rel="icon"
            type="image/png"
            sizes="32x32"
            href="/favicon/favicon-32x32.png"
          />
          <link
            rel="icon"
            type="image/png"
            sizes="96x96"
            href="/favicon/favicon-96x96.png"
          />
          <link
            rel="icon"
            type="image/png"
            sizes="16x16"
            href="/favicon/favicon-16x16.png"
          />
          <link rel="manifest" href="/manifest.json" />
          <meta name="msapplication-TileColor" content="#00b7ff" />
          <meta
            name="msapplication-TileImage"
            content="/favicon/ms-icon-144x144.png"
          />
          <meta name="theme-color" content="#00b7ff" />

          <script
            async
            src={`https://www.googletagmanager.com/gtag/js?id=${config.googleAnalyticsId}`}
          />
          <script
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

          <style
            type="text/css"
            dangerouslySetInnerHTML={{
              __html:
                'body { margin: 0 !important; -webkit-tap-highlight-color: rgba(0, 0, 0, 0); -webkit-touch-callout: none; }',
            }}
          />
        </Head>
        <body>
          <Main />
          <NextScript />
        </body>
      </Html>
    )
  }
}
