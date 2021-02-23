import React from 'react'
import App, { AppContext } from 'next/app'
import 'normalize.css'
import '../src/assets/css/global.css'
import Head from 'next/head'
import Router from 'next/router'

class MyApp extends App {
  static async getInitialProps({ Component, ctx }: AppContext) {
    const pageProps = Component.getInitialProps
      ? await Component.getInitialProps(ctx)
      : {}

    return {
      pageProps,
    }
  }

  componentDidMount() {
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

  render() {
    const { Component, pageProps } = this.props
    return (
      <>
        <Head>
          <meta
            name="viewport"
            content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=0, minimum-scale=1, viewport-fit=cover"
          />
        </Head>
        <Component {...pageProps} />
      </>
    )
  }
}

export default MyApp
