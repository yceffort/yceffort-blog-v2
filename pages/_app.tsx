import 'src/tailwind.css'

import React from 'react'
import App from 'next/app'
import Head from 'next/head'
import Router from 'next/router'
import { ThemeProvider } from 'next-themes'
import { DefaultSeo } from 'next-seo'

import { SEO } from '#components/SEO'
import LayoutWrapper from '#components/LayoutWrapper'

class MyApp extends App {
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
      <ThemeProvider attribute="class">
        <Head>
          <meta content="width=device-width, initial-scale=1" name="viewport" />
        </Head>
        <DefaultSeo {...SEO} />
        <LayoutWrapper>
          <Component {...pageProps} />
        </LayoutWrapper>{' '}
      </ThemeProvider>
    )
  }
}

export default MyApp
