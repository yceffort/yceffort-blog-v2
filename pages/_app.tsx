import React from 'react'
import { AppProps } from 'next/app'
import 'normalize.css'
import '../src/assets/css/global.css'
import Head from 'next/head'

function App({ Component, pageProps }: AppProps) {
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

export default App
