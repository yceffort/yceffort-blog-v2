import React from 'react'
import { AppProps } from 'next/app'
import 'normalize.css'
import '../src/assets/css/global.css'

function App({ Component, pageProps }: AppProps) {
  return <Component {...pageProps} />
}

export default App
