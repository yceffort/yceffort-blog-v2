import Document, { Html, Main, NextScript } from 'next/document'

import MainHead from '#components/Head'

export default class MyDocument extends Document {
  public render() {
    return (
      <Html lang="ko">
        <MainHead />
        <body className="bg-white text-black antialiased dark:bg-gray-900 dark:text-white">
          <Main />
          <NextScript />
        </body>
      </Html>
    )
  }
}
