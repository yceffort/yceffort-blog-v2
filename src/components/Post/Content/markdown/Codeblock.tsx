import React from 'react'
import SyntaxHighlighter from 'react-syntax-highlighter'
import { arduinoLight as codeStyle } from 'react-syntax-highlighter/dist/cjs/styles/hljs'

export default function CodeBlock({
  language,
  value,
}: {
  language: string
  value: string
}) {
  // TODO: TOC 처리
  return language !== 'toc' ? (
    <SyntaxHighlighter language={'js'} style={codeStyle}>
      {value}
    </SyntaxHighlighter>
  ) : null
}
