import React from 'react'
import styled from 'styled-components'

import 'prismjs/themes/prism-coy.css'

const ContentContainer = styled.div`
  max-width: 59.0625rem;
  padding: 0 0.9375rem;
  margin: 0 auto;

  @media screen and (min-width: 960px) {
    padding: 0;
  }
`

const ContentTitle = styled.h1`
  font-size: 2rem;
  max-width: 40rem;
  font-weight: 600;
  text-align: center;
  line-height: 2.68125rem;
  margin: 1.625rem auto 0;

  @media screen and (min-width: 960px) {
    font-size: 3rem;
    line-height: 3.65625rem;
    margin-top: 3.65625rem;
    margin-bottom: 2.4375rem;
  }
`

const ContentBody = styled.div`
  figure {
    margin-bottom: 1.625rem;
  }

  figure blockquote {
    font-style: italic;
    text-align: center;
    margin-top: 0;
    padding: 1.625rem 0;
  }

  figure blockquote p {
    max-width: 40rem;
    font-size: 1.6817rem;
    margin-top: 0;
    margin-bottom: 1.625rem;
    line-height: 2.4375rem;
  }

  a {
    text-decoration: underline;
  }

  * {
    max-width: 40rem;
    margin-left: auto !important;
    margin-right: auto !important;
  }

  img {
    max-width: 100%;
  }

  @media screen and (min-width: 960px) {
    font-size: 1.125rem;
    line-height: 1.82813rem;
    margin-bottom: 1.82813rem;

    p {
      font-size: 1.125rem;
      line-height: 1.82813rem;
      margin-bottom: 1.82813rem;
    }
  }
`

export default function Content({
  body,
  title,
}: {
  body: string
  title: string
}) {
  return (
    <ContentContainer>
      <ContentTitle>{title}</ContentTitle>
      <ContentBody dangerouslySetInnerHTML={{ __html: body }}></ContentBody>
    </ContentContainer>
  )
}
