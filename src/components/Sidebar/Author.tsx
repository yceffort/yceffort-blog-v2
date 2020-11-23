import Image from 'next/image'
import React from 'react'
import styled from 'styled-components'
import Router from 'next/router'

import config from '#src/config'

const AuthorTitle = styled.h1`
  font-size: 1.125em;
  font-weight: 600;
  line-height: 1.82813em;
  margin: 0.8125rem 0;
  cursor: pointer;
`

const AuthorSubTitle = styled.p`
  color: #888;
  line-height: 1.625rem;
  margin-bottom: 1.625rem;
`

export default function Author() {
  const {
    author: { name, photo, bio },
  } = config
  return (
    <>
      <Image alt={name} src={photo} width={75} height={75} />
      <AuthorTitle onClick={() => Router.push('/')}>{name}</AuthorTitle>
      <AuthorSubTitle>{bio}</AuthorSubTitle>
    </>
  )
}
