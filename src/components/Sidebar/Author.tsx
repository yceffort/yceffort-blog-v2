import Link from 'next/link'
import Image from 'next/image'
import React from 'react'
import styled from 'styled-components'

import config from '../../config'

const AuthorPhoto = styled(Image)`
  display: inline-block;
  margin-bottom: 0;
  border-radius: 50%;
  background-clip: padding-box;
  width: 75px;
  height: 75px;
  cursor: pointer;
`

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
      <Link href="/">
        <AuthorPhoto alt={name} src={photo} width={75} height={75} />
      </Link>
      <Link href="/">
        <AuthorTitle>{name}</AuthorTitle>
      </Link>
      <AuthorSubTitle>{bio}</AuthorSubTitle>
    </>
  )
}
