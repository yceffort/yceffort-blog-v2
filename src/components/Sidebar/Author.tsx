import React from 'react'
import styled from 'styled-components'

import config from '../../config'

const AuthorPhoto = styled.img`
  display: inline-block;
  margin-bottom: 0;
  border-radius: 50%;
  background-clip: padding-box;
  width: 75px;
  height: 75px;
`

const AuthorTitle = styled.h1`
  font-size: 1.125em;
  font-weight: 600;
  line-height: 1.82813em;
  margin: 0.8125rem 0;
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
      <AuthorPhoto alt={name} src={photo} />
      <AuthorTitle>{name}</AuthorTitle>
      <AuthorSubTitle>{bio}</AuthorSubTitle>
    </>
  )
}
