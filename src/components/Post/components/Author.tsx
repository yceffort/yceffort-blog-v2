import React from 'react'
import styled from 'styled-components'

import config from '../../../config'
import getContactHref from '../../../utils/Contact'

const AuthorContainer = styled.div`
  border-top: 1px solid #e6e6e6;
  max-width: 59.0625rem;
  padding-top: 1.25rem;
  line-height: 1.625rem;
  margin-top: 1.625rem;
  margin-bottom: 3.25rem;

  @media screen and (min-width: 685px) {
    margin-left: auto;
    margin-right: auto;
  }
`

const AuthorBio = styled.p`
  line-height: 1.625rem;
  margin-bottom: 1.625rem;
`

const AuthorBioLnk = styled.a`
  display: block;
  text-decoration: underline;
  color: #5d93ff;
`
export default function Author() {
  const {
    author: {
      bio,
      name,
      contacts: { github },
    },
  } = config
  return (
    <AuthorContainer>
      <AuthorBio>
        {bio}
        <AuthorBioLnk
          href={getContactHref('github', github)}
          rel="noopener noreferrer"
          target="_blank"
        >
          <strong>{name}</strong> on github
        </AuthorBioLnk>
      </AuthorBio>
    </AuthorContainer>
  )
}
