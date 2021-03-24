import React from 'react'
import styled from 'styled-components'

import config from '#src/config'
import getContactHref from '#constants/Contact'
import getIcon from '#constants/Icon'
import Icon from '#components/Icon'

const ContactDiv = styled.div`
  margin-bottom: 1.625rem;
`

const ContactList = styled.ul`
  display: flex;
  flex-flow: row wrap;
  flex-grow: 0;
  flex-shrink: 0;
  list-style: none;
  padding: 0;
  margin: 0.625rem -0.1875rem;
  width: 8.75rem;

  li {
    padding: 0;
    margin: 0.25rem;
    display: flex;
    align-content: center;
    align-items: center;
    justify-content: center;
    height: 2.1875rem;
    width: 2.1875rem;
    line-height: 2.1875rem;
    border-radius: 50%;
    text-align: center;
    border: 1px solid #ebebeb;
  }
`

const ContactLink = styled.a`
  border: 0;
  display: flex;
  color: #222;
`

export default function Contacts() {
  const {
    author: { contacts },
  } = config

  return (
    <ContactDiv>
      <ContactList>
        <li>
          <ContactLink
            href={getContactHref('email', contacts.email)}
            rel="noopener noreferrer"
            target="_blank"
          >
            <Icon name={'email'} icon={getIcon('email')} />
          </ContactLink>
        </li>

        <li>
          <ContactLink
            href={getContactHref('github', contacts.github)}
            rel="noopener noreferrer"
            target="_blank"
          >
            <Icon name={'github'} icon={getIcon('github')} />
          </ContactLink>
        </li>
      </ContactList>
    </ContactDiv>
  )
}
