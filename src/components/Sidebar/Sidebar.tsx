import React from 'react'
import styled from 'styled-components'

import Author from './Author'
import Contacts from './Contacts'
import Copyright from './Copyright'
import Menu from './Menu'

const SidebarDiv = styled.div`
  width: 100%;

  /* 685px */
  @media screen and (min-width: 685px) {
    width: calc(41.625% - 1.09375rem);

    &:nth-child(1n) {
      float: left;
      margin-right: 1.875rem;
      clear: none;
    }

    &:last-child {
      margin-right: 0;
    }

    &:nth-child(12n) {
      margin-right: 0;
      float: right;
    }

    &:nth-child(12n + 1) {
      clear: both;
    }
  }

  /* 960px */
  @media screen and (min-width: 960px) {
    width: calc(33.3% - 1.25rem);

    &:nth-child(1n) {
      float: left;
      margin-right: 1.875rem;
      clear: none;
    }

    &:last-child {
      margin-right: 0;
    }

    &:nth-child(3n) {
      margin-right: 0;
      float: right;
    }

    &:nth-child(3n + 1) {
      clear: both;
    }
  }
`

const InnerSidebarDiv = styled.div`
  position: relative;
  padding: 1.5625rem 1.25rem 0;

  @media screen and (min-width: 685px) {
    padding: 30px 20px 0;
    &:after {
      background: #e6e6e6;
      background: linear-gradient(180deg, #e6e6e6 0, #e6e6e6 48%, #fff);
      position: absolute;
      content: '';
      width: 0.0625rem;
      height: 33.75rem;
      top: 30px;
      right: -10px;
      bottom: 0;
    }
  }

  @media screen and (min-width: 960px) {
    padding: 2.5rem;
  }
`

export default function Sidebar() {
  return (
    <SidebarDiv>
      <InnerSidebarDiv>
        <Author />
        <Menu />
        <Contacts />
        <Copyright />
      </InnerSidebarDiv>
    </SidebarDiv>
  )
}
