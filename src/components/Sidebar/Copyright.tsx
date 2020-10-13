import React from 'react'
import styled from 'styled-components'

import config from '../../../config'

const CopyrightDiv = styled.div`
  color: #b6b6b6;
  font-size: 0.875rem;
`
export default function Copyright() {
  const { copyright } = config
  return <CopyrightDiv>{copyright}</CopyrightDiv>
}
