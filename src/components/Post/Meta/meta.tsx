import { format, toDate } from 'date-fns'
import React from 'react'
import styled from 'styled-components'

const MetaDate = styled.p`
  font-style: italic;
`

export default function Meta({ dateTime }: { dateTime: number }) {
  return (
    <div>
      <MetaDate>Published {format(toDate(dateTime), 'd MMMM yyyy')}</MetaDate>
    </div>
  )
}
