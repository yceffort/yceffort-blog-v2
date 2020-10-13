import React from 'react'
import styled from 'styled-components'

const IconSVG = styled.svg`
  display: inline-block;
  width: 1em;
  height: 1em;
  stroke-width: 0;
  stroke: currentColor;
  fill: currentColor;
  font-style: normal;
  font-weight: normal;
  speak: none;
  margin-right: 0.2em;
  text-align: center;
  font-variant: normal;
  text-transform: none;
  line-height: 1em;
  margin-left: 0.2em;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
`

export default function Icon({
  name,
  icon,
}: {
  name: string
  icon: {
    viewBox?: string
    path?: string
  }
}) {
  return (
    <IconSVG viewBox={icon.viewBox}>
      <title>{name}</title>
      <path d={icon.path} />
    </IconSVG>
  )
}
