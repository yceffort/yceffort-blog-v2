import React from 'react'
import styled, { css } from 'styled-components'

import PAGINATION from '../../constants/pagination'

const PaginationContainer = styled.div`
  margin-top: 3.25rem;
  display: flex;
`

const PaginationButton = styled.div<{ $direction: string }>`
  width: 50%;
  ${({ $direction }) => css`
    text-align: ${$direction};
  `}
`

const PaginationLink = styled.a<{ $disabled: boolean }>`
  color: #f7a046;
  font-size: 1.625rem;
  font-weight: 700;

  &::focus,
  &::hover {
    color: #5d93ff;
  }

  ${({ $disabled }) =>
    $disabled
      ? css`
          pointer-events: none;
          color: #bbb;
        `
      : null}
`
export default function Pagination({
  prevPagePath,
  nextPagePath,
  hasNextPage,
  hasPrevPage,
}: {
  prevPagePath: string
  nextPagePath: string
  hasNextPage: boolean
  hasPrevPage: boolean
}) {
  return (
    <PaginationContainer>
      <PaginationButton $direction={'left'}>
        <PaginationLink href={prevPagePath} $disabled={!hasPrevPage}>
          {PAGINATION.PREV_PAGE}
        </PaginationLink>
      </PaginationButton>

      <PaginationButton $direction={'right'}>
        <PaginationLink href={nextPagePath} $disabled={!hasNextPage}>
          {PAGINATION.NEXT_PAGE}
        </PaginationLink>
      </PaginationButton>
    </PaginationContainer>
  )
}
