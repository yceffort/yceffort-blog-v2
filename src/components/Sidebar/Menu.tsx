import { useRouter } from 'next/router'
import React from 'react'
import styled, { css } from 'styled-components'

import config from '../../config'

const MenuNav = styled.nav`
  margin-bottom: 1.625rem;
`

const MenuList = styled.ul`
  list-style: none;
  padding: 0;
  margin: 0;

  li {
    padding: 0;
    margin: 10px 0;
  }
`

const MenuItem = styled.a<{ $active?: boolean }>`
  font-size: 1rem;
  color: #222;
  font-weight: 400;
  border: 0;

  ${({ $active }) =>
    $active
      ? css`
          color: #222;
          border-bottom: 1px solid #222;
        `
      : null}

  &:hover,
  &:focus {
    color: #5d93ff;
    border-bottom: 1px solid #5d93ff;
  }
  /* TODO: 메뉴 활성화 시에 $color-base처리 */
`

export default function Menu() {
  const { menu: menus } = config
  const { pathname } = useRouter()

  console.log(pathname)
  return (
    <MenuNav>
      <MenuList>
        {menus.map((menu, index) => (
          <li key={index}>
            <MenuItem
              href={menu.path}
              $active={
                menu.path === pathname ||
                (menu.path === '/' && pathname.includes('page'))
              }
            >
              {menu.label}
            </MenuItem>
          </li>
        ))}
      </MenuList>
    </MenuNav>
  )
}
