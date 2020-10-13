import React from 'react'
import styled from 'styled-components'

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

const MenuItem = styled.a`
  font-size: 1rem;
  color: #222;
  font-weight: 400;
  border: 0;

  &:hover,
  &:focus {
    color: #5d93ff;
    border-bottom: 1px solid #5d93ff;
  }
  /* TODO: 메뉴 활성화 시에 $color-base처리 */
`

export default function Menu() {
  const { menu: menus } = config

  return (
    <MenuNav>
      <MenuList>
        {menus.map((menu, index) => (
          <li key={index}>
            <MenuItem href={menu.path}>{menu.label}</MenuItem>
          </li>
        ))}
      </MenuList>
    </MenuNav>
  )
}
