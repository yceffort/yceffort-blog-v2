'use client'

import {useTheme} from 'next-themes'

import Moon from '#components/icons/themes/moon'
import Sun from '#components/icons/themes/sun'
import {Theme} from '#constants/index'

const ThemeSwitch = () => {
  const {theme, setTheme} = useTheme()

  function handleButtonClick() {
    setTheme(theme === Theme.dark ? Theme.light : Theme.dark)
  }

  return (
    <button
      aria-label="Toggle Dark Mode"
      type="button"
      className="ml-1 mr-1 h-8 w-8 rounded-sm p-1 sm:ml-4"
      onClick={handleButtonClick}
    >
      {theme === Theme.light ? <Sun /> : <Moon />}
    </button>
  )
}

export default ThemeSwitch
