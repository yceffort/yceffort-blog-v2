import { useTheme } from 'next-themes'

import Sun from '#components/icons/themes/sun'
import Moon from '#components/icons/themes/moon'
import { Theme } from '#constants/Theme'

const ThemeSwitch = () => {
  const { theme, setTheme } = useTheme()

  function handleButtonClick() {
    setTheme(theme === Theme.dark ? Theme.light : Theme.dark)
  }

  return (
    <button
      aria-label="Toggle Dark Mode"
      type="button"
      className="w-8 h-8 p-1 ml-1 mr-1 rounded sm:ml-4"
      onClick={handleButtonClick}
    >
      {theme === Theme.light ? <Sun /> : <Moon />}
    </button>
  )
}

export default ThemeSwitch
