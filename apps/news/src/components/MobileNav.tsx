'use client'

import {MobileNav as SharedMobileNav} from '@yceffort/shared/components'

import {mobileNavClassNames} from '@/components/MobileNav.styles'
import {SiteConfig} from '@/config'

export default function MobileNav() {
  return (
    <SharedMobileNav menu={SiteConfig.menu} classNames={mobileNavClassNames} />
  )
}
