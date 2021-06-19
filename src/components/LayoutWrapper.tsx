import Image from 'next/image'
import { ReactNode } from 'react'

import Link from './Link'
import SectionContainer from './SectionContainer'
import Footer from './Footer'
import MobileNav from './MobileNav'
import ThemeSwitch from './ThemeSwitch'

import SiteConfig from '#src/config'

const LayoutWrapper = ({ children }: { children: ReactNode }) => {
  return (
    <SectionContainer>
      <div className="flex flex-col justify-between h-screen">
        <header className="flex items-center justify-between py-10">
          <div>
            <Link href="/" aria-label="Tailwind CSS Blog">
              <div className="flex items-center justify-between">
                <div className="mr-3">
                  <Image
                    src={SiteConfig.author.photo}
                    alt="avatar"
                    width={40}
                    height={40}
                    className="w-10 h-10 rounded-full"
                  />
                </div>
                <div className="hidden h-6 text-2xl font-semibold sm:block">
                  {SiteConfig.title}
                </div>
              </div>
            </Link>
          </div>
          <div className="flex items-center text-base leading-5">
            <div className="hidden sm:block">
              {SiteConfig.menu.map((link) => (
                <Link
                  key={link.label}
                  href={link.path}
                  className="p-1 font-medium text-gray-900 sm:p-4 dark:text-gray-100"
                >
                  {link.label}
                </Link>
              ))}
            </div>
            <ThemeSwitch />
            <MobileNav />
          </div>
        </header>
        <main className="mb-auto">{children}</main>
        <Footer />
      </div>
    </SectionContainer>
  )
}

export default LayoutWrapper
