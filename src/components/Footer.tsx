import Link from 'next/link'

import SocialIcon from '#components/icons'
import { SiteConfig } from '#src/config'

export default function Footer() {
  return (
    <footer>
      <div className="mt-16 flex flex-col items-center">
        <div className="mb-3 flex space-x-4">
          <SocialIcon
            kind="mail"
            href={`mailto:${SiteConfig.author.contacts.email}`}
            size={6}
          />
          <SocialIcon
            kind="github"
            href={SiteConfig.author.contacts.github}
            size={6}
          />
          <SocialIcon
            kind="twitter"
            href={SiteConfig.author.contacts.twitter}
            size={6}
          />
        </div>
        <div className="mb-2 flex space-x-2 text-sm text-gray-500 dark:text-gray-400">
          <div>{SiteConfig.author.name}</div>
          <div>{` • `}</div>
          <div>{`© ${new Date().getFullYear()}`}</div>
          <div>{` • `}</div>
          <Link href="/">{SiteConfig.url}</Link>
        </div>
      </div>
    </footer>
  )
}
