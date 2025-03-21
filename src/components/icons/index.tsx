import type {FC} from 'react'

import FacebookIcon from '#components/icons/facebook'
import GithubIcon from '#components/icons/github'
import LinkedinIcon from '#components/icons/linkedin'
import MailIcon from '#components/icons/mail'
import TwitterIcon from '#components/icons/twitter'
import YoutubeIcon from '#components/icons/youtube'

type IconType = 'mail' | 'github' | 'facebook' | 'youtube' | 'linkedin' | 'twitter'

const Components: Record<IconType, FC<{className: string}>> = {
    mail: MailIcon,
    github: GithubIcon,
    facebook: FacebookIcon,
    youtube: YoutubeIcon,
    linkedin: LinkedinIcon,
    twitter: TwitterIcon,
}

const SocialIcon = ({kind, href, size = 8}: {kind: IconType; href: string; size?: number}) => {
    if (!href) {
        return null
    }

    const SocialSvg = Components[kind]

    return (
        <a
            className="text-sm text-gray-500 transition hover:text-gray-600"
            target="_blank"
            rel="noopener noreferrer"
            href={href}
        >
            <span className="sr-only">{kind}</span>
            <SocialSvg
                className={`fill-current text-gray-700 hover:text-blue-500 dark:text-gray-200 dark:hover:text-blue-400 h-${size} w-${size}`}
            />
        </a>
    )
}

export default SocialIcon
