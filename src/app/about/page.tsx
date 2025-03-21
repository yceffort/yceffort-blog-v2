import Image from 'next/image'
import Link from 'next/link'

import SocialIcon from '#components/icons'
import profile from '#public/profile.png'
import {SiteConfig} from '#src/config'

export default function Page() {
    return (
        <>
            <div className="flex flex-col items-center space-x-2 pt-8">
                <Image
                    src={profile}
                    placeholder="blur"
                    alt="avatar"
                    width={192}
                    height={192}
                    className="h-48 w-48 rounded-full"
                />
                <h3 className="pt-4 pb-2 text-2xl font-bold leading-8 tracking-tight">{SiteConfig.author.name}</h3>
                <div className="text-gray-500 dark:text-gray-400">Fullstack Engineer</div>
                <div className="text-gray-500 dark:text-gray-400">Seoul, Korea</div>
                <div className="flex space-x-3 pt-6">
                    <SocialIcon kind="mail" href={`mailto:${SiteConfig.author.contacts.email}`} />
                    <SocialIcon kind="github" href={SiteConfig.author.contacts.github} />
                    <SocialIcon kind="twitter" href={SiteConfig.author.contacts.twitter} />
                </div>
            </div>
            <div className="prose max-w-none pt-8 pb-8 dark:prose-dark xl:col-span-2">
                <p>Hello, I am a full stack engineer working in Korea.</p>
                <p>
                    Yesterday All my troubles seemed so far away Now it looks as though they&apos;re here to stay Oh, I
                    believe in yesterday Suddenly I&apos;m not half the man I used to be There&apos;s a shadow hanging
                    over me Oh, yesterday came suddenly Why she had to go I don&apos;t know, she wouldn&apos;t say I
                    said something wrong Now I long for yesterday
                </p>
                <p>
                    Yesterday love was such an easy game to play. Now I need a place to hide away. Oh, I believe in
                    yesterday.
                </p>
                <p>
                    <Link href="https://yceffort.notion.site/9fc4262c01744a63a849cdccdde5c85f">
                        Detailed Resume (Notion)
                    </Link>
                </p>
            </div>
        </>
    )
}
