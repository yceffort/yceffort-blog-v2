import Image from 'next/image'

import SiteConfig from '#src/config'
import { PageSeo } from '#components/SEO'
import SocialIcon from '#components/icons'
import CustomLink from '#components/Link'
import profile from '#public/profile.png'

export default function About() {
  return (
    <>
      <PageSeo
        title={`About - ${SiteConfig.author.name}`}
        description={`About me - ${SiteConfig.author.name}`}
        url={`${SiteConfig.url}/about`}
      />
      <div className="divide-y">
        <div className="pt-6 pb-8 space-y-2 md:space-y-5">
          <h1 className="text-3xl font-extrabold leading-9 tracking-tight text-gray-900 dark:text-gray-100 sm:text-4xl sm:leading-10 md:text-6xl md:leading-14">
            About
          </h1>
        </div>
        <div className="items-start space-y-2 xl:grid xl:grid-cols-3 xl:gap-x-8 xl:space-y-0">
          <div className="flex flex-col items-center pt-8 space-x-2">
            <Image
              src={profile}
              placeholder="blur"
              alt="avatar"
              width={192}
              height={192}
              className="w-48 h-48 rounded-full"
            />
            <h3 className="pt-4 pb-2 text-2xl font-bold leading-8 tracking-tight">
              {SiteConfig.author.name}
            </h3>
            <div className="text-gray-500 dark:text-gray-400">
              Frontend Engineer
            </div>
            <div className="text-gray-500 dark:text-gray-400">Seoul, Korea</div>
            <div className="flex pt-6 space-x-3">
              <SocialIcon
                kind="mail"
                href={`mailto:${SiteConfig.author.contacts.email}`}
              />
              <SocialIcon
                kind="github"
                href={SiteConfig.author.contacts.github}
              />
            </div>
          </div>
          <div className="pt-8 pb-8 prose dark:prose-dark max-w-none xl:col-span-2">
            <p>Hello, I am a front-end developer working in Korea.</p>
            <p>
              It&apos;s very embarrassing to introduce myself. Please visit my
              notion for detailed resume.
            </p>
            <p>
              Yesterday All my troubles seemed so far away Now it looks as
              though they&apos;re here to stay Oh, I believe in yesterday
              Suddenly I&apos;m not half the man I used to be There&apos;s a
              shadow hanging over me Oh, yesterday came suddenly Why she had to
              go I don&apos;t know, she wouldn&apos;t say I said something wrong
              Now I long for yesterday
            </p>
            <p>
              <CustomLink href="https://www.notion.so/yceffort/9fc4262c01744a63a849cdccdde5c85f">
                Resume in Notion
              </CustomLink>
            </p>
          </div>
        </div>
      </div>
    </>
  )
}
