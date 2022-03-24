import { format } from 'date-fns'
import Image from 'next/image'
import { PropsWithChildren } from 'react'

import { FrontMatter } from '#src/type'
import SectionContainer from '#components/SectionContainer'
import { BlogSeo } from '#components/SEO'
import Tag from '#components/Tag'
import CustomLink from '#components/Link'
import PageTitle from '#components/PageTitle'
import { SiteConfig } from '#src/config'
import { getThumbnailUrl } from '#utils/common'
import MathLoader from '#components/layouts/Post/math'
import profile from '#public/profile.png'

export default function PostLayout({
  children,
  frontMatter,
  slug,
}: PropsWithChildren<{
  slug: string
  frontMatter: FrontMatter
}>) {
  const { date, title, tags, description } = frontMatter
  const updatedAt = format(new Date(date), 'yyyy-MM-dd')

  const thumbnailUrl = getThumbnailUrl({
    tags: frontMatter.tags.map((tag) => tag.trim()),
    title: frontMatter.title,
    path: slug,
    slug,
  })

  const link = `https://github.com/yceffort/yceffort-blog-v2/issues/new?labels=%F0%9F%92%AC%20Discussion&title=[Discussion] issue on ${title}&assignees=yceffort&body=${SiteConfig.url}/${slug}`

  return (
    <SectionContainer>
      <BlogSeo
        title={title}
        summary={description}
        date={updatedAt}
        updatedAt={updatedAt}
        url=""
        tags={tags}
        images={[thumbnailUrl]}
      />
      <MathLoader />
      <article>
        <div className="xl:divide-y xl:divide-gray-200 xl:dark:divide-gray-700">
          <header className="pt-6 xl:pb-6">
            <div className="space-y-1 text-center">
              <dl className="space-y-10">
                <div>
                  <dt className="sr-only">Published on</dt>
                  <dd className="text-base font-medium leading-6 text-gray-500 dark:text-gray-400">
                    <time dateTime={updatedAt}>{updatedAt}</time>
                  </dd>
                </div>
              </dl>
              <div>
                <PageTitle>{title}</PageTitle>
              </div>
            </div>
          </header>
          <div
            className="divide-y divide-gray-200 pb-8 dark:divide-gray-700 xl:grid xl:grid-cols-4 xl:gap-x-6 xl:divide-y-0"
            style={{ gridTemplateRows: 'auto 1fr' }}
          >
            <dl className="pt-6 pb-10 xl:border-b xl:border-gray-200 xl:pt-11 xl:dark:border-gray-700">
              <dt className="sr-only">Author</dt>
              <dd>
                <ul className="flex justify-center space-x-8 sm:space-x-12 xl:block xl:space-x-0 xl:space-y-8">
                  <li className="flex items-center space-x-2">
                    <Image
                      src={profile}
                      placeholder="blur"
                      alt="avatar"
                      width={40}
                      height={40}
                      className="h-10 w-10 rounded-full"
                    />
                    <dl className="whitespace-nowrap text-sm font-medium leading-5">
                      <dt className="sr-only">Name</dt>
                      <dd className="text-gray-900 dark:text-gray-100">
                        {SiteConfig.author.name}
                      </dd>
                    </dl>
                  </li>
                </ul>
              </dd>
            </dl>
            <div className="divide-y divide-gray-200 dark:divide-gray-700 xl:col-span-3 xl:row-span-2 xl:pb-0">
              <div className="prose max-w-none pt-10 pb-8 dark:prose-dark">
                {children}
              </div>
              <div className="pt-6 pb-6 text-sm text-gray-700 dark:text-gray-300">
                <CustomLink href={link}>Issue on GitHub</CustomLink>
              </div>
            </div>
            <footer>
              <div className="divide-gray-200 text-sm font-medium leading-5 dark:divide-gray-700 xl:col-start-1 xl:row-start-2 xl:divide-y">
                {tags && (
                  <div className="py-4 xl:py-8">
                    <h2 className="text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400">
                      Tags
                    </h2>
                    <div className="flex flex-wrap">
                      {tags.map((tag) => (
                        <Tag key={tag} text={tag} />
                      ))}
                    </div>
                  </div>
                )}
              </div>
              <div className="pt-4 xl:pt-8">
                <CustomLink
                  href="/"
                  className="text-blue-500 hover:text-blue-600 dark:hover:text-blue-400"
                >
                  &larr; Back to the blog
                </CustomLink>
              </div>
            </footer>
          </div>
        </div>
      </article>
    </SectionContainer>
  )
}
