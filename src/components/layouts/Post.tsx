import { format } from 'date-fns'
import Image from 'next/image'
import { PropsWithChildren } from 'react'

import { FrontMatter } from '#commons/types'
import SectionContainer from '#components/SectionContainer'
import { BlogSeo } from '#components/SEO'
import Tag from '#components/Tag'
import CustomLink from '#components/Link'
import PageTitle from '#components/PageTitle'
import SiteConfig from '#src/config'
import { getThumbnailURL } from '#utils/common'
import MathLoader from '#components/layouts/Post/math'

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

  const thumbnailUrl = getThumbnailURL({
    tags: frontMatter.tags.map((tag) => tag.trim()),
    title: frontMatter.title,
    path: slug,
    slug,
  })

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
            className="pb-8 divide-y divide-gray-200 xl:divide-y-0 dark:divide-gray-700 xl:grid xl:grid-cols-4 xl:gap-x-6"
            style={{ gridTemplateRows: 'auto 1fr' }}
          >
            <dl className="pt-6 pb-10 xl:pt-11 xl:border-b xl:border-gray-200 xl:dark:border-gray-700">
              <dt className="sr-only">Author</dt>
              <dd>
                <ul className="flex justify-center space-x-8 xl:block sm:space-x-12 xl:space-x-0 xl:space-y-8">
                  <li className="flex items-center space-x-2">
                    {/* <img
                      src={siteMetdata.image}
                      alt="avatar"
                      className="w-10 h-10 rounded-full"
                    /> */}
                    <dl className="text-sm font-medium leading-5 whitespace-nowrap">
                      <dt className="sr-only">Name</dt>
                      <dd className="text-gray-900 dark:text-gray-100">
                        {SiteConfig.author.name}
                      </dd>
                    </dl>
                  </li>
                </ul>
              </dd>
            </dl>
            <div className="divide-y divide-gray-200 dark:divide-gray-700 xl:pb-0 xl:col-span-3 xl:row-span-2">
              <div className="pt-10 pb-8 prose dark:prose-dark max-w-none">
                {children}
              </div>
              <div className="pt-6 pb-6 text-sm text-gray-700 dark:text-gray-300">
                <CustomLink
                  href={`https://github.com/yceffort/yceffort-blog-v2/issues/new?labels=%F0%9F%92%AC%20Discussion&title=[Discussion]&assignees=yceffort&body=${SiteConfig.url}/${slug}`}
                >
                  {'Issue on GitHub'}
                </CustomLink>
              </div>
            </div>
            <footer>
              <div className="text-sm font-medium leading-5 divide-gray-200 xl:divide-y dark:divide-gray-700 xl:col-start-1 xl:row-start-2">
                {tags && (
                  <div className="py-4 xl:py-8">
                    <h2 className="text-xs tracking-wide text-gray-500 uppercase dark:text-gray-400">
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
