import React from 'react'
import { format } from 'date-fns'

import { Post } from '#commons/types'
import CustomLink from '#components/Link'
import Tag from '#components/Tag'

export default function ListLayout({
  posts,
  title,
  pageNo = 1,
  hasNextPage = false,
  prevPath,
  nextPath,
}: {
  pageNo: number
  hasNextPage: boolean
  posts: Post[]
  title: string
  prevPath?: string
  nextPath?: string
}) {
  return (
    <>
      <div className="divide-y">
        <div className="pt-6 pb-8 space-y-2 md:space-y-5">
          <h1 className="text-3xl font-extrabold leading-9 tracking-tight text-gray-900 dark:text-gray-100 sm:text-4xl sm:leading-10 md:text-6xl md:leading-14">
            {title}
          </h1>
        </div>
        <ul>
          {posts.map(({ fields: { slug }, frontMatter: frontmatter }) => {
            const { date, title, description, tags } = frontmatter
            const updatedAt = format(new Date(date), 'yyyy-MM-dd')
            return (
              <li key={slug} className="py-4">
                <article className="space-y-2 xl:grid xl:grid-cols-4 xl:space-y-0 xl:items-baseline">
                  <dl>
                    <dt className="sr-only">Published on</dt>
                    <dd className="text-base font-medium leading-6 text-gray-500 dark:text-gray-400">
                      <time dateTime={updatedAt}>{updatedAt}</time>
                    </dd>
                  </dl>
                  <div className="space-y-3 xl:col-span-3">
                    <div>
                      <h3 className="text-2xl font-bold leading-8 tracking-tight">
                        <CustomLink
                          href={`/${slug}`}
                          className="text-gray-900 dark:text-gray-100"
                        >
                          {title}
                        </CustomLink>
                      </h3>
                      <div className="flex flex-wrap">
                        {tags.map((tag) => (
                          <Tag key={tag} text={tag} />
                        ))}
                      </div>
                    </div>
                    <div className="prose text-gray-500 max-w-none dark:text-gray-400">
                      {description}
                    </div>
                  </div>
                </article>
              </li>
            )
          })}
        </ul>
      </div>
      <div className="flex">
        <div className="flex md:w-1/2 text-base font-medium leading-6 justify-start">
          {pageNo !== 1 && (
            <CustomLink
              href={prevPath || `/pages/${pageNo - 1}`}
              className="text-blue-500 hover:text-blue-600 dark:hover:text-blue-400"
              aria-label="all posts"
            >
              Page {pageNo - 1} &larr;
            </CustomLink>
          )}
        </div>

        <div className="flex md:w-1/2 text-base font-medium leading-6 justify-end">
          {hasNextPage && (
            <CustomLink
              href={nextPath || `/pages/${pageNo + 1}`}
              className="text-blue-500 hover:text-blue-600 dark:hover:text-blue-400"
              aria-label="all posts"
            >
              Page {pageNo + 1} &rarr;
            </CustomLink>
          )}
        </div>
      </div>
    </>
  )
}
