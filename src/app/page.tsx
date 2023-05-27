import { format } from 'date-fns'
import Link from 'next/link'
import Script from 'next/script'

import { getAllPosts } from '#utils/Post'
import { DEFAULT_NUMBER_OF_POSTS } from '#src/constants'
import Tag from '#components/Tag'
import { SiteConfig } from '#src/config'

export default async function Page() {
  const allPosts = await getAllPosts()
  const recentPosts = allPosts.slice(0, DEFAULT_NUMBER_OF_POSTS)
  return (
    <>
      <div className="divide-y divide-gray-200 dark:divide-gray-700">
        <div className="space-y-2 pt-6 pb-8 md:space-y-5">
          <h1 className="text-3xl font-extrabold leading-9 tracking-tight text-gray-900 dark:text-gray-100 sm:text-4xl sm:leading-10 md:text-6xl md:leading-14">
            Latest
          </h1>
          <p className="text-lg leading-7 text-gray-500 dark:text-gray-400">
            {SiteConfig.subtitle}
          </p>
        </div>
        <ul className="divide-y divide-gray-200 dark:divide-gray-700">
          {recentPosts.map(
            ({
              frontMatter: { date, title, tags, description },
              fields: { slug },
            }) => {
              const updatedAt = format(new Date(date), 'yyyy-MM-dd')
              return (
                <li key={slug} className="py-12">
                  <article>
                    <div className="space-y-2 xl:grid xl:grid-cols-4 xl:items-baseline xl:space-y-0">
                      <dl>
                        <dt className="sr-only">Published on</dt>
                        <dd className="text-base font-medium leading-6 text-gray-500 dark:text-gray-400">
                          <time dateTime={updatedAt}>{updatedAt}</time>
                        </dd>
                      </dl>
                      <div className="space-y-5 xl:col-span-3">
                        <div className="space-y-6">
                          <div>
                            <h2 className="text-2xl font-bold leading-8 tracking-tight">
                              <Link
                                href={`/${slug}`}
                                className="text-gray-900 dark:text-gray-100"
                              >
                                {title}
                              </Link>
                            </h2>
                            <div className="flex flex-wrap">
                              {tags.map((tag) => (
                                <Tag key={tag} text={tag} />
                              ))}
                            </div>
                          </div>
                          <div className="prose max-w-none text-gray-500 dark:text-gray-400">
                            {description}
                          </div>
                        </div>
                        <div className="text-base font-medium leading-6">
                          <Link
                            href={`/${slug}`}
                            className="text-blue-500 hover:text-blue-600 dark:hover:text-blue-400"
                            aria-label={`Read "${title}"`}
                          >
                            Read more &rarr;
                          </Link>
                        </div>
                      </div>
                    </div>
                  </article>
                </li>
              )
            },
          )}
        </ul>
      </div>
      <div className="flex justify-end text-base font-medium leading-6">
        <Link
          href="/pages/1"
          className="text-blue-500 hover:text-blue-600 dark:hover:text-blue-400"
          aria-label="all posts"
        >
          All Posts &rarr;
        </Link>
      </div>
      <Script
        src={`https://www.googletagmanager.com/gtag/js?id=${SiteConfig.googleAnalyticsId}`}
        strategy="afterInteractive"
      />
      <Script id="google-analytics" strategy="afterInteractive">
        {`
          window.dataLayer = window.dataLayer || [];
          function gtag(){window.dataLayer.push(arguments);}
          gtag('js', new Date());

          gtag('config', 'GA_MEASUREMENT_ID');
        `}
      </Script>
    </>
  )
}
