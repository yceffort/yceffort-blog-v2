import { GetStaticProps } from 'next'
import { format } from 'date-fns'

import { Post } from '#commons/types'
import { PageSeo } from '#components/SEO'
import { SiteConfig } from '#src/config'
import CustomLink from '#components/Link'
import Tag from '#components/Tag'
import { DEFAULT_NUMBER_OF_POSTS } from '#commons/const'
import { getAllPosts } from '#utils/Post'

const MAX_DISPLAY = 5

export default function Home({ posts }: { posts: Array<Post> }) {
  return (
    <>
      <PageSeo
        title="Home"
        description={SiteConfig.subtitle}
        url={SiteConfig.url}
      />
      <div className="divide-y divide-gray-200 dark:divide-gray-700">
        <div className="pt-6 pb-8 space-y-2 md:space-y-5">
          <h1 className="text-3xl font-extrabold leading-9 tracking-tight text-gray-900 dark:text-gray-100 sm:text-4xl sm:leading-10 md:text-6xl md:leading-14">
            Latest
          </h1>
          <p className="text-lg leading-7 text-gray-500 dark:text-gray-400">
            {SiteConfig.subtitle}
          </p>
        </div>
        <ul className="divide-y divide-gray-200 dark:divide-gray-700">
          {posts
            .slice(0, MAX_DISPLAY)
            .map(({ frontMatter: frontmatter, fields: { slug } }) => {
              const { date, title, tags, description } = frontmatter
              const updatedAt = format(new Date(date), 'yyyy-MM-dd')
              return (
                <li key={slug} className="py-12">
                  <article>
                    <div className="space-y-2 xl:grid xl:grid-cols-4 xl:space-y-0 xl:items-baseline">
                      <dl>
                        <dt className="sr-only">Published on</dt>
                        <dd className="text-base font-medium leading-6 text-gray-500 dark:text-gray-400">
                          <time dateTime={updatedAt}>
                            {/* {updated} */}
                            {updatedAt}
                          </time>
                        </dd>
                      </dl>
                      <div className="space-y-5 xl:col-span-3">
                        <div className="space-y-6">
                          <div>
                            <h2 className="text-2xl font-bold leading-8 tracking-tight">
                              <CustomLink
                                href={`/${slug}`}
                                className="text-gray-900 dark:text-gray-100"
                              >
                                {title}
                              </CustomLink>
                            </h2>
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
                        <div className="text-base font-medium leading-6">
                          <CustomLink
                            href={`/${slug}`}
                            className="text-blue-500 hover:text-blue-600 dark:hover:text-blue-400"
                            aria-label={`Read "${title}"`}
                          >
                            Read more &rarr;
                          </CustomLink>
                        </div>
                      </div>
                    </div>
                  </article>
                </li>
              )
            })}
        </ul>
      </div>
      <div className="flex justify-end text-base font-medium leading-6">
        <CustomLink
          href="/pages/1"
          className="text-blue-500 hover:text-blue-600 dark:hover:text-blue-400"
          aria-label="all posts"
        >
          All Posts &rarr;
        </CustomLink>
      </div>
    </>
  )
}

export const getStaticProps: GetStaticProps = async () => {
  const recentPosts = (await getAllPosts()).slice(0, DEFAULT_NUMBER_OF_POSTS)

  return {
    props: { posts: recentPosts.map((post) => ({ ...post, path: '' })) },
  }
}
