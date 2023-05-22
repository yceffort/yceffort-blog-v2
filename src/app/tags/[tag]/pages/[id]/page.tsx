import { notFound } from 'next/navigation'
import { format } from 'date-fns'

import { getAllPosts, getAllTagsFromPosts } from '#utils/Post'
import { DEFAULT_NUMBER_OF_POSTS } from '#src/constants'
import CustomLink from '#components/Link'
import Tag from '#components/Tag'

export async function generateStaticParams() {
  const allTags = await getAllTagsFromPosts()
  const posts = await getAllPosts()

  const paths: Array<{ tag: string; id: string }> = []
  allTags.forEach(({ tag }) => {
    const tagsCount: number = posts.filter((post) =>
      post.frontMatter.tags.find((t) => t === tag),
    ).length

    ;[
      ...new Array(Math.round(tagsCount / DEFAULT_NUMBER_OF_POSTS)).keys(),
    ].forEach((i) => {
      paths.push({ tag, id: `${i + 1}` })
    })
  })

  return paths
}

export default async function Page({
  params,
}: {
  params: { tag: string; id: string }
}) {
  const allPosts = await getAllPosts()
  const { tag = 'javascript', id = '1' } = params
  const pageNo = parseInt(id)

  const postsWithTag = allPosts.filter((post) =>
    post.frontMatter.tags.find((t) => t === tag),
  )

  if (
    isNaN(pageNo) ||
    pageNo > Math.ceil(postsWithTag.length / DEFAULT_NUMBER_OF_POSTS) ||
    pageNo < 1
  ) {
    return notFound()
  }
  const startIndex = (pageNo - 1) * DEFAULT_NUMBER_OF_POSTS
  const endIndex = startIndex + DEFAULT_NUMBER_OF_POSTS

  const posts = postsWithTag.slice(startIndex, endIndex)

  const hasNextPage =
    Math.ceil(postsWithTag.length / DEFAULT_NUMBER_OF_POSTS) > pageNo

  const title = `${
    tag[0].toUpperCase() + tag.split(' ').join('-').slice(1)
  } ${pageNo}`

  return (
    <>
      <div className="divide-y">
        <div className="space-y-2 pt-6 pb-8 md:space-y-5">
          <h1 className="text-3xl font-extrabold leading-9 tracking-tight text-gray-900 dark:text-gray-100 sm:text-4xl sm:leading-10 md:text-6xl md:leading-14">
            {title}
          </h1>
        </div>
        <ul>
          {posts.map(
            (
              {
                fields: { slug },
                frontMatter: { date, title, description, tags },
              },
              index,
            ) => {
              const updatedAt = format(new Date(date), 'yyyy-MM-dd')
              return (
                <li key={`${slug}_${index}`} className="py-4">
                  <article className="space-y-2 xl:grid xl:grid-cols-4 xl:items-baseline xl:space-y-0">
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
                      <div className="prose max-w-none text-gray-500 dark:text-gray-400">
                        {description}
                      </div>
                    </div>
                  </article>
                </li>
              )
            },
          )}
        </ul>
      </div>
      <div className="flex">
        <div className="flex w-1/2 justify-start text-base font-medium leading-6">
          {pageNo !== 1 && (
            <CustomLink
              href={`/tags/${tag}/pages/${pageNo - 1}`}
              className="text-blue-500 hover:text-blue-600 dark:hover:text-blue-400"
              aria-label="all posts"
            >
              Page {pageNo - 1} &larr;
            </CustomLink>
          )}
        </div>

        <div className="flex w-1/2 justify-end text-base font-medium leading-6">
          {hasNextPage && (
            <CustomLink
              href={`/tags/${tag}/pages/${pageNo + 1}`}
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
