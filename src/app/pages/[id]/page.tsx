import { notFound } from 'next/navigation'

import { getAllPosts } from '#utils/Post'
import { DEFAULT_NUMBER_OF_POSTS } from '#src/constants'
import CustomLink from '#components/Link'
import MathLoader from '#components/layouts/Post/math'
import ListLayout from '#components/layouts/ListLayout'

export async function generateMetadata({
  params: { id },
}: {
  params: { id: string }
}) {
  return {
    title: `Page ${id}`,
  }
}

export async function generateStaticParams() {
  const posts = await getAllPosts()

  return [
    ...new Array(Math.round(posts.length / DEFAULT_NUMBER_OF_POSTS)).keys(),
  ].map((i) => ({ id: `${i + 1}` }))
}

export default async function Page({ params }: { params: { id: string } }) {
  const allPosts = await getAllPosts()
  const pageNo = parseInt(params.id)

  if (
    isNaN(pageNo) ||
    pageNo > Math.ceil(allPosts.length / DEFAULT_NUMBER_OF_POSTS) ||
    pageNo < 1
  ) {
    return notFound()
  }

  const startIndex = (pageNo - 1) * DEFAULT_NUMBER_OF_POSTS
  const endIndex = startIndex + DEFAULT_NUMBER_OF_POSTS

  const posts = allPosts.slice(startIndex, endIndex)

  const hasNextPage =
    Math.floor(allPosts.length / DEFAULT_NUMBER_OF_POSTS) > pageNo

  const title = `Page ${pageNo}`

  return (
    <>
      <MathLoader />
      <ListLayout posts={posts} title={title} />
      <div className="flex">
        <div className="flex w-1/2 justify-start text-base font-medium leading-6">
          {pageNo !== 1 && (
            <CustomLink
              href={`/pages/${pageNo - 1}`}
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
              href={`/pages/${pageNo + 1}`}
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
