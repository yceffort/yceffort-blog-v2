import { notFound } from 'next/navigation'

import { getAllPosts, getAllTagsFromPosts } from '#utils/Post'
import { DEFAULT_NUMBER_OF_POSTS } from '#src/constants'
import CustomLink from '#components/Link'
import ListLayout from '#components/layouts/ListLayout'

export async function generateMetadata({
  params: { tag, id },
}: {
  params: { tag: string; id: string }
}) {
  return {
    title: `${tag}: Page ${id}`,
  }
}

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
      <ListLayout posts={posts} title={title} />
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
