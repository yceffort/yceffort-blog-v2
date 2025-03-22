import {notFound} from 'next/navigation'

import ListLayout from '#components/layouts/ListLayout'
import PageNumber from '#components/layouts/PageNumber'
import {DEFAULT_NUMBER_OF_POSTS} from '#src/constants'
import {getAllPosts, getAllTagsFromPosts} from '#utils/Post'

export const dynamic = 'error'

export async function generateMetadata(props: {
  params: Promise<{tag: string; id: string}>
}) {
  const params = await props.params

  const {tag, id} = params

  return {
    title: `${tag}: Page ${id}`,
  }
}

export async function generateStaticParams() {
  const allTags = await getAllTagsFromPosts()
  const posts = await getAllPosts()

  const paths: {tag: string; id: string}[] = []
  allTags.forEach(({tag}) => {
    const tagsCount: number = posts.filter((post) =>
      post.frontMatter.tags.find((t) => t === tag),
    ).length

    ;[
      ...new Array(Math.round(tagsCount / DEFAULT_NUMBER_OF_POSTS)).keys(),
    ].forEach((i) => {
      paths.push({tag, id: `${i + 1}`})
    })
  })

  return paths
}

export default async function Page(props: {
  params: Promise<{tag: string; id: string}>
}) {
  const params = await props.params
  const allPosts = await getAllPosts()
  const {tag = 'javascript', id = '1'} = params
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

  const title = `${tag[0].toUpperCase() + tag.split(' ').join('-').slice(1)} ${pageNo}`

  return (
    <>
      <ListLayout posts={posts} title={title} />
      <PageNumber
        pageNo={pageNo}
        next={`/tags/${tag}/pages/${pageNo + 1}`}
        prev={`/tags/${tag}/pages/${pageNo - 1}`}
        hasNextPage={hasNextPage}
      />
    </>
  )
}
