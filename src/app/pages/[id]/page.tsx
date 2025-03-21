import {notFound} from 'next/navigation'

import ListLayout from '#components/layouts/ListLayout'
import PageNumber from '#components/layouts/PageNumber'
import {DEFAULT_NUMBER_OF_POSTS} from '#src/constants'
import {getAllPosts} from '#utils/Post'

export const dynamic = 'error'

export async function generateMetadata({params: {id}}: {params: {id: string}}) {
    return {
        title: `Page ${id}`,
    }
}

export async function generateStaticParams() {
    const posts = await getAllPosts()

    return [...new Array(Math.round(posts.length / DEFAULT_NUMBER_OF_POSTS)).keys()].map((i) => ({id: `${i + 1}`}))
}

export default async function Page({params}: {params: {id: string}}) {
    const allPosts = await getAllPosts()
    const pageNo = parseInt(params.id)

    if (isNaN(pageNo) || pageNo > Math.ceil(allPosts.length / DEFAULT_NUMBER_OF_POSTS) || pageNo < 1) {
        return notFound()
    }

    const startIndex = (pageNo - 1) * DEFAULT_NUMBER_OF_POSTS
    const endIndex = startIndex + DEFAULT_NUMBER_OF_POSTS

    const posts = allPosts.slice(startIndex, endIndex)

    const hasNextPage = Math.floor(allPosts.length / DEFAULT_NUMBER_OF_POSTS) > pageNo

    const title = `Page ${pageNo}`

    return (
        <>
            <ListLayout posts={posts} title={title} />
            <PageNumber
                pageNo={pageNo}
                next={`/pages/${pageNo + 1}`}
                prev={`/pages/${pageNo - 1}`}
                hasNextPage={hasNextPage}
            />
        </>
    )
}
