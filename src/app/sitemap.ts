import type {MetadataRoute} from 'next'

import {getAllPosts, getAllTagsFromPosts} from '#utils/Post'

export default async function sitemap(): Promise<MetadataRoute.Sitemap> {
    const posts = await getAllPosts()
    const tags = await getAllTagsFromPosts()

    return [
        {
            url: 'https://yceffort.kr',
            lastModified: new Date(),
        },
        {
            url: 'https://yceffort.kr/about',
            lastModified: new Date(),
        },
        ...posts.map((post) => {
            return {
                url: `https://yceffort.kr/${post.fields.slug}`,
                lastModified: new Date(post.frontMatter.date),
            }
        }),
        ...tags.map((tag) => {
            return {
                url: `https://yceffort.kr/tags/${tag}`,
            }
        }),
    ]
}
