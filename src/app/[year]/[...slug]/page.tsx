import Image from 'next/image'
import Link from 'next/link'
import {notFound} from 'next/navigation'

import {format} from 'date-fns'
import {MDXRemote} from 'next-mdx-remote/rsc'
import rehypeAutolinkHeadings from 'rehype-autolink-headings'
import rehypeKatex from 'rehype-katex'
import prism from 'rehype-prism-plus'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import remarkSlug from 'remark-slug'
import remarkToc from 'remark-toc'

import MathLoader from '#components/layouts/Post/math'
import MDXComponents from '#components/MDXComponents'
import PageTitle from '#components/PageTitle'
import Tag from '#components/Tag'
import profile from '#public/profile.png'
import {SiteConfig} from '#src/config'
import imageMetadata from '#utils/imageMetadata'
import {parseCodeSnippet} from '#utils/Markdown'
import {findPostByYearAndSlug, getAllPosts} from '#utils/Post'

export const dynamic = 'error'

export async function generateMetadata(props: {params: Promise<{year: string; slug: string[]}>}) {
    const params = await props.params

    const {year, slug} = params

    const post = await findPostByYearAndSlug(year, slug)

    if (!post) {
        return {}
    }

    return {
        title: post.frontMatter.title,
    }
}

export async function generateStaticParams() {
    const allPosts = await getAllPosts()
    const result = allPosts.reduce<{year: string; slug: string[]}[]>((prev, {fields: {slug}}) => {
        const [year, ...slugs] = `${slug.replace('.md', '')}`.split('/')

        prev.push({year, slug: slugs})
        return prev
    }, [])

    return result
}

export default async function Page(props: {params: Promise<{year: string; slug: string[]}>}) {
    const params = await props.params

    const {year, slug} = params

    const post = await findPostByYearAndSlug(year, slug)

    if (!post) {
        return notFound()
    }

    const {
        frontMatter: {title, tags, date},
        body,
        path,
    } = post

    const updatedAt = format(new Date(date), 'yyyy-MM-dd')
    const link = `https://github.com/yceffort/yceffort-blog-v2/issues/new?labels=%F0%9F%92%AC%20Discussion&title=[Discussion] issue on ${title}&assignees=yceffort&body=${SiteConfig.url}/${slug}`

    return (
        <>
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
                        className="divide-y divide-gray-200 pb-8 dark:divide-gray-700 xl:grid xl:grid-cols-4 xl:gap-x-6 xl:divide-y-0"
                        style={{gridTemplateRows: 'auto 1fr'}}
                    >
                        <dl className="pt-6 pb-10 xl:border-b xl:border-gray-200 xl:pt-11 xl:dark:border-gray-700">
                            <dt className="sr-only">Author</dt>
                            <dd>
                                <ul className="flex justify-center space-x-8 sm:space-x-12 xl:block xl:space-x-0 xl:space-y-8">
                                    <li className="flex items-center space-x-2">
                                        <Image
                                            src={profile}
                                            placeholder="blur"
                                            alt="avatar"
                                            width={40}
                                            height={40}
                                            className="h-10 w-10 rounded-full"
                                        />
                                        <dl className="whitespace-nowrap text-sm font-medium leading-5">
                                            <dt className="sr-only">Name</dt>
                                            <dd className="text-gray-900 dark:text-gray-100">
                                                {SiteConfig.author.name}
                                            </dd>
                                        </dl>
                                    </li>
                                </ul>
                            </dd>
                        </dl>
                        <div className="divide-y divide-gray-200 dark:divide-gray-700 xl:col-span-3 xl:row-span-2 xl:pb-0">
                            <div className="prose max-w-none pt-10 pb-8 dark:prose-dark">
                                {/* eslint-disable-next-line @typescript-eslint/ban-ts-comment */}
                                {/* @ts-ignore */}
                                <MDXRemote
                                    source={body}
                                    components={MDXComponents}
                                    options={{
                                        mdxOptions: {
                                            remarkPlugins: [remarkMath, remarkToc, remarkSlug, remarkGfm],
                                            rehypePlugins: [
                                                rehypeKatex,
                                                prism,
                                                parseCodeSnippet,
                                                rehypeAutolinkHeadings,
                                                imageMetadata(path),
                                            ],
                                        },
                                    }}
                                />
                            </div>
                            <div className="pt-6 pb-6 text-sm text-gray-700 dark:text-gray-300">
                                <Link href={link}>Issue on GitHub</Link>
                            </div>
                        </div>
                        <footer>
                            <div className="divide-gray-200 text-sm font-medium leading-5 dark:divide-gray-700 xl:col-start-1 xl:row-start-2 xl:divide-y">
                                {tags && (
                                    <div className="py-4 xl:py-8">
                                        <h2 className="text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400">
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
                                <Link href="/" className="text-blue-500 hover:text-blue-600 dark:hover:text-blue-400">
                                    &larr; Back to the blog
                                </Link>
                            </div>
                        </footer>
                    </div>
                </div>
            </article>
        </>
    )
}
