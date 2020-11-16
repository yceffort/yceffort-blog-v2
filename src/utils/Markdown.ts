import { join } from 'path'

import { statSync, readdirSync, readFile } from 'promise-fs'
import frontMatter from 'front-matter'
import unified from 'unified'
import markdown from 'remark-parse'
import math from 'remark-math'
import remark2rehype from 'remark-rehype'
import katex from 'rehype-katex'
import html from 'rehype-stringify'
// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-ignore
import highlightCode from '@mapbox/rehype-prism'
import toc from 'remark-toc'
// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-ignore
import slug from 'remark-slug'
import gfm from 'remark-gfm'

import { FrontMatter, Post, TagWithCount } from '../types/types'

const POST_PATH = `${process.cwd()}/content/posts/articles`
const DIR_REPLACE_STRING = '/posts/articles'

export async function getAllPosts(): Promise<Array<Post>> {
  const files = getFilesRecursively(POST_PATH).reverse()
  const posts: Array<Post> = []

  for await (const f of files) {
    const file = await readFile(f, { encoding: 'utf8' })
    const { attributes, body } = frontMatter(file)
    const fm: FrontMatter = attributes as any
    const { tags: fmTags, published, date } = fm

    if (published) {
      const tags = (fmTags || []).map((tag) => tag.trim())

      const slug = f
        .slice(f.indexOf(DIR_REPLACE_STRING) + DIR_REPLACE_STRING.length + 1)
        .replace('.md', '')

      const result: Post = {
        frontmatter: {
          ...fm,
          tags,
          date: new Date(date).getTime() - 9 * 60 * 60 * 1000, // 한국시간
        },
        body,
        fields: {
          slug,
        },
        path: f,
      }

      posts.push(result)
    }
  }

  return posts.sort((a, b) => b.frontmatter.date - a.frontmatter.date)
}

export async function getAllTagsFromPosts(): Promise<Array<TagWithCount>> {
  const tags = (await getAllPosts()).reduce((prev, curr) => {
    curr.frontmatter.tags.forEach((tag) => {
      prev.push(tag)
    })
    return prev
  }, [] as string[])

  const tagWithCount = [...new Set(tags)].map((tag) => ({
    tag,
    count: tags.filter((t) => t === tag).length,
  }))

  return tagWithCount.sort((a, b) => b.count - a.count)
}

export async function parseMarkdownToHTML(body: string): Promise<string> {
  const result = (
    await unified()
      .use(markdown)
      .use(toc)
      .use(slug)
      .use(math)
      .use(gfm)
      .use(remark2rehype, {
        allowDangerousHtml: true,
      })
      .use(katex, { strict: false })
      .use(highlightCode)
      .use(html, { allowDangerousHtml: true })
      .process(body)
  ).toString()

  return result
}

function getFilesRecursively(path: string) {
  const getFiles = (path: string) =>
    readdirSync(path)
      .map((name) => join(path, name))
      .filter((path: string) => statSync(path).isFile())

  const isDirectory = (path: string) => statSync(path).isDirectory()

  const getDirectories = (path: string) =>
    readdirSync(path)
      .map((name) => join(path, name))
      .filter(isDirectory)

  const dirs = getDirectories(path)

  const files: string[] = dirs
    .map((dir) => getFilesRecursively(dir))
    .reduce((a, b) => a.concat(b), [])

  return files.concat(getFiles(path)).filter((f) => f.endsWith('.md'))
}
