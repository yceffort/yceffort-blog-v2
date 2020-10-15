import { join } from 'path'

import { statSync, readdirSync, readFile } from 'promise-fs'
import frontMatter from 'front-matter'
import unified from 'unified'
import markdown from 'remark-parse'
import math from 'remark-math'
import remark2rehype from 'remark-rehype'
import katex from 'rehype-katex'
import stringify from 'rehype-stringify'
// TODO: 타입추가 필요. 현재 강제로 임포트 중
// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-ignore
import highlightCode from '@mapbox/rehype-prism'
import toc from 'remark-toc'
// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-ignore
import slug from 'remark-slug'

import { Post } from '../types/types'

const POST_PATH = `${process.cwd()}/content/posts/articles`
const DIR_REPLACE_STRING = '/posts/articles'

export async function getAllPosts(): Promise<Array<Post>> {
  const files = getFilesRecursively(POST_PATH).reverse()

  const posts = (
    await Promise.all(
      files.map(async (f) => {
        const file = await readFile(f, { encoding: 'utf8' })
        const { attributes: fm, body } = frontMatter(file)

        const tags = (((fm as any).tags as string[]) || []).map((tag) =>
          tag.trim(),
        )

        const slug = f
          .slice(f.indexOf(DIR_REPLACE_STRING) + DIR_REPLACE_STRING.length + 1)
          .replace('.md', '')

        return {
          frontmatter: {
            ...(fm as any),
            tags,
            date: new Date((fm as any).date).getTime(),
          },
          body,
          fields: {
            slug,
            categorySlug: (fm as any).category,
            tagSlugs: tags,
          },
          path: f,
        }
      }),
    )
  ).sort((a, b) => b.frontmatter.date - a.frontmatter.date)

  return posts
}

export async function getAllTagsFromPosts(): Promise<string[]> {
  const tags = (await getAllPosts()).reduce((prev, curr) => {
    curr.frontmatter.tags.forEach(
      (tag) => !prev.includes(tag) && prev.push(tag),
    )
    return prev
  }, [] as string[])

  return tags
}

export async function parseMarkdownToHTML(body: string): Promise<string> {
  const result = (
    await unified()
      .use(markdown)
      .use(toc)
      .use(slug)
      .use(math)
      .use(remark2rehype, {
        allowDangerousHtml: true,
      })
      .use(katex, { strict: false })
      .use(highlightCode)
      .use(stringify, { allowDangerousHtml: true })
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
