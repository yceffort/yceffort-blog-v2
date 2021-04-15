import { join } from 'path'
import fs from 'fs'

import qs from 'query-string'
import memoize from 'memoizee'
import frontMatter from 'front-matter'

import { FrontMatter, Post, TagWithCount } from '../common/types'

const DIR_REPLACE_STRING = '/posts'

const POST_PATH = `${process.cwd()}${DIR_REPLACE_STRING}`
const {
  promises: { readFile },
  readdirSync,
  statSync,
} = fs

export const getAllPosts: () => Promise<Array<Post>> = memoize(retreiveAllPosts)

export async function getAllDraftPosts() {
  const files = getFilesRecursively(POST_PATH).reverse()
  const draftPosts: Array<string> = []

  for await (const f of files) {
    const file = await readFile(f, { encoding: 'utf8' })
    const { attributes } = frontMatter(file)
    const fm: FrontMatter = attributes as any
    const { published } = fm

    const slug = f
      .slice(f.indexOf(DIR_REPLACE_STRING) + DIR_REPLACE_STRING.length + 1)
      .replace('.md', '')

    if (!published) {
      draftPosts.push(slug)
    }
  }

  console.table(draftPosts)
  return draftPosts
}

export async function getAllTagsFromPosts(): Promise<Array<TagWithCount>> {
  const tags: string[] = (await getAllPosts()).reduce<string[]>(
    (prev, curr) => {
      curr.frontMatter.tags.forEach((tag: string) => {
        prev.push(tag)
      })
      return prev
    },
    [],
  )

  const tagWithCount = [...new Set(tags)].map((tag) => ({
    tag,
    count: tags.filter((t) => t === tag).length,
  }))

  return tagWithCount.sort((a, b) => b.count - a.count)
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

export async function retreiveAllPosts(): Promise<Array<Post>> {
  const files = getFilesRecursively(POST_PATH).reverse()
  const posts: Array<Post> = []

  for await (const f of files) {
    const file = await readFile(f, { encoding: 'utf8' })
    const { attributes, body } = frontMatter(file)
    const fm: FrontMatter = attributes as any
    const { tags: fmTags, published, date } = fm

    const slug = f
      .slice(f.indexOf(DIR_REPLACE_STRING) + DIR_REPLACE_STRING.length + 1)
      .replace('.md', '')

    if (published) {
      const tags: string[] = (fmTags || []).map((tag: string) => tag.trim())

      const result: Post = {
        frontMatter: {
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

  return posts.sort((a, b) => b.frontMatter.date - a.frontMatter.date)
}
