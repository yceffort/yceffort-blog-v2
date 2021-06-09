import fs from 'fs'

import glob from 'glob'
import memoize from 'memoizee'
import frontMatter from 'front-matter'

import { FrontMatter, Post, TagWithCount } from '../common/types'

const DIR_REPLACE_STRING = '/posts'

const POST_PATH = `${process.cwd()}${DIR_REPLACE_STRING}`

async function retreiveAllPosts(): Promise<Array<Post>> {
  const files = glob.sync(`${POST_PATH}/**/*.md`).reverse()
  const posts: Array<Post> = []

  for await (const f of files) {
    const file = await fs.promises.readFile(f, { encoding: 'utf8' })
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

export const getAllPosts: () => Promise<Array<Post>> = memoize(retreiveAllPosts)

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
