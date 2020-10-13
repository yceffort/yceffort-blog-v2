import { join } from 'path'

import { statSync, readdirSync, readFile } from 'promise-fs'
import frontMatter from 'front-matter'

import { FrontMatter } from '../types/types'

const POST_PATH = `${process.cwd()}/content/posts/articles`

export async function getAllFrontMatters(): Promise<Array<FrontMatter>> {
  const files = getFilesRecursively(POST_PATH).reverse()

  return (
    await Promise.all(
      files.map(async (f) => {
        const file = await readFile(f, { encoding: 'utf8' })
        const { attributes: fm } = frontMatter(file)
        return { ...(fm as any), date: (fm as any).date.getTime(), path: f }
      }),
    )
  ).sort((a: FrontMatter, b: FrontMatter) => b.date - a.date)
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
