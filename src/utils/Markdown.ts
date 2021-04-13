import { join } from 'path'

import renderToString from 'next-mdx-remote/render-to-string'
import { statSync, readdirSync, readFile } from 'promise-fs'
import frontMatter from 'front-matter'
import remarkMath from 'remark-math'
// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-ignore
import rehypeKatex from 'rehype-katex'
// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-ignore
import prism from '@mapbox/rehype-prism'
import toc from 'remark-toc'
// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-ignore
import slug from 'remark-slug'
import memoize from 'memoizee'
import { MdxRemote } from 'next-mdx-remote/types'
import visit from 'unist-util-visit'

import { FrontMatter, Post, TagWithCount } from '#commons/types'
import MDXComponents from '#components/MDXComponents'

const DIR_REPLACE_STRING = '/posts'
const POST_PATH = `${process.cwd()}${DIR_REPLACE_STRING}`

export const getAllPosts = memoize(retreiveAllPosts)

type TokenType =
  | 'tag'
  | 'attr-name'
  | 'attr-value'
  | 'deleted'
  | 'inserted'
  | 'punctuation'
  | 'keyword'
  | 'string'
  | 'function'
  | 'boolean'
  | 'comment'

const tokenClassNames: { [key in TokenType]: string } = {
  tag: 'text-code-red',
  'attr-name': 'text-code-yellow',
  'attr-value': 'text-code-green',
  deleted: 'text-code-red',
  inserted: 'text-code-green',
  punctuation: 'text-code-white',
  keyword: 'text-code-purple',
  string: 'text-code-green',
  function: 'text-code-blue',
  boolean: 'text-code-red',
  comment: 'text-gray-400 italic',
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

// const imgToJSX = (_: any) => (tree: any) => {
//   visit(
//     tree,
//     // only visit p tags that contain an img element
//     (node: any) =>
//       node.type === 'paragraph' &&
//       node.children.some((n) => n.type === 'image'),
//     (node) => {
//       const imageNode = node.children.find((n) => n.type === 'image')

//       // only local files
//       if (fs.existsSync(`${process.cwd()}/public${imageNode.url}`)) {
//         const dimensions = sizeOf(`${process.cwd()}/public${imageNode.url}`)

//         // Convert original node to next/image
//         imageNode.type = 'jsx'
//         imageNode.value = `<Image
//           alt={\`${imageNode.alt}\`}
//           src={\`${imageNode.url}\`}
//           width={${dimensions.width}}
//           height={${dimensions.height}}
//       />`

//         // Change node type from p to div to avoid nesting error
//         node.type = 'div'
//         node.children = [imageNode]
//       }
//     },
//   )
// }

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
      const tags = (fmTags || []).map((tag) => tag.trim())

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

export async function parseMarkdownToMDX(
  body: string,
): Promise<MdxRemote.Source> {
  return renderToString(body, {
    components: MDXComponents,
    mdxOptions: {
      remarkPlugins: [toc, slug, remarkMath],
      rehypePlugins: [
        rehypeKatex,
        prism,
        () => {
          return (tree) => {
            visit(tree, 'element', (node: any) => {
              const [token, type]: [string, TokenType] =
                node.properties.className || []
              if (token === 'token') {
                node.properties.className = [tokenClassNames[type]]
              }
            })
          }
        },
      ],
      compilers: [],
    },
  })
}
