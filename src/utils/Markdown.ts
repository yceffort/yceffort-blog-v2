import { serialize } from 'next-mdx-remote/serialize'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-ignore
import prism from '@mapbox/rehype-prism'
import toc from 'remark-toc'
// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-ignore
import slug from 'remark-slug'
// import { MdxRemote } from 'next-mdx-remote/types'
import visit from 'unist-util-visit'
import { Node } from 'unist'
import sizeOf from 'image-size'

import imageInfo from '../../public/imageInfo.json'
import MDXComponents from '../components/MDXComponents'

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

const postPrefix = 'posts/'

function getSizeOfImage(name: string) {
  const path = `public${name}`
  try {
    if (process.env.NODE_ENV === 'production') {
      return (imageInfo as any)[path]
    } else {
      return sizeOf(path)
    }
  } catch (e) {
    console.error(`Error while get Size of image. path: ${name} error: ${e}`)
    return {
      height: undefined,
      width: undefined,
    }
  }
}

export async function parseMarkdownToMDX(body: string, path: string) {
  return serialize(body, {
    // scope: MDXComponents,
    mdxOptions: {
      remarkPlugins: [
        toc,
        slug,
        remarkMath,
        () => {
          return (tree: Node) => {
            visit(
              tree,
              (node: any) =>
                node.type === 'paragraph' &&
                node.children.some((n: any) => n.type === 'image'),
              (node: any) => {
                const imageNode = node.children.find(
                  (n: any) => n.type === 'image',
                )

                if (!imageNode.url.startsWith('http')) {
                  const startIndex = path.indexOf(postPrefix)
                  const endIndex = path.lastIndexOf('/')
                  const tempImgPath = path.slice(
                    startIndex + postPrefix.length,
                    endIndex,
                  )

                  // /year/month/day 로 되어 있던 이미지 지원.
                  const tempImgSplit = tempImgPath.split('/')
                  const imgPath =
                    tempImgSplit.length > 2
                      ? [tempImgSplit[0], tempImgSplit[1]].join('/')
                      : tempImgPath

                  const imageIndex = imageNode.url.indexOf('/') + 1

                  const imageURL = `/${imgPath}/${imageNode.url.slice(
                    imageIndex,
                  )}`

                  const imageSize = getSizeOfImage(imageURL)

                  imageNode.type = 'jsx'
                  imageNode.value = `<Image
                  alt={\`${imageNode.alt}\`}
                  src={\`${imageURL}\`}
                  width={${imageSize.width}}
                  height={${imageSize.height}}
                />`

                  node.type = 'div'
                  node.children = [imageNode]
                }
              },
            )
          }
        },
      ],
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
