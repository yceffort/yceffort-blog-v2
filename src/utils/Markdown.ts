import renderToString from 'next-mdx-remote/render-to-string'
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
import { MdxRemote } from 'next-mdx-remote/types'
import visit from 'unist-util-visit'

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
