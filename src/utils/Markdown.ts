import { serialize } from 'next-mdx-remote/serialize'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import toc from 'remark-toc'
import slug from 'remark-slug'
import { visit } from 'unist-util-visit'
import { Node } from 'unist'
import remarkGfm from 'remark-gfm'
import rehypeAutolinkHeadings from 'rehype-autolink-headings'
import prism from 'rehype-prism-plus'

import imageMetadata from '#utils/imageMetadata'

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

function parseCodeSnippet() {
  return (tree: Node) => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    visit(tree, 'element', (node: any) => {
      const [token, type]: [string, TokenType] = node.properties.className || []
      if (token === 'token') {
        node.properties.className = [tokenClassNames[type]]
      }
    })
  }
}

export async function parseMarkdownToMdx(body: string, path: string) {
  return serialize(body, {
    mdxOptions: {
      remarkPlugins: [remarkMath, toc, slug, remarkGfm],
      rehypePlugins: [
        rehypeKatex,
        prism,
        parseCodeSnippet,
        rehypeAutolinkHeadings,
        imageMetadata(path),
      ],
    },
  })
}
