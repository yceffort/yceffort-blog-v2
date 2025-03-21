import {serialize} from 'next-mdx-remote/serialize'
import rehypeAutolinkHeadings from 'rehype-autolink-headings'
import rehypeKatex from 'rehype-katex'
import prism from 'rehype-prism-plus'
import slug from 'rehype-slug'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import toc from 'remark-toc'
import {visit} from 'unist-util-visit'

import type {Node} from 'unist'

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

const tokenClassNames: Record<TokenType, string> = {
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
} as const

export function parseCodeSnippet() {
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
            rehypePlugins: [rehypeKatex, prism, parseCodeSnippet, rehypeAutolinkHeadings, imageMetadata(path)],
            // @see https://github.com/hashicorp/next-mdx-remote/issues/307#issuecomment-1363415249
            development: false,
        },
    })
}
