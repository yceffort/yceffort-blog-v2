import { serialize } from 'next-mdx-remote/serialize'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import prism from '@mapbox/rehype-prism'
import toc from 'remark-toc'
import slug from 'remark-slug'
import visit from 'unist-util-visit'
import sizeOf from 'image-size'

const tokenClassNames = {
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

function getSizeOfImage(name) {
  const path = `public${name}`
  try {
    // if (process.env.NODE_ENV === 'production') {
    //   return imageInfo[path]
    // } else {
    return sizeOf(path)
    // }
  } catch (e) {
    console.error(`Error while get Size of image. path: ${name} error: ${e}`) // eslint-disable-line no-console
    return {
      height: undefined,
      width: undefined,
    }
  }
}

export async function parseMarkdownToMdx(body, path) {
  return serialize(body, {
    mdxOptions: {
      remarkPlugins: [
        remarkMath,
        toc,
        slug,
        () => {
          return (tree) => {
            visit(
              tree,
              (node) =>
                node.type === 'paragraph' &&
                node.children.some((n) => n.type === 'image'),
              (node) => {
                const imageNode = node.children.find((n) => n.type === 'image')

                if (!imageNode.url.startsWith('http')) {
                  const startIndex = path.indexOf(postPrefix)
                  const endIndex = path.lastIndexOf('/')
                  const tempImgPath = path.slice(
                    startIndex + postPrefix.length,
                    endIndex,
                  )

                  const tempImgSplit = tempImgPath.split('/')
                  const imgPath =
                    tempImgSplit.length > 2
                      ? [tempImgSplit[0], tempImgSplit[1]].join('/')
                      : tempImgPath

                  const imageIndex = imageNode.url.indexOf('/') + 1

                  const imageUrl = `/${imgPath}/${imageNode.url.slice(
                    imageIndex,
                  )}`

                  const imageSize = getSizeOfImage(imageUrl)

                  imageNode.type = 'jsx'
                  imageNode.value = `<Image
                  alt={\`${imageNode.alt}\`}
                  src={\`${imageUrl}\`}
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
            visit(tree, 'element', (node) => {
              const [token, type] = node.properties.className || []
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
