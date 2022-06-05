import { promisify } from 'util'

import { imageSize } from 'image-size'
import type { Processor } from 'unified'
import type { Node } from 'unist'
import { visit } from 'unist-util-visit'
import type { VFile } from 'vfile'

const sizeOf = promisify(imageSize)

interface ImageNode extends Node {
  type: 'element'
  tagName: 'img'
  properties: {
    src: string
    height?: number
    width?: number
  }
}

function isImageNode(node: Node): node is ImageNode {
  const img = node as ImageNode
  return (
    img.type === 'element' &&
    img.tagName === 'img' &&
    img.properties &&
    typeof img.properties.src === 'string'
  )
}

async function addMetadata(node: ImageNode, postPath: string): Promise<void> {
  const {
    properties: { src },
  } = node

  if (src.startsWith('http')) {
    return
  }

  const startIndex = postPath.indexOf('/posts')
  const endIndex = postPath.lastIndexOf('/')
  const tempImgPath = postPath.slice(startIndex + '/posts'.length, endIndex)

  const tempImgSplit = tempImgPath.split('/')

  const imgPath =
    tempImgSplit.length > 2
      ? [tempImgSplit[1], tempImgSplit[2]].join('/')
      : tempImgPath

  const imageIndex = src.indexOf('/') + 1

  const imageUrl = `/${imgPath}/${src.slice(imageIndex)}`

  const imageSize = await sizeOf(`public${imageUrl}`)

  if (imageSize) {
    node.properties.width = imageSize.width
    node.properties.height = imageSize.height
  }
  node.properties.src = imageUrl
}

export default function imageMetadata(path: string) {
  return function (this: Processor) {
    return async function transformer(tree: Node, _: VFile): Promise<Node> {
      const imgNodes: ImageNode[] = []
      visit(tree, 'element', (node) => {
        if (isImageNode(node)) {
          imgNodes.push(node)
        }
      })

      for await (const node of imgNodes) {
        await addMetadata(node, path)
      }

      return tree
    }
  }
}
