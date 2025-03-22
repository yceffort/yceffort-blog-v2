import {promisify} from 'util'

import {imageSize} from 'image-size'
import {visit} from 'unist-util-visit'

import type {Root, Element} from 'hast'
import type {Plugin} from 'unified'
import type {VFile} from 'vfile'

const sizeOf = promisify(imageSize)

interface ImageElement extends Element {
    tagName: 'img'
    properties: {
        src?: string
        width?: number
        height?: number
    }
}

interface ImageMetadataOptions {
    path: string
}

async function addMetadata(node: ImageElement, postPath: string) {
    const {src} = node.properties
    if (!src) {
        return
    }

    if (src.startsWith('http')) {
        return
    }

    const startIndex = postPath.indexOf('/posts')
    const endIndex = postPath.lastIndexOf('/')
    const tempImgPath = postPath.slice(startIndex + '/posts'.length, endIndex)
    const tempImgSplit = tempImgPath.split('/')
    const imgPath = tempImgSplit.length > 2 ? [tempImgSplit[1], tempImgSplit[2]].join('/') : tempImgPath

    const imageIndex = src.indexOf('/') + 1
    const imageUrl = `/${imgPath}/${src.slice(imageIndex)}`

    try {
        const size = await sizeOf(`public${imageUrl}`)
        if (size) {
            node.properties.width = size.width
            node.properties.height = size.height
        }
    } catch {
        // nothing to do
    }

    node.properties.src = imageUrl
}

const imageMetadataPlugin: Plugin<[ImageMetadataOptions], Root, Root> = function imageMetadataPlugin(options) {
    return async function transformer(tree: Root, _file: VFile): Promise<Root> {
        const imgNodes: ImageElement[] = []

        visit(tree, 'element', (node: Element) => {
            if (node.tagName === 'img' && node.properties && typeof node.properties.src === 'string') {
                imgNodes.push(node as ImageElement)
            }
        })

        for (const node of imgNodes) {
            await addMetadata(node, options.path)
        }

        return tree
    }
}

export default imageMetadataPlugin
