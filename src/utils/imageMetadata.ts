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

// 옵션으로 path 하나를 받는다고 가정
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
        // 파일이 없거나 에러가 나면 무시
    }

    node.properties.src = imageUrl
}

/**
 * 튜플 형태(`[imageMetadata, { path }]`)로 쓸 플러그인.
 * Plugin<[ImageMetadataOptions], Root, Root>로 선언해서
 *  - 인자로 ImageMetadataOptions 객체 1개를 받고
 *  - Transformer<Root, Root>를 리턴.
 */
const imageMetadataPlugin: Plugin<[ImageMetadataOptions], Root, Root> = function imageMetadataPlugin(options) {
    // options: { path: string }

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
