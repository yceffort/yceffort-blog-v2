import fs from 'fs'

import glob from 'glob'
import sizeOf from 'image-size'

interface ImagesInterface {
  [key: string]: {
    width?: number
    height?: number
    type?: string
  }
}

async function getSizeOfImages() {
  const result = ['png', '.jpg', '.jpeg'].reduce<ImagesInterface>(
    (prev: ImagesInterface, extension: string) => {
      const files = glob.sync(`public/**/*.${extension}`)

      files.forEach((path) => {
        const size = sizeOf(path)
        prev[path] = size
      })
      return prev
    },
    {},
  )

  await fs.promises.writeFile('public/imageInfo.json', JSON.stringify(result), {
    encoding: 'utf-8',
  })
}

getSizeOfImages()
