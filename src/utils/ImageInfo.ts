import fs from 'fs'

import glob from 'glob'
import sizeOf from 'image-size'
import { ISizeCalculationResult } from 'image-size/dist/types/interface'

async function getSizeOfImages() {
  const result = ['png', 'jpg', 'jpeg', 'gif', 'PNG', 'webp', 'svg'].reduce<{
    [key: string]: ISizeCalculationResult
  }>((prev, extension) => {
    const files = glob.sync(`public/**/*.${extension}`)

    files.forEach((path) => {
      const size = sizeOf(path)
      prev[path] = size
    })
    return prev
  }, {})

  await fs.promises.writeFile('public/imageInfo.json', JSON.stringify(result), {
    encoding: 'utf-8',
  })
}

getSizeOfImages()
