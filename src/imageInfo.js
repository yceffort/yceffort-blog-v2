const fs = require('fs')

const glob = require('glob')
const sizeOf = require('image-size')

async function getSizeOfImages() {
  const result = ['png', 'jpg', 'jpeg', 'gif', 'PNG', 'webp', 'svg'].reduce(
    (prev, extension) => {
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
