const functions = require('firebase-functions')
const admin = require('firebase-admin')
const chromium = require('chrome-aws-lambda')
const cloudinary = require('cloudinary')
const fetch = require('node-fetch')
const queryString = require('query-string')

const CLOUDINARY_CLOUD = process.env.CLOUDINARY_CLOUD || 'yceffort'
const CLOUDINARY_KEY = process.env.CLOUDINARY_KEY || '135156986266348'
const CLOUDINARY_SECRET =
  process.env.CLOUDINARY_SECRET || 'ZH50DrK1FUILXqdxTDJtijuo4VY'

// const local = process.env.NODE_ENV !== 'production'

async function main() {
  admin.initializeApp()
  // ;(async () => {
  //   await chromium.font(
  //     'https://rawcdn.githack.com/yceffort/yceffort-blog-v2/f07d74f18bfdabe1e639dc18fdf559fe008a8287/font/GamjaFlower-Regular.ttf',
  //   )
  // })()

  cloudinary.v2.config({
    cloud_name: CLOUDINARY_CLOUD, // eslint-disable-line
    api_key: CLOUDINARY_KEY, // eslint-disable-line
    api_secret: CLOUDINARY_SECRET, // eslint-disable-line
  })
}

try {
  main()
} catch (e) {
  console.error(e)
}

const takeScreenshot = async function (url) {
  const chromiumPath = await chromium.executablePath

  const browser = await chromium.puppeteer.launch({
    executablePath: chromiumPath,
    args: chromium.args,
    defaultViewport: chromium.defaultViewport,
    headless: chromium.headless,
  })

  const page = await browser.newPage()
  await page.setViewport({ height: 630, width: 1200 })
  await page.goto(url)
  const buffer = await page.screenshot({ encoding: 'base64' })
  await browser.close()
  return `data:image/png;base64,${buffer}`
}

const getImage = async function (title) {
  const url = `https://res.cloudinary.com/${CLOUDINARY_CLOUD}/image/upload/social-images/${title}.png`
  const response = await fetch(url)

  return response.ok ? url : null
}

const putImage = async function (title, buffer) {
  const cloudinaryOptions = {
    public_id: `social-images/${title}`, // eslint-disable-line
    unique_filename: false, // eslint-disable-line
  }
  const response = await cloudinary.v2.uploader.upload(
    buffer,
    cloudinaryOptions,
  )

  return response.url
}

exports.screenshot = functions.https.onRequest(async (req, res) => {
  const query = req.query
  const title = encodeURI(query.slug)
  const existingImage = await getImage(title)

  const postUrl = `http://yceffort.kr/generate-screenshot?${queryString.stringify(
    query,
  )}`

  if (existingImage) {
    res.redirect(existingImage)
  }

  try {
    const screenshot = await takeScreenshot(postUrl)
    const uploadedImage = await putImage(title, screenshot)
    res.redirect(uploadedImage)
  } catch (e) {
    console.error(e)
    res.json({ error: e.toString() })
  }
})
