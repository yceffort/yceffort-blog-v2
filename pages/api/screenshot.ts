import queryString from 'querystring'

import type { NextApiRequest, NextApiResponse } from 'next'
import chromium from 'chrome-aws-lambda'
import slugify from 'slugify'
import cloudinary from 'cloudinary'
import fetch from 'isomorphic-fetch'

const CLOUDINARY_CLOUD = process.env.CLOUDINARY_CLOUD || 'yceffort'
const CLOUDINARY_KEY = process.env.CLOUDINARY_KEY || ''
const CLOUDINARY_SECRET = process.env.CLOUDINARY_SECRET || ''

cloudinary.v2.config({
  cloud_name: CLOUDINARY_CLOUD, // eslint-disable-line
  api_key: CLOUDINARY_KEY, // eslint-disable-line
  api_secret: CLOUDINARY_SECRET, // eslint-disable-line
})

const local = process.env.NODE_ENV === 'development'

const takeScreenshot = async function (url: string) {
  const browser = await chromium.puppeteer.launch({
    executablePath: local
      ? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
      : await chromium.executablePath,
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

const getImage = async function (title: string) {
  const url = `https://res.cloudinary.com/${CLOUDINARY_CLOUD}/image/upload/social-images/${title}.png`
  console.log(url)
  const response = await fetch(url)

  return response.ok ? url : null
}

const putImage = async function (title: string, buffer: any) {
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

export default async (req: NextApiRequest, res: NextApiResponse) => {
  const realQuery = JSON.parse(JSON.stringify(req.query).replace(/amp;/gi, ''))
  const slugTitle = slugify(realQuery.title)
  const exisitingImage = await getImage(slugTitle)
  const postUrl = `https://yceffort.kr/generate-screenshot?${queryString.stringify(
    realQuery,
  )}`

  if (exisitingImage) {
    res.setHeader('location', exisitingImage)
    return res.status(308).redirect(exisitingImage)
  }

  const screenshot = await takeScreenshot(postUrl)
  const uploadedImage = await putImage(slugTitle, screenshot)

  res.setHeader('location', uploadedImage)
  return res.status(308).redirect(uploadedImage)
}
