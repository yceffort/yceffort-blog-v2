import queryString from 'querystring'

import type { NextApiRequest, NextApiResponse } from 'next'
import puppeteer from 'puppeteer'
import slugify from 'slugify'
import cloudinary from 'cloudinary'
import fetch from 'isomorphic-fetch'

const CLOUDINARY_CLOUD = process.env.CLOUDINARY_CLOUD || 'yceffort'
const CLOUDINARY_KEY = process.env.CLOUDINARY_KEY || '789373479369932'
const CLOUDINARY_SECRET =
  process.env.CLOUDINARY_SECRET || 'Aff3KgwUhhEL-IXGiU7mmuUbbQ0'

cloudinary.v2.config({
  cloud_name: CLOUDINARY_CLOUD, // eslint-disable-line
  api_key: CLOUDINARY_KEY, // eslint-disable-line
  api_secret: CLOUDINARY_SECRET, // eslint-disable-line
})

const takeScreenshot = async function (url: string) {
  const browser = await puppeteer.launch()
  const page = await browser.newPage()
  await page.setViewport({ height: 630, width: 1200 })
  await page.goto(url)
  const buffer = await page.screenshot({ encoding: 'base64' })
  await browser.close()
  return `data:image/png;base64,${buffer}`
}

const getImage = async function (title: string) {
  const url = `https://res.cloudinary.com/${process.env.CLOUDINARY_CLOUD}/image/upload/social-images/${title}.png`
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

type Data = {
  imageUrl: string
}

export default async (req: NextApiRequest, res: NextApiResponse<Data>) => {
  const local = process.env.NODE_ENV === 'development'
  if (local) {
    res.status(404).json({ imageUrl: '' })
  }
  const slugTitle = slugify(req.query.title as string)
  const exisitingImage = await getImage(slugTitle)
  const postUrl = `https://yceffort.kr/generate-screenshot?${queryString.stringify(
    req.query,
  )}`

  if (exisitingImage) {
    res.status(200).json({ imageUrl: exisitingImage })
  }

  const screenshot = await takeScreenshot(postUrl)
  const uploadedImage = await putImage(slugTitle, screenshot)

  res.setHeader('location', uploadedImage)
  return res.status(308).json({ imageUrl: uploadedImage })
}
