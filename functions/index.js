const functions = require('firebase-functions')
const admin = require('firebase-admin')
const chromium = require('chrome-aws-lambda')
const cloudinary = require('cloudinary')
const queryString = require('query-string')

const CLOUDINARY_CLOUD = process.env.CLOUDINARY_CLOUD || 'yceffort'
const CLOUDINARY_KEY = process.env.CLOUDINARY_KEY || ''
const CLOUDINARY_SECRET = process.env.CLOUDINARY_SECRET || ''

const APPLE_HEALTH_SECRET = process.env.APPLE_HEALTH_SECRET || ''

admin.initializeApp()

const db = admin.firestore()

cloudinary.v2.config({
  cloud_name: CLOUDINARY_CLOUD,
  api_key: CLOUDINARY_KEY,
  api_secret: CLOUDINARY_SECRET,
})

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

const putImage = async function (title, buffer) {
  const cloudinaryOptions = {
    public_id: `social-images/${title}`,
    unique_filename: false,
  }
  const response = await cloudinary.v2.uploader.upload(
    buffer,
    cloudinaryOptions,
  )

  return response.url
}

exports.screenshot = functions.https.onRequest(async (req, res) => {
  const query = {}
  Object.keys(req.query).forEach(
    (key) => (query[key.replace(/amp;/, '')] = req.query[key]),
  )

  const title = encodeURI(query.slug)
  const firebaseTitle = title.replace(/\//gi, '-')
  const screenshotRef = db.collection('screenshot')

  const exist = await screenshotRef.doc(firebaseTitle).get()

  if (exist.exists) {
    return res.redirect(exist.data().url)
  }

  try {
    const postUrl = `http://yceffort.kr/generate-screenshot?${queryString.stringify(
      query,
    )}`
    const screenshot = await takeScreenshot(postUrl)
    const uploadedImage = await putImage(title, screenshot)
    screenshotRef.doc(firebaseTitle).set({
      url: uploadedImage,
    })
    res.redirect(uploadedImage)
  } catch (e) {
    console.error(e)
    res.json({ error: e.toString() })
  }
})

exports.health = functions.https.onRequest(async (req, res) => {
  // {'health': {'run': '28.88118937860876', 'timestamps': '2021-03-17T08:45:00+09:00', 'unit': 'km'}}
  const {
    health: { run, timestamps },
  } = req.body

  const { secret } = req.headers

  if (secret !== APPLE_HEALTH_SECRET) {
    return res.send('permission denied! 🤬')
  }

  const healthRef = db.collection('apple_health')

  const date = new Date(timestamps)
  const timeZoneFromDB = +9.0
  const tzDifference = timeZoneFromDB * 60 + date.getTimezoneOffset()
  const offsetDate = new Date(date.getTime() + tzDifference * 60 * 1000)

  const key = `${offsetDate.getFullYear()}${
    offsetDate.getMonth() + 1
  }${offsetDate.getDate()}`

  healthRef.doc('daily').set({
    [key]: run,
  })

  res.send('성공')
})
