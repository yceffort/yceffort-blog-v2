const functions = require('firebase-functions')
const admin = require('firebase-admin')
const puppeteer = require('puppeteer')
const cloudinary = require('cloudinary')
const queryString = require('query-string')

admin.initializeApp()

const db = admin.firestore()

async function putImage(title, buffer) {
  const CLOUDINARY_CLOUD = functions.config().env.config.CLOUDINARY_CLOUD
  const CLOUDINARY_KEY = functions.config().env.config.CLOUDINARY_KEY
  const CLOUDINARY_SECRET = functions.config().env.config.CLOUDINARY_SECRET

  cloudinary.v2.config({
    cloud_name: CLOUDINARY_CLOUD,
    api_key: CLOUDINARY_KEY,
    api_secret: CLOUDINARY_SECRET,
  })

  const cloudinaryOptions = {
    public_id: `social-images/${title}`,
    unique_filename: false,
  }
  const response = await cloudinary.v2.uploader.upload(
    buffer,
    cloudinaryOptions,
  )

  console.log('upload success!', response)

  return response.url
}

async function takeScreenshot(url) {
  const browser = await puppeteer.launch({ args: ['--no-sandbox'] })

  const page = await browser.newPage()

  await page.setViewport({ height: 630, width: 1200 })

  await page.goto(url)

  await page._client.send('ServiceWorker.enable')
  await page._client.send('ServiceWorker.stopAllWorkers')

  const imageBuffer = await page.screenshot({ encoding: 'base64' })

  await browser.close()

  return `data:image/png;base64,${imageBuffer}`
}

exports.screenshot = functions
  .region('asia-northeast3')
  .runWith({
    timeoutSeconds: 120,
    memory: '1GB',
  })
  .https.onRequest(async (req, res) => {
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
      res.json({ error: JSON.stringify(e) })
    }
  })
