import admin from 'firebase-admin'
import { NextApiRequest, NextApiResponse } from 'next'
import { addDays, parse } from 'date-fns'

const CLIENT_EMAIL = process.env.CLIENT_EMAIL
const PRIVATE_KEY = process.env.PRIVATE_KEY

if ((admin.apps || []).length === 0 && PRIVATE_KEY && CLIENT_EMAIL) {
  admin.initializeApp({
    credential: admin.credential.cert({
      projectId: 'yceffort',
      clientEmail: CLIENT_EMAIL,
      privateKey: PRIVATE_KEY,
    }),
  })
}

export default async (req: NextApiRequest, res: NextApiResponse) => {
  const startDate = getStartDate(req.query.start_date as string)
  const endDate = req.query.end_date
    ? getEndDate(req.query.end_date as string)
    : addDays(startDate, 6)

  console.log(startDate, endDate)

  const db = admin.firestore()

  const doc = await db
    .collection('/apple_health/daily/data')
    .where('date', '>=', startDate)
    .where('date', '<=', endDate)
    .get()

  const result = doc.docs.map((doc) => {
    const data = doc.data()
    return {
      ...data,
      date: data['date'].toDate(),
    }
  })

  res.status(200).json(result)
}

function getStartDate(targetDate?: string) {
  const date = targetDate
    ? parse(targetDate, 'yyyy-MM-dd', new Date())
    : new Date()
  const timeZoneFromDB = +9.0
  const tzDifference = timeZoneFromDB * 60 + date.getTimezoneOffset()
  const offsetDate = new Date(date.getTime() + tzDifference * 60 * 1000)
  offsetDate.setHours(0, 0, 0, 0)
  return offsetDate
}

function getEndDate(targetDate?: string) {
  const date = targetDate
    ? parse(targetDate, 'yyyy-MM-dd', new Date())
    : new Date()
  const timeZoneFromDB = +9.0
  const tzDifference = timeZoneFromDB * 60 + date.getTimezoneOffset()
  const offsetDate = new Date(date.getTime() + tzDifference * 60 * 1000)
  offsetDate.setHours(23, 59, 59, 0)
  return offsetDate
}
