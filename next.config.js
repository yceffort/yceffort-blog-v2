/* eslint-disable @typescript-eslint/naming-convention */
const withPWA = require('next-pwa')
const isProduction = process.env.NODE_ENV !== 'production'
isProduction && require('dotenv').config()

module.exports = withPWA({
  env: {
    HOST_URL: isProduction
      ? `https://${process.env.VERCEL_URL}`
      : process.env.VERCEL_URL,
    PROJECT_ID: process.env.PROJECT_ID,
    PRIVATE_KEY: process.env.PRIVATE_KEY,
    CLIENT_EMAIL: process.env.CLIENT_EMAIL,
  },
  pwa: {
    dest: 'public',
    disable: process.env.NODE_ENV !== 'production',
  },
  async redirects() {
    return [
      {
        source: '/tag/:tag',
        destination: '/tag/:tag/page/1',
        permanent: true,
      },
      {
        source: '/category/:tag',
        destination: '/tag/:tag/page/1',
        permanent: true,
      },
      {
        source: '/category/:tag/page/:no',
        destination: '/tag/:tag/page/:no',
        permanent: true,
      },
      {
        source: '/categories',
        destination: '/tags',
        permanent: true,
      },
    ]
  },
})
