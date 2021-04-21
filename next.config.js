/* eslint-disable @typescript-eslint/naming-convention */
const withPWA = require('next-pwa')
const isProduction = process.env.NODE_ENV === 'production'

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
        source: '/contacts',
        destination: '/about',
        permanent: true,
      },
      {
        source: '/tag/:tag',
        destination: '/tag/:tag/pages/1',
        permanent: true,
      },
      {
        source: '/tag/:tag/page/:no',
        destination: '/tags/:tag/pages/:no',
        permanent: true,
      },
      {
        source: '/tags/:tag',
        destination: '/tags/:tag/pages/1',
        permanent: true,
      },
      {
        source: '/category/:tag',
        destination: '/tags/:tag/pages/1',
        permanent: true,
      },
      {
        source: '/category/:tag/page/:no',
        destination: '/tags/:tag/page/:no',
        permanent: true,
      },
      {
        source: '/categories',
        destination: '/tags',
        permanent: true,
      },
      {
        source: '/2018/:month/:title/:day',
        destination: '/blogs/2018/:month/:title/:day',
        permanent: true,
      },
      {
        source: '/2019/:month/:title/:day',
        destination: '/blogs/2019/:month/:title/:day',
        permanent: true,
      },
      {
        source: '/2020/:month/:title',
        destination: '/blogs/2020/:month/:title',
        permanent: true,
      },
      {
        source: '/2021/:month/:title',
        destination: '/blogs/2021/:month/:title',
        permanent: true,
      },
    ]
  },
  future: {
    webpack5: true,
  },
  webpack: (config) => {
    config.module.rules.push({
      test: /\.(png|jpe?g|gif|mp4)$/i,
      use: [
        {
          loader: 'file-loader',
          options: {
            publicPath: '/_next',
            name: 'static/media/[name].[hash].[ext]',
          },
        },
      ],
    })

    config.module.rules.push({
      test: /\.svg$/,
      use: ['@svgr/webpack'],
    })

    return config
  },
})
