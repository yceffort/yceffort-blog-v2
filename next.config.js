const withPWA = require('next-pwa')

module.exports = withPWA({
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
