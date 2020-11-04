const withPWA = require('next-pwa')

module.exports = withPWA({
  pwa: {
    dest: 'public',
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
        source: '/categories',
        destination: '/tags',
        permanent: true,
      },
    ]
  },
})
