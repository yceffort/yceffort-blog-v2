module.exports = {
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
    ]
  },
}
