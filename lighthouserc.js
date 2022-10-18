module.exports = {
  ci: {
    collect: {
      url: ['http://localhost:3000'],
      collect: {
        numberOfRuns: 5,
      },
    },
    upload: {
      startServerCommand: 'npm run start',
      target: 'temporary-public-storage',
    },
  },
}
