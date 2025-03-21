module.exports = {
    ci: {
        collect: {
            // @see: https://github.com/GoogleChrome/lighthouse-ci/issues/799
            settings: {
                hostname: '127.0.0.1',
            },
            url: ['http://127.0.0.1:3000'],
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
