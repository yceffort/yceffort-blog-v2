module.exports = {
    reactStrictMode: true,
    async redirects() {
        return [
            {
                source: '/tag/:tag',
                destination: '/tags/:tag/pages/1',
                permanent: true,
            },
            {
                source: '/tag/:tag/page/:no',
                destination: '/tags/:tag/pages/:no',
                permanent: true,
            },
            {
                source: '/tags/:tag/pages/((?!\\d).*)',
                destination: '/tags/:tag/pages/1',
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
        ]
    },
}
