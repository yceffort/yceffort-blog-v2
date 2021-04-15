import { writeFile } from 'fs/promises'

import { getAllPosts } from './utils/Post'

export async function createSiteMap() {
  const posts = await getAllPosts()

  const sitemap = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" 
  xmlns:news="http://www.google.com/schemas/sitemap-news/0.9"
  xmlns:xhtml="http://www.w3.org/1999/xhtml" 
  xmlns:mobile="http://www.google.com/schemas/sitemap-mobile/1.0"
  xmlns:image="http://www.google.com/schemas/sitemap-image/1.1"
  xmlns:video="http://www.google.com/schemas/sitemap-video/1.1">
  <url>
    <loc>https://yceffort.kr/about</loc>
  </url>
  <url>
    <loc>https://yceffort.kr/contacts</loc>
  </url>
  <url>
    <loc>https://yceffort.kr/tags</loc>
  </url>
    ${posts
      .map(({ fields: { slug } }) => {
        return `<url>
                    <loc>${`https://yceffort.kr/${slug}`}</loc>
                    <changefreq>daily</changefreq>
                    <priority>0.7</priority>
                </url>`
      })
      .join('\n')}
</urlset>
    `

  await writeFile('public/sitemap.xml', sitemap, { encoding: 'utf-8' })
}

createSiteMap()
