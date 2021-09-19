import fs from 'fs'

import { getAllPosts, getAllTagsFromPosts } from './Post'

async function createSiteMap() {
  const posts = await getAllPosts()
  const slugs = posts.map((p) => p.fields.slug)
  const postUrls = slugs.map(
    (slug) =>
      `<url><loc>${`https://yceffort.kr/${slug}`}</loc><changefreq>daily</changefreq><priority>0.7</priority></url>`,
  )
  const tags = await getAllTagsFromPosts()
  const tagUrls = tags.map(
    ({ tag }) =>
      `<url><loc>${`https://yceffort.kr/tags/${tag}`}</loc><changefreq>daily</changefreq><priority>0.3</priority></url>`,
  )

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
      <loc>https://yceffort.kr/tags</loc>
    </url>${postUrls.join('\n')}${tagUrls.join('\n')}</urlset>`

  await fs.promises.writeFile('public/sitemap.xml', sitemap, {
    encoding: 'utf-8',
  })
}

createSiteMap()
