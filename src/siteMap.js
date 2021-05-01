const fs = require('fs')

const frontMatter = require('front-matter')
const glob = require('glob')

async function createSiteMap() {
  const rawPosts = glob.sync('./posts/**/*.md')
  const slugs = rawPosts.reduce((prev, path) => {
    const file = fs.readFileSync(path, { encoding: 'utf-8' })
    const fm = frontMatter(file)

    if (fm.attributes.published) {
      const slug = path
        .slice(path.indexOf('/posts') + '/posts'.length + 1)
        .replace('.md', '')
      return [...prev, slug]
    } else {
      return prev
    }
  }, [])

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
    ${slugs
      .map((slug) => {
        return `<url><loc>${`https://yceffort.kr/${slug}`}</loc><changefreq>daily</changefreq><priority>0.7</priority></url>`
      })
      .join('\n')}
</urlset>
    `

  await fs.promises.writeFile('public/sitemap.xml', sitemap, {
    encoding: 'utf-8',
  })
}

createSiteMap()
