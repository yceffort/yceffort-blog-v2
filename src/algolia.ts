import algolia from 'algoliasearch'

import { getAllPosts } from './utils/Markdown'

const ALGOLIA_API_KEY = process.env.ALGOLIA_API_KEY || ''
;(async () => {
  try {
    const client = algolia('4KH5R593RQ', ALGOLIA_API_KEY)

    const index = client.initIndex('blog')

    const blogData = await getAllPosts()

    const algoliaObject = blogData.map(
      ({ frontmatter: { title, tags, description }, body }) => ({
        title,
        tags,
        description,
        body,
      }),
    )

    await index.saveObjects(algoliaObject, {
      autoGenerateObjectIDIfNotExist: true,
    })
  } catch (e) {
    throw new Error(e)
  }
})()
