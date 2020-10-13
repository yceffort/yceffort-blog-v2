export interface FrontMatter {
  title: string
  category: string
  tags: string[]
  published: boolean
  date: number
  description: string
  template: string
  path: string
}

export interface Post {
  fields: {
    slug: string
    categorySlug?: string
    tagSlugs?: string[]
  }
  frontmatter: FrontMatter
  body: string
  path: string
}
