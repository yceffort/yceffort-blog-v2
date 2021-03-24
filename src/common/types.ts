export interface FrontMatter {
  title: string
  category: string
  tags: string[]
  published: boolean
  date: number
  description: string
  template: string
  path: string
  socialImageUrl?: string
  socialImageCredit?: string
}

export interface Post {
  fields: {
    slug: string
  }
  frontmatter: FrontMatter
  body: string
  parsedBody?: string
  path: string
}

export interface TagWithCount {
  tag: string
  count: number
}
