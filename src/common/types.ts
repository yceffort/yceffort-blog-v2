export interface FrontMatter {
  title: string
  category: string
  tags: string[]
  published: boolean
  date: string
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
  frontMatter: FrontMatter
  body: string
  path: string
}

export interface TagWithCount {
  tag: string
  count: number
}
