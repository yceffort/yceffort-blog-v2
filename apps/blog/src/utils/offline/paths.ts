const POST_PATH = /^(\/en)?\/\d{4}\/\d{2}(\/\d{2})?\/[^/]+$/

export const isPostPath = (pathname: string) => POST_PATH.test(pathname)

export const localeOfPath = (pathname: string): 'ko' | 'en' =>
  pathname === '/en' || pathname.startsWith('/en/') ? 'en' : 'ko'

export const homePathOf = (locale: 'ko' | 'en') =>
  locale === 'en' ? '/en' : '/'

export const savedPathOf = (locale: 'ko' | 'en') =>
  locale === 'en' ? '/en/saved' : '/saved'
