import {getContactHref} from '@yceffort/shared/utils'

const isDev = process.env.NODE_ENV === 'development'

export const SiteConfig = {
  url: isDev ? 'http://localhost:3003' : 'https://news.yceffort.kr',
  title: 'yceffort news',
  description: '매일 아침 9시, 주장이 있는 글만 골라 핵심을 요약합니다.',
  googleAnalyticsId: 'G-ND58S24JBX',
  menu: [
    {label: '일간', path: '/daily'},
    {label: '주간', path: '/weekly'},
    {
      label: 'Blog',
      path: isDev ? 'http://localhost:3000' : 'https://yceffort.kr',
    },
    {
      label: 'Research',
      path: isDev ? 'http://localhost:3001' : 'https://research.yceffort.kr',
    },
  ],
  author: {
    name: 'yceffort',
    contacts: {
      email: 'root@yceffort.kr',
      twitter: getContactHref('twitter', 'yceffort_dev'),
      github: getContactHref('github', 'yceffort'),
    },
  },
}
