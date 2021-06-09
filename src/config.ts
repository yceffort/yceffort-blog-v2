import getContactHref from '#constants/Contact'

export const SiteConfig = {
  url: 'https://yceffort.kr',
  pathPrefix: '/',
  title: 'yceffort',
  subtitle: 'yceffort',
  copyright: 'yceffort © All rights reserved.',
  disqusShortname: '',
  postsPerPage: 5,
  googleAnalyticsId: 'UA-139493546-1',
  useKatex: false,
  menu: [
    {
      label: 'Posts',
      path: '/pages/1',
    },
    {
      label: 'Tags',
      path: '/tags',
    },
    {
      label: 'Study',
      path: 'https://www.notion.so/yceffort/9f2de57230a241b18f7321f591064486',
    },
    {
      label: 'about',
      path: '/about',
    },
  ],
  author: {
    name: 'yceffort',
    photo: '/profile.png',
    bio: 'frontend engineer',
    contacts: {
      email: 'root@yceffort.kr',
      facebook: '',
      telegram: '',
      twitter: '',
      github: getContactHref('github', 'yceffort'),
      rss: '',
      linkedin: '',
      instagram: '',
      line: '',
      gitlab: '',
      codepen: '',
      youtube: '',
      soundcloud: '',
    },
  },
}

export default SiteConfig
