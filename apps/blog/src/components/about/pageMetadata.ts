import type {Metadata} from 'next'

import {SiteConfig} from '@/config'
import {buildOgImageUrl} from '@/utils/og'
import {pathPrefixOf, type Locale} from '@/utils/postPaths'

const pages = {
  about: {
    title: 'About',
    description: {
      ko: '프론트엔드 엔지니어 yceffort. 웹의 동작 원리와 성능, 개발 도구, AI 시대의 판단과 학습을 탐구하며 글과 책, 오픈소스로 경험을 나눕니다.',
      en: 'yceffort, a frontend engineer who explores how the web works, performance, developer tools, and judgment and learning in the age of AI, and shares it through writing, books, and open source.',
    },
  },
  resume: {
    title: 'Resume',
    description: {
      ko: '프론트엔드 엔지니어 yceffort의 경력과 기여. 서비스 개발과 운영, 팀의 공통 개발 기반 구축, 저술, 번역과 오픈소스 활동을 소개합니다.',
      en: 'Career and contributions of frontend engineer yceffort: building and running services, shared foundations for teams, books, translation, and open source.',
    },
  },
}

// about과 resume은 한국어와 영어가 같은 구성이라 메타데이터를 한 곳에서 만든다
export function pageMetadata(
  page: keyof typeof pages,
  locale: Locale,
): Metadata {
  const {title, description} = pages[page]
  const path = `${pathPrefixOf(locale)}/${page}`
  const fullTitle = `${title} - ${SiteConfig.title}`
  return {
    title: fullTitle,
    description: description[locale],
    openGraph: {
      title: fullTitle,
      description: description[locale],
      url: `${SiteConfig.url}${path}`,
      images: [
        {
          url: buildOgImageUrl({
            title: fullTitle,
            description: description[locale],
            path,
            type: 'page',
          }),
          width: 1200,
          height: 630,
        },
      ],
    },
    alternates: {
      canonical: `${SiteConfig.url}${path}`,
      languages: {
        ko: `${SiteConfig.url}/${page}`,
        en: `${SiteConfig.url}/en/${page}`,
      },
    },
  }
}
