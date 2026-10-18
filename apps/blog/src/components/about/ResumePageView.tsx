import * as stylex from '@stylexjs/stylex'

import {AboutTabs} from '@/components/about/AboutTabs'
import {Resume} from '@/components/about/Resume'
import * as resumeStyles from '@/components/about/Resume.styles'
import * as ambientStyles from '@/components/layout/ambient.styles'
import {SiteConfig} from '@/config'
import type {Locale} from '@/utils/postPaths'

const header = {
  ko: {
    eyebrow: 'FRONTEND ENGINEER / WRITER',
    note: '만들고 운영하며 쌓아 온 경험을 기록합니다.',
  },
  en: {
    eyebrow: 'FRONTEND ENGINEER AND WRITER',
    note: 'A record of the experience I gained by building and running things.',
  },
}

const sx = stylex.create({
  div: {
    '@layer utilities': {
      marginTop: 'calc(var(--spacing) * 4)',
    },
  },
})
export function ResumePageView({locale}: {locale: Locale}) {
  return (
    <div className={`page-view ${ambientStyles.page_view}`}>
      <header className={`resume-pageHeader ${resumeStyles.resume_pageHeader}`}>
        <div>
          <p className={`resume-eyebrow ${resumeStyles.resume_eyebrow}`}>
            {header[locale].eyebrow}
          </p>
          <h1 className={resumeStyles.element_h1}>
            {SiteConfig.author.name}
            <span className={resumeStyles.resume_pageHeader_span}>.</span>
          </h1>
          <p className={`resume-headerNote ${resumeStyles.resume_headerNote}`}>
            {header[locale].note}
          </p>
        </div>
        <div className={`resume-contacts ${resumeStyles.resume_contacts}`}>
          <a
            href={`mailto:${SiteConfig.author.contacts.email}`}
            className={resumeStyles.resume_link_hover}
          >
            {SiteConfig.author.contacts.email} <span aria-hidden="true">↗</span>
          </a>
          <a
            href={SiteConfig.author.contacts.github}
            className={resumeStyles.resume_link_hover}
          >
            {'GitHub '}
            <span aria-hidden="true">↗</span>
          </a>
        </div>
      </header>

      <AboutTabs active="resume" locale={locale} />

      <div className={stylex.props(sx.div).className}>
        <Resume locale={locale} />
      </div>
    </div>
  )
}
