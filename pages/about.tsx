import React from 'react'

import Layout from '#components/Layout'
import Page from '#components/Page'
import Sidebar from '#components/Sidebar/Sidebar'
import config from '#src/config'

export default function About() {
  return (
    <Layout
      title="about"
      description={config.author.name}
      url="https://yceffort.kr/about"
    >
      <Sidebar />
      <Page title="About">
        <div>
          <p>Frontend Engineer in Korea.</p>
          <p>Study Everyday, Dream Always.</p>
          <p>
            <a
              href="https://www.notion.so/9fc4262c01744a63a849cdccdde5c85f"
              target="_blank"
              rel="nofollow noopener noreferrer"
            >
              Resume
            </a>
          </p>
        </div>
      </Page>
    </Layout>
  )
}
