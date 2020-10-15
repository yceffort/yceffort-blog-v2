import React from 'react'

import Layout from '../src/components/Layout'
import Page from '../src/components/Page'
import Sidebar from '../src/components/Sidebar/Sidebar'
import config from '../src/config'
export default function Contacts() {
  return (
    <Layout title="contact" description={config.author.name}>
      <Sidebar />
      <Page title="Contact">
        <div>
          <ul>
            <li>
              <a
                href="mailto:root@yceffort.kr"
                target="_blank"
                rel="nofollow noopener noreferrer"
              >
                email
              </a>
            </li>
            <li>
              <a
                href="https://yceffort.kr"
                target="_blank"
                rel="nofollow noopener noreferrer"
              >
                blog
              </a>
            </li>
            <li>
              <a
                href="https://github.com/yceffort"
                target="_blank"
                rel="nofollow noopener noreferrer"
              >
                github
              </a>
            </li>
          </ul>
        </div>
      </Page>
    </Layout>
  )
}
