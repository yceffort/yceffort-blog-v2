import React from 'react'
import { GetStaticProps } from 'next'

import config from '../config'
import Layout from '../src/components/Layout/Layout'
import { DEFAULT_NUMBER_OF_POSTS } from '../src/types/const'
import { FrontMatter } from '../src/types/types'
import { getAllFrontMatters } from '../src/utils/frontMatters'
import Sidebar from '../src/components/Sidebar/Sidebar'
import Page from '../src/components/Page/Page'

export default function Index({
  recentPosts,
}: {
  recentPosts: Array<FrontMatter>
}) {
  return (
    <Layout title={config.title} description={config.subtitle}>
      <Sidebar />
      <Page>
        {recentPosts.map((fm) => (
          <div key={fm.path}>{fm.title}</div>
        ))}
      </Page>
    </Layout>
  )
}

export const getStaticProps: GetStaticProps = async () => {
  const recentPosts = (await getAllFrontMatters()).slice(
    0,
    DEFAULT_NUMBER_OF_POSTS,
  )

  return {
    props: { recentPosts },
  }
}
