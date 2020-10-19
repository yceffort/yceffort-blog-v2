import React from 'react'
import { GetStaticProps } from 'next'
import dynamic from 'next/dynamic'

import config from '../src/config'
import Layout from '../src/components/Layout'
import { DEFAULT_NUMBER_OF_POSTS } from '../src/types/const'
import { Post } from '../src/types/types'
import { getAllPosts } from '../src/utils/Markdown'
import Sidebar from '../src/components/Sidebar/Sidebar'
import Page from '../src/components/Page'
import Feed from '../src/components/Feed'
import Pagination from '../src/components/Pagination'

export default function Index({ recentPosts }: { recentPosts: Array<Post> }) {
  return (
    <Layout
      title={config.title}
      description={config.subtitle}
      url="https://yceffort.kr"
    >
      <Sidebar />
      <Page>
        <Feed posts={recentPosts} />
        <Pagination
          hasPrevPage={false}
          hasNextPage={true}
          prevPagePath={'/'}
          nextPagePath={'/page/1'}
        />
      </Page>
    </Layout>
  )
}

export const getStaticProps: GetStaticProps = async () => {
  const recentPosts = (await getAllPosts()).slice(0, DEFAULT_NUMBER_OF_POSTS)

  return {
    props: { recentPosts },
  }
}
