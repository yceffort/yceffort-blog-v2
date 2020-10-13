import React from 'react'
import { GetStaticProps } from 'next'

import config from '../config'
import Layout from '../src/components/Layout/Layout'
import { DEFAULT_NUMBER_OF_POSTS } from '../src/types/const'
import { Post } from '../src/types/types'
import { getAllPosts } from '../src/utils/frontMatters'
import Sidebar from '../src/components/Sidebar/Sidebar'
import Page from '../src/components/Page/Page'
import Feed from '../src/components/Feed/Feed'
import Pagination from '../src/components/Pagination/Pagination'

export default function Index({ recentPosts }: { recentPosts: Array<Post> }) {
  return (
    <Layout title={config.title} description={config.subtitle}>
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
