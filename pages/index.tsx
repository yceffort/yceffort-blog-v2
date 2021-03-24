import React from 'react'
import { GetStaticProps } from 'next'

import config from '#src/config'
import Layout from '#components/Layout'
import { DEFAULT_NUMBER_OF_POSTS } from '#common/const'
import { Post } from '#types/types'
import { getAllPosts } from '#utils/Markdown'
import Sidebar from '#components/Sidebar/Sidebar'
import Page from '#components/Page'
import Feed from '#components/Feed'
import Pagination from '#components/Pagination'

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
    props: { recentPosts: recentPosts.map((post) => ({ ...post, path: '' })) },
  }
}
