import { GetStaticProps } from 'next'

import React from 'react'
import Layout from '../src/components/Layout/Layout'
import { DEFAULT_NUMBER_OF_POSTS } from '../src/types/const'
import { FrontMatter } from '../src/types/types'
import { getAllFrontMatters } from '../src/utils/front-matters'

export default function Index({recentPosts}: {recentPosts: Array<FrontMatter>}) {

  return <Layout title="yceffort">{recentPosts.map((fm) =>  <>{fm.title}</>)}</Layout>
}


export const getStaticProps: GetStaticProps = async ({ params }) => {
  const recentPosts = (await getAllFrontMatters()).slice(0, DEFAULT_NUMBER_OF_POSTS)

  return {
    props: { recentPosts },
  }
}