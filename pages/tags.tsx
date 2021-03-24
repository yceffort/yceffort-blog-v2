import { GetStaticProps } from 'next'
import Link from 'next/link'
import React from 'react'
import styled from 'styled-components'

import Layout from '#components/Layout'
import Page from '#components/Page'
import Sidebar from '#components/Sidebar/Sidebar'
import config from '#src/config'
import { TagWithCount } from '#commons/types'
import { getAllTagsFromPosts } from '#utils/Markdown'

const TagList = styled.ul`
  list-style: none;
  padding-left: 0px;

  li::before {
    content: '#️⃣';
    display: inline-block;
    margin-right: 0.5rem;
  }
`
export default function Tags({ tags }: { tags: Array<TagWithCount> }) {
  return (
    <Layout title={`Tags-${config.title}`} url="https://yceffort.kr/tags">
      <Sidebar />
      <Page title="tags">
        <TagList>
          {tags.map(({ tag, count }, index) => (
            <li key={index}>
              <Link href={`/tag/${tag}/page/1`}>{`${tag} (${count})`}</Link>
            </li>
          ))}
        </TagList>
      </Page>
    </Layout>
  )
}

export const getStaticProps: GetStaticProps = async () => {
  const tags = await getAllTagsFromPosts()
  return {
    props: { tags },
  }
}
