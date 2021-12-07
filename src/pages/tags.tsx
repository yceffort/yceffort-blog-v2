import { GetStaticProps } from 'next'

import { getAllTagsFromPosts } from '#utils/Post'
import { PageSeo } from '#components/SEO'
import { SiteConfig } from '#src/config'
import Tag from '#components/Tag'
import CustomLink from '#components/Link'
import { TagWithCount } from '#commons/types'

export default function Tags({ tags }: { tags: TagWithCount[] }) {
  return (
    <>
      <PageSeo
        title={`Tags - ${SiteConfig.author.name}`}
        description="Things I blog about"
        url={`${SiteConfig.url}/tags`}
      />
      <div className="flex flex-col items-start justify-start divide-y divide-gray-200 dark:divide-gray-700 md:justify-center md:items-center md:divide-y-0 md:flex-row md:space-x-6 md:mt-24">
        <div className="pt-6 pb-8 space-x-2 md:space-y-5">
          <h1 className="text-3xl font-extrabold leading-9 tracking-tight text-gray-900 dark:text-gray-100 sm:text-4xl sm:leading-10 md:text-6xl md:leading-14 md:border-r-2 md:px-6">
            Tags
          </h1>
        </div>
        <div className="flex flex-wrap max-w-lg">
          {Object.keys(tags).length === 0 && 'No tags found.'}
          {tags.map(({ tag, count }) => {
            return (
              <div key={tag} className="mt-2 mb-2 mr-5">
                <Tag text={tag} />
                <CustomLink
                  href={`/tag/${tag}`}
                  className="-ml-2 text-sm font-semibold text-gray-600 uppercase dark:text-gray-300"
                >
                  {count}
                </CustomLink>
              </div>
            )
          })}
        </div>
      </div>
    </>
  )
}

export const getStaticProps: GetStaticProps = async () => {
  const tags = await getAllTagsFromPosts()
  return {
    props: { tags },
  }
}
