import Link from 'next/link'

import Tag from '#components/Tag'
import { getAllTagsFromPosts } from '#utils/Post'

export default async function Page() {
  const tags = await getAllTagsFromPosts()

  return (
    <div className="flex flex-col items-start justify-start divide-y divide-gray-200 dark:divide-gray-700 md:mt-24 md:flex-row md:items-center md:justify-center md:space-x-6 md:divide-y-0">
      <div className="space-x-2 pt-6 pb-8 md:space-y-5">
        <h1 className="text-3xl font-extrabold leading-9 tracking-tight text-gray-900 dark:text-gray-100 sm:text-4xl sm:leading-10 md:border-r-2 md:px-6 md:text-6xl md:leading-14">
          Tags
        </h1>
      </div>
      <div className="flex max-w-lg flex-wrap">
        {Object.keys(tags).length === 0 && 'No tags found.'}
        {tags.map(({ tag, count }) => {
          return (
            <div key={tag} className="mt-2 mb-2 mr-5">
              <Tag text={tag} />
              <Link
                href={`/tag/${tag}`}
                className="-ml-2 text-sm font-semibold uppercase text-gray-600 dark:text-gray-300"
              >
                {count}
              </Link>
            </div>
          )
        })}
      </div>
    </div>
  )
}
