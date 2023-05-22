import Tag from '#components/Tag'
import CustomLink from '#components/Link'
import { getAllTagsFromPosts } from '#utils/Post'

export default async function Page() {
  const tags = await getAllTagsFromPosts()

  return (
    <div className="flex max-w-lg flex-wrap">
      {Object.keys(tags).length === 0 && 'No tags found.'}
      {tags.map(({ tag, count }) => {
        return (
          <div key={tag} className="mt-2 mb-2 mr-5">
            <Tag text={tag} />
            <CustomLink
              href={`/tag/${tag}`}
              className="-ml-2 text-sm font-semibold uppercase text-gray-600 dark:text-gray-300"
            >
              {count}
            </CustomLink>
          </div>
        )
      })}
    </div>
  )
}
