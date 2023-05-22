import CustomLink from '#components/Link'

export default function PageNumber({
  pageNo,
  hasNextPage,
  next,
  prev,
}: {
  pageNo: number
  next: string
  prev?: string
  hasNextPage?: boolean
}) {
  return (
    <div className="flex">
      <div className="flex w-1/2 justify-start text-base font-medium leading-6">
        {pageNo !== 1 && (
          <CustomLink
            href={prev}
            className="text-blue-500 hover:text-blue-600 dark:hover:text-blue-400"
            aria-label="all posts"
          >
            Page {pageNo - 1} &larr;
          </CustomLink>
        )}
      </div>

      <div className="flex w-1/2 justify-end text-base font-medium leading-6">
        {hasNextPage && (
          <CustomLink
            href={next}
            className="text-blue-500 hover:text-blue-600 dark:hover:text-blue-400"
            aria-label="all posts"
          >
            Page {pageNo + 1} &rarr;
          </CustomLink>
        )}
      </div>
    </div>
  )
}
