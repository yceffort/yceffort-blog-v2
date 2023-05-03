import Link from 'next/link'

const Tag = ({ text }: { text: string }) => {
  return (
    <Link
      href={`/tags/${text}/pages/1`}
      className="mr-3 text-sm font-medium uppercase text-blue-500 hover:text-blue-600 dark:hover:text-blue-400"
    >
      {text.split(' ').join('-')}
    </Link>
  )
}

export default Tag
