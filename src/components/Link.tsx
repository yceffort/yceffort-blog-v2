import Link from 'next/link'
import { HTMLProps } from 'react'

const CustomLink = ({ href, ...rest }: HTMLProps<HTMLAnchorElement>) => {
  const isInternalLink = href && href.startsWith('/')
  const isAnchorLink = href && href.startsWith('#')

  if (isInternalLink) {
    return (
      <Link
        href={href}
        passHref
        className={rest.className}
        onClick={rest.onClick}
      >
        {rest.children}
      </Link>
    )
  }

  if (isAnchorLink) {
    return <a href={href} {...rest} />
  }

  return <a target="_blank" rel="noopener noreferrer" href={href} {...rest} />
}

export default CustomLink
