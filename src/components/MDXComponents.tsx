import { HTMLProps } from 'react'
import Link from 'next/link'
import Image from 'next/image'

function NextImage(props: HTMLProps<HTMLImageElement>) {
  const { src } = props
  const width = Number(props.width)
  const height = Number(props.height)

  if (src) {
    if (src.startsWith('http')) {
      // eslint-disable-next-line @next/next/no-img-element
      return <img src={src} alt={src} width={width} height={height} />
    } else {
      return (
        <Image
          width={width}
          height={height}
          alt={props.alt || ''}
          crossOrigin="anonymous"
          src={src}
          placeholder="empty"
        />
      )
    }
  } else {
    return <p>Currently, image is not available. {src}</p>
  }
}

const MdxComponents = {
  img: NextImage,
  a: (props: HTMLProps<HTMLAnchorElement>) => {
    const { href, ...rest } = props

    if (!href) {
      return null
    }

    const isAnchorLink = href.startsWith('#')

    if (isAnchorLink) {
      return <a href={href} {...rest} />
    }

    return (
      <Link
        href={href}
        className={props.className}
        target={props.target}
        rel={props.rel}
      >
        {props.children}
      </Link>
    )
  },
}

export default MdxComponents
