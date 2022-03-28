import Image from 'next/image'
import { HTMLProps } from 'react'

import CustomLink from './Link'

function NextImage(props: HTMLProps<HTMLImageElement>) {
  const { src } = props

  if (src) {
    if (src.startsWith('http')) {
      return <img src={src} alt={src} />
    } else {
      return (
        <Image
          {...props}
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
  a: CustomLink,
}

export default MdxComponents
