import Image from 'next/image'
import { HTMLProps } from 'react'

import CustomLink from './Link'

function NextImage(props: HTMLProps<HTMLImageElement>) {
  return (
    <Image
      {...props}
      crossOrigin="anonymous"
      src={props.src || ''}
      placeholder="empty"
    />
  )
}

const MdxComponents = {
  img: NextImage,
  a: CustomLink,
}

export default MdxComponents
