import Image from 'next/image'
import { HTMLProps } from 'react'

import CustomLink from './Link'

function NextImage(props: HTMLProps<HTMLImageElement>) {
  return props.src ? (
    <Image
      {...props}
      crossOrigin="anonymous"
      src={props.src}
      placeholder="empty"
    />
  ) : (
    <p>Currently, image is not available. {props.src}</p>
  )
}

const MdxComponents = {
  img: NextImage,
  a: CustomLink,
}

export default MdxComponents
