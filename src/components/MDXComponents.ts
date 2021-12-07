import Image from 'next/image'

import CustomLink from './Link'

const MdxComponents = {
  Image, // eslint-disable-line @typescript-eslint/naming-convention
  a: CustomLink,
}

export default MdxComponents
