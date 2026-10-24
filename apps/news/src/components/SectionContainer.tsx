import type {ReactNode} from 'react'

import {sectionContainer} from '@/components/SectionContainer.styles'

export default function SectionContainer({children}: {children: ReactNode}) {
  return <div className={sectionContainer}>{children}</div>
}
