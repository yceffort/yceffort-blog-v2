import type {Metadata} from 'next'

import * as ambientStyles from '@/components/layout/ambient.styles'
import {SavedPostsView} from '@/components/pwa/SavedPostsView'

export const metadata: Metadata = {
  title: 'Saved posts - yceffort',
  robots: {index: false},
}

export default function SavedPage() {
  return (
    <div className={`page-view ${ambientStyles.page_view}`}>
      <SavedPostsView locale="en" />
    </div>
  )
}
