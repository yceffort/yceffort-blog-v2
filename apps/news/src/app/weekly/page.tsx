import type {Metadata} from 'next'

import * as styles from '@/components/Issue.styles'
import {IssueArchive} from '@/components/IssueList'
import {listIssues} from '@/lib/content'

export const metadata: Metadata = {title: '주간'}

export default function Page() {
  const issues = listIssues('weekly')
  return (
    <>
      <header className={styles.header}>
        <div className={styles.eyebrow}>Weekly</div>
        <h1 className={styles.title}>주간</h1>
        <p className={styles.subtitle}>
          {`월요일 아침에 지난 한 주의 흐름을 묶습니다. 지금까지 ${issues.length}호.`}
        </p>
      </header>
      <IssueArchive issues={issues} />
    </>
  )
}
