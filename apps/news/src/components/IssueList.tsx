import Link from 'next/link'

import * as styles from '@/components/IssueList.styles'
import type {Issue} from '@/lib/content'
import {formatDate, issueHref, issueTitle} from '@/lib/format'

function rowDate(issue: Issue) {
  if (issue.kind === 'weekly') {
    return issue.slug
  }
  return formatDate(issue.date, 'yyyy-MM-dd')
}

export function IssueRow({issue}: {issue: Issue}) {
  return (
    <Link href={issueHref(issue)} className={styles.row}>
      <span className={styles.row_date}>{rowDate(issue)}</span>
      <span className={styles.row_title}>
        {issue.kind === 'weekly' ? issueTitle(issue) : issue.lead}
      </span>
      <span className={styles.row_count}>{`${issue.count}편`}</span>
    </Link>
  )
}

export function IssueSection({
  title,
  issues,
  moreHref,
}: {
  title: string
  issues: Issue[]
  moreHref?: string
}) {
  if (issues.length === 0) {
    return null
  }
  return (
    <section className={styles.section}>
      <h2 className={styles.heading}>
        <span>{title}</span>
        <span className={styles.heading_rule} aria-hidden="true" />
        {moreHref && (
          <Link href={moreHref} className={styles.heading_more}>
            전체 보기
          </Link>
        )}
      </h2>
      {issues.map((issue) => (
        <IssueRow key={issue.slug} issue={issue} />
      ))}
    </section>
  )
}

// 월별로 묶은 전체 목록
export function IssueArchive({issues}: {issues: Issue[]}) {
  const groups = new Map<string, Issue[]>()
  for (const issue of issues) {
    const key = formatDate(issue.date, 'yyyy년 M월')
    groups.set(key, [...(groups.get(key) ?? []), issue])
  }
  return (
    <>
      {[...groups].map(([label, group]) => (
        <section key={label}>
          <h2 className={styles.month}>{label}</h2>
          {group.map((issue) => (
            <IssueRow key={issue.slug} issue={issue} />
          ))}
        </section>
      ))}
    </>
  )
}
