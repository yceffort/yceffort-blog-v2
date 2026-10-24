import Link from 'next/link'

import * as styles from '@/components/Issue.styles'
import type {Issue} from '@/lib/content'
import {issueHref, issueSubtitle, issueTitle, issueYear} from '@/lib/format'
import {renderIssueBody} from '@/lib/markdown'

const KIND_LABEL = {daily: 'Daily', weekly: 'Weekly'} as const

function Dot() {
  return <span className={styles.eyebrow_dot} aria-hidden="true" />
}

export default function IssueView({
  issue,
  prev,
  next,
}: {
  issue: Issue
  prev: Issue | null
  next: Issue | null
}) {
  const subtitle = issueSubtitle(issue)
  return (
    <article>
      <header className={styles.header}>
        <div className={styles.eyebrow}>
          <span>{KIND_LABEL[issue.kind]}</span>
          <Dot />
          <span>{issueYear(issue)}</span>
          <Dot />
          <span>{`${issue.count}편`}</span>
        </div>
        <h1 className={styles.title}>{issueTitle(issue)}</h1>
        {subtitle && <p className={styles.subtitle}>{subtitle}</p>}
      </header>
      <div className="news-body">{renderIssueBody(issue.body)}</div>
      {(prev || next) && (
        <nav className={styles.nav} aria-label="이전 호와 다음 호">
          {prev ? (
            <Link href={issueHref(prev)} className={styles.nav_link}>
              <span className={styles.nav_label}>이전 호</span>
              <span className={styles.nav_title}>{issueTitle(prev)}</span>
              <span className={styles.nav_sub}>{prev.lead}</span>
            </Link>
          ) : (
            <span />
          )}
          {next && (
            <Link href={issueHref(next)} className={styles.nav_link_next}>
              <span className={styles.nav_label}>다음 호</span>
              <span className={styles.nav_title}>{issueTitle(next)}</span>
              <span className={styles.nav_sub}>{next.lead}</span>
            </Link>
          )}
        </nav>
      )}
    </article>
  )
}
