import {format, parseISO} from 'date-fns'
import {ko} from 'date-fns/locale'

import type {Issue} from '@/lib/content'

export function formatDate(date: string, pattern: string) {
  return format(parseISO(date), pattern, {locale: ko})
}

export function issueTitle(issue: Issue) {
  if (issue.kind === 'weekly') {
    const week = Number(issue.slug.split('-W')[1])
    return `${week}주차`
  }
  return formatDate(issue.date, 'M월 d일 EEEE')
}

export function issueSubtitle(issue: Issue) {
  if (issue.kind === 'weekly' && issue.period) {
    return `${formatDate(issue.period.start, 'M월 d일')}부터 ${formatDate(issue.period.end, 'M월 d일')}까지`
  }
  return null
}

export function issueHref(issue: Issue) {
  return `/${issue.kind}/${issue.slug}`
}

export function issueYear(issue: Issue) {
  return issue.slug.slice(0, 4)
}
