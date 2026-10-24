import type {Metadata} from 'next'
import {notFound} from 'next/navigation'

import IssueView from '@/components/IssueView'
import {getIssue, getNeighbors, listIssues} from '@/lib/content'
import {issueTitle} from '@/lib/format'

interface Params {
  date: string
}

export function generateStaticParams(): Params[] {
  return listIssues('daily').map((issue) => ({date: issue.slug}))
}

export async function generateMetadata({
  params,
}: {
  params: Promise<Params>
}): Promise<Metadata> {
  const {date} = await params
  const issue = getIssue('daily', date)
  if (!issue) {
    return {}
  }
  return {title: issueTitle(issue), description: issue.lead}
}

export default async function Page({params}: {params: Promise<Params>}) {
  const {date} = await params
  const issue = getIssue('daily', date)
  if (!issue) {
    notFound()
  }
  const {prev, next} = getNeighbors('daily', date)
  return <IssueView issue={issue} prev={prev} next={next} />
}
