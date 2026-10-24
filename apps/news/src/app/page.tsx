import {IssueSection} from '@/components/IssueList'
import IssueView from '@/components/IssueView'
import {getLatest, getNeighbors, listIssues} from '@/lib/content'

export default function Page() {
  const latest = getLatest('daily')
  if (!latest) {
    return <p>아직 발행된 호가 없습니다.</p>
  }
  const {prev, next} = getNeighbors('daily', latest.slug)
  const recentDaily = listIssues('daily').slice(1, 8)
  const recentWeekly = listIssues('weekly').slice(0, 3)
  return (
    <>
      <IssueView issue={latest} prev={prev} next={next} />
      <IssueSection title="지난 호" issues={recentDaily} moreHref="/daily" />
      <IssueSection title="주간" issues={recentWeekly} moreHref="/weekly" />
    </>
  )
}
