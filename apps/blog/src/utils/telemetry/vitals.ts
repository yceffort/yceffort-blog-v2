import type {Attributes} from '@opentelemetry/api'
import {
  type MetricWithAttribution,
  onCLS,
  onFCP,
  onINP,
  onLCP,
  onTTFB,
} from 'web-vitals/attribution'

import {createLcpGate} from './lcp-gate'

export type Report = (metric: MetricWithAttribution, extra?: Attributes) => void

/**
 * 같은 지표 id 는 처음 한 번만 넘긴다. CLS 와 INP 는 페이지가 다시 보였다가 숨으면
 * 같은 id 로 다시 보고된다. 두 번 세면 페이지 하나가 건수 둘이 된다.
 */
export function oncePerId(report: Report): Report {
  const seen = new Set<string>()

  return (metric, extra) => {
    const key = `${metric.name}:${metric.id}`
    if (seen.has(key)) return
    seen.add(key)
    report(metric, extra)
  }
}

/** quiet 로 확정한 LCP 는 곧바로 flush 한다. 기다리는 동안 앱이 닫히면 유실된다. */
export function observeVitals(report: Report, flush: () => void): void {
  const once = oncePerId(report)

  onTTFB(once)
  onFCP(once)
  onCLS(once)
  onINP(once)

  const lcp = createLcpGate((metric, by) => {
    report(metric, {'cwv.lcp.finalized_by': by})
    if (by === 'quiet') flush()
  })
  onLCP(lcp.final)
  onLCP(lcp.candidate, {reportAllChanges: true})
}
