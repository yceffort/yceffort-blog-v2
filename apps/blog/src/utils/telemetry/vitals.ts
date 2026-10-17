import type {Attributes} from '@opentelemetry/api'
import {
  type LCPMetricWithAttribution,
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

// SPA 안의 이동도 따로 잰다(Chromium 151 부터 기본으로 켜진 Soft Navigations API). 다른 브라우저에서는 효과가 없다.
const SOFT = {reportSoftNavs: true}

/**
 * quiet 로 확정한 LCP 는 곧바로 flush 한다. 기다리는 동안 앱이 닫히면 유실된다.
 * LCP 확정은 내비게이션마다 따로다. `navigationKey` 가 같은 지표끼리 한 gate 를 쓴다.
 */
export function observeVitals(
  report: Report,
  flush: () => void,
  navigationKey: (metric: MetricWithAttribution) => string,
): void {
  const once = oncePerId(report)

  // SPA 이동의 TTFB 는 0 으로 보고되므로(web-vitals README) 첫 로드만 잰다.
  onTTFB(once)
  onFCP(once, SOFT)
  onCLS(once, SOFT)
  onINP(once, SOFT)

  const gates = new Map<string, ReturnType<typeof createLcpGate>>()
  const gateFor = (metric: LCPMetricWithAttribution) => {
    const key = navigationKey(metric)
    let gate = gates.get(key)
    if (!gate) {
      gate = createLcpGate((settled, by) => {
        report(settled, {'cwv.lcp.finalized_by': by})
        if (by === 'quiet') flush()
      })
      gates.set(key, gate)
    }
    return gate
  }
  onLCP((metric) => gateFor(metric).final(metric), SOFT)
  onLCP((metric) => gateFor(metric).candidate(metric), {
    ...SOFT,
    reportAllChanges: true,
  })
}
