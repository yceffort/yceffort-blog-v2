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

/** 같은 id 의 다시 온 보고. `first` 는 처음 넘긴 보고의 값이다. */
export type ReportUpdate = (
  metric: MetricWithAttribution,
  first: Pick<MetricWithAttribution, 'value' | 'rating'>,
  index: number,
) => void

/**
 * 같은 지표 id 는 처음 한 번만 `report` 로 넘긴다. CLS 와 INP 는 페이지가 다시 보였다가 숨으면
 * 같은 id 로 다시 보고된다. 두 번 세면 페이지 하나가 건수 둘이 된다. 그래서 Mimir 의 CLS, INP 는
 * 처음 숨겨진 시점의 값이다. 다시 온 보고는 `update` 로 넘겨, 그 값이 얼마나 자주 얼마나 바뀌는지 잰다.
 */
export function oncePerId(report: Report, update?: ReportUpdate): Report {
  const seen = new Map<
    string,
    {value: number; rating: MetricWithAttribution['rating']; count: number}
  >()

  return (metric, extra) => {
    const key = `${metric.name}:${metric.id}`
    const first = seen.get(key)
    if (first) {
      first.count++
      update?.(metric, {value: first.value, rating: first.rating}, first.count)
      return
    }
    seen.set(key, {value: metric.value, rating: metric.rating, count: 1})
    report(metric, extra)
  }
}

// SPA 안의 이동도 따로 잰다(Chromium 151 부터 기본으로 켜진 Soft Navigations API). 다른 브라우저에서는 효과가 없다.
const SOFT = {reportSoftNavs: true}

/**
 * quiet 값과 정식값의 관계. LCP 후보는 커지기만 하므로 quiet 값은 정식값보다 크지 않다.
 * `lower` 는 quiet 값을 최종값으로 썼다면 더 작은 LCP 를 기록했을 페이지 로드다.
 */
function quietAttributes(value: number, quietValue?: number): Attributes {
  if (quietValue === undefined) return {'cwv.lcp.quiet': 'none'}
  return {
    'cwv.lcp.quiet': quietValue === value ? 'same' : 'lower',
    'cwv.lcp.quiet_value': Math.round(quietValue),
  }
}

/**
 * LCP 확정은 내비게이션마다 따로다. `navigationKey` 가 같은 지표끼리 한 gate 를 쓴다.
 */
export function observeVitals(
  report: Report,
  reportUpdate: ReportUpdate,
  navigationKey: (metric: MetricWithAttribution) => string,
): void {
  const once = oncePerId(report, reportUpdate)

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
      gate = createLcpGate((final, quietValue) =>
        report(final, quietAttributes(final.value, quietValue)),
      )
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
