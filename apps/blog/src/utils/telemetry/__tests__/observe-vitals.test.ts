import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest'
import type {
  LCPMetricWithAttribution,
  MetricWithAttribution,
} from 'web-vitals/attribution'

type Callback = (metric: MetricWithAttribution) => void
const lcpSubscriptions: Array<{callback: Callback; allChanges: boolean}> = []

vi.mock('web-vitals/attribution', () => ({
  onTTFB: vi.fn(),
  onFCP: vi.fn(),
  onCLS: vi.fn(),
  onINP: vi.fn(),
  onLCP: (callback: Callback, opts?: {reportAllChanges?: boolean}) =>
    lcpSubscriptions.push({callback, allChanges: !!opts?.reportAllChanges}),
}))

const {observeVitals} = await import('../vitals')

const lcp = (navigationId: number, value: number) =>
  ({
    name: 'LCP',
    id: `lcp-${navigationId}`,
    navigationId,
    value,
  }) as LCPMetricWithAttribution
const candidate = (m: LCPMetricWithAttribution) =>
  lcpSubscriptions.filter((s) => s.allChanges).forEach((s) => s.callback(m))
const final = (m: LCPMetricWithAttribution) =>
  lcpSubscriptions.filter((s) => !s.allChanges).forEach((s) => s.callback(m))

describe('observeVitals 의 LCP', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    lcpSubscriptions.length = 0
  })
  afterEach(() => vi.useRealTimers())

  it('내비게이션마다 quiet 값을 따로 들고, 정식값만 한 번씩 내보낸다', () => {
    const report = vi.fn()
    observeVitals(report, vi.fn(), (m) => `nav:${m.navigationId}`)

    // 첫 로드: 800 이 3초 멈춘 뒤 정식값 6000
    candidate(lcp(1, 800))
    vi.advanceTimersByTime(3000)
    candidate(lcp(1, 6000))
    // SPA 이동: 1200 이 3초 안에 정식 확정
    candidate(lcp(2, 1200))
    vi.advanceTimersByTime(1000)
    final(lcp(2, 1200))
    final(lcp(1, 6000))
    vi.advanceTimersByTime(10_000)

    expect(
      report.mock.calls.map(([m, extra]) => [m.navigationId, m.value, extra]),
    ).toEqual([
      [2, 1200, {'cwv.lcp.quiet': 'none'}],
      [1, 6000, {'cwv.lcp.quiet': 'lower', 'cwv.lcp.quiet_value': 800}],
    ])
  })
})
