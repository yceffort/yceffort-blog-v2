import type {LCPMetricWithAttribution} from 'web-vitals/attribution'

export type LcpFinalizer = 'web-vitals' | 'quiet'

/**
 * LCP 를 페이지당 한 번만 내보낸다. 먼저 확정한 쪽이 이긴다.
 *
 * web-vitals 는 첫 입력이나 hidden 에서 LCP 를 확정한다. 앱 웹뷰를 입력 없이 떠나면서
 * hidden 이 오지 않으면 LCP 는 영영 확정되지 않는다. 그래서 후보가 `quietMs` 동안
 * 바뀌지 않으면 그 후보로 확정한다. 새 후보가 오면 타이머를 다시 건다.
 */
export function createLcpGate(
  emit: (metric: LCPMetricWithAttribution, finalizedBy: LcpFinalizer) => void,
  quietMs = 3000,
) {
  let settled = false
  let timer: ReturnType<typeof setTimeout> | undefined

  const settle = (metric: LCPMetricWithAttribution, by: LcpFinalizer) => {
    if (settled) return
    settled = true
    clearTimeout(timer)
    emit(metric, by)
  }

  return {
    /** `onLCP(..., { reportAllChanges: true })` 의 후보 */
    candidate: (metric: LCPMetricWithAttribution) => {
      if (settled) return
      clearTimeout(timer)
      timer = setTimeout(() => settle(metric, 'quiet'), quietMs)
    },
    /** `onLCP(...)` 의 정식 확정값 */
    final: (metric: LCPMetricWithAttribution) => {
      settle(metric, 'web-vitals')
    },
  }
}
