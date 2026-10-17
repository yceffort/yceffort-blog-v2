import type {LCPMetricWithAttribution} from 'web-vitals/attribution'

/**
 * LCP 는 web-vitals 가 확정한 값(첫 입력이나 hidden)만 내보낸다.
 *
 * 처음으로 후보가 `quietMs` 동안 바뀌지 않았을 때의 값도 함께 넘긴다. 앱 웹뷰처럼 hidden 이 오지 않는
 * 환경에서는 이 값을 최종값으로 쓴다(blog-experiments 의 cwv-otel-grafana). 일반 브라우저에서는 독자가
 * 클릭 없이 3초 넘게 읽기만 해도 이 값이 먼저 정해져, 나중에 그려진 더 큰 요소를 놓친다. 그래서 여기서는
 * 정식값과 비교하는 진단용으로만 남긴다.
 */
export function createLcpGate(
  emit: (metric: LCPMetricWithAttribution, quietValue?: number) => void,
  quietMs = 3000,
) {
  let quietValue: number | undefined
  let timer: ReturnType<typeof setTimeout> | undefined

  return {
    /** `onLCP(..., { reportAllChanges: true })` 의 후보 */
    candidate: (metric: LCPMetricWithAttribution) => {
      if (quietValue !== undefined) return
      clearTimeout(timer)
      timer = setTimeout(() => (quietValue = metric.value), quietMs)
    },
    /** `onLCP(...)` 의 정식 확정값 */
    final: (metric: LCPMetricWithAttribution) => {
      clearTimeout(timer)
      emit(metric, quietValue)
    },
  }
}
