import type {Attributes} from '@opentelemetry/api'
import type {MetricWithAttribution} from 'web-vitals/attribution'

/**
 * 메트릭 라벨. 값의 종류가 정해진 것만 쓴다. 라벨 조합 하나가 시리즈 하나라
 * selector 나 URL 을 넣으면 시리즈 수가 화면 요소 수만큼 늘어난다.
 */
export function toLabels(
  metric: MetricWithAttribution,
  pageLabels: Attributes,
): Attributes {
  const labels: Attributes = {
    ...pageLabels,
    'navigation.type': metric.navigationType,
    rating: metric.rating,
  }

  if (metric.name === 'INP' && metric.attribution.interactionType) {
    labels['interaction.type'] = metric.attribution.interactionType
  }

  return labels
}
