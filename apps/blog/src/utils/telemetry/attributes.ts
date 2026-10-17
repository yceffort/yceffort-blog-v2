import type {Attributes} from '@opentelemetry/api'
import type {MetricWithAttribution} from 'web-vitals/attribution'

import {maskSelector, stripQuery} from './mask'

/**
 * ms 값은 정수로 맞춘다. OTLP 로 바꿀 때 정수는 intValue, 소수는 doubleValue 가 되는데,
 * TraceQL 은 `> 2500` 과 `> 2500.0` 을 다른 타입으로 비교한다. 섞여 있으면 필터가 일부만 잡는다.
 */
const ms = (value: number | undefined) =>
  value === undefined ? undefined : Math.round(value)

/**
 * 지표 하나를 span 속성으로 펼친다. trace 쪽에만 싣는 값이라 카디널리티를 신경 쓰지 않는다.
 * 메트릭 라벨은 `labels.ts` 에서 따로 고른다.
 */
export function toAttributes(metric: MetricWithAttribution): Attributes {
  const common: Attributes = {
    'cwv.name': metric.name,
    'cwv.value': metric.name === 'CLS' ? metric.value : ms(metric.value),
    'cwv.rating': metric.rating,
    'cwv.id': metric.id,
    'cwv.navigation_type': metric.navigationType,
  }

  switch (metric.name) {
    case 'LCP': {
      const a = metric.attribution
      return {
        ...common,
        'cwv.lcp.target': maskSelector(a.target),
        'cwv.lcp.url': stripQuery(a.url),
        'cwv.lcp.time_to_first_byte': ms(a.timeToFirstByte),
        'cwv.lcp.resource_load_delay': ms(a.resourceLoadDelay),
        'cwv.lcp.resource_load_duration': ms(a.resourceLoadDuration),
        'cwv.lcp.element_render_delay': ms(a.elementRenderDelay),
      }
    }
    case 'CLS': {
      const a = metric.attribution
      return {
        ...common,
        'cwv.cls.largest_shift_target': maskSelector(a.largestShiftTarget),
        'cwv.cls.largest_shift_value': a.largestShiftValue,
        'cwv.cls.largest_shift_time': ms(a.largestShiftTime),
        'cwv.cls.load_state': a.loadState,
      }
    }
    case 'INP': {
      const a = metric.attribution
      return {
        ...common,
        'cwv.inp.target': maskSelector(a.interactionTarget),
        'cwv.inp.type': a.interactionType,
        'cwv.inp.input_delay': ms(a.inputDelay),
        'cwv.inp.processing_duration': ms(a.processingDuration),
        'cwv.inp.presentation_delay': ms(a.presentationDelay),
        'cwv.inp.load_state': a.loadState,
        // 아래는 Long Animation Frame 을 지원하는 브라우저에서만 채워진다.
        'cwv.inp.script_duration': ms(a.totalScriptDuration),
        'cwv.inp.style_and_layout_duration': ms(a.totalStyleAndLayoutDuration),
        'cwv.inp.paint_duration': ms(a.totalPaintDuration),
        'cwv.inp.unattributed_duration': ms(a.totalUnattributedDuration),
        'cwv.inp.longest_script.subpart': a.longestScript?.subpart,
        'cwv.inp.longest_script.invoker': a.longestScript?.entry.invoker,
      }
    }
    case 'FCP': {
      const a = metric.attribution
      return {
        ...common,
        'cwv.fcp.time_to_first_byte': ms(a.timeToFirstByte),
        'cwv.fcp.first_byte_to_fcp': ms(a.firstByteToFCP),
        'cwv.fcp.load_state': a.loadState,
      }
    }
    case 'TTFB': {
      const a = metric.attribution
      return {
        ...common,
        'cwv.ttfb.waiting': ms(a.waitingDuration),
        'cwv.ttfb.cache': ms(a.cacheDuration),
        'cwv.ttfb.dns': ms(a.dnsDuration),
        'cwv.ttfb.connection': ms(a.connectionDuration),
        'cwv.ttfb.request': ms(a.requestDuration),
      }
    }
  }
}
