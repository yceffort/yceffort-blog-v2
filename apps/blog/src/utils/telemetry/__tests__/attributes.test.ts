import {describe, expect, it} from 'vitest'
import type {
  CLSMetricWithAttribution,
  LCPMetricWithAttribution,
} from 'web-vitals/attribution'

import {toAttributes} from '../attributes'

const numbers = (attrs: Record<string, unknown>) =>
  Object.entries(attrs).filter(([, v]) => typeof v === 'number') as [
    string,
    number,
  ][]

describe('toAttributes', () => {
  it('ms 값은 모두 정수로 보낸다(OTLP intValue 로 타입을 맞춘다)', () => {
    const lcp = {
      name: 'LCP',
      value: 2412.7,
      rating: 'good',
      id: 'v1',
      navigationType: 'navigate',
      attribution: {
        timeToFirstByte: 120.4,
        resourceLoadDelay: 30.5,
        resourceLoadDuration: 900.1,
        elementRenderDelay: 1361.7,
      },
    } as unknown as LCPMetricWithAttribution

    for (const [key, value] of numbers(toAttributes(lcp))) {
      expect(Number.isInteger(value), key).toBe(true)
    }
  })

  it('CLS 점수는 소수 그대로 둔다', () => {
    const cls = {
      name: 'CLS',
      value: 0.083,
      rating: 'good',
      id: 'v1',
      navigationType: 'navigate',
      attribution: {largestShiftValue: 0.05, largestShiftTime: 812.3},
    } as unknown as CLSMetricWithAttribution
    const attrs = toAttributes(cls)

    expect(attrs['cwv.value']).toBe(0.083)
    expect(attrs['cwv.cls.largest_shift_value']).toBe(0.05)
    expect(attrs['cwv.cls.largest_shift_time']).toBe(812)
  })
})
