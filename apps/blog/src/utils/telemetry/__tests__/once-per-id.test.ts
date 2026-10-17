import {describe, expect, it, vi} from 'vitest'
import type {MetricWithAttribution} from 'web-vitals/attribution'

import {oncePerId} from '../vitals'

const metric = (name: string, id: string, value: number) =>
  ({name, id, value}) as MetricWithAttribution

describe('oncePerId', () => {
  it('같은 지표의 같은 id 는 처음 보고만 넘긴다', () => {
    const report = vi.fn()
    const once = oncePerId(report)

    once(metric('CLS', 'v1', 0.05))
    once(metric('CLS', 'v1', 0.2))

    expect(report).toHaveBeenCalledExactlyOnceWith(
      metric('CLS', 'v1', 0.05),
      undefined,
    )
  })

  it('id 가 같아도 지표가 다르면 따로 센다', () => {
    const report = vi.fn()
    const once = oncePerId(report)

    once(metric('CLS', 'v1', 0.05))
    once(metric('INP', 'v1', 120))

    expect(report).toHaveBeenCalledTimes(2)
  })
})
