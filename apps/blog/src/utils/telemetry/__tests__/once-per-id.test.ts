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

describe('oncePerId 의 갱신', () => {
  it('같은 id 로 다시 온 보고는 report 대신 update 로 넘기고, 처음 값과 몇 번째인지를 함께 준다', () => {
    const report = vi.fn()
    const update = vi.fn()
    const once = oncePerId(report, update)

    once({...metric('CLS', 'v1', 0.01), rating: 'good'})
    once({...metric('CLS', 'v1', 0.4), rating: 'poor'})
    once({...metric('CLS', 'v1', 0.5), rating: 'poor'})

    expect(report).toHaveBeenCalledOnce()
    expect(
      update.mock.calls.map(([m, first, index]) => [m.value, first, index]),
    ).toEqual([
      [0.4, {value: 0.01, rating: 'good'}, 2],
      [0.5, {value: 0.01, rating: 'good'}, 3],
    ])
  })
})
