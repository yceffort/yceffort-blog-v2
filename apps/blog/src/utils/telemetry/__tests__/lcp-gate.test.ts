import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest'
import type {LCPMetricWithAttribution} from 'web-vitals/attribution'

import {createLcpGate} from '../lcp-gate'

const lcp = (value: number) =>
  ({name: 'LCP', id: 'v1', value}) as LCPMetricWithAttribution

describe('createLcpGate', () => {
  beforeEach(() => vi.useFakeTimers())
  afterEach(() => vi.useRealTimers())

  it('후보가 3초 동안 멈춰도 내보내지 않고, 정식 확정값을 quiet 값과 함께 내보낸다', () => {
    const emit = vi.fn()
    const gate = createLcpGate(emit)

    gate.candidate(lcp(800))
    vi.advanceTimersByTime(3000)
    expect(emit).not.toHaveBeenCalled()

    gate.candidate(lcp(6000))
    gate.final(lcp(6000))
    expect(emit).toHaveBeenCalledExactlyOnceWith(lcp(6000), 800)
  })

  it('quiet 값은 처음으로 3초 멈춘 후보다', () => {
    const emit = vi.fn()
    const gate = createLcpGate(emit)

    gate.candidate(lcp(800))
    vi.advanceTimersByTime(3000)
    gate.candidate(lcp(6000))
    vi.advanceTimersByTime(3000)
    gate.final(lcp(6000))

    expect(emit).toHaveBeenCalledExactlyOnceWith(lcp(6000), 800)
  })

  it('후보가 3초 안에 바뀌면 타이머를 다시 건다', () => {
    const emit = vi.fn()
    const gate = createLcpGate(emit)

    gate.candidate(lcp(800))
    vi.advanceTimersByTime(2000)
    gate.candidate(lcp(1600))
    vi.advanceTimersByTime(2999)
    gate.final(lcp(1600))

    expect(emit).toHaveBeenCalledExactlyOnceWith(lcp(1600), undefined)
  })

  it('정식 확정이 먼저 오면 걸려 있던 타이머는 아무것도 남기지 않는다', () => {
    const emit = vi.fn()
    const gate = createLcpGate(emit)

    gate.candidate(lcp(800))
    gate.final(lcp(800))
    vi.advanceTimersByTime(10_000)

    expect(emit).toHaveBeenCalledExactlyOnceWith(lcp(800), undefined)
  })
})
