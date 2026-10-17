import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest'
import type {LCPMetricWithAttribution} from 'web-vitals/attribution'

import {createLcpGate} from '../lcp-gate'

const lcp = (value: number) =>
  ({name: 'LCP', id: 'v1', value}) as LCPMetricWithAttribution

describe('createLcpGate', () => {
  beforeEach(() => vi.useFakeTimers())
  afterEach(() => vi.useRealTimers())

  it('후보가 3초 동안 바뀌지 않으면 마지막 후보로 확정한다', () => {
    const emit = vi.fn()
    const gate = createLcpGate(emit)

    gate.candidate(lcp(800))
    vi.advanceTimersByTime(2000)
    gate.candidate(lcp(1600))
    vi.advanceTimersByTime(2999)
    expect(emit).not.toHaveBeenCalled()

    vi.advanceTimersByTime(1)
    expect(emit).toHaveBeenCalledExactlyOnceWith(lcp(1600), 'quiet')
  })

  it('quiet 로 확정한 뒤 web-vitals 가 확정해도 다시 내보내지 않는다', () => {
    const emit = vi.fn()
    const gate = createLcpGate(emit)

    gate.candidate(lcp(800))
    vi.advanceTimersByTime(3000)
    gate.final(lcp(800))

    expect(emit).toHaveBeenCalledExactlyOnceWith(lcp(800), 'quiet')
  })

  it('web-vitals 가 먼저 확정하면 걸려 있던 quiet 타이머는 아무것도 내보내지 않는다', () => {
    const emit = vi.fn()
    const gate = createLcpGate(emit)

    gate.candidate(lcp(800))
    gate.final(lcp(800))
    vi.advanceTimersByTime(10_000)

    expect(emit).toHaveBeenCalledExactlyOnceWith(lcp(800), 'web-vitals')
  })
})
