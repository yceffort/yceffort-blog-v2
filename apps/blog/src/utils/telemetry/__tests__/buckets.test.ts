import {describe, expect, it} from 'vitest'
import {
  CLSThresholds,
  FCPThresholds,
  INPThresholds,
  LCPThresholds,
  TTFBThresholds,
} from 'web-vitals'

import {BUCKETS} from '../buckets'

const thresholds = {
  CLS: CLSThresholds,
  FCP: FCPThresholds,
  INP: INPThresholds,
  LCP: LCPThresholds,
  TTFB: TTFBThresholds,
}

describe('BUCKETS', () => {
  it.each(Object.entries(thresholds))(
    '%s 의 good, poor 경계가 버킷 경계에 들어 있다',
    (name, [good, poor]) => {
      const bounds = BUCKETS[name as keyof typeof BUCKETS]
      expect(bounds).toContain(good)
      expect(bounds).toContain(poor)
    },
  )

  it.each(Object.entries(BUCKETS))('%s 경계는 오름차순이다', (_, bounds) => {
    expect(bounds).toEqual([...bounds].sort((a, b) => a - b))
  })
})
