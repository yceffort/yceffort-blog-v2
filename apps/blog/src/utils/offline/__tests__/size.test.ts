import {describe, expect, it} from 'vitest'

import {formatBytes} from '../size'

describe('formatBytes', () => {
  it('uses 1000-based KB below 1 MB and MB above', () => {
    expect(formatBytes(0)).toBe('0 KB')
    expect(formatBytes(819_200)).toBe('819 KB')
    expect(formatBytes(3_891_000)).toBe('3.9 MB')
    expect(formatBytes(16_000_000)).toBe('16.0 MB')
  })
})
