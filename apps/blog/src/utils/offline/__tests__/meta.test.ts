import {describe, expect, it} from 'vitest'

import {readMeta, writeMeta} from '../meta'
import {createFakeStorage} from './fakeCaches'

describe('meta', () => {
  it('falls back to defaults without storage', () => {
    const defaults = {
      count: 5,
      images: true,
      locale: 'ko',
      auto: [],
      excluded: [],
      lastSync: null,
      announced: false,
      sizes: {},
    }
    expect(readMeta(null)).toEqual(defaults)
    expect(() => writeMeta(null, readMeta(null))).not.toThrow()
  })

  it('reads stored meta without sizes as empty sizes', () => {
    const storage = createFakeStorage()
    storage.setItem('offline-saved-meta', JSON.stringify({count: 10}))
    expect(readMeta(storage).sizes).toEqual({})
    expect(readMeta(storage).count).toBe(10)
  })
})
