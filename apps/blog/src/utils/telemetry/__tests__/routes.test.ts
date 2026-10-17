import {describe, expect, it} from 'vitest'

import {routeOf} from '../routes'

describe('routeOf', () => {
  it.each([
    ['/', 'home'],
    ['/pages/3', 'home'],
    ['/en', 'home'],
    ['/en/pages/2', 'home'],
    ['/2026/10/cwv-to-tempo-and-mimir', 'post'],
    ['/en/2026/09/unused-javascript-cost', 'post'],
    ['/tags', 'tag'],
    ['/tags/react/pages/2', 'tag'],
    ['/en/tags/react', 'tag'],
    ['/series', 'series'],
    ['/series/sw-caching', 'series'],
    ['/about', 'other'],
    ['/archive', 'other'],
    ['/2026', 'other'],
    ['/english', 'other'],
  ])('%s 는 %s', (pathname, route) => {
    expect(routeOf(pathname)).toBe(route)
  })
})
