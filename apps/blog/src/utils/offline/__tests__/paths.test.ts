import {describe, expect, it} from 'vitest'

import {isPostPath} from '../paths'

describe('isPostPath', () => {
  it.each([
    ['/2018/05/31/Levenshtein-distance', true],
    ['/en/2018/05/31/x', true],
    ['/2020/08/typescript-4.0', true],
    ['/2026/10/a/b/c', false],
    ['/pages/2', false],
    ['/saved', false],
    ['/', false],
  ])('%s is %s', (pathname, expected) => {
    expect(isPostPath(pathname)).toBe(expected)
  })
})
