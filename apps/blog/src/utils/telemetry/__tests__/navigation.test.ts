import {describe, expect, it} from 'vitest'
import type {MetricWithAttribution} from 'web-vitals/attribution'

import {createNavigationTracker} from '../navigation'

const metric = (fields: Partial<MetricWithAttribution>) =>
  ({
    name: 'LCP',
    navigationId: 1,
    navigationType: 'navigate',
    navigationURL: 'https://yceffort.kr/',
    ...fields,
  }) as MetricWithAttribution

describe('createNavigationTracker', () => {
  it('첫 로드는 하나의 키로 모은다', () => {
    const tracker = createNavigationTracker('/')
    expect(tracker.resolve(metric({navigationType: 'reload'}))).toEqual({
      key: 'hard',
      kind: 'hard',
      startTime: 0,
      pathname: '/',
    })
  })

  it('SPA 이동은 navigationId 마다 다른 키와 그 이동의 URL 을 쓴다', () => {
    const tracker = createNavigationTracker('/')
    const soft = tracker.resolve(
      metric({
        navigationType: 'soft-navigation',
        navigationId: 7,
        navigationURL: 'https://yceffort.kr/2026/10/post?x=1',
        navigationStartTime: 4200,
      }),
    )
    expect(soft).toEqual({
      key: 'soft:7',
      kind: 'soft',
      startTime: 4200,
      pathname: '/2026/10/post',
    })
  })

  it('bfcache 복원은 navigationId 가 같아도 복원할 때마다 다른 키가 된다', () => {
    const tracker = createNavigationTracker('/')
    const restored = metric({navigationType: 'back-forward-cache'})

    tracker.restored(9000, '/tags')
    const first = tracker.resolve(restored)
    tracker.restored(15000, '/series')
    const second = tracker.resolve(restored)

    expect(first).toEqual({
      key: 'restore:1',
      kind: 'restore',
      startTime: 9000,
      pathname: '/tags',
    })
    expect(second.key).toBe('restore:2')
    expect(second.pathname).toBe('/series')
  })
})
