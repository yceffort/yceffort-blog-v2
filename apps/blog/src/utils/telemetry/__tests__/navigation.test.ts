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

  it('bfcache 복원은 navigationStartTime 으로 가르고 경로는 pageshow 에서 저장한 값을 쓴다', () => {
    const tracker = createNavigationTracker('/')
    const restored = (navigationStartTime: number) =>
      metric({navigationType: 'back-forward-cache', navigationStartTime})

    tracker.restored(9000, '/tags')
    tracker.restored(15000, '/series')

    expect(tracker.resolve(restored(9000))).toEqual({
      key: 'restore:9000',
      kind: 'restore',
      startTime: 9000,
      pathname: '/tags',
    })
    expect(tracker.resolve(restored(15000))).toEqual({
      key: 'restore:15000',
      kind: 'restore',
      startTime: 15000,
      pathname: '/series',
    })
    expect(tracker.resolve(restored(9000)).pathname).toBe('/tags')
  })

  it('복원 경로를 아직 저장하지 않았으면 첫 로드 경로로 둔다', () => {
    const tracker = createNavigationTracker('/')
    expect(
      tracker.resolve(
        metric({
          navigationType: 'back-forward-cache',
          navigationStartTime: 4000,
        }),
      ),
    ).toEqual({
      key: 'restore:4000',
      kind: 'restore',
      startTime: 4000,
      pathname: '/',
    })
  })
})
