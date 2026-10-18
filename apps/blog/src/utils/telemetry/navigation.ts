import type {MetricWithAttribution} from 'web-vitals/attribution'

export type NavigationKind = 'hard' | 'soft' | 'restore'

/** 지표가 속한 내비게이션. trace 하나가 내비게이션 하나다. */
export interface NavigationRef {
  key: string
  kind: NavigationKind
  /** `performance.timeOrigin` 기준 시작 시각(ms). 첫 로드는 0 이다. */
  startTime: number
  pathname: string
}

function pathnameOf(url: string | undefined, fallback: string): string {
  if (!url) return fallback
  try {
    return new URL(url).pathname
  } catch {
    return fallback
  }
}

/**
 * 지표마다 어느 내비게이션의 것인지 정한다.
 *
 * SPA 안의 이동(soft navigation)은 web-vitals 가 `navigationId` 와 `navigationURL` 을 새로 준다.
 * bfcache 복원은 첫 로드의 `navigationId` 를 그대로 쓰지만(web-vitals 6.2.3 `initMetric`),
 * `navigationStartTime` 에 복원 시각(`pageshow` 의 `timeStamp`)이 들어 있어 그 값으로 가른다.
 * 복원된 문서는 SPA 이동 뒤의 URL 일 수 있어 경로는 `pageshow` 에서 읽어 같은 `timeStamp` 로 저장한다.
 */
export function createNavigationTracker(initialPathname: string) {
  const restorePathnames = new Map<number, string>()

  return {
    /** `pageshow` 에서 `event.persisted` 일 때 `event.timeStamp` 와 그때의 경로를 넘긴다. */
    restored(timeStamp: number, pathname: string) {
      restorePathnames.set(timeStamp, pathname)
    },

    resolve(metric: MetricWithAttribution): NavigationRef {
      if (metric.navigationType === 'soft-navigation') {
        return {
          key: `soft:${metric.navigationId}`,
          kind: 'soft',
          startTime: metric.navigationStartTime ?? 0,
          pathname: pathnameOf(metric.navigationURL, initialPathname),
        }
      }
      if (metric.navigationType === 'back-forward-cache') {
        const start = metric.navigationStartTime ?? 0
        return {
          key: `restore:${start}`,
          kind: 'restore',
          startTime: start,
          pathname: restorePathnames.get(start) ?? initialPathname,
        }
      }
      return {
        key: 'hard',
        kind: 'hard',
        startTime: 0,
        pathname: pathnameOf(metric.navigationURL, initialPathname),
      }
    },
  }
}
