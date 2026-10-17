import {
  type Attributes,
  type Context,
  ROOT_CONTEXT,
  trace,
  type Tracer,
} from '@opentelemetry/api'

import type {NavigationRef} from './navigation'

/**
 * 내비게이션 하나를 trace 하나로 묶는다. 지표 span 은 모두 그 내비게이션의 `cwv.page` 자식이다.
 * 첫 로드, SPA 안의 이동, bfcache 복원이 각각 따로 trace 가 된다.
 *
 * root 는 그 내비게이션의 첫 지표가 나올 때 만들고 그 자리에서 끝낸다. 첫 지표 span 도 같은 tick 에
 * 끝나므로 BatchSpanProcessor 의 같은 배치로 나간다. tail sampling 이 판정할 때 이 trace 에는
 * span 이 이미 두 개 있다. root 를 내비게이션 시작 때 따로 보내면 혼자 판정받아 trace 째 버려진다.
 *
 * span 길이는 지표 값이다. 그래서 TraceQL 에서 `duration` 분위수가 곧 지표 분위수가 된다.
 */
export function createPageTrace(tracer: Tracer) {
  const parents = new Map<string, Context>()

  return function record(
    navigation: NavigationRef,
    base: Attributes,
    name: string,
    durationMs: number,
    attributes: Attributes,
  ): void {
    let parent = parents.get(navigation.key)
    if (!parent) {
      const root = tracer.startSpan('cwv.page', {
        startTime: performance.timeOrigin + navigation.startTime,
        attributes: {'page.navigation': navigation.kind},
      })
      root.end()
      parent = trace.setSpan(ROOT_CONTEXT, root)
      parents.set(navigation.key, parent)
    }

    const end = performance.timeOrigin + performance.now()
    tracer
      .startSpan(
        name,
        {startTime: end - durationMs, attributes: {...base, ...attributes}},
        parent,
      )
      .end(end)
  }
}
