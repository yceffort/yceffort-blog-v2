import {
  AggregationTemporality,
  MeterProvider,
  type PushMetricExporter,
  type ResourceMetrics,
} from '@opentelemetry/sdk-metrics'
import {describe, expect, it} from 'vitest'

import {FlushOnlyMetricReader} from '../flush-reader'

/** 응답(콜백)을 돌려주지 않고 쥐고 있는 exporter. 떠나는 문서처럼 응답이 오기 전 상태를 만든다. */
function stalledExporter() {
  const exports: Array<Record<string, number>> = []
  const pending: Array<() => void> = []
  const exporter: PushMetricExporter = {
    export(metrics: ResourceMetrics, done) {
      const counts = metrics.scopeMetrics
        .flatMap((s) => s.metrics)
        .map((m) => [
          m.descriptor.name,
          m.dataPoints.reduce(
            (n, p) => n + (p.value as {count: number}).count,
            0,
          ),
        ])
      exports.push(Object.fromEntries(counts))
      pending.push(() => done({code: 0}))
    },
    forceFlush: async () => {},
    shutdown: async () => {},
    selectAggregationTemporality: () => AggregationTemporality.DELTA,
  }
  const respond = () => pending.splice(0).forEach((p) => p())
  return {exporter, exports, respond}
}

const tick = () => new Promise((resolve) => setTimeout(resolve, 0))

function setup() {
  const stalled = stalledExporter()
  const provider = new MeterProvider({
    readers: [new FlushOnlyMetricReader(stalled.exporter)],
  })
  return {...stalled, provider, meter: provider.getMeter('test')}
}

describe('FlushOnlyMetricReader', () => {
  it('앞선 export 의 응답을 기다리지 않고 다음 flush 를 보낸다', async () => {
    const {provider, meter, exports, respond} = setup()

    meter.createHistogram('cwv.ttfb').record(100)
    const first = provider.forceFlush()
    await tick()
    meter.createHistogram('cwv.cls').record(0.1)
    const second = provider.forceFlush()
    await tick()

    // 첫 응답이 오기 전에 두 번째 export 가 이미 나갔고, 각 export 에는 그사이 기록한 값만 있다.
    expect(exports).toEqual([{'cwv.ttfb': 1}, {'cwv.cls': 1}])

    respond()
    await Promise.all([first, second])
  })

  it('flush 하지 않으면 보내지 않고, 기록한 것이 없으면 flush 해도 보내지 않는다', async () => {
    const {provider, meter, exports} = setup()

    await provider.forceFlush()
    meter.createHistogram('cwv.lcp').record(1800)
    await new Promise((resolve) => setTimeout(resolve, 50))

    expect(exports).toEqual([])
  })
})
