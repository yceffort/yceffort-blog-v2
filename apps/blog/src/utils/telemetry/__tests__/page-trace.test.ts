import type {ExportResult} from '@opentelemetry/core'
import {
  BasicTracerProvider,
  BatchSpanProcessor,
  InMemorySpanExporter,
  type ReadableSpan,
  type SpanExporter,
} from '@opentelemetry/sdk-trace-base'
import {describe, expect, it} from 'vitest'

import {createPageTrace} from '../page-trace'

/** export 호출 하나가 collector 로 가는 요청 하나다. 호출별로 span 을 모은다. */
class BatchRecorder implements SpanExporter {
  batches: ReadableSpan[][] = []
  export(spans: ReadableSpan[], done: (result: ExportResult) => void) {
    this.batches.push(spans)
    done({code: 0})
  }
  async shutdown() {}
}

const toMs = ([s, ns]: [number, number]) => s * 1000 + ns / 1e6

describe('createPageTrace', () => {
  it('root 와 첫 지표 span 이 같은 export 호출로 나간다', async () => {
    const exporter = new BatchRecorder()
    const provider = new BasicTracerProvider({
      spanProcessors: [new BatchSpanProcessor(exporter)],
    })
    const record = createPageTrace(provider.getTracer('test'), () => ({}))

    record('cwv.fcp', 120, {})
    await provider.forceFlush()

    expect(exporter.batches).toHaveLength(1)
    expect(exporter.batches[0].map((s) => s.name).sort()).toEqual([
      'cwv.fcp',
      'cwv.page',
    ])
  })

  it('지표 span 은 모두 같은 root 의 자식이고 길이가 지표 값이다', async () => {
    const exporter = new InMemorySpanExporter()
    const provider = new BasicTracerProvider({
      spanProcessors: [new BatchSpanProcessor(exporter)],
    })
    const record = createPageTrace(provider.getTracer('test'), () => ({
      'page.route': 'home',
    }))

    record('cwv.fcp', 120, {})
    record('cwv.lcp', 2400, {'cwv.lcp.finalized_by': 'quiet'})
    await provider.forceFlush()

    const spans = exporter.getFinishedSpans()
    const root = spans.find((s) => s.name === 'cwv.page')!
    const children = spans.filter((s) => s.name !== 'cwv.page')

    expect(spans.filter((s) => s.name === 'cwv.page')).toHaveLength(1)
    for (const child of children) {
      expect(child.spanContext().traceId).toBe(root.spanContext().traceId)
      expect(child.parentSpanContext?.spanId).toBe(root.spanContext().spanId)
      expect(child.attributes['page.route']).toBe('home')
    }
    const lcp = children.find((s) => s.name === 'cwv.lcp')!
    expect(toMs(lcp.duration)).toBeCloseTo(2400, 0)
  })
})
