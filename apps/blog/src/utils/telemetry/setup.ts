import type {Histogram} from '@opentelemetry/api'
import {
  AggregationTemporalityPreference,
  OTLPMetricExporter,
} from '@opentelemetry/exporter-metrics-otlp-http'
import {OTLPTraceExporter} from '@opentelemetry/exporter-trace-otlp-http'
import {resourceFromAttributes} from '@opentelemetry/resources'
import {MeterProvider} from '@opentelemetry/sdk-metrics'
import {
  BasicTracerProvider,
  BatchSpanProcessor,
} from '@opentelemetry/sdk-trace-base'
import type {Metric} from 'web-vitals'

import {toAttributes} from './attributes'
import {BUCKETS} from './buckets'
import {pageContextFor, type PageContext, readDeviceContext} from './context'
import {FlushOnlyMetricReader} from './flush-reader'
import {toLabels} from './labels'
import {createNavigationTracker, type NavigationRef} from './navigation'
import {createPageTrace} from './page-trace'
import {observeVitals} from './vitals'

interface Options {
  collectorUrl: string
}

/** CLS 는 점수라서 span 길이로 넣을 때 1000 을 곱한다. TraceQL 의 초 단위 결과가 그대로 CLS 가 된다. */
function spanDurationMs(metric: {name: Metric['name']; value: number}): number {
  return metric.name === 'CLS' ? metric.value * 1000 : metric.value
}

export function startTelemetry({collectorUrl}: Options): void {
  const device = readDeviceContext()
  const navigations = createNavigationTracker(window.location.pathname)
  // 복원 지표의 키와 시작 시각은 web-vitals 가 넣어 주는 navigationStartTime(pageshow 의 timeStamp)에서 읽고,
  // 경로만 여기서 같은 timeStamp 로 저장한다. web-vitals 도 window 에 capture 로 pageshow 를 걸고 TTFB 는 그 안에서
  // 바로 보고하므로, 먼저 등록하는 이 리스너도 capture 로 걸어 같은 타깃에서 먼저 돌게 한다.
  window.addEventListener(
    'pageshow',
    (event) => {
      if (event.persisted)
        navigations.restored(event.timeStamp, window.location.pathname)
    },
    true,
  )
  const contexts = new Map<string, PageContext>()
  const contextOf = (navigation: NavigationRef) => {
    let context = contexts.get(navigation.key)
    if (!context) {
      context = pageContextFor(navigation, device)
      contexts.set(navigation.key, context)
    }
    return context
  }
  const resource = resourceFromAttributes({'service.name': 'yceffort-blog'})

  const tracerProvider = new BasicTracerProvider({
    resource,
    spanProcessors: [
      new BatchSpanProcessor(
        new OTLPTraceExporter({url: `${collectorUrl}/v1/traces`}),
        // 브라우저용 processor 는 기본으로 document 의 visibilitychange 와 pagehide 에 자기 flush 를 건다.
        // flush 시점은 아래 visibilitychange 리스너 하나에서만 정하도록 끈다. 주기 export(5초)는 그대로다.
        {disableAutoFlushOnDocumentHide: true},
      ),
    ],
  })
  const meterProvider = new MeterProvider({
    resource,
    readers: [
      new FlushOnlyMetricReader(
        new OTLPMetricExporter({
          url: `${collectorUrl}/v1/metrics`,
          // 브라우저는 수명이 짧고 탭마다 따로 센다. 누적값을 들고 있을 주체가 없어 delta 로 보낸다.
          temporalityPreference: AggregationTemporalityPreference.DELTA,
        }),
      ),
    ],
  })

  const meter = meterProvider.getMeter('yceffort-blog/cwv')
  const histograms = Object.fromEntries(
    (Object.keys(BUCKETS) as Metric['name'][]).map((name) => [
      name,
      meter.createHistogram(`cwv.${name.toLowerCase()}`, {
        unit: name === 'CLS' ? '1' : 'ms',
        advice: {explicitBucketBoundaries: BUCKETS[name]},
      }),
    ]),
  ) as Record<Metric['name'], Histogram>

  const record = createPageTrace(tracerProvider.getTracer('yceffort-blog/cwv'))

  const flush = () => {
    void tracerProvider.forceFlush()
    void meterProvider.forceFlush()
  }
  // web-vitals 는 window 에서 capture 로 받아 CLS, INP 를 먼저 보고한다. 여기는 그 뒤에 불린다.
  // metric 은 이 flush 에서만 나간다(FlushOnlyMetricReader). span 은 주기 export 와 이 flush 로 나간다.
  //
  // pagehide 에서는 flush 하지 않는다. 문서를 떠날 때 pagehide 가 visibilitychange 보다 먼저 와서,
  // pagehide 에서 보내면 visibilitychange 에서 막 기록한 CLS, INP 가 빠진다
  // (blog-experiments 의 cwv-otel-grafana/experiments/e7-unload-flush).
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'hidden') flush()
  })

  observeVitals(
    (metric, extra) => {
      const navigation = navigations.resolve(metric)
      const context = contextOf(navigation)
      record(
        navigation,
        context.attributes,
        `cwv.${metric.name.toLowerCase()}`,
        spanDurationMs(metric),
        {...toAttributes(metric), ...extra},
      )
      histograms[metric.name].record(
        metric.value,
        toLabels(metric, context.labels),
      )
    },
    // 같은 id 로 다시 온 CLS, INP 는 histogram 에 넣지 않고 trace 에만 남긴다.
    (metric, first, index) => {
      const navigation = navigations.resolve(metric)
      record(
        navigation,
        contextOf(navigation).attributes,
        `cwv.${metric.name.toLowerCase()}.update`,
        spanDurationMs(metric),
        {
          ...toAttributes(metric),
          'cwv.update.index': index,
          'cwv.update.first_value':
            metric.name === 'CLS' ? first.value : Math.round(first.value),
          'cwv.update.first_rating': first.rating,
          'cwv.update.rating_changed': first.rating !== metric.rating,
        },
      )
    },
    (metric) => navigations.resolve(metric).key,
  )
}
