import type {Histogram} from '@opentelemetry/api'
import {
  AggregationTemporalityPreference,
  OTLPMetricExporter,
} from '@opentelemetry/exporter-metrics-otlp-http'
import {OTLPTraceExporter} from '@opentelemetry/exporter-trace-otlp-http'
import {resourceFromAttributes} from '@opentelemetry/resources'
import {
  MeterProvider,
  PeriodicExportingMetricReader,
} from '@opentelemetry/sdk-metrics'
import {
  BasicTracerProvider,
  BatchSpanProcessor,
} from '@opentelemetry/sdk-trace-base'
import type {Metric} from 'web-vitals'

import {toAttributes} from './attributes'
import {BUCKETS} from './buckets'
import {pageContextFor, type PageContext, readDeviceContext} from './context'
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
  // web-vitals 의 bfcache 처리보다 먼저 복원을 센다. web-vitals 는 복원 뒤 다음 프레임에 보고한다.
  window.addEventListener('pageshow', (event) => {
    if (event.persisted)
      navigations.restored(event.timeStamp, window.location.pathname)
  })
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
      ),
    ],
  })
  const meterProvider = new MeterProvider({
    resource,
    readers: [
      new PeriodicExportingMetricReader({
        exporter: new OTLPMetricExporter({
          url: `${collectorUrl}/v1/metrics`,
          // 브라우저는 수명이 짧고 탭마다 따로 센다. 누적값을 들고 있을 주체가 없어 delta 로 보낸다.
          temporalityPreference: AggregationTemporalityPreference.DELTA,
        }),
        exportIntervalMillis: 10_000,
      }),
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
  //
  // pagehide 에서는 flush 하지 않는다. 문서를 떠날 때 pagehide 가 visibilitychange 보다 먼저 온다.
  // sdk-metrics 2.9.0 부터 forceFlush 는 진행 중인 export 의 응답을 기다린 뒤에 다시 collect 한다.
  // pagehide 에서 보낸 요청의 응답이 오기 전에 문서가 사라지므로, visibilitychange 에서 막 기록한
  // CLS, INP 는 나가지 못한다(blog-experiments 의 cwv-otel-grafana/experiments/e7-unload-flush).
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
    flush,
    (metric) => navigations.resolve(metric).key,
  )
}
