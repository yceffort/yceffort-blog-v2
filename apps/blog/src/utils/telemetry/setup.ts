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
import {toLabels} from './labels'
import {createPageTrace} from './page-trace'
import type {Route} from './routes'
import {observeVitals} from './vitals'

interface Options {
  collectorUrl: string
  /** 메트릭 라벨에 쓰는 화면 이름. URL 을 그대로 쓰지 않고 `routeOf` 로 줄여서 넘긴다. */
  route: Route
}

type NavigatorExtras = Navigator & {
  connection?: {effectiveType?: string}
  deviceMemory?: number
}

function baseAttributes(route: Route) {
  const nav = navigator as NavigatorExtras
  return {
    'page.route': route,
    'network.effective_type': nav.connection?.effectiveType ?? 'unknown',
    'device.memory': nav.deviceMemory ?? -1,
    'device.cores': nav.hardwareConcurrency ?? -1,
  }
}

/** CLS 는 점수라서 span 길이로 넣을 때 1000 을 곱한다. TraceQL 의 초 단위 결과가 그대로 CLS 가 된다. */
function spanDurationMs(metric: {name: Metric['name']; value: number}): number {
  return metric.name === 'CLS' ? metric.value * 1000 : metric.value
}

export function startTelemetry({collectorUrl, route}: Options): void {
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

  const record = createPageTrace(
    tracerProvider.getTracer('yceffort-blog/cwv'),
    () => baseAttributes(route),
  )

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

  observeVitals((metric, extra) => {
    record(`cwv.${metric.name.toLowerCase()}`, spanDurationMs(metric), {
      ...toAttributes(metric),
      ...extra,
    })
    histograms[metric.name].record(metric.value, toLabels(metric, route))
  }, flush)
}
