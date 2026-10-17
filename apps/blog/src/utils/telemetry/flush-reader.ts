import {MetricReader, type PushMetricExporter} from '@opentelemetry/sdk-metrics'

/**
 * forceFlush 할 때만 collect 하고 export 한다. 주기 export 는 없다.
 *
 * PeriodicExportingMetricReader(sdk-metrics 2.9.0 부터)의 forceFlush 는 진행 중인 export 의 응답을
 * 기다린 뒤에 collect 한다. 주기 export 가 나가 있는 동안 hidden 이 오면 막 기록한 CLS, INP 는 그 응답이
 * 올 때까지 collect 되지 않고, 그 사이 문서가 사라지면 나가지 못한다. 일반 브라우저는 떠날 때 hidden 이
 * 오므로 hidden 에서만 보내고, 앞선 export 가 끝나기를 기다리지 않는다. delta 라서 collect 가 겹쳐도
 * 같은 값을 두 번 보내지 않는다.
 */
export class FlushOnlyMetricReader extends MetricReader {
  constructor(private readonly exporter: PushMetricExporter) {
    super({
      aggregationSelector: exporter.selectAggregation?.bind(exporter),
      aggregationTemporalitySelector:
        exporter.selectAggregationTemporality?.bind(exporter),
    })
  }

  protected async onForceFlush(): Promise<void> {
    const {resourceMetrics} = await this.collect()
    if (resourceMetrics.scopeMetrics.length === 0) return
    await new Promise<void>((resolve) =>
      this.exporter.export(resourceMetrics, () => resolve()),
    )
  }

  protected async onShutdown(): Promise<void> {
    await this.exporter.shutdown()
  }
}
