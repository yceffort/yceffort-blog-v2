---
title: 'Sending Core Web Vitals <em>at the Moment of Leaving the Page</em>'
tags:
  - web-performance
  - debugging
  - browser
published: true
date: 2026-10-18 18:10:00
description: 'Traced why CLS and INP go missing when the page is left, through the transport of OTel JS and the order in which flush runs.'
series: 'Core Web Vitals 직접 수집하기'
seriesOrder: 2
art:
  undraw: mail-sent
---

## Table of Contents

## Flushing on pagehide Dropped CLS and INP

I built a reproduction environment that collects Core Web Vitals (CWV from here on) with OpenTelemetry JS (OTel JS from here on) 2.12.0. Following the internal guide, it called flush in two places so that any remaining metrics were sent right away: when the page is hidden (`visibilitychange` to hidden) and on `pagehide`. TTFB, FCP and LCP arrived, but CLS and INP did not. There were no errors in the browser console or in the collector logs.

The counts come from page loads that leave before the 10-second periodic export runs. Each page load pressed a button once and left 1.5 seconds later, and I checked how much each metric's count in Mimir grew per page load. As [part 1](/2026/10/web-vitals-report-timing) showed, web-vitals 6.2.3 defers the processing of interaction entries until idle or a 1-second timeout, and leaving before that means INP is never reported. To keep that apart from a loss in transport, the page left after the timeout had passed.

| flush timing                  | TTFB | FCP | LCP | INP   | CLS   |
| ----------------------------- | ---- | --- | --- | ----- | ----- |
| pagehide and visibilitychange | 1    | 1   | 1   | **0** | **0** |
| visibilitychange only         | 1    | 1   | 1   | 1     | 1     |

> Each mode was run twice with the same result. In both modes the same page load's trace reached Tempo with all five spans. The measurement code is `run.mjs` in `cwv-otel-grafana/experiments/e7-unload-flush`.

Removing the single line that flushes on pagehide made all five metrics arrive. With OTel JS 1.30.1, which the internal code used, the same flush code loses nothing. But 1.30.1 had a problem of its own: it reported failed sends as successes. The transport and the flush behavior of each version had to be checked together.

Code that sends CWV splits into two parts: one that collects values and exports them periodically, and one that pushes out whatever is left when the page is being left. In OTel JS, the `PeriodicExportingMetricReader` for metrics and the `BatchSpanProcessor` for traces collect and export, and the collection code calls `forceFlush` at the moment of leaving.

When leaving, the request has to keep going after the document is gone, so it is sent with [`navigator.sendBeacon`](https://developer.mozilla.org/en-US/docs/Web/API/Navigator/sendBeacon) or `fetch` with [`keepalive`](https://developer.mozilla.org/en-US/docs/Web/API/RequestInit#keepalive). Which one is used depends on the version of the OTel JS exporter. The collection code also has to decide which event to flush in, [`pagehide`](https://developer.mozilla.org/en-US/docs/Web/API/Window/pagehide_event) or the hidden state of [`visibilitychange`](https://developer.mozilla.org/en-US/docs/Web/API/Document/visibilitychange_event). I checked how each of these two affected the result above.

> The internal work ran from late August to mid October 2026 in an internal verification environment. Every number in this post was re-measured in the public reproduction environment ([blog-experiments/cwv-otel-grafana](https://github.com/yceffort/blog-experiments/tree/cwv-otel-grafana-v1/cwv-otel-grafana)).
>
> The measurement code for the exporter and the event order (`e2-exporter`, `e3-lifecycle`) is at the tag `cwv-otel-grafana-v1`. The code for the first table (`e7-unload-flush`) was changed to leave 1.5 seconds after the click and to count arrivals in Tempo as well, and was re-run on 2026-10-18 at commit [`e4e48c6`](https://github.com/yceffort/blog-experiments/tree/e4e48c6de625e90ff3c9b8931e2f40381481d996/cwv-otel-grafana/experiments/e7-unload-flush). The code for the overlap with the periodic export (`e14-inflight-export`) is at commit [`6cafea7`](https://github.com/yceffort/blog-experiments/tree/6cafea7dc5a1fbb6dc45fbce2d9393d2c65785ed/cwv-otel-grafana/experiments/e14-inflight-export) and was measured the same day.
>
> The reproduction environment runs web-vitals 6.2.3, OTel JS 2.12.0 (OTLP exporter 0.223.0), OpenTelemetry Collector contrib 0.162.0 (0.163.1 for the e7 re-run), Mimir 3.2.2 and Tempo 3.1.0. OTel JS 1.30.1 (exporter 0.57.2) was used alongside for comparison, and the browser is Chromium 156 from Playwright 1.64.1. Every quoted source is pinned to the tag of these versions.

## Recording the Same Measurement as a Trace and a Metric

The collection code records a trace and a metric from a single callback of the web-vitals attribution build, the build that supplies the data for root-cause analysis. Next.js's [`useReportWebVitals`](https://nextjs.org/docs/app/api-reference/functions/use-report-web-vitals) measures with the copy of web-vitals that Next.js bundles itself (4.2.1 as of 16.3.5). Mixing that hook with a separately installed web-vitals gives different values and `id`s for the same page load. To join the values in the two stores by `id`, both had to come from the same measurement.

One page load is recorded as one trace, with one span added per metric. The span's length is set to the metric's value, so TraceQL's `quantile_over_time(duration, .75)` gives the metric's p75. CLS is a score, so it is stored as the score times 1000 in milliseconds, and the value TraceQL returns in seconds equals the original CLS score. The span attributes carry every attribution value. For LCP that is the element selector, the resource URL and the time split into the four LCP phases.

```ts
// cwv-otel-grafana/demo-web/src/telemetry/page-trace.ts
export function createPageTrace(tracer: Tracer, getBaseAttributes: () => Attributes) {
  let parent: Context | undefined;
  let base: Attributes = {};

  return function record(name: string, durationMs: number, attributes: Attributes): void {
    if (!parent) {
      const root = tracer.startSpan('cwv.page', { startTime: performance.timeOrigin });
      root.end();
      parent = trace.setSpan(ROOT_CONTEXT, root);
      base = getBaseAttributes();
    }

    const end = performance.timeOrigin + performance.now();
    tracer
      .startSpan(name, { startTime: end - durationMs, attributes: { ...base, ...attributes } }, parent)
      .end(end);
  };
}
```

```ts
// cwv-otel-grafana/demo-web/src/telemetry/setup.ts
/** CLS 는 점수라서 span 길이로 넣을 때 1000 을 곱한다. TraceQL 의 초 단위 결과가 그대로 CLS 가 된다. */
function spanDurationMs(metric: { name: Metric['name']; value: number }): number {
  return metric.name === 'CLS' ? metric.value * 1000 : metric.value;
}
```

A metric span starts at the report time minus the value and ends at the report time. Its length therefore equals the metric value, but its start is not when the metric actually started. What can be trusted in this trace is the length and the attributes of each span, not its position on the timeline. The root span `cwv.page` is created when the first metric arrives and ended on the spot. Why it is not sent on its own when the page opens comes up with tail sampling in part 3.

Metrics are one histogram per metric. In the browser, every tab counts on its own and the collection code ends with the page, so nothing can keep holding a cumulative value. So it sends delta temporality, the increment since the previous export, and leaves the conversion to cumulative to the collector. The bucket boundaries include the good and poor thresholds of web-vitals.

```ts
// cwv-otel-grafana/demo-web/src/telemetry/buckets.ts
export const BUCKETS: Record<Metric['name'], number[]> = {
  TTFB: [100, 200, 400, 600, 800, 1000, 1400, 1800, 2500, 4000],
  FCP: [400, 800, 1200, 1500, 1800, 2400, 3000, 4000, 6000],
  LCP: [500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 7000, 10000],
  INP: [50, 100, 150, 200, 300, 400, 500, 750, 1000, 2000],
  CLS: [0.01, 0.025, 0.05, 0.1, 0.15, 0.25, 0.5, 1],
};
```

With 2500 as a boundary, the single `le="2500"` bucket divided by the total count is the exact good ratio, with no interpolation. Values between boundaries are estimated by linear interpolation inside a bucket, so the ranges where p75 usually lands are denser. Labels are limited to values with a fixed set of possibilities: the screen name (`page.route`), the navigation type, the rating and the INP input type. Each label combination is one series, so putting a selector in a label would grow the number of series by the number of elements on the screen.

```ts
// cwv-otel-grafana/demo-web/src/telemetry/setup.ts
  const histograms = Object.fromEntries(
    (Object.keys(BUCKETS) as Metric['name'][]).map((name) => [
      name,
      meter.createHistogram(`cwv.${name.toLowerCase()}`, {
        unit: name === 'CLS' ? '1' : 'ms',
        advice: { explicitBucketBoundaries: BUCKETS[name] },
      }),
    ]),
  ) as Record<Metric['name'], Histogram>;
```

The bucket boundaries are passed as `advice` when the histogram is created. Without them, sdk-metrics uses its default boundaries (15 values from 0 to 10000, the same for every metric), which do not include thresholds like 200 for INP or 0.1 for CLS. The web-vitals callback does both recordings at once.

```ts
// cwv-otel-grafana/demo-web/src/telemetry/setup.ts
  observeVitals((metric, extra) => {
    record(`cwv.${metric.name.toLowerCase()}`, spanDurationMs(metric), { ...toAttributes(metric), ...extra });
    histograms[metric.name].record(metric.value, toLabels(metric, route));
  }, flush);
```

`observeVitals` attaches this callback to each metric's `on*` function, passing each `id` through only once.

```ts
// cwv-otel-grafana/demo-web/src/telemetry/vitals.ts
export function oncePerId(report: Report): Report {
  const seen = new Set<string>();

  return (metric, extra) => {
    const key = `${metric.name}:${metric.id}`;
    if (seen.has(key)) return;
    seen.add(key);
    report(metric, extra);
  };
}
```

As part 1 showed, CLS and INP are reported again with the same `id` every time the page becomes visible and is hidden again. A histogram counts every call to `record`, so left alone, one page load would count as two. Only the first report is counted, and a value that got worse later is not included. Measuring the cost of that choice is part 4's story.

Metrics are exported every 10 seconds by `PeriodicExportingMetricReader`, and traces are batched and exported by `BatchSpanProcessor`. In both cases, if the page is left while waiting for the next interval, whatever remains has to be flushed by hand.

## CLS and INP Are Reported on hidden

web-vitals does not report its five metrics at the same moment. Without `reportAllChanges`, they are reported as follows.

| Metric | Reported                                                   | Source (v6.2.3) |
| ------ | ---------------------------------------------------------- | --------------- |
| TTFB   | Once, after `load`                                         | `src/onTTFB.ts` |
| FCP    | Once, at the first contentful paint                        | `src/onFCP.ts`  |
| LCP    | On idle after the first `keydown` or `click`, or on hidden | `src/onLCP.ts`  |
| CLS    | On hidden                                                  | `src/onCLS.ts`  |
| INP    | On hidden                                                  | `src/onINP.ts`  |

Part 1 followed each report timing through the source. What matters for the sending code is that CLS and INP need a hidden event to be reported. On a page that reported FCP, CLS is reported then, and on a page with a processed interaction, INP is reported then. On a page left without any input, LCP is also first reported at the moment of leaving. With no interaction at all, INP never moves from its initial -1 and is not reported, and in 6.2.3 it is also not reported when the page is left within 1 second of the last interaction. Values first reported at departure cannot have gone out with an earlier periodic export, so if the final send fails they are simply missing from the collected data.

## sendBeacon in 1.30.1 Reports Failure as Success

The internal code used OTel JS 1.30.1 with OTLP exporter 0.57.2. The browser exporter of that version sends with `navigator.sendBeacon` when there is no `headers` option and with XHR when there is[^4]. In the reproduction environment I put a fake collector on another origin, varied its response, and looked at what the exporter reported.

| exporter                          | collector returns 200 | 400         | CORS rejection                      |
| --------------------------------- | --------------------- | ----------- | ----------------------------------- |
| 0.57.2, no `headers` (sendBeacon) | success               | **success** | **success** (server received 0)     |
| 0.57.2, `headers: {}` (XHR)       | success               | failed      | failed                              |
| 0.223.0 (keepalive fetch)         | success               | failed      | failed after 5 attempts (about 8 s) |

> Each combination was run twice with the same result. The measurement code is in `cwv-otel-grafana/experiments/e2-exporter`.

The first row is written out in the exporter's transport code[^4].

```ts
// opentelemetry-js experimental/v0.57.2 experimental/packages/otlp-exporter-base/src/transport/send-beacon-transport.ts
  send(data: Uint8Array): Promise<ExportResponse> {
    return new Promise<ExportResponse>(resolve => {
      if (
        navigator.sendBeacon(
          this._params.url,
          new Blob([data], { type: this._params.blobType })
        )
      ) {
        // no way to signal retry, treat everything as success
        diag.debug('SendBeacon success');
        resolve({
          status: 'success',
        });
      } else {
        resolve({
          status: 'failure',
          error: new Error('SendBeacon failed'),
        });
      }
    });
  }
```

All that sendBeacon's return value tells you is whether the request made it into the browser's queue. No response ever comes back, so, as the comment says, there is nothing to base a retry on, and every queued request becomes a success. A 400 response and a CORS rejection both get queued, so they look like successes, as in the first row of the table. Because 0.57.2 reports a queued request as success, the exporter returns success even when CORS blocked the request and the server received nothing. The internal setup sent with XHR normally so that failures stayed visible, and switched to sendBeacon only when the page was hidden. Even so, the result of the send at the moment of leaving could not be known.

## The Current Exporter Sends with fetch keepalive

The branch is gone in current versions. From 0.204.0, fetch is preferred over XHR when `headers` is present, 0.211.0 removed the XHR transport, and 0.213.0 replaced sendBeacon with keepalive fetch as well[^1]. keepalive fetch keeps the request going after the document is gone and still returns the response status. That is why 0.223.0 reported the 400 as failed right away in the table above.

A CORS rejection is treated as a network error and retried. The retry interval and count come from constants and a loop in the transport code[^5].

```ts
// opentelemetry-js experimental/v0.223.0 experimental/packages/otlp-exporter-base/src/retrying-transport.ts
const MAX_ATTEMPTS = 5;
const INITIAL_BACKOFF = 1000;
const MAX_BACKOFF = 5000;
const BACKOFF_MULTIPLIER = 1.5;
const JITTER = 0.2;
```

```ts
// opentelemetry-js experimental/v0.223.0 experimental/packages/otlp-exporter-base/src/retrying-transport.ts
    while (result.status === 'retryable' && attempts > 0) {
      attempts--;

      // use maximum of computed backoff and 0 to avoid negative timeouts
      const backoff = Math.max(
        Math.min(nextBackoff * (1 + getJitter()), MAX_BACKOFF),
        0
      );
      nextBackoff = nextBackoff * BACKOFF_MULTIPLIER;
      const retryInMillis = result.retryInMillis ?? backoff;

      // return when expected retry time is after the export deadline.
      const remainingTimeoutMillis = deadline - Date.now();
      if (retryInMillis > remainingTimeoutMillis) {
        diag.info(
          `Export retry time ${Math.round(retryInMillis)}ms exceeds remaining timeout ${Math.round(
            remainingTimeoutMillis
          )}ms, not retrying further.`
        );
        return result;
      }
```

The interval starts at 1 second, grows 1.5x each time with ±20% jitter, and is capped at 5 seconds. Waiting 1, 1.5, 2.25 and 3.375 seconds for 4 retries takes about 8.1 seconds, and the fifth interval of 5 seconds is longer than what is left of the 10-second export limit, so the loop ends there. The table's 5 attempts including the first, and the roughly 8 seconds, match this arithmetic. What ends the retries is the export limit, which arrives before the attempt count. The 10 seconds is the exporter's shared configuration default, `timeoutMillis: 10000`[^9].

The [Fetch specification](https://fetch.spec.whatwg.org/#http-network-or-cache-fetch) allows at most 64KiB of body across all in-flight keepalive requests combined. The fetch transport source in 0.223.0 also notes that Chrome accepts at most 9 keepalive requests per renderer process. That quotes a Chromium developer's explanation in [whatwg/fetch#679](https://github.com/whatwg/fetch/issues/679), and I did not confirm it in the Chromium source. The exporter keeps count of the cumulative size and the number of in-flight keepalive requests, and sends without keepalive when a request would go over 60KiB (leaving headroom for headers) or 9 requests[^5].

```ts
// opentelemetry-js experimental/v0.223.0 experimental/packages/otlp-exporter-base/src/transport/fetch-transport.ts
/**
 * Maximum total body size for concurrent keepalive requests.
 * Browsers enforce a 64KiB cumulative limit across all pending keepalive requests.
 * We use 60KB to leave headroom for headers.
 * @see https://github.com/whatwg/fetch/issues/679
 * @see https://blog.huli.tw/2025/01/06/en/navigator-sendbeacon-64kib-and-source-code/
 */
const MAX_KEEPALIVE_BODY_SIZE = 60 * 1024;

/**
 * Maximum concurrent keepalive requests.
 * Chrome enforces 9 concurrent keepalive fetch requests per renderer process.
 * @see https://github.com/whatwg/fetch/issues/679
 * Quote: "If the renderer process is processing more than 9 requests with keepalive set, we reject a new request"
 */
const MAX_KEEPALIVE_REQUESTS = 9;
```

```ts
// opentelemetry-js experimental/v0.223.0 experimental/packages/otlp-exporter-base/src/transport/fetch-transport.ts
    const requestSize = data.byteLength;

    // Determine if we can use keepalive based on cumulative browser limits.
    // We must check BEFORE adding to pending totals to avoid exceeding limits.
    const wouldExceedSize =
      pendingBodySize + requestSize > MAX_KEEPALIVE_BODY_SIZE;
    const wouldExceedCount = pendingKeepaliveCount >= MAX_KEEPALIVE_REQUESTS;
    const useKeepalive = !wouldExceedSize && !wouldExceedCount;
```

The two variables `pendingBodySize` and `pendingKeepaliveCount` live at module level, so the trace exporter and the metric exporter on the same page share the budget. A request over the limit goes out with keepalive dropped and no error raised. A request sent without keepalive is cancelled together with the document, so if requests pile up at the moment of leaving, this limit can become another point of loss. In the reproduction environment the requests at departure were one trace request (2,987B) and one metric request (1,327B), about 4.3KB together. That is 7% of 60KiB and nowhere near the count limit either (measured along with the overlap experiment below). The size depends on the selectors and URLs carried in span attributes, so it differs from site to site.

Within these limits, the request keeps going after the document is gone and whether it failed is visible, which is the advantage. But CLS and INP still did not arrive in the reproduction environment with the current exporter, so the next thing to check was the order in which flush runs.

## pagehide Comes Before visibilitychange

OTel JS's BatchSpanProcessor registers two listeners to export the remaining spans when the page is hidden. The 2.12.0 code is as follows[^2].

```ts
// opentelemetry-js v2.12.0 packages/sdk-trace/src/platform/browser/export/BatchSpanProcessor.ts
document.addEventListener(
  'visibilitychange',
  this._visibilityChangeListener
);

// use 'pagehide' event as a fallback for Safari; see https://bugs.webkit.org/show_bug.cgi?id=116769
document.addEventListener('pagehide', this._pageHideListener);
```

WebKit bug [116769](https://bugs.webkit.org/show_bug.cgi?id=116769), the one the comment points at, is about the visibility state of a document in the page cache being wrong, and MDN's browser compatibility data ([browser-compat-data](https://github.com/mdn/browser-compat-data), BCD from here on) 8.1.6 cites that bug and three others as the grounds for stating that before Safari 14.1 (iOS 14.5) `visibilitychange` did not fire when navigating away from a document[^6]. The internal guide's instruction to flush in both hidden and pagehide probably came from the same place. But `pagehide` is an event that fires on Window. I recorded in sessionStorage which listeners fired in which order when following a link in the same tab, and read it back from the next document.

```text
window.capture.beforeunload:visible
window.beforeunload:visible
window.capture.pagehide:visible
window.pagehide:visible
window.capture.visibilitychange:hidden
document.visibilitychange:hidden
window.visibilitychange:hidden
window.capture.unload:hidden
window.unload:hidden
```

> Run twice in Chromium 156 with the same result. `cwv-otel-grafana/experiments/e3-lifecycle`.

The `pagehide` listener registered on `document` never fired. By the spec, pagehide fires on Window and does not pass through document. So BatchSpanProcessor's Safari fallback listener most likely does nothing in other browsers either, though Chromium is the only browser I measured.

The events came in the order `pagehide`, then `visibilitychange`. At pagehide the document is still visible, and web-vitals reports CLS and INP at the `visibilitychange` that follows. Flushing in each of the two events means one send attempt before CLS and INP are recorded and another one after.

## From 2.9.0, forceFlush Waits for the Export in Progress

It looks like the second flush should simply send CLS and INP, but sdk-metrics' `forceFlush` waits until any export already in progress has finished[^3]. At the moment of leaving the page, that wait can keep the freshly recorded values from being sent.

```ts
// opentelemetry-js v2.12.0 packages/sdk-metrics/src/export/PeriodicExportingMetricReader.ts
protected async onForceFlush(): Promise<void> {
  // Wait for any in-progress export to finish first so that we never run
  // collect + export concurrently with it.
  await this._awaitOngoingExport();
  // forceFlush SHOULD collect and export the latest metrics. If a concurrent
  // caller already started a fresh export while we were waiting above, await
  // that one instead of starting yet another collect + export cycle;
  // otherwise run our own.
  if (this._ongoingExportPromise) {
    await this._awaitOngoingExport();
  } else {
    await this._runOnce();
  }
  await this._exporter.forceFlush();
}
```

The flush called from pagehide sends the first request, carrying TTFB, FCP and LCP. Then in visibilitychange web-vitals records CLS and INP and flush is called again, and that call waits in `_awaitOngoingExport()` until the first request's response arrives. No collect happens while it waits, so CLS and INP are not in any request yet.

To see the wait itself, I kept the page alive, dispatched `pagehide` and `visibilitychange` as separate tasks 50ms apart, and delayed the first metric request's response by 2 seconds with Playwright (`split-task.mjs`). The first request, sent from `pagehide`, carried TTFB, FCP and LCP, and the second carried CLS and INP. The second request went out only about 2ms after the first request's response had arrived (2.0ms and 1.9ms in two runs). The trace flush called from the same `visibilitychange` did not wait, and the request carrying the CLS and INP spans went out 3ms after the event.

When the page is actually left, the document appears to disappear while that first response is awaited, so the second collect never runs. The first request, already started, keeps going as a keepalive fetch, so only TTFB, FCP and LCP arrive. The first table, where only some of the metrics arrived while every trace did, is explained by the same difference.

The wait was added in 2.9.0 (#6655). In 2.8.0, the version right before it, `onForceFlush` is three lines[^8].

```ts
// opentelemetry-js v2.8.0 packages/sdk-metrics/src/export/PeriodicExportingMetricReader.ts
  protected async onForceFlush(): Promise<void> {
    await this._runOnce();
    await this._exporter.forceFlush();
  }
```

The `onForceFlush` of sdk-metrics 1.30.1, which the internal code used, has no such wait either. So with the same flush code, upgrading sdk-metrics to 2.9.0 or later loses CLS and INP under the condition reproduced above. 2.12.0 keeps the same guarantee on the periodic export path too, but in a different way: when an export is in progress, that interval's run is skipped[^3].

```ts
// opentelemetry-js v2.12.0 packages/sdk-metrics/src/export/PeriodicExportingMetricReader.ts
  private async _doRun(): Promise<void> {
    if (this._ongoingExportPromise) {
      api.diag.debug(
        'PeriodicExportingMetricReader: export already in progress, skipping'
      );
      return;
    }
```

The periodic export skips, and `forceFlush` waits. The guarantee that collect and export never run at the same time feels natural in a server process, I think. A skipped interval comes around again, and forceFlush can wait and then gather the latest values. But a document about to be torn down has neither a next interval nor time to wait. While it waits for the response, the next collect and send never run.

Traces are different. `BatchSpanProcessor`'s `forceFlush` only splits the queued spans into batches and exports them right away, without looking at any export in progress[^7].

```ts
// opentelemetry-js v2.12.0 packages/sdk-trace/src/export/BatchSpanProcessorBase.ts
  forceFlush(): Promise<void> {
    if (this._shutdownOnce.isCalled) {
      return this._shutdownOnce.promise;
    }
    return this._flushAll();
  }
```

```ts
// opentelemetry-js v2.12.0 packages/sdk-trace/src/export/BatchSpanProcessorBase.ts
  private _flushAll(): Promise<void> {
    return new Promise((resolve, reject) => {
      const promises = [];
      // calculate number of batches
      const count = Math.ceil(
        this._finishedSpans.length / this._maxExportBatchSize
      );
      for (let i = 0, j = count; i < j; i++) {
        promises.push(this._flushOneBatch());
      }
```

That is why all five spans reached Tempo in both modes of the first table. The flushes were called on the same page, in the same event, in the same way, and the reason trace and metric came out differently is that the `forceFlush` of the two SDK packages treats an export in progress differently.

## Flushing Only on visibilitychange

The reproduction environment was changed to flush only on `visibilitychange`.

```ts
// cwv-otel-grafana/demo-web/src/telemetry/setup.ts
// pagehide 에서는 flush 하지 않는다. 문서를 떠날 때 pagehide 가 visibilitychange 보다 먼저 온다.
// sdk-metrics 2.9.0 부터 forceFlush 는 진행 중인 export 의 응답을 기다린 뒤에 다시 collect 한다.
// pagehide 에서 보낸 요청의 응답이 오기 전에 문서가 사라지므로, visibilitychange 에서 막 기록한
// CLS, INP 는 나가지 못한다.
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'hidden') flush();
});
```

Now there is a single flush at the moment of leaving, and that one carries CLS and INP too. The listener can go on `document` because visibilitychange is an event that fires on document. web-vitals receives visibilitychange on window in the capture phase, so by the order above this listener is called after web-vitals has recorded CLS and INP.

Traces, however, are sent before the collection code's listener runs. As seen above, `BatchSpanProcessor` registers its flush listener on document's visibilitychange when it is created. The collection code's listener is registered after it, so the SDK exports the queued spans first. By the time the collection code calls the trace `forceFlush`, the queue is empty, so the outcome is the same, but the moment the spans actually go out is decided by the SDK's listener. In 2.12.0 that listener can be turned off with the `disableAutoFlushOnDocumentHide` option (in `onInit` of the same file[^2]). The blog turns it off so that a single listener in the collection code flushes spans and metrics together (part 4).

Not flushing on pagehide means that in browsers that do not fire `visibilitychange` when leaving, like Safari before 14.1 (iOS 14.5), the values waiting for the periodic export are lost. In those browsers, though, web-vitals never receives hidden either and does not report CLS or INP, so what is additionally lost is whichever of TTFB, FCP and LCP had not been sent yet. I judged that loss to be smaller than losing CLS and INP every time. App webviews that do not fire hidden when leaving are covered in part 3.

That judgment rests on web-vitals 6.2.3 reporting CLS and INP on `visibilitychange`. The fix on main examined in part 1 ([#809](https://github.com/GoogleChrome/web-vitals/pull/809)) makes the visibility watcher call the `onHidden` function on `pagehide` as well. From the next version on, CLS and INP will be reported at `pagehide`, ahead of `visibilitychange`, when leaving a document.

In Chromium `visibilitychange` follows `pagehide`, so the current code still sends those values. But in browsers that do not fire `visibilitychange` on leaving, the CLS and INP that web-vitals reports on `pagehide` would never be exported. Leaving pagehide out would then lose exactly these two metrics, so upgrading web-vitals means revisiting the flush timing and checking the arrival counts per metric again.

Even without flushing on pagehide, `onForceFlush` still waits whenever another export is in progress. The 10-second periodic export may be in flight when hidden arrives; the flush that exports a provisional value when the LCP candidate has not changed for 3 seconds (part 3) may be waiting for its response; and the flush from an earlier hidden may not have finished when the tab is hidden, brought back and left again. If the document disappears in the meantime, CLS and INP are lost. The first table's experiment only covered leaving before the first periodic export. Overlap with the periodic export was measured separately in the next section.

Turning off the periodic export removes the overlap with periodic sends, but the wait on an earlier flush remains. In app webviews that never fire hidden on leaving, the periodic export is the only chance to send anything, so turning it off was hard there. When the same collection code was put on this blog, which ordinary browsers visit, the periodic export was removed and the reader replaced with one that does not wait for an earlier export's response. That implementation is in part 4.

## Leaving While the Periodic Export Waits for Its Response

I built the race that remains after removing the pagehide flush in the reproduction environment. A proxy in front of the collector delayed only the response to the first 10-second periodic metric export of each visit by D, and the page left L after that export went out. The browser's requests were not intercepted. Each page load pressed the button 1 second after load, and after leaving I counted how much each metric's count in Mimir grew.

| Response delay D | Left L after the export | TTFB | FCP | LCP | INP   | CLS   | Requests sent on leaving |
| ---------------- | ----------------------- | ---- | --- | --- | ----- | ----- | ------------------------ |
| 2000ms           | 500ms                   | 1    | 1   | 1   | **0** | **0** | trace only               |
| 2000ms           | 1500ms                  | 1    | 1   | 1   | **0** | **0** | trace only               |
| 2000ms           | 2500ms                  | 1    | 1   | 1   | 1     | 1     | trace, metric            |
| 2000ms           | 3500ms                  | 1    | 1   | 1   | 1     | 1     | trace, metric            |
| 300ms            | 100ms                   | 1    | 1   | 1   | **0** | **0** | trace only               |
| 0                | 100ms                   | 1    | 1   | 1   | 1     | 1     | trace, metric            |

> Each condition was run twice with the same result. `cwv-otel-grafana/experiments/e14-inflight-export`. Delaying the response with Playwright's `route` was not used: calling `unroute` inside the handler let the request go through untouched, so no delay applied.

The boundary is not the moment of leaving but the moment the response arrives. Visits that left before the response sent only the trace request on leaving, and the metric request never went out. The INP and CLS spans are in Tempo while the histogram with the same values is missing from Mimir. Visits that left after the response sent the trace and metric requests together within 10ms of leaving, and all five metrics arrived. Even with a response delay of only 300ms, leaving within it gives the same result.

How much is lost is set by the probability that an export is in progress at the moment of leaving. If departures are spread evenly across the 10-second interval, that probability is the periodic export's response delay divided by 10 seconds: 0.5% at 50ms, 20% at 2 seconds. I did not measure the actual distribution of response delays, so the blog's number is unknown. What can be said is that this loss grows as the network gets slower.

## What Has to Be Counted Again When a Version Goes Up

The exporter switched to keepalive fetch and made the result of a send visible, and sdk-metrics made flush wait for an earlier export so that collect and export never overlap. Each change was needed on its own, but in code that calls flush twice right before leaving the page, the second change became the reason the second send never started. That gap showed up in no error log.

A request sent while leaving has its document disappear before the response arrives, so confirming the result from inside the page is hard. Whenever the collection code or the OTel version changes, it is worth actually leaving pages and comparing, metric by metric, the number sent with the number that reached the store. That is how the missing CLS and INP were found this time. Checking only TTFB, FCP and LCP would have made the transport look fine.

The same comparison can keep running in production. If spans also go out with the periodic export while metrics go out only in the flush at departure, then putting the number of first-load traces stored in Tempo next to the number of first-load samples stored in Mimir shows, as the difference, the page loads whose departure send never arrived. The blog keeps these two numbers side by side on its dashboard, and the 48 hours of the second day of collection came to 56 against 40. The breakdown is in part 4. A result that can never be known from inside the browser is visible as the difference in counts between the two stores.

[^1]: Items #5807, #6317 and #6391 in [opentelemetry-js experimental/v0.223.0 `experimental/CHANGELOG.md`](https://github.com/open-telemetry/opentelemetry-js/blob/experimental/v0.223.0/experimental/CHANGELOG.md)

[^2]: [opentelemetry-js v2.12.0 `packages/sdk-trace/src/platform/browser/export/BatchSpanProcessor.ts`](https://github.com/open-telemetry/opentelemetry-js/blob/v2.12.0/packages/sdk-trace/src/platform/browser/export/BatchSpanProcessor.ts#L36-L42)

[^3]: [opentelemetry-js v2.12.0 `packages/sdk-metrics/src/export/PeriodicExportingMetricReader.ts`](https://github.com/open-telemetry/opentelemetry-js/blob/v2.12.0/packages/sdk-metrics/src/export/PeriodicExportingMetricReader.ts#L259-L273). The behavior came in with the 2.9.0 entry (#6655) in [`CHANGELOG.md`](https://github.com/open-telemetry/opentelemetry-js/blob/v2.12.0/CHANGELOG.md). The same file in 2.8.0 has no `_awaitOngoingExport`. The `_doRun` that skips a periodic run is at [L165-L171 of the same file](https://github.com/open-telemetry/opentelemetry-js/blob/v2.12.0/packages/sdk-metrics/src/export/PeriodicExportingMetricReader.ts#L165-L171)

[^4]: [opentelemetry-js experimental/v0.57.2 `experimental/packages/otlp-exporter-base/src/configuration/create-legacy-browser-delegate.ts`](https://github.com/open-telemetry/opentelemetry-js/blob/experimental/v0.57.2/experimental/packages/otlp-exporter-base/src/configuration/create-legacy-browser-delegate.ts#L38-L50) picks XHR when `headers` is present or `sendBeacon` is absent. The place that reports sendBeacon's return value as success is [`src/transport/send-beacon-transport.ts`](https://github.com/open-telemetry/opentelemetry-js/blob/experimental/v0.57.2/experimental/packages/otlp-exporter-base/src/transport/send-beacon-transport.ts#L30-L49)

[^5]: The keepalive limits in [opentelemetry-js experimental/v0.223.0 `experimental/packages/otlp-exporter-base/src/transport/fetch-transport.ts`](https://github.com/open-telemetry/opentelemetry-js/blob/experimental/v0.223.0/experimental/packages/otlp-exporter-base/src/transport/fetch-transport.ts#L16-L30) and the keepalive decision at [L71-L88](https://github.com/open-telemetry/opentelemetry-js/blob/experimental/v0.223.0/experimental/packages/otlp-exporter-base/src/transport/fetch-transport.ts#L71-L88), the retry constants (5 attempts, from 1000ms by 1.5x, at most 5000ms) in [`src/retrying-transport.ts`](https://github.com/open-telemetry/opentelemetry-js/blob/experimental/v0.223.0/experimental/packages/otlp-exporter-base/src/retrying-transport.ts#L10-L14) and the retry loop at [L42-L69](https://github.com/open-telemetry/opentelemetry-js/blob/experimental/v0.223.0/experimental/packages/otlp-exporter-base/src/retrying-transport.ts#L42-L69)

[^7]: `forceFlush` in [opentelemetry-js v2.12.0 `packages/sdk-trace/src/export/BatchSpanProcessorBase.ts`](https://github.com/open-telemetry/opentelemetry-js/blob/v2.12.0/packages/sdk-trace/src/export/BatchSpanProcessorBase.ts#L72-L77) and `_flushAll` at [L143-L157](https://github.com/open-telemetry/opentelemetry-js/blob/v2.12.0/packages/sdk-trace/src/export/BatchSpanProcessorBase.ts#L143-L157)

[^8]: [opentelemetry-js v2.8.0 `packages/sdk-metrics/src/export/PeriodicExportingMetricReader.ts`](https://github.com/open-telemetry/opentelemetry-js/blob/v2.8.0/packages/sdk-metrics/src/export/PeriodicExportingMetricReader.ts#L201-L204)

[^9]: `getSharedConfigurationDefaults` in [opentelemetry-js experimental/v0.223.0 `experimental/packages/otlp-exporter-base/src/configuration/shared-configuration.ts`](https://github.com/open-telemetry/opentelemetry-js/blob/experimental/v0.223.0/experimental/packages/otlp-exporter-base/src/configuration/shared-configuration.ts#L68-L74)

[^6]: `visibilitychange_event` in [MDN browser-compat-data v8.1.6 `api/Document.json`](https://github.com/mdn/browser-compat-data/blob/v8.1.6/api/Document.json). The entry for Safari before 14.1 says "Doesn't fire the `visibilitychange` event when navigating away from a document, so also include code to check for the `pagehide` event" and cites WebKit bugs 116769, 151234, 151610 and 194897. iOS Safari mirrors this data, so the boundary there is 14.5.
