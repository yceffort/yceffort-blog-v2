---
title: 'Self-Hosting <em>Core Web Vitals</em> Collection for a Quiet Blog'
tags:
  - web-performance
  - devops
  - debugging
  - backend
published: true
date: 2026-10-23 13:31:00
description: 'Moved a Core Web Vitals collection setup designed for shared observability infrastructure to my personal blog. Fixed a dashboard left empty by sparse traffic and changed when metrics are collected for ordinary browsers. This post records the changes from October 17 to 22, query omissions investigated on October 23, and what five days of data can support.'
series: 'Core Web Vitals 직접 수집하기'
seriesOrder: 4
art:
  undraw: no-data
---

## Table of Contents

## The Dashboard Was Empty on the First Day

After adding the collection code to the blog and visiting a few times, I opened the Mimir dashboard. Every panel was empty. The same dashboard had shown numbers when I sent 300 page loads through the reproduction environment, and I had copied the collection pipeline over unchanged. The difference was how often people visited. After fixing the pipeline and clearing Mimir to start again, it collected 7 page loads in almost two hours.

Through part 3, I could not change the shared observability infrastructure, and the app webview sometimes gave no signal when a user left. For the blog, I ran the collection services on a single server costing \$12 a month and could change everything from the collector to the dashboards. Visitors also used ordinary browsers. I needed to check whether the measures added to avoid losing webview data were still useful, and whether the same aggregation method worked with sparse visits.

This post records connecting the collection server to the blog on October 17 and 18, 2026, then improving deployment and input validation on October 22. On October 23, I also investigated the Tempo error encountered while querying those five days of data. Dated section headings follow commit order. At the end, I revisit those choices using roughly 121 hours of collected data.

I first changed storage and queries so that even a few observations would be counted. Then I adjusted quiet LCP finalization and periodic metric exports for the blog and built panels to monitor loss. The work also required checking the dashboards and deployment process alongside the collection code.

> The server configuration is in [`infra/otel`](https://github.com/yceffort/blog/tree/e02c600a7ff73eaf8d507d153a3a21d26a3670c8/infra/otel), and the browser collection code is in [`apps/blog/src/utils/telemetry`](https://github.com/yceffort/blog/tree/e02c600a7ff73eaf8d507d153a3a21d26a3670c8/apps/blog/src/utils/telemetry). These links are pinned to the last commit on October 18. The input validation and deployment changes added on October 22 refer to [`infra/otel` at `e9f75f77`](https://github.com/yceffort/blog/tree/e9f75f77a6d391e7eae0d9da38a49c8ff5fb9388/infra/otel).
>
> Versions: web-vitals 6.2.3, OTel JS 2.12.0 (OTLP exporter 0.223.0), OpenTelemetry Collector contrib 0.163.1, Tempo 3.1.0, Mimir 3.2.2, Grafana 13.2.4 and Caddy 2.11.7. The server is a Debian 13 instance with 2GB of memory in the Amazon Lightsail Seoul region.

## October 17, 12:38: Moving the Collection Server into the Blog Repository

I initially started the server with a deployment override in the reproduction repository. To separate experiments from the live setup, I moved it into `infra/otel` in the blog repository. The server uses sparse checkout to fetch only that directory and runs it with `docker compose up`. There are five services: Caddy, the collector, Tempo, Mimir and Grafana.

I chose Lightsail because the server, disk, public IPv4 address and monthly transfer allowance are included in one price. That made the cost of a small server easier to estimate. Extras such as snapshots and transfer overages are charged separately ([Lightsail pricing](https://aws.amazon.com/lightsail/pricing/)).

| Option                        | Monthly cost                        | Considerations                                                                                                                                                                                    |
| ----------------------------- | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Lightsail `small_3_0`         | \$12                                | Includes 2GB of memory, 2 vCPUs, a 60GB disk, 3TB of transfer and 1 public IPv4 address                                                                                                           |
| EC2 on the same provider      | Charges added to the instance price | EBS storage and public IPv4 are separate; IPv4 costs \$0.005 an hour, about \$3.6 a month                                                                                                         |
| Oracle Cloud Always Free (A1) | Free                                | An instance can be reclaimed if, over 7 days, CPU utilization at the 95th percentile, network utilization and memory utilization are all below 20%[^1]. This server spends most of its time idle. |

> The Lightsail price was checked with its `get-bundles` API in Seoul. The EC2 public IPv4 price came from the AWS VPC pricing page[^2]. These are October 2026 prices; checking again on October 22 returned the same values.

That afternoon, memory use was 115MiB for Mimir, 243MiB for Grafana, 158MiB for Tempo, 86MiB for the collector and 19MiB for Caddy. Together, they used about 620MiB of the server's 2GB. Grafana was the largest service with its default installation. Grafana 13 downloads preinstalled app plugins at startup and launches separate processes for data source plugins. I disabled plugin preinstallation (`GF_PLUGINS_PREINSTALL_DISABLED`) and 11 built-in data source plugins other than Mimir and Tempo (`GF_PLUGINS_DISABLE_PLUGINS`). With the same data sources and dashboards loaded and queried once, the measurements were:

| Grafana 13.2.4       | After 5 minutes | After 6 minutes | Data source plugin processes | Preinstalled plugins |
| -------------------- | --------------- | --------------- | ---------------------------- | -------------------- |
| Default installation | 222MiB          | 261MiB          | 13                           | 5                    |
| Fewer plugins        | 159MiB          | 201MiB          | 2                            | 0                    |

> Both instances started at the same time in local Docker. Memory use fluctuated by around 40MiB even within a minute.

Removing plugins did not make Grafana particularly small. On the live server, it still used 243-292MiB that afternoon and remained the largest container on the 2GB machine.

### Anyone Can Send to the Collection Endpoint

The collection endpoint is public because browsers send to it directly. Anyone could copy a shared secret embedded in the browser code, so such a key could not distinguish real blog visits. In this setup, I narrowed the requests accepted without adding visitor authentication. Caddy forwards only the two OTLP/HTTP paths to the collector.

```text
# infra/otel/caddy/Caddyfile
# Forward only the two OTLP/HTTP paths. The collector handles CORS.
{$OTEL_DOMAIN} {
    @otlp path /v1/traces /v1/metrics
    handle @otlp {
        # The Fetch specification caps in-flight keepalive bodies at 64KiB in total.
        request_body {
            max_size 256KB
        }
        reverse_proxy collector:4318
    }
    respond 404
}
```

Other paths return 404, and bodies larger than 256KB are rejected with 413. The collector allows only the blog origin through CORS, but that does not block requests sent directly with curl. At this point, only the browser code restricted labels. A manually constructed OTLP request could carry arbitrary metric names and labels. I added server input validation on October 22.

Grafana binds only to 127.0.0.1 on the server and is accessed through an SSH tunnel. Caddy access logging is disabled, and the collector does not add the connecting IP as an attribute. This collection path is configured not to record visitor IP addresses separately.

### Tempo's Default Query Range Left Too Few Observations

With the defaults in Tempo 3.1.0, trace searches reject ranges longer than 7 days, and TraceQL metrics reject ranges longer than 24 hours.

```text
range specified by start and end exceeds 168h0m0s. received start=1791535470 end=1792226640
metrics query time range exceeds the maximum allowed duration of 24h0m0s
```

> These responses came from an 8-day trace search and a 2-day TraceQL metrics query against Tempo 3.1.0 with default settings.

Once divided by label, a day's traffic on this blog left only a few observations. I increased both retention and the query range to one year (`max_search_duration`, `max_metrics_duration` and `block_retention`). Increasing retention alone would keep data that the queries could not reach.

## October 17, 12:42: Connecting the Blog

I adapted the reproduction environment's browser code to the blog. It runs only in production, and the OTel SDK loads through a dynamic import because the initial screen does not need it. In the October 22 build at commit `466ad1b1`, this chunk was 100,571B raw and 29,365B gzipped. The collection code and SDK versions had not changed since the second day. Even when loaded late, web-vitals can collect earlier LCP and CLS entries from the buffer. However, buffered Event Timing entries include only interactions of at least 104ms, so shorter interactions before the SDK loads are absent from the INP candidates. I classify screens into five `page.route` values: home, post, tag, series and other.

## October 17, 13:42: The Empty Dashboard

This was the empty dashboard from the introduction. To investigate, I simulated a site with sparse visits, sent the same 8 LCP observations through four collection pipelines and counted what each dashboard query returned. Five label combinations received one observation each; a sixth received three, 7 minutes apart.

| Pipeline                                                                             | Dashboard count |
| ------------------------------------------------------------------------------------ | --------------- |
| Reproduction setup: `delta_to_cumulative`, remote write and `increase()`             | 0.00            |
| Same setup, with cumulative state expiry (`max_stale`) extended to 7 days            | 2.43            |
| `max_stale` of 7 days, OTLP ingestion and a 0 sample inserted at the start timestamp | 10.66           |
| Store delta directly without cumulative conversion, then use `sum_over_time()`       | 8.00            |

> I ran this three times. Only the first run of the reproduction setup reused a run label and inherited the preceding state. The other two runs agreed. The experiment is `cwv-otel-grafana/experiments/e8-sparse-counts`.

When a label combination has only one sample, `increase()` omits that series from the result. It returns no value with fewer than two samples. The table's 0.00 is the panel rendering an empty result as zero. The default engine in Mimir 3.2.2 also returned an empty result when I applied `increase()` to 20 series with one sample each.

Extending cumulative state expiry still leaves one sample in combinations seen only once. Adding a zero sample at the start timestamp makes the increase visible, but `increase()` extrapolated to the query boundaries and counted 8 observations as 10.66. A delta histogram contains the count and buckets recorded during each export interval, so summing the stored values gives the measurement count. One sample can contain several measurements when a browser exports multiple navigations together.

I removed `delta_to_cumulative` from the collector and enabled Mimir's delta storage (`-distributor.otel-native-delta-ingestion`). The dashboard counts with `sum_over_time()`. This is still experimental in Mimir 3.2.2, so upgrades require checking for changes to ingestion and querying.

I changed timestamps as well. Browser clocks differ, and Mimir rejects a sample with an earlier timestamp than one already accepted for the same series. The collector now replaces the browser timestamp with its receive time.

```yaml
# infra/otel/collector/config.yaml
  transform/receive_time:
    metric_statements:
      - context: datapoint
        statements:
          - set(start_time, Now())
          - set(time, Now())
```

There are two limitations. The timestamp now represents receipt by the collector rather than the page load. Also, two values for the same series arriving within the same millisecond become samples with the same timestamp, and Mimir drops one. In the experiment from part 3, sending 10 observations per second to one series lost 1.3-1.5% (`cwv-otel-grafana/experiments/e10-concurrent-writers`). I expected this to be uncommon at the blog's traffic level. I return to it when adding the loss panels.

## October 17, 13:43: What Is Slow, and for Whom?

With only five screen categories, the route alone was not enough to narrow down slow visits. I added metric labels with bounded sets of values: language (`page.lang`, `en` when the path starts with `/en`), device class (`device.class`, three groups divided at viewport widths of 768px and 1024px) and browser engine (`browser.engine`). Browser names alone do not identify the engine. Referrer category, whether the navigation passed through a service worker, network details and device details go only into span attributes because they have more possible values or answer narrower questions.

I also remove StyleX atomic classes from LCP element selectors.

```ts
// apps/blog/src/utils/telemetry/mask.ts
    // StyleX atomic classes (`x1j9s4mx`) describe individual styles, not stable element identities.
    // Keeping them splits the same element across deployments; digit masking can also turn them into `x{n}`.
    .replace(/\.x[0-9a-z]{5,8}(?![\w-])/g, '')
```

The blog uses StyleX atomic classes, and web-vitals includes those classes in its selectors. Leaving them in would split the same element into different entries across deployments in Tempo's panel grouped by LCP element.

## October 17, 14:17: Bots, SPA Navigations and bfcache

I made three changes here. First, bots: visits from PageSpeed Insights (`Chrome-Lighthouse`), headless browsers and search engine renderers were also being collected. These observations can easily shift percentiles on a quiet blog. I added checks for the user agent and `navigator.webdriver` before loading the SDK. These checks also filter out Playwright, so the later reader comparison uses a separate demo page.

Second, SPA navigation: links within the blog use client-side navigation, so only the initial load had been measured. Enabling web-vitals' `reportSoftNavs` measures FCP, LCP, CLS and INP for each navigation on Chromium 151 and later. TTFB is reported as zero for these navigations, so I measure it only for the initial load. Third, bfcache restores: initial loads, SPA navigations and bfcache restores each get their own navigation and trace. SPA navigations use the `navigationId` from web-vitals. At this stage, bfcache restores used a counter incremented by `pageshow` events with `persisted` set. The dashboard gained a variable for selecting initial loads or SPA navigations.

## October 17, 16:17: Adjusting Collection Timing for Ordinary Browsers

While reviewing the code and dashboards, I reconsidered whether the blog still needed quiet finalization and periodic exports, both added for the webview.

### Keeping Quiet LCP for Diagnosis

The quiet LCP mechanism from part 3 finalized a value when its candidate had not changed for 3 seconds. It was intended to retain LCP from app webviews that never became hidden. By the afternoon of the first day, 7 of the 9 LCP observations in the blog's Tempo had also been finalized this way. Readers can spend more than 3 seconds without interacting, so I expected quiet finalization to precede web-vitals frequently.

Ordinary browsers can finalize LCP on the first input or when hidden (see `finalizeLCP` in web-vitals 6.2.3's [`onLCP.ts`](https://github.com/GoogleChrome/web-vitals/blob/v6.2.3/src/onLCP.ts)). Missing those signals remains possible, but quiet finalization still risks missing a larger element rendered later, as part 3 demonstrated. For the blog, I chose to wait for the final report.

Only LCP finalized by web-vitals goes into the histogram. The quiet timer remains, but its first candidate that stays unchanged for 3 seconds is recorded only as a diagnostic span attribute. `cwv.lcp.quiet` is `none`, `same` or `lower`. A `lower` value means quiet finalization would have recorded a smaller LCP. The five-day results later in this post show how often that happened.

### Sending Metrics on Hidden Without Waiting for the Previous Response

In part 2, I removed the pagehide flush and kept flushing only on visibilitychange. But sdk-metrics' `forceFlush` waits for an export already in progress. If a periodic export is running when hidden arrives, collection of CLS and INP is delayed. The app webview needed a chance to export without hidden; for the blog, I prioritized export on hidden. I removed periodic metric exports and switched to a reader that starts collection and export without waiting for the preceding response. Spans still export every 5 seconds.

```ts
// apps/blog/src/utils/telemetry/flush-reader.ts
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
```

Delta exports contain values recorded since the preceding collection, allowing the next collection to start without waiting for the response. Since the blog filters automated browsers, I copied this reader to the reproduction environment's demo page and compared it with the periodic reader in Chromium (`cwv-otel-grafana/experiments/e15-flush-only-reader`). The page becomes hidden 1 second after loading and sends its first flush; a proxy in front of the collector delays the response by 5 seconds. After 300ms, the page becomes visible and a button is clicked. Another 1.5 seconds later, it navigates away. The first flush carries TTFB, FCP and CLS; the second carries LCP and INP.

| Reader                          | First flush response delay | Second request on leaving | LCP and INP arriving in Mimir |
| ------------------------------- | -------------------------- | ------------------------- | ----------------------------- |
| `PeriodicExportingMetricReader` | 5s                         | Never sent                | 0, 0                          |
| `PeriodicExportingMetricReader` | 0                          | 10ms after leaving        | 1, 1                          |
| `FlushOnlyMetricReader`         | 5s                         | 12ms after leaving        | 1, 1                          |
| `FlushOnlyMetricReader`         | 0                          | 11-13ms after leaving     | 1, 1                          |

> On October 22, each condition was run twice in Chromium from Playwright 1.64.1, with the same result. Headless Chromium did not become hidden on a tab switch, so the first hidden event was simulated by overriding `visibilityState` and dispatching `visibilitychange`. Because the event was not trusted, web-vitals did not finalize LCP at that point, as described in part 1. LCP therefore traveled in the request sent on leaving. With a real tab switch, LCP would also be in the first flush. The experiment is at blog-experiments commit [`be34dc4`](https://github.com/yceffort/blog-experiments/tree/be34dc4/cwv-otel-grafana/experiments/e15-flush-only-reader).

If hidden never arrives, no metrics are sent. Spans can arrive earlier through periodic export. I decided to compare trace and metric counts, but that difference alone cannot distinguish a tab still open at query time from a failed transmission.

### Measuring Updates and Timestamp Collisions Without Adding a Gateway

I measured two other issues without changing how they were handled. The first was CLS and INP reported again with the same id. The histogram records only the first report, so later changes after returning to the tab are absent. The second was duplicate millisecond timestamps. The gateway from part 3 retains the latest value per measurement id and aggregates it by series before exporting, addressing both issues. This requires operating another stateful service and managing its update window and state lost on restart. The difference in accuracy is also difficult to assess with only a few visits per hour. I kept repeat reports as `cwv.cls.update` and `cwv.inp.update` spans, recording the first value, whether the rating changed and the report index. I also made Mimir's discarded samples visible in the dashboard. I revisit both in the five-day results.

I also left the INP loss found in [part 1](/2026/10/web-vitals-report-timing) in place. The blog uses web-vitals 6.2.3 with the default INP reporting options. Leaving within 1 second of a new interaction can destroy the document before that input reaches the INP candidates. Using `reportAllChanges` only for INP does not fit a histogram that records the first report. I plan to upgrade web-vitals when a version containing fix #809 becomes available, then repeat the departing-page experiment from part 1 and count reports by metric.

## October 17, 17:05: The Loss Panel Hid Missing Collection

The loss panel showed zero even when collection of its own metrics had stopped. Its query could not distinguish no discarded samples from no loss metrics received.

Every minute, the collector scrapes Mimir's `cortex_discarded_samples_total` and its own refused and failed-export counters, then stores them in Mimir. The initial query was:

```promql
max by (reason) (max_over_time(cortex_discarded_samples_total{job="mimir", user="blog"}[$__range])) or on() label_replace(vector(0), "reason", "없음", "", "")
```

First, `or vector(0)` renders zero even when loss metrics have never been scraped. A new server missing the scrape configuration and a stopped collector both appeared as "none: 0." Second, `max_over_time` returns the past maximum in the query window, retaining the old value even after a Mimir restart resets the counter. A third issue affected a different panel. The dashboard generator assigned refIds only in its final panel. Adding the loss panel after it left all five queries in the measurement-count panel with refId A. Grafana rejected that panel with `400 query.duplicateRefId`. Checking queries without opening the dashboard had missed it.

The corrected query displays a value only when the last scrape succeeded (`up == 1`) and is less than 3 minutes old. It uses the latest counter value. This is cumulative since Mimir started, regardless of the selected query range, so the panel title says so.

```promql
(max by (reason) (cortex_discarded_samples_total{job="mimir", user="blog"}) or on() label_replace(vector(0), "reason", "없음", "", "")) and on() max((up{job="mimir"} == 1) and (time() - timestamp(up{job="mimir"}) < 180))
```

I started the same configuration locally and compared the queries with fixtures. Before the first scrape, the corrected panel showed "not collected."

| Situation                                         | Original query | Corrected query |
| ------------------------------------------------- | -------------- | --------------- |
| 4 and 6 minutes after stopping the collector      | None: 0        | Not collected   |
| Discard 10 samples, restart Mimir, then discard 5 | 10             | 5               |

> I sent the panel queries through Grafana 13.2.4's `/api/ds/query`. After the fixes, all 30 Mimir panels returned 200, including on the live server's Grafana.

## October 18, 13:50: Restored-Page TTFB Was Counted as an Initial Load

The next day, I fixed bfcache restore metrics being split across different traces. web-vitals reports restored-page TTFB directly inside its `pageshow` listener (`onBFCacheRestore` in part 1). The collection code also counted restores in a window `pageshow` listener and used that count as a key. But web-vitals used capture and the collection code used bubble. Under the current DOM specification, capture listeners on the same target run first regardless of registration order. TTFB therefore arrived before the counter incremented and received a different key from FCP and LCP, which were reported two frames later. One restore became two traces.

I switched the key to `navigationStartTime`, supplied by web-vitals for restore metrics. As the code in part 1 showed, this is the restore `pageshow` event's `timeStamp`, shared by all metrics from that restore. The collection code stores only the path. Since a restored document may have navigated within the SPA, I save the current path under the same timestamp and look it up when a metric arrives. I also switched this listener to capture and registered it before web-vitals. Capture listeners on the same target run in registration order, so the path is stored before any metric reported right after the restore.

```ts
// apps/blog/src/utils/telemetry/navigation.ts
      if (metric.navigationType === 'back-forward-cache') {
        const start = metric.navigationStartTime ?? 0
        return {
          key: `restore:${start}`,
          kind: 'restore',
          startTime: start,
          pathname: restorePathnames.get(start) ?? initialPathname,
        }
      }
```

Fixing the key exposed another issue. As part 1 showed, web-vitals supplies zero for restored-page TTFB instead of measuring it. Yet the dashboard's initial-load selector included `back-forward-cache`. Restored FCP and LCP also measure something different from their initial-load counterparts. Including or excluding 2 restores in a 48-hour window produced these values:

| 48-hour initial-load p75 | Including restores | Excluding restores |
| ------------------------ | ------------------ | ------------------ |
| TTFB                     | 375ms              | 383ms              |
| LCP                      | 2.50s              | 2.75s              |
| FCP                      | 1.65s              | 1.80s              |

> Read from the live Grafana at the same time, changing only the variable. There were around 40 initial-load TTFB observations, so even 2 restores moved p75 this much.

I stopped recording restored-page TTFB, for the same reason I exclude SPA TTFB, and split the selector into initial load, SPA navigation, bfcache restore and all. FCP, reported for each restore, provides the restore count. This keeps values with different measurement definitions out of the same distribution, even when their metric names match.

## October 18, 13:50: Preview Deployments Are Production Builds Too

The collection code was also running in Vercel preview deployments. It checks whether `NODE_ENV` is production, which is also true for preview builds. The collector allows only the blog origin, so the browser blocks the requests. As part 2 showed, the exporter treats a CORS rejection as a network error and makes 5 attempts over roughly 8 seconds: the initial request and 4 retries. The values did not reach storage, but every preview loaded the SDK and attempted requests that would fail. I now check the origin before loading the SDK. At the same point, I catch dynamic import failures so a missing telemetry chunk does not produce an `unhandledrejection` for the reader.

```tsx
// apps/blog/src/components/analytics/CwvTelemetry.tsx
    // Preview deployments are production builds too. Export only from the blog origin.
    if (window.location.origin !== SiteConfig.url) return
```

## October 18, 13:51: Setting the Preflight Cache and Server Resource Limits

On the server, I changed the preflight cache, memory limit and log retention. The exporter's cross-origin requests use `Content-Type: application/json`, which requires a preflight (OPTIONS). Without `Access-Control-Max-Age`, the collector's CORS response is cached for only the default 5 seconds[^3]. Spans export every 5 seconds too, leaving little opportunity to reuse the cached result. I set `cors.max_age` to Chromium's 2-hour cap and checked an OPTIONS response for `access-control-max-age: 7200`. I did not separately count the reduction in OPTIONS requests in a real browser.

The second change was the collector's memory cap. `memory_limiter` used `limit_percentage: 80`; its README recommends this setting for containers whose host environment determines their memory limit[^4]. But compose had no collector memory limit. Without one, the collector could use the server's full memory, making 80% mean 80% of 2GB and leaving too little for the other containers. It had used 86MiB the previous afternoon, so I set a container cap of 512MiB. Subtracting `spike_limit_percentage: 20` from the 80% limit gives a refusal threshold of about 307MiB. Refused data appears in the loss panels fixed the day before.

Third, Docker's default `json-file` logging does not rotate. Since this server was configured for one-year data retention, logs could eventually fill the disk. Each container now rotates three 10MB log files.

## October 18, 13:54: One Listener Controls the Flush

Part 2 described how the `BatchSpanProcessor` listener for document `pagehide` is not called. Its document `visibilitychange` listener does run. Registered when the SDK is created, it runs before the collection code's listener. The collection code then calls trace `forceFlush` against a queue already drained by the SDK. The outgoing values were the same, but the comment saying the collection code controlled flush timing was inaccurate for spans. I disabled the SDK listener with `disableAutoFlushOnDocumentHide` in version 2.12.0, leaving one collection-code listener to start both trace and metric flushes.

```ts
// apps/blog/src/utils/telemetry/setup.ts
      new BatchSpanProcessor(
        new OTLPTraceExporter({url: `${collectorUrl}/v1/traces`}),
        // By default, the browser processor flushes on document visibilitychange and pagehide.
        // Disable those listeners so our visibilitychange listener controls flushing. Keep the 5-second periodic export.
        {disableAutoFlushOnDocumentHide: true},
      ),
```

## October 18, 16:29: Opening the Dashboard Produced 429s

I opened the corrected dashboard in the live Grafana and immediately scrolled to the bottom. Three panels failed with `too many outstanding requests`: Mimir had returned 429. Its logs recorded 16 rejected requests within 10ms at 15:51:03, with none in the preceding 48 hours. Reloading or scrolling slowly did not reproduce it, even with 7-day or 30-day ranges.

I first checked the burst of subrequests and Mimir's queue limit. The live `/config` showed queries split at 24-hour boundaries (`split_queries_by_interval`), then sharded 16 ways (`parallelize_shardable_queries`, `query_sharding_total_shards`), with at most 100 outstanding requests per tenant (`max_outstanding_requests_per_tenant`). A 48-hour window crossing those boundaries can span 3 intervals, producing up to 48 subrequests from one panel query. The measurement-count panel has five queries, making up to 240. This distributes work among queriers, but this Mimir had one process and one querier (`max_concurrent: 8`), with the result cache disabled. The subrequests converged on the same queue.

```yaml
# infra/otel/mimir/mimir.yaml
frontend:
  split_queries_by_interval: 0
  parallelize_shardable_queries: false
```

After disabling time splitting and sharding, I sent the live dashboard's 36 Mimir queries concurrently from the server: once with a 48-hour range and twice with a 30-day range. All 108 requests returned 200, and no requests were rejected. I did not run the identical experiment before the change. The evidence was the observed 429s and the subrequest calculation from the defaults.

### Traces Arrived Without Corresponding Metrics

Two new panels showed 56 initial-load traces in Tempo and 40 initial-load observations in Mimir. They were intended to count the same loads, yet differed by 16. Comparing labels and timestamps showed that 6 traces preceded the deployment of the 13:42 fix on October 17. Mimir's first sample was at 13:56.

The other 10 were spread from the evening of October 17 through the afternoon of October 18, mostly desktop Chromium visits to posts. Periodic span export can run before hidden, so these could be tabs not yet hidden or metric requests that started on hidden but did not complete. The Mimir discarded-sample and collector refused-data counters I checked were zero, but that did not identify the cause. I recorded the difference of 10 out of 50 comparable visits and revisited it with five days of data.

## October 22, 16:25: Applying Configuration Through GitHub Actions

Until then, every configuration change required SSH access followed by a manual `git pull` and `docker compose up -d`. On days with several revisions, it was easy to skip that step and write about a repository state the server was not running. I added a GitHub Actions workflow that connects over SSH and applies changes when `infra/otel` reaches main (`.github/workflows/deploy-otel.yml` and `infra/otel/deploy.sh`).

I created a dedicated deployment key. Its entry in `authorized_keys` uses a forced command, so whatever command the workflow sends, the server runs only `git pull` followed by `deploy.sh`. Port forwarding and pty allocation are disabled. This key invokes the defined deployment command instead of providing an arbitrary shell. The workflow's requested commit hash is passed through `SSH_ORIGINAL_COMMAND`; the script logs a warning if HEAD after the pull differs.

```text
# Server: ~/.ssh/authorized_keys
command="cd ~/blog && b=$(git rev-parse HEAD) && git pull --ff-only --quiet && exec bash infra/otel/deploy.sh $b",no-port-forwarding,no-agent-forwarding,no-X11-forwarding,no-pty ssh-ed25519 AAAA... github-actions deploy-otel
```

After `docker compose up -d`, `deploy.sh` compares the commits before and after the pull and restarts only services whose configuration directories changed. Changing a bind-mounted configuration file does not by itself make compose recreate the container. Changes to a service in compose.yaml are handled by `up -d`.

The first run failed with `Connection timed out` because port 22 allowed only my IP address. GitHub publishes the runners' IP ranges, but they are broad and change, so instead of narrowing the firewall rule I opened port 22 to every address. Password authentication stays disabled and the deployment key is restricted by the forced command, but my own key's shell access is exposed on the same port. The next run pulled the server from `e02c600a` to `f3c6f8ff`. No configuration files had changed, so no containers restarted. Whether a configuration change actually triggered a restart would be tested by the next change.

## October 22, 19:09: The Collector Was Not Validating Labels

On the first day, labels were restricted only in browser code. The collector accepted the supplied `service.name` and labels, so a manually constructed OTLP request could add arbitrary values or labels to the blog's `cwv.lcp`. With 100 to 180 observations per metric per day, fake values can affect p75. Even without verifying that a request came from a real visit, there was no need to accept names and labels the browser code never sends. Mimir was still using these default limits:

| Mimir 3.2.2 default         | Value            |
| --------------------------- | ---------------- |
| Series per tenant           | 150,000          |
| Labels per series           | 30               |
| Label value length          | 2,048 characters |
| Received samples per second | 10,000           |

Limiting bodies to 256KB does not limit request frequency or the values inside them. Spans sent to Tempo were not checked against the blog's names and attributes either. I added collector rules for the names and labels used by the browser code. The `filter` processor rejects metrics with the wrong service name, a name outside the five metrics, a type other than histogram, temporality other than delta, or label values outside the allowlists. After validation, the `transform` processor removes unknown label keys with `keep_keys` and limits span attributes to 64 entries and 512 characters.

```yaml
# infra/otel/collector/config.yaml
  filter/schema:
    error_mode: ignore
    metrics:
      metric:
        - 'resource.attributes["service.name"] != "yceffort-blog"'
        - 'not IsMatch(name, "^cwv\\.(ttfb|fcp|lcp|inp|cls)$")'
        - 'type != METRIC_DATA_TYPE_HISTOGRAM'
        - 'aggregation_temporality != AGGREGATION_TEMPORALITY_DELTA'
      datapoint:
        - 'not IsString(attributes["page.route"]) or not IsMatch(attributes["page.route"], "^(home|post|tag|series|other)$")'
```

I validated the configuration locally with the same image, replaced the exporter with debug output and sent 12 request variants. Only the valid metric request and the one containing an unknown key survived; the unknown key was removed. Requests with another service name, an unknown metric name, cumulative temporality, gauge data, disallowed label values or missing labels were dropped. Two `cwv.*` spans in the same trace passed, and a 2,000-character attribute was truncated to 512 characters. The experiment verified those rejection rules and removal of unknown datapoint label keys.

Fake values using allowed labels still pass. This also does not guarantee that the total number of series stays below a few thousand. The configuration does not validate the values of retained resource attributes or histogram bucket boundaries, and it does not limit request frequency. The input checks implemented here do not constitute a bound on storage volume.

The deployment process needed another check too. The workflow succeeded, but the collector still showed "Up 4 days." After `cd infra/otel`, `deploy.sh` ran `git diff -- infra/otel`, resolving the pathspec as `infra/otel/infra/otel`. Once the 19:10 commit changed the pathspec to `.`, the script printed the two changed files and restarted the collector. Workflow success alone had not established that configuration was applied.

## Successful Tempo Queries Still Omitted Blocks

On October 22, querying five days of data at once through Tempo 3.1.0's TraceQL metrics API failed with `end must be greater than start`. Both instant and range queries failed. Investigating on October 23 showed that the problem lay in stored block time ranges rather than the length of the query window.

Between 16:22 and 16:36 on October 22, the Tempo logs recorded 369 failed subrequests, all for the same three blocks. Each block's metadata had a `startTime` later than its `endTime`. One held only a single INP span. That span ended at 12:23:45, but the block's start time was 13:32:03.

### Span Timestamps and Block Bounds Were Different

The live-store's [`writeHeadBlock`](https://github.com/grafana/tempo/blob/v3.1.0/modules/livestore/instance.go) derives time bounds from the spans in a trace, then constrains those bounds relative to ingestion time. The adjusted bounds contribute to the block metadata. The original timestamps inside the spans remain unchanged.

If the start is more than `ingestion_time_range_slack` (2 minutes by default) before ingestion, it is raised to that lower bound. The end is lowered only when it exceeds the upper bound in the future. For a late span, the start can therefore move forward while the end remains in the past. Applying that rule to this span suggests ingestion around 13:34; the block completion log was at 13:34:13. With only this span in a block cut every 30 seconds (`max_block_duration`), the reversed bounds remained in the block metadata.

Tempo 3.1.0's block-builder raises the end to the start if adjustment would leave it earlier ([`adjustTimeRangeForSlack`](https://github.com/grafana/tempo/blob/v3.1.0/modules/blockbuilder/tenant_store.go)). The live-store in the same version lacks that handling. I could not establish why the span's timestamp was about 70 minutes behind ingestion. Browser clock skew, the behavior of `performance.now()` during sleep and delayed request delivery remain possible explanations.

### Shorter Windows Avoided the Error by Omitting Data

The query-frontend selects a block when `block start <= query end` and `block end >= query start` ([`blockMetasForSearch`](https://github.com/grafana/tempo/blob/v3.1.0/modules/frontend/frontend.go)). A block with reversed bounds is selected only when the query window contains the entire reversed interval. For an instant query, those reversed bounds reach the subrequest, the querier rejects it and the whole query returns 500.

For a range query, [`TrimToBlockOverlap`](https://github.com/grafana/tempo/blob/v3.1.0/pkg/traceql/engine_metrics.go) aligns the block bounds to the step. If the resulting start and end are equal, [`buildBackendRequests`](https://github.com/grafana/tempo/blob/v3.1.0/modules/frontend/metrics_query_range_sharder.go) creates no request for that block. If the start remains greater than the end, the error remains. Receiving 200 after changing the step therefore does not establish that the block was successfully queried.

The three reversed intervals observed on October 22 ran from October 21 at 16:26 to October 22 at 08:22, from October 21 at 20:31 to October 22 at 10:01, and from 12:23 to 13:32 on October 22. None fit entirely within the two-hour windows aligned to 13:00. Those queries succeeded because they did not select the affected blocks; two hours was not a query limit. Checking that one two-hour result equaled the sum of two one-hour results did not establish completeness either, since both approaches could omit the same blocks.

On October 23, two of the original three blocks had been compacted away and one new block with reversed bounds had appeared. In that state, the responses were:

| Query condition                                   | Response |
| ------------------------------------------------- | -------- |
| October 22, 12:00 to 14:00, 2 hours               | 500      |
| October 17, 13:00 to October 22, 12:00, 119 hours | 200      |
| A 121-hour interval, 1-hour step                  | 200      |
| The same 121-hour interval, 5-minute step         | 500      |

I have kept the counts queried on October 22 below. They record what those queries returned; they have not been verified as complete stored counts after recovering the reversed blocks. That said, each of the three blocks reversed on October 22 held a single trace (totalObjects in meta.json and the compactor log), so what those queries omitted was the spans of those three traces. Interpreting the difference between Tempo and Mimir must account for omissions during querying as well as transmission.

## Five Days of Collected Data

These values cover roughly 121 hours, from the first Mimir sample at 13:56 on October 17 to 15:00 on October 22. Tempo and Mimir counts use the same 61 two-hour slices, from 13:00 on October 17 to 15:00 on October 22. All other queries below also end at 15:00. Counts and ratios derived from Tempo are subject to the query omissions described above.

### Trace and Metric Count Differences Vary by Metric

| Metric | Tempo spans | Mimir measurements | Difference |
| ------ | ----------- | ------------------ | ---------- |
| TTFB   | 624         | 526                | 98 (15.7%) |
| FCP    | 954         | 896                | 58 (6.1%)  |
| LCP    | 902         | 880                | 22 (2.4%)  |
| CLS    | 793         | 778                | 15 (1.9%)  |
| INP    | 707         | 702                | 5 (0.7%)   |

The Mimir measurement counts are sums of the stored delta histograms' counts, not the number of time-series samples. The table includes SPA navigations and bfcache restores as well as initial loads.

TTFB and FCP are reported shortly after loading, so their spans can arrive through periodic export before the tab becomes hidden. LCP can finalize on first input or hidden; CLS and INP can also report at points such as SPA navigation. Even when both flushes start on hidden, traces and metrics use separate HTTP requests, so their arrival is not atomic. Reporting and export timing, as well as the Tempo query omissions described above, can affect the differences. The table alone cannot assign a cause to each one.

Spans that started before Mimir's first sample (13:56 on October 17) number 4 for TTFB, 4 for FCP, 3 for LCP, 1 for CLS and 1 for INP (2 of the 6 traces seen on October 18 precede the 13:00 start of the comparison window). Excluding them on that timestamp basis leaves 620 TTFB spans against 526 measurements, a gap of 94 or about 15.2%; FCP is 950 against 896, a gap of 54 (5.7%); LCP 19 (2.1%), CLS 14 (1.8%) and INP 4 (0.6%). Since I cannot distinguish tabs still open from failed transmissions, I did not treat this as a loss rate on departure. The Mimir discarded-sample, collector refused-data and collector failed-export counters I checked were all zero. However, these counters reset on restart and do not cover every loss path, so they do not prove there was no server-side loss over the entire period.

The stores also use different timestamp bases. Tempo spans retain browser timestamps, while Mimir metrics use the collector's receive time. Even with identical query boundaries, the same measurement near a boundary may fall into different windows. The preceding adjustment uses span start timestamps; it is not a comparison of matched visits.

Mimir had 524 initial-load TTFB measurements and 453 initial-load FCP measurements. The TTFB count is the table's 526 minus 2 bfcache restore values collected before I stopped recording them. The difference is 71, or 13.5% of the TTFB count. Metrics do not store per-visit ids, so this difference alone cannot establish that exactly 71 visits had TTFB but no FCP. Because web-vitals does not report FCP when the page is hidden before the first paint, visits opened in background tabs are one possible explanation. I did not verify that visit by visit.

### Of 90 Label Combinations, 38 Had Only One Observation

There were 90 label combinations with FCP measurements. Of those, 38 had only one observation in five days, while the largest had 132. The average was about 2 observations per combination per day. With cumulative conversion configured as in the first-day experiment, those 38 single-sample series would have been absent from `increase()` results. The live data supported the reason for choosing delta storage on a quiet blog.

### SPA Navigations Were 40% of Recorded Navigations

The Tempo query returned 1,076 navigations: 623 initial loads, 433 SPA navigations (40%) and 12 bfcache restores (1.9% of initial loads). Another 8 lacked the attribute because they preceded the separation of SPA navigation and bfcache restores. Without enabling `reportSoftNavs` on the first day, those 433 navigations would not have had separate measurements. By FCP observations, Chromium accounted for 852 (89%), WebKit for 62 and Gecko for 36; the remaining 4 predate the engine attribute. Since SPA navigation is recorded only on Chromium 151 and later, 40% is not the SPA share of all visits. WebKit and Gecko SPA navigations are not collected this way.

I counted referrers using initial-load TTFB spans. Of the 614 initial loads carrying the attribute (excluding 8 from before the attribute existed and 2 bfcache restores), 277 were direct (45%), 234 came from search (38%), 54 from within the blog, 44 from other sources and 5 from AI services. Of those, 291 (47%) went through a service worker.

### Quiet LCP Was Lower in 0.6% of Comparable Observations

Among 902 LCP spans, `cwv.lcp.quiet` was `none` in 341, `same` in 546 and `lower` in 5. Another 10 preceded the attribute. Of the 892 with the attribute, the quiet timer finished before the final report in 551 cases (62%). In 5 cases, about 0.6% of those 892, the quiet value was lower than the final value. All five final values were rated poor. Quiet finalization would have recorded smaller LCP values for those visits.

Even at a small percentage, that still understates slow visits. I kept waiting for the final report and retained quiet values only as diagnostic attributes. This comparison also requires the final LCP to have arrived. These numbers cannot tell me how many additional visits quiet finalization would have captured when no final report arrived.

### CLS Ratings Changed in 4.4% of Reporting Navigations

Since the same id can report more than once, I counted updates by trace. One trace represents one navigation: an initial load, SPA navigation or bfcache restore. Of the 793 navigations reporting CLS, 77 had updates (9.7%), and 35 had a changed rating (4.4% of the total). INP had 56 navigations with updates, of which 2 changed rating. There were 96 CLS update spans and 62 INP update spans.

Recording only the first report in the histogram excludes later CLS increases. The 4.4% is the share of navigations whose rating changed; it excludes increases within the same rating. It cannot tell us how much p75 would change.

For now, I use Mimir's first-report values to identify screens needing work and inspect updates in Tempo alongside them. Initial-load CLS p75 was already in the needs improvement range. To identify what to fix, I needed the element and timestamp of the largest shift reported by web-vitals, which remain available in update spans. The shifted element is not necessarily what caused the shift, so it still needs reproduction and investigation.

The gateway from part 3 can replace a measurement with its latest value, but it also requires managing a stateful server, a 30-minute finalization delay and pending state lost on restart. I chose to keep first-report aggregates and update traces together without adding that service. This histogram therefore cannot be described as the final CLS distribution over the full lifetime of each navigation.

### CLS Was the Blog's Weak Point

Initial-load p75 was 455ms for TTFB, 1,191ms for FCP, 1,499ms for LCP, 75ms for INP and 0.126 for CLS. LCP was good in 90% of observations (399 of 445), and INP in 96% (326 of 339). CLS was good in only 68% (250 of 365), with p75 at 0.126 in the needs improvement range. LCP was similar on desktop and mobile, at 1,526ms and 1,473ms respectively; SPA LCP p75 was 467ms. These percentiles are estimated from histogram buckets and exclude updates after the first report. Within the collected data, there was enough reason to investigate CLS first. Tempo can narrow down the screen and shifted element; the actual cause analysis belongs in another post.

### The Services Used 627MiB

On the afternoon of October 22, memory use was 249MiB for Grafana, 192MiB for Tempo, 93MiB for Mimir, 74MiB for the collector and 19MiB for Caddy, totaling 627MiB. This was close to the first afternoon's 620MiB. Tempo had risen from 158MiB to 192MiB, while Mimir had fallen from 115MiB to 93MiB. Disk usage was 6.5GB out of 59GB. According to its own metrics, scraped into Mimir every minute, the collector accepted 4,885 spans and 93,495 metric points (including the loss metrics scraped every minute) between its restart at 16:32 on October 18 and 15:00 on October 22; the previous container had accepted 285 spans. The tail-sampling counter read from the container on October 22 showed 0 traces rejected for having fewer than 2 spans. The server's base price is \$12 a month. Five days of usage is not enough to establish the disk capacity needed for one year of retention.

## What the Small Sample Can Support

The first requirement after moving to the blog was to include even infrequent visits in the counts. With delta storage, I could query the 38 label combinations that had only one observation in five days. For ordinary browsers, I stopped finalizing LCP with quiet and changed metric export to start on hidden without waiting for the preceding response.

I did not choose a setup that aggregates the final value of every measurement. CLS and INP updates with the same id remain in traces, while Mimir counts the first report. Even with that limitation, the data supported investigating CLS first. If I need a distribution that incorporates subsequent updates, I will need to revisit the aggregation method.

The loss panels and automatic deployment needed verification too. I saw a panel display zero when collection had stopped, and a workflow succeed without applying changed configuration. Being able to change the server settings also meant creating the relevant conditions and checking what was actually displayed or applied.

Before investigating the remaining count differences, I need to account for the blocks omitted from Tempo queries. Even then, traces and metrics use different timestamp bases, and the gap can include open tabs and failed transmissions. These five days support investigating CLS with the current aggregates. Calculating loss across the entire collection path requires resolving query omissions and comparing measurements from the same visits.

[^1]: [Oracle Cloud Infrastructure, Always Free Resources](https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm), under "Reclamation of Idle Compute Instances."

[^2]: [Amazon VPC pricing](https://aws.amazon.com/vpc/pricing/), under Public IPv4 Address.

[^3]: [MDN, Access-Control-Max-Age](https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/Access-Control-Max-Age). The default is 5 seconds. Chromium caps it at 2 hours (7200 seconds) from version 76; Firefox caps it at 24 hours.

[^4]: [opentelemetry-collector v0.163.0, `processor/memorylimiterprocessor/README.md`](https://github.com/open-telemetry/opentelemetry-collector/blob/v0.163.0/processor/memorylimiterprocessor/README.md), which recommends percentage limits when the container's host environment determines the memory allocation.
