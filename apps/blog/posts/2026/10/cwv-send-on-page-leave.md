---
title: '<em>페이지를 떠나는 순간</em> 핵심 웹 지표를 보내기'
tags:
  - web-performance
  - browser
published: false
date: 2026-10-17 09:00:00
description: '페이지를 떠날 때 CLS와 INP가 누락되는 원인을 OTel JS의 전송 방식과 flush 순서에서 찾았다.'
series: 'Core Web Vitals 직접 수집하기'
seriesOrder: 2
---

## Table of Contents

## pagehide에서 flush했더니 CLS와 INP가 빠졌다

핵심 웹 지표(Core Web Vitals, 이하 CWV)를 수집하는 재현 환경을 OpenTelemetry JS(이하 OTel JS) 2.12.0으로 만들었다. 사내 가이드에 따라 페이지가 숨겨질 때(`visibilitychange`의 hidden)와 `pagehide` 두 곳에서 남은 metric을 즉시 전송하도록 flush를 호출했다. 그런데 TTFB, FCP, LCP는 도착했지만 CLS와 INP는 도착하지 않았다. 브라우저 콘솔과 collector 로그에도 오류는 없었다.

건수는 10초 주기 export가 실행되기 전에 떠나는 페이지 로드를 만들어 셌다. 버튼을 한 번 누르고 1.5초 뒤에 떠나게 하고, 페이지 로드마다 Mimir의 지표별 건수가 얼마나 늘었는지 봤다. [1편](/2026/10/web-vitals-report-timing)에서 본 대로 web-vitals 6.2.3은 상호작용 항목의 처리를 idle이나 1초 timeout까지 미룬다. 그 전에 떠나면 INP가 보고되지 않으므로, 이 문제와 전송 중 누락을 구별하려고 timeout이 지난 뒤에 떠났다.

| flush 시점                  | TTFB | FCP | LCP | INP   | CLS   |
| --------------------------- | ---- | --- | --- | ----- | ----- |
| pagehide와 visibilitychange | 1    | 1   | 1   | **0** | **0** |
| visibilitychange만          | 1    | 1   | 1   | 1     | 1     |

> 모드마다 2번 실행해 같은 결과를 얻었다. 같은 페이지 로드의 trace는 두 모드 모두 다섯 span이 Tempo에 도착했다. 측정 코드는 `cwv-otel-grafana/experiments/e7-unload-flush`의 `run.mjs`다.

pagehide에서 flush하는 줄 하나를 지우자 다섯 지표가 모두 도착했다. 사내 코드가 쓰던 OTel JS 1.30.1에서는 같은 flush 코드로 이런 누락이 생기지 않는다. 다만 1.30.1에는 전송 실패를 성공으로 보고하는 문제가 있었다. 버전별 전송 방식과 flush 동작을 함께 확인해야 했다.

CWV 전송 코드는 값을 모아 주기적으로 내보내는 부분과, 페이지를 떠날 때 남은 값을 보내는 부분으로 나뉜다. OTel JS에서는 metric의 `PeriodicExportingMetricReader`와 trace의 `BatchSpanProcessor`가 값을 모아 내보내고, 수집 코드는 떠나는 시점에 `forceFlush`를 호출한다.

페이지를 떠날 때는 문서가 사라진 뒤에도 요청이 계속되어야 하므로 [`navigator.sendBeacon`](https://developer.mozilla.org/en-US/docs/Web/API/Navigator/sendBeacon)이나 `fetch`의 [`keepalive`](https://developer.mozilla.org/en-US/docs/Web/API/RequestInit#keepalive)를 쓴다. 전송 방식은 OTel JS exporter 버전에 따라 달라진다. 수집 코드에서는 [`pagehide`](https://developer.mozilla.org/en-US/docs/Web/API/Window/pagehide_event)나 [`visibilitychange`](https://developer.mozilla.org/en-US/docs/Web/API/Document/visibilitychange_event)의 hidden 중 어느 이벤트에서 flush할지 정해야 한다. 이 두 가지가 앞의 결과에 어떤 영향을 주는지 확인했다.

> 사내 작업은 2026년 8월 말부터 10월 중순까지 사내 검증 환경에서 했다. 이 글의 수치는 모두 공개 재현 환경([blog-experiments/cwv-otel-grafana](https://github.com/yceffort/blog-experiments/tree/cwv-otel-grafana-v1/cwv-otel-grafana))에서 다시 잰 값이다.
>
> exporter와 이벤트 순서의 측정 코드(`e2-exporter`, `e3-lifecycle`)는 태그 `cwv-otel-grafana-v1`에 있고, 처음 표의 측정 코드(`e7-unload-flush`)는 떠나는 시점을 클릭 1.5초 뒤로 늘리고 Tempo 도착까지 세도록 고쳐 커밋 [`e4e48c6`](https://github.com/yceffort/blog-experiments/tree/e4e48c6de625e90ff3c9b8931e2f40381481d996/cwv-otel-grafana/experiments/e7-unload-flush)에서 2026-10-18에 다시 쟀다. 주기 export와 겹치는 경우의 측정 코드(`e14-inflight-export`)는 커밋 [`c52fbff`](https://github.com/yceffort/blog-experiments/tree/c52fbfffc428b173bf4406406b667b6f64f2c03d/cwv-otel-grafana/experiments/e14-inflight-export)에 있고 같은 날 쟀다.
>
> 재현 환경의 버전은 web-vitals 6.2.3, OTel JS 2.12.0(OTLP exporter 0.223.0), OpenTelemetry Collector contrib 0.162.0(e7 재측정은 0.163.1), Mimir 3.2.2, Tempo 3.1.0이다. 비교할 때는 OTel JS 1.30.1(exporter 0.57.2)을 함께 썼고, 브라우저는 Playwright 1.64.1의 Chromium 156이다. 인용한 소스는 모두 이 버전의 태그로 고정했다.

## 같은 측정값을 trace와 metric으로 기록하기

수집 코드는 지표의 원인 분석 정보를 제공하는 web-vitals attribution 빌드의 콜백 하나에서 trace와 metric을 함께 기록한다. Next.js의 [`useReportWebVitals`](https://nextjs.org/docs/app/api-reference/functions/use-report-web-vitals)는 Next.js가 자체 번들한 web-vitals 사본(16.3.5 기준 4.2.1)으로 값을 잰다. 이 훅과 직접 설치한 web-vitals를 섞어 쓰면 같은 페이지 로드라도 값과 `id`가 어긋난다. 두 저장소의 값을 `id`로 연결하려면 같은 측정 결과를 사용해야 했다.

페이지 로드 하나를 trace 하나로 기록하고, 지표마다 span을 하나씩 추가한다. span의 길이를 지표 값에 맞췄기 때문에 TraceQL의 `quantile_over_time(duration, .75)`로 지표의 p75를 구할 수 있다. CLS는 점수에 1000을 곱한 값을 ms 단위로 넣었으므로, TraceQL이 초 단위로 반환한 값이 원래 CLS 점수와 같다. span 속성에는 attribution 값을 모두 넣는다. LCP라면 요소 selector, 리소스 URL, LCP를 네 단계로 나눈 시간이 포함된다.

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

지표 span은 보고된 시각에서 값만큼 거슬러 올라간 시각에 시작해 보고된 시각에 끝난다. 그래서 길이는 지표 값과 같지만, 시작 시각은 지표가 실제로 시작된 시각이 아니다. 이 trace에서 믿을 수 있는 것은 span의 길이와 속성이고, 타임라인 위의 위치는 보지 않는다. root인 `cwv.page`는 첫 지표가 올 때 만들어 그 자리에서 끝내는데, 페이지가 열릴 때 따로 보내지 않는 이유는 3편의 tail sampling에서 나온다.

metric은 지표마다 히스토그램 하나로 기록한다. 브라우저에서는 탭마다 따로 집계하고 페이지를 떠나면 수집 코드도 종료되므로, 누적값을 계속 유지하기 어렵다. 그래서 직전 전송 이후의 증분을 보내는 delta temporality를 쓰고, 누적값으로 바꾸는 일은 collector에 맡긴다. 버킷 경계에는 web-vitals의 good, poor 기준값을 넣었다.

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

경계에 2500이 있으면 `le="2500"` 버킷 하나를 전체 건수로 나눈 값이 보간 없이 정확한 good 비율이 된다. 경계 사이 값은 버킷 안에서 선형 보간으로 추정되므로, p75가 자주 걸리는 구간은 촘촘하게 두었다. 라벨은 화면 이름(`page.route`), 탐색 종류, 등급, INP의 입력 종류처럼 값의 종류가 정해진 것만 쓴다. 라벨 조합 하나가 시리즈 하나라서, selector를 라벨로 넣으면 시리즈 수가 화면 요소 수만큼 늘어난다.

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

버킷 경계는 히스토그램을 만들 때 `advice`로 넘긴다. 이 값이 없으면 sdk-metrics의 기본 경계(0부터 10000까지 15개, 지표와 상관없이 같다)가 쓰이는데, 거기에는 INP의 200과 CLS의 0.1 같은 기준값이 없다. web-vitals의 콜백은 두 기록을 한 번에 한다.

```ts
// cwv-otel-grafana/demo-web/src/telemetry/setup.ts
  observeVitals((metric, extra) => {
    record(`cwv.${metric.name.toLowerCase()}`, spanDurationMs(metric), { ...toAttributes(metric), ...extra });
    histograms[metric.name].record(metric.value, toLabels(metric, route));
  }, flush);
```

`observeVitals`는 지표별 `on*` 함수에 이 콜백을 거는데, 같은 `id`는 처음 한 번만 넘긴다.

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

1편에서 본 대로 CLS와 INP는 페이지가 다시 보였다가 숨겨질 때마다 같은 `id`로 다시 보고된다. 히스토그램은 `record`를 부른 횟수를 세므로, 그대로 두면 페이지 로드 하나가 건수 둘이 된다. 그래서 첫 보고만 세고, 나중에 나빠진 값은 들어가지 않는다. 이 선택의 대가를 재는 이야기는 4편에 있다.

metric은 `PeriodicExportingMetricReader`가 10초마다 내보내고, trace는 `BatchSpanProcessor`가 모았다가 내보낸다. 둘 다 주기를 기다리는 동안 페이지를 떠나면 남은 값을 직접 flush해야 한다.

## CLS와 INP는 hidden에서 보고된다

web-vitals는 다섯 지표를 같은 시점에 보고하지 않는다. `reportAllChanges` 없이 쓸 때 보고 시점은 다음과 같다.

| 지표 | 보고 시점                                     | 근거 (v6.2.3)   |
| ---- | --------------------------------------------- | --------------- |
| TTFB | `load` 뒤 한 번                               | `src/onTTFB.ts` |
| FCP  | 첫 contentful paint 때 한 번                  | `src/onFCP.ts`  |
| LCP  | 첫 `keydown`이나 `click` 뒤 idle, 또는 hidden | `src/onLCP.ts`  |
| CLS  | hidden 때                                     | `src/onCLS.ts`  |
| INP  | hidden 때                                     | `src/onINP.ts`  |

각 지표의 보고 시점은 1편에서 소스로 확인했다. 전송 코드에서 고려할 점은 CLS와 INP가 hidden을 받아야 보고된다는 것이다. FCP가 보고된 페이지에서는 CLS가, 처리된 상호작용이 있었던 페이지에서는 INP가 이때 보고된다. 입력 없이 떠나는 페이지에서는 LCP도 떠나는 순간 처음 보고된다. 상호작용이 한 번도 없으면 INP는 초기값 -1에서 바뀌지 않아 보고되지 않고, 6.2.3에서는 마지막 상호작용 뒤 1초 안에 떠나도 보고되지 않는다. 이렇게 떠날 때 처음 보고된 값은 앞선 주기 export로 보낼 수 없으므로, 마지막 전송이 실패하면 수집 결과에서 빠진다.

## 1.30.1의 sendBeacon은 실패를 성공으로 보고한다

사내 코드는 OTel JS 1.30.1과 OTLP exporter 0.57.2를 쓰고 있었다. 이 버전의 브라우저 exporter는 `headers` 옵션이 없으면 `navigator.sendBeacon`으로, 있으면 XHR로 보낸다[^4]. 재현 환경에서 다른 origin에 가짜 collector를 띄우고 응답을 바꿔 가며 exporter가 보고하는 결과를 봤다.

| exporter                           | collector가 200 | 400         | CORS 거부                  |
| ---------------------------------- | --------------- | ----------- | -------------------------- |
| 0.57.2, `headers` 없음(sendBeacon) | success         | **success** | **success**(서버 수신 0건) |
| 0.57.2, `headers: {}`(XHR)         | success         | failed      | failed                     |
| 0.223.0(keepalive fetch)           | success         | failed      | 5번 시도 후 failed(약 8초) |

> 각 조합을 2번 돌려 같은 결과를 얻었다. 측정 코드는 `cwv-otel-grafana/experiments/e2-exporter`에 남겼다.

첫 줄의 결과는 exporter의 전송 코드에 그대로 적혀 있다[^4].

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

sendBeacon의 반환값으로 알 수 있는 것은 요청이 브라우저 큐에 들어갔는지까지다. 응답은 돌려받지 못하므로 주석대로 재시도할 근거가 없고, 큐에 들어간 요청은 모두 success가 된다. 400 응답도, CORS 거부도 큐에는 들어가므로 표의 첫 줄처럼 성공으로 보인다. 0.57.2는 큐에 들어간 요청을 성공으로 보고하므로, CORS에 막혀 서버가 한 건도 받지 못해도 exporter는 success를 반환한다. 사내에서는 평소에 XHR로 보내 실패를 확인하고, 페이지가 숨겨졌을 때만 sendBeacon으로 전환했다. 이 구성에서도 떠나는 순간의 전송 결과는 확인할 수 없었다.

## 최신 exporter는 fetch keepalive로 보낸다

최신 버전에서는 이 분기가 사라졌다. 0.204.0부터 `headers`가 있으면 XHR보다 fetch를 먼저 사용했고, 0.211.0에서 XHR 전송을 제거했다. 0.213.0에서는 sendBeacon도 keepalive fetch로 바꿨다[^1]. keepalive fetch는 문서가 사라진 뒤에도 요청을 계속하며 응답 상태도 반환한다. 앞의 표에서 0.223.0이 400을 바로 failed로 보고한 이유다.

CORS 거부는 네트워크 오류로 처리해 재시도한다. 재시도 간격과 횟수는 전송 코드의 상수와 반복문에서 정해진다[^5].

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

간격은 1초에서 시작해 1.5배씩 늘고 ±20% 흔들리며, 상한은 5초다. 1초, 1.5초, 2.25초, 3.375초를 기다리며 4번 다시 보내면 약 8.1초가 지나고, 다섯 번째 간격 5초는 export 제한 10초에서 남은 시간보다 길어 반복문이 거기서 끝난다. 표의 최초 요청까지 총 5번 시도와 약 8초는 이 계산과 맞는다. 재시도를 끝내는 조건은 횟수보다 export 제한이 먼저 온다. 10초는 exporter 공통 설정의 기본값 `timeoutMillis: 10000`이다[^9].

[Fetch 명세](https://fetch.spec.whatwg.org/#http-network-or-cache-fetch)는 진행 중인 keepalive 요청의 본문을 모두 합쳐 64KiB까지만 허용한다. 0.223.0의 fetch 전송 소스에는 Chrome이 렌더러 프로세스당 keepalive 요청을 9개까지만 받는다는 설명도 있다. [whatwg/fetch#679](https://github.com/whatwg/fetch/issues/679)에 있는 Chromium 개발자의 설명을 인용한 것으로, Chromium 소스에서 직접 확인하지는 못했다. exporter는 진행 중인 keepalive 요청의 누적 크기와 개수를 세고, 헤더를 위한 여유를 남긴 60KiB 또는 9개 한도를 넘으면 keepalive 없이 보낸다[^5].

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

두 변수 `pendingBodySize`와 `pendingKeepaliveCount`는 모듈 수준에 있어서, 같은 페이지의 trace exporter와 metric exporter가 한도를 함께 쓴다. 한도를 넘긴 요청은 오류 없이 keepalive만 빠진 채 나간다. keepalive 없이 보낸 요청은 문서가 사라지면 함께 취소되므로, 떠나는 순간에 요청이 몰리면 전송이 누락될 수 있다. 재현 환경에서 떠나는 순간의 요청은 trace 하나(2,987B)와 metric 하나(1,327B)로, 합쳐 약 4.3KB다. 60KiB의 7%이고 개수 한도와도 멀다(뒤의 겹침 실험에서 함께 쟀다). 크기는 span 속성에 실리는 selector와 URL의 길이에 달려 있어 사이트마다 다르다.

이 한도 안에서는 문서가 사라진 뒤에도 요청을 이어 가면서 실패 여부를 확인할 수 있다는 장점이 있다. 다만 최신 exporter를 쓴 재현 환경에서도 CLS와 INP가 도착하지 않았으므로, 다음으로 flush가 실행되는 순서를 확인했다.

## pagehide는 visibilitychange보다 먼저 온다

OTel JS의 BatchSpanProcessor는 페이지가 숨겨질 때 남은 span을 내보내려고 리스너 두 개를 건다. 2.12.0의 코드는 다음과 같다[^2].

```ts
// opentelemetry-js v2.12.0 packages/sdk-trace/src/platform/browser/export/BatchSpanProcessor.ts
document.addEventListener(
  'visibilitychange',
  this._visibilityChangeListener
);

// use 'pagehide' event as a fallback for Safari; see https://bugs.webkit.org/show_bug.cgi?id=116769
document.addEventListener('pagehide', this._pageHideListener);
```

주석이 가리키는 WebKit 버그 [116769](https://bugs.webkit.org/show_bug.cgi?id=116769)는 page cache에 들어간 문서의 가시성 상태가 틀리는 문제이고, MDN의 브라우저 호환성 데이터([browser-compat-data](https://github.com/mdn/browser-compat-data), 이하 BCD) 8.1.6은 이 버그를 포함한 네 건을 근거로 Safari 14.1(iOS 14.5) 이전에는 문서를 떠날 때 `visibilitychange`가 오지 않았다고 적는다[^6]. 사내 가이드가 hidden과 pagehide 두 곳에서 flush하라고 한 것도 같은 이유였을 것이다. 그런데 `pagehide`는 Window에서 발생하는 이벤트다. 같은 탭에서 링크로 이동할 때 어떤 리스너가 어떤 순서로 불리는지 sessionStorage에 기록해 다음 문서에서 읽어 봤다.

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

> Chromium 156에서 2번 실행해 같은 결과를 얻었다. `cwv-otel-grafana/experiments/e3-lifecycle`.

`document`에 등록한 `pagehide` 리스너는 한 번도 불리지 않았다. pagehide는 명세상 Window에서 발생하고 document를 거치지 않는다. 따라서 BatchSpanProcessor의 Safari 대응 리스너는 다른 브라우저에서도 동작하지 않을 것으로 보이지만, 직접 측정한 것은 Chromium뿐이다.

이벤트는 `pagehide`, `visibilitychange` 순서로 발생했다. pagehide 시점에는 문서가 아직 visible이고, web-vitals는 그다음 `visibilitychange`에서 CLS와 INP를 보고한다. 두 이벤트에서 각각 flush하면 CLS와 INP를 기록하기 전에 한 번, 기록한 뒤에 다시 한 번 전송을 시도하게 된다.

## 2.9.0부터 forceFlush는 진행 중인 export를 기다린다

두 번째 flush가 CLS와 INP를 보내면 될 것 같지만, sdk-metrics의 `forceFlush`는 이미 진행 중인 export가 끝날 때까지 기다린다[^3]. 페이지를 떠나는 순간에는 이 대기 때문에 새로 기록한 값을 보내지 못할 수 있다.

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

pagehide에서 부른 flush가 TTFB, FCP, LCP를 담은 요청을 먼저 보낸다. 이어서 visibilitychange에서 web-vitals가 CLS와 INP를 기록하고 다시 flush를 부르는데, 이 호출은 첫 요청의 응답이 올 때까지 `_awaitOngoingExport()`에서 기다린다. 기다리는 동안에는 collect가 일어나지 않으니 CLS와 INP는 아직 어떤 요청에도 실려 있지 않다.

대기 시간을 확인하려고 페이지를 유지한 채 `pagehide`와 `visibilitychange`를 50ms 간격의 서로 다른 task에서 보내고, Playwright로 첫 metric 요청의 응답을 2초 늦췄다(`split-task.mjs`). `pagehide`에서 나간 첫 요청에는 TTFB, FCP, LCP가, 두 번째 요청에는 CLS와 INP가 들어 있었다. 두 번째 요청은 첫 요청의 응답이 도착하고 약 2ms 뒤에야 나갔다(2번 실행해 2.0ms, 1.9ms). 같은 `visibilitychange`에서 부른 trace flush는 기다리지 않았고, CLS와 INP span을 담은 요청은 이벤트 3ms 뒤에 나갔다.

실제로 페이지를 떠날 때는 첫 응답을 기다리는 동안 문서가 사라져 두 번째 collect를 실행하지 못한 것으로 보인다. 이미 시작한 첫 요청은 keepalive fetch로 계속 전송되므로 TTFB, FCP, LCP만 도착한다. 앞의 표에서 metric은 일부만 도착하고 trace는 모두 도착한 결과도 이 차이로 설명할 수 있다.

이 대기는 2.9.0에서 추가됐다(#6655). 바로 앞 버전인 2.8.0의 `onForceFlush`는 세 줄이다[^8].

```ts
// opentelemetry-js v2.8.0 packages/sdk-metrics/src/export/PeriodicExportingMetricReader.ts
  protected async onForceFlush(): Promise<void> {
    await this._runOnce();
    await this._exporter.forceFlush();
  }
```

사내 코드가 쓰던 sdk-metrics 1.30.1의 `onForceFlush`에도 이 대기는 없다. 따라서 같은 flush 코드라도 sdk-metrics를 2.9.0 이상으로 올리면 앞에서 재현한 조건에서 CLS와 INP가 누락된다. 2.12.0은 주기 export 경로에서도 같은 보장을 지키는데, 방식이 다르다. 진행 중인 export가 있으면 그 주기의 실행을 건너뛴다[^3].

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

주기 export는 건너뛰고 `forceFlush`는 기다린다. "collect와 export를 동시에 돌리지 않는다"는 보장은 서버 프로세스에서는 자연스럽다고 생각한다. 주기 실행은 다음 주기에 다시 오고, forceFlush는 기다렸다가 최신 값을 모으면 된다. 하지만 종료를 앞둔 문서에는 다음 주기도, 기다릴 시간도 없다. 응답을 기다리는 동안 다음 수집과 전송을 실행하지 못한다.

trace는 다르다. `BatchSpanProcessor`의 `forceFlush`는 큐에 남은 span을 배치 크기로 나눠 바로 내보낼 뿐, 진행 중인 export를 보지 않는다[^7].

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

그래서 처음 표의 두 모드 모두 다섯 span이 Tempo에 도착했다. 같은 페이지에서 같은 이벤트에 같은 방식으로 부른 flush인데, trace와 metric이 다른 결과를 낸 이유는 SDK 두 패키지의 `forceFlush`가 진행 중인 export를 다르게 다루기 때문이었다.

## visibilitychange에서만 flush한다

재현 환경은 `visibilitychange`에서만 flush하도록 바꿨다.

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

이렇게 하면 떠나는 순간에 부르는 flush가 한 번이 되고, 그 한 번이 CLS와 INP까지 담는다. 리스너를 `document`에 걸어도 되는 것은 visibilitychange가 document에서 발생하는 이벤트이기 때문이다. web-vitals는 window에서 capture 단계로 visibilitychange를 받으므로, 앞의 순서대로라면 web-vitals가 CLS와 INP를 기록한 뒤에 이 리스너가 불린다.

다만 trace는 수집 코드의 리스너가 실행되기 전에 전송된다. 앞에서 본 대로 `BatchSpanProcessor`는 생성될 때 document의 visibilitychange에 flush 리스너를 등록한다. 수집 코드의 리스너는 그 뒤에 등록되므로 SDK가 먼저 큐에 있는 span을 내보낸다. 이후 수집 코드에서 trace의 `forceFlush`를 호출할 때는 큐가 비어 있어 결과는 같지만, 실제 span 전송 시점은 SDK의 리스너가 정한다. 2.12.0에서는 `disableAutoFlushOnDocumentHide` 옵션으로 이 리스너를 끌 수 있다(같은 파일의 `onInit`[^2]). 블로그에서는 이 옵션으로 끄고, 수집 코드의 리스너 하나가 span과 metric을 함께 flush하게 했다(4편).

pagehide에서 flush하지 않으면 Safari 14.1(iOS 14.5) 이전처럼 떠날 때 `visibilitychange`가 오지 않는 브라우저에서는 주기 export를 기다리던 값이 누락된다. 다만 이런 브라우저에서는 web-vitals도 hidden을 받지 못해 CLS와 INP를 보고하지 않으므로, 추가로 잃는 것은 TTFB, FCP, LCP 중 아직 전송하지 않은 값이다. CLS와 INP를 매번 잃는 것보다는 이 손실이 작다고 판단했다. 떠날 때 hidden이 오지 않는 앱 웹뷰의 문제는 3편에서 다룬다.

이 판단은 web-vitals 6.2.3이 CLS와 INP를 `visibilitychange`에서 보고하는 동작을 기준으로 했다. 1편에서 살펴본 main의 수정([#809](https://github.com/GoogleChrome/web-vitals/pull/809))은 가시성 감시가 `pagehide`에서도 `onHidden` 함수를 부르도록 바꿨다. 다음 버전부터는 문서를 떠날 때 CLS와 INP가 `visibilitychange`보다 앞선 `pagehide` 시점에 보고된다.

Chromium에서는 `pagehide` 뒤에 `visibilitychange`가 오므로 현재 코드로도 그 값을 전송할 수 있다. 하지만 떠날 때 `visibilitychange`가 오지 않는 브라우저에서는 web-vitals가 `pagehide`에서 보고한 CLS와 INP를 내보내지 못한다. pagehide를 제외하면 오히려 이 두 지표를 잃게 되므로, web-vitals를 올릴 때는 flush 시점을 다시 검토하고 지표별 도착 건수를 확인해야 한다.

pagehide에서 flush하지 않아도 다른 export가 진행 중이면 `onForceFlush`는 여전히 기다린다. hidden이 오는 순간 10초 주기 export가 진행 중일 수 있고, LCP 후보가 3초 동안 바뀌지 않아 잠정값을 내보내는 flush(3편)가 응답을 기다리고 있을 수도 있다. 탭을 숨겼다가 돌아와 다시 떠날 때 앞선 hidden의 flush가 끝나지 않은 경우도 같다. 그사이 문서가 사라지면 CLS와 INP가 누락된다. 처음 표의 실험은 첫 주기 export 전에 떠나는 조건에서만 했다. 주기 export와 겹치는 조건은 다음 절에서 따로 쟀다.

주기 export를 끄면 주기 전송과 겹치는 경우는 없앨 수 있지만, 앞선 flush를 기다리는 문제는 남는다. 떠날 때 hidden이 오지 않는 앱 웹뷰에서는 주기 export가 값을 보낼 유일한 기회라서 이를 끄기도 어려웠다. 일반 브라우저로 방문하는 이 블로그에 같은 수집 코드를 적용할 때는 주기 export를 없애고, 앞선 export의 응답을 기다리지 않는 reader로 바꿨다. 그 구현은 4편에서 다룬다.

## 주기 export의 응답을 기다리는 동안 떠나면

pagehide의 flush를 뺀 뒤에도 남는 경쟁을 재현 환경에서 만들어 봤다. collector 앞에 프록시를 두고 방문마다 첫 10초 주기 metric export의 응답만 D만큼 늦춘 뒤, 그 export가 나간 시각으로부터 L 뒤에 떠나게 했다. 브라우저의 요청은 가로채지 않았다. 페이지 로드 1초 뒤에 버튼을 누르고, 떠난 뒤 Mimir의 지표별 건수가 얼마나 늘었는지 셌다.

| 응답 지연 D | export 뒤 떠난 시각 L | TTFB | FCP | LCP | INP   | CLS   | 떠날 때 나간 요청 |
| ----------- | --------------------- | ---- | --- | --- | ----- | ----- | ----------------- |
| 2000ms      | 500ms                 | 1    | 1   | 1   | **0** | **0** | trace만           |
| 2000ms      | 1500ms                | 1    | 1   | 1   | **0** | **0** | trace만           |
| 2000ms      | 2500ms                | 1    | 1   | 1   | 1     | 1     | trace, metric     |
| 2000ms      | 3500ms                | 1    | 1   | 1   | 1     | 1     | trace, metric     |
| 300ms       | 100ms                 | 1    | 1   | 1   | **0** | **0** | trace만           |
| 0           | 100ms                 | 1    | 1   | 1   | 1     | 1     | trace, metric     |

> 조건마다 2번 실행해 같은 결과를 얻었다. `cwv-otel-grafana/experiments/e14-inflight-export`. Playwright의 `route`로 응답을 늦추는 방법은 쓰지 않았다. 핸들러 안에서 `unroute`를 부르자 요청이 그대로 나가 지연이 걸리지 않았다.

경계는 떠나는 시각이 아니라 응답이 도착한 시각이다. 응답이 오기 전에 떠난 방문은 떠날 때 trace 요청만 나갔고 metric 요청은 끝내 나가지 않았다. INP와 CLS span은 Tempo에 있는데 같은 값의 histogram은 Mimir에 없다. 응답이 온 뒤에 떠난 방문은 떠나는 순간 trace와 metric 요청이 10ms 안에 함께 나갔고 다섯 지표가 모두 도착했다. 응답 지연이 300ms뿐이어도 그 안에 떠나면 같은 결과다.

이 손실이 얼마나 되는지는 떠나는 순간에 export가 진행 중일 확률로 정해진다. 떠나는 시각이 10초 주기 안에 고르게 퍼져 있다고 보면, 그 확률은 주기 export의 응답 지연을 10초로 나눈 값이다. 응답이 50ms면 0.5%, 2초면 20%다. 실제 응답 지연의 분포는 재지 않았으므로 블로그의 값은 알 수 없고, 느린 네트워크일수록 커지는 손실이라는 것만 말할 수 있다.

## 버전을 올릴 때 다시 세야 하는 것

exporter가 keepalive fetch를 사용하면서 전송 결과를 확인할 수 있게 됐고, sdk-metrics는 collect와 export가 겹치지 않도록 앞선 export를 기다리게 됐다. 각각 필요한 변경이지만, 페이지를 떠나기 직전에 flush를 두 번 호출하는 코드에서는 두 번째 전송을 시작하지 못하는 원인이 됐다. 이 누락은 오류 로그로 확인할 수 없었다.

페이지를 떠날 때 보낸 요청은 응답이 오기 전에 문서가 사라지므로, 페이지 안에서 전송 결과까지 확인하기 어렵다. 수집 코드나 OTel 버전을 바꿀 때는 실제로 페이지를 떠나게 하고, 지표별로 보낸 건수와 저장소에 도착한 건수를 비교할 필요가 있다. 이번에도 그렇게 세면서 CLS와 INP의 누락을 발견했다. TTFB, FCP, LCP만 확인했다면 전송이 정상이라고 판단했을 것이다.

같은 비교를 운영 중에도 계속할 수 있다. span은 주기 export로도 나가고 metric은 떠나는 순간의 flush로만 나가게 해 두면, Tempo에 저장된 첫 로드의 trace 수와 Mimir에 저장된 첫 로드의 표본 수를 나란히 놓는 것만으로 떠나는 순간의 전송이 오지 않은 페이지 로드가 그 차이로 드러난다. 블로그는 이 두 수를 대시보드에 나란히 두었고, 수집 이틀째의 48시간은 56건 대 40건이었다. 그 내역은 4편에 있다. 브라우저 안에서는 끝내 알 수 없는 결과가, 두 저장소의 건수 차이로는 보인다.

[^1]: [opentelemetry-js experimental/v0.223.0 `experimental/CHANGELOG.md`](https://github.com/open-telemetry/opentelemetry-js/blob/experimental/v0.223.0/experimental/CHANGELOG.md)의 #5807, #6317, #6391 항목

[^2]: [opentelemetry-js v2.12.0 `packages/sdk-trace/src/platform/browser/export/BatchSpanProcessor.ts`](https://github.com/open-telemetry/opentelemetry-js/blob/v2.12.0/packages/sdk-trace/src/platform/browser/export/BatchSpanProcessor.ts#L36-L42)

[^3]: [opentelemetry-js v2.12.0 `packages/sdk-metrics/src/export/PeriodicExportingMetricReader.ts`](https://github.com/open-telemetry/opentelemetry-js/blob/v2.12.0/packages/sdk-metrics/src/export/PeriodicExportingMetricReader.ts#L259-L273). 이 동작은 [`CHANGELOG.md`](https://github.com/open-telemetry/opentelemetry-js/blob/v2.12.0/CHANGELOG.md)의 2.9.0 항목(#6655)에서 들어왔다. 2.8.0의 같은 파일에는 `_awaitOngoingExport`가 없다. 주기 실행을 건너뛰는 `_doRun`은 [같은 파일의 L165-L171](https://github.com/open-telemetry/opentelemetry-js/blob/v2.12.0/packages/sdk-metrics/src/export/PeriodicExportingMetricReader.ts#L165-L171)

[^4]: [opentelemetry-js experimental/v0.57.2 `experimental/packages/otlp-exporter-base/src/configuration/create-legacy-browser-delegate.ts`](https://github.com/open-telemetry/opentelemetry-js/blob/experimental/v0.57.2/experimental/packages/otlp-exporter-base/src/configuration/create-legacy-browser-delegate.ts#L38-L50)가 `headers`가 있거나 `sendBeacon`이 없으면 XHR을 고른다. sendBeacon의 반환값을 success로 보고하는 자리는 [`src/transport/send-beacon-transport.ts`](https://github.com/open-telemetry/opentelemetry-js/blob/experimental/v0.57.2/experimental/packages/otlp-exporter-base/src/transport/send-beacon-transport.ts#L30-L49)

[^5]: [opentelemetry-js experimental/v0.223.0 `experimental/packages/otlp-exporter-base/src/transport/fetch-transport.ts`](https://github.com/open-telemetry/opentelemetry-js/blob/experimental/v0.223.0/experimental/packages/otlp-exporter-base/src/transport/fetch-transport.ts#L16-L30)의 keepalive 한도와 [L71-L88](https://github.com/open-telemetry/opentelemetry-js/blob/experimental/v0.223.0/experimental/packages/otlp-exporter-base/src/transport/fetch-transport.ts#L71-L88)의 keepalive 판정, [`src/retrying-transport.ts`](https://github.com/open-telemetry/opentelemetry-js/blob/experimental/v0.223.0/experimental/packages/otlp-exporter-base/src/retrying-transport.ts#L10-L14)의 재시도 상수(5회, 1000ms부터 1.5배, 최대 5000ms)와 [L42-L69](https://github.com/open-telemetry/opentelemetry-js/blob/experimental/v0.223.0/experimental/packages/otlp-exporter-base/src/retrying-transport.ts#L42-L69)의 재시도 반복문

[^7]: [opentelemetry-js v2.12.0 `packages/sdk-trace/src/export/BatchSpanProcessorBase.ts`](https://github.com/open-telemetry/opentelemetry-js/blob/v2.12.0/packages/sdk-trace/src/export/BatchSpanProcessorBase.ts#L72-L77)의 `forceFlush`와 [L143-L157](https://github.com/open-telemetry/opentelemetry-js/blob/v2.12.0/packages/sdk-trace/src/export/BatchSpanProcessorBase.ts#L143-L157)의 `_flushAll`

[^8]: [opentelemetry-js v2.8.0 `packages/sdk-metrics/src/export/PeriodicExportingMetricReader.ts`](https://github.com/open-telemetry/opentelemetry-js/blob/v2.8.0/packages/sdk-metrics/src/export/PeriodicExportingMetricReader.ts#L201-L204)

[^9]: [opentelemetry-js experimental/v0.223.0 `experimental/packages/otlp-exporter-base/src/configuration/shared-configuration.ts`](https://github.com/open-telemetry/opentelemetry-js/blob/experimental/v0.223.0/experimental/packages/otlp-exporter-base/src/configuration/shared-configuration.ts#L68-L74)의 `getSharedConfigurationDefaults`

[^6]: [MDN browser-compat-data v8.1.6 `api/Document.json`](https://github.com/mdn/browser-compat-data/blob/v8.1.6/api/Document.json)의 `visibilitychange_event`. Safari 14.1 이전 항목에 "Doesn't fire the `visibilitychange` event when navigating away from a document, so also include code to check for the `pagehide` event"라고 적혀 있고 근거로 WebKit 버그 116769, 151234, 151610, 194897을 든다. iOS Safari는 이 데이터를 mirror로 받아 14.5가 경계다.
