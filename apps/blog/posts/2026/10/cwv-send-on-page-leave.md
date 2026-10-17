---
title: '<em>페이지를 떠나는 순간</em> CWV를 보내기'
tags:
  - web-performance
  - browser
published: false
date: 2026-10-17 09:00:00
description: 'CLS와 INP는 페이지를 떠나는 순간에 값이 생기므로, 그 순간의 전송이 실패하면 통째로 빠진다. OTel JS 1.30.1의 sendBeacon, 최신 exporter의 fetch keepalive, pagehide와 visibilitychange의 순서, sdk-metrics 2.9.0의 forceFlush 대기를 재현 환경에서 재고 정리했다.'
series: 'Core Web Vitals 직접 수집하기'
seriesOrder: 2
---

## Table of Contents

## CLS와 INP만 도착하지 않았다

재현 환경의 수집 코드를 OTel JS 2.12.0으로 만들고, 사내 가이드에 적혀 있던 대로 페이지가 숨겨질 때(`visibilitychange`의 hidden)와 `pagehide` 두 곳에서 metric을 flush하게 했다. 그다음 10초 주기 export가 오기 전에 떠나는 방문을 만들고, 방문마다 Mimir의 지표별 건수가 얼마나 늘었는지 셌다.

| flush 시점                  | TTFB | FCP | LCP | INP   | CLS   |
| --------------------------- | ---- | --- | --- | ----- | ----- |
| pagehide와 visibilitychange | 1    | 1   | 1   | **0** | **0** |
| visibilitychange만          | 1    | 1   | 1   | 1     | 1     |

> 모드마다 두 번 실행해 같은 결과를 얻었다. `cwv-otel-grafana/experiments/e7-unload-flush`.

TTFB, FCP, LCP는 한 건씩 늘었는데 CLS와 INP는 한 건도 오지 않았다. 브라우저 콘솔에도, collector 로그에도 오류는 없었다. pagehide에서 flush하는 줄 하나를 지우자 다섯 지표가 모두 도착했다. 사내 코드가 쓰던 OTel JS 1.30.1에서는 같은 flush 코드로 이런 일이 생기지 않는다. 대신 1.30.1에는 떠나는 순간의 전송 실패를 성공으로 보고하는 문제가 있고, 최신 버전은 그 문제를 고쳤다. 전송의 문제를 고친 버전에서 flush에 새 손실이 생긴 것이다.

이 글은 페이지를 떠나는 순간에 CWV 값이 어떻게 나가는지를 두 버전에서 따라가며, 이 표가 나온 이유를 찾아간 기록이다.

> 사내 작업은 2026년 8월 말부터 10월 중순까지 사내 검증 환경에서 했다. 이 글의 수치는 모두 공개 재현 환경([blog-experiments/cwv-otel-grafana](https://github.com/yceffort/blog-experiments/tree/cwv-otel-grafana-v1/cwv-otel-grafana))에서 다시 잰 값이다.
>
> 재현 환경의 버전은 web-vitals 6.2.3, OTel JS 2.12.0(OTLP exporter 0.223.0), OpenTelemetry Collector contrib 0.162.0, Mimir 3.2.2다. 비교할 때는 OTel JS 1.30.1(exporter 0.57.2)을 함께 썼고, 브라우저는 Playwright 1.64.1의 Chromium 156이다. 인용한 소스는 모두 이 버전의 태그로 고정했다.

## 수집 코드의 모양

수집 코드는 web-vitals의 attribution 빌드 콜백 하나에서 trace와 metric을 함께 기록한다. Next.js의 `useReportWebVitals`는 Next.js가 자체 번들한 web-vitals 사본으로 값을 재기 때문에, 이 훅과 직접 설치한 web-vitals를 섞어 쓰면 같은 페이지 로드라도 값과 `id`가 어긋난다. 두 저장소의 값을 `id`로 이어 붙이려면 측정 소스가 하나여야 했다.

trace는 페이지 로드 하나가 trace 하나이고, 지표마다 span이 하나씩 달린다. span의 길이를 지표 값으로 맞췄기 때문에 TraceQL의 `quantile_over_time(duration, .75)`가 그대로 지표의 p75가 된다. CLS는 점수라서 1000을 곱해 ms로 넣었고, TraceQL이 초 단위로 돌려주는 값이 그대로 CLS 점수로 읽힌다. 속성에는 attribution 값을 모두 싣는다. LCP라면 요소 selector, 리소스 URL, 그리고 LCP를 네 단계로 나눈 시간이다.

metric은 지표마다 히스토그램 하나다. 브라우저는 수명이 짧고 탭마다 따로 세기 때문에 누적값을 들고 있을 주체가 없다. 그래서 delta temporality로 보내고, 누적으로 바꾸는 일은 collector에 맡긴다. 버킷 경계에는 web-vitals의 good, poor 기준값을 반드시 넣었다.

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

metric은 `PeriodicExportingMetricReader`가 10초마다 내보내고, trace는 `BatchSpanProcessor`가 모았다가 내보낸다. 둘 다 주기를 기다리는 동안 페이지를 떠나면 남은 값을 직접 flush해야 한다.

## 값의 절반은 떠날 때 생긴다

web-vitals는 다섯 지표를 같은 시점에 보고하지 않는다. `reportAllChanges` 없이 쓸 때 보고 시점은 다음과 같다.

| 지표 | 보고 시점                             | 근거 (v6.2.3)   |
| ---- | ------------------------------------- | --------------- |
| TTFB | `load` 뒤 한 번                       | `src/onTTFB.ts` |
| FCP  | 첫 contentful paint 때 한 번          | `src/onFCP.ts`  |
| LCP  | 첫 `keydown`이나 `click`, 또는 hidden | `src/onLCP.ts`  |
| CLS  | hidden 때                             | `src/onCLS.ts`  |
| INP  | hidden 때                             | `src/onINP.ts`  |

각 지표가 왜 그 시점에 확정되는지는 1편에서 소스로 따라갔다. 이 글에서 필요한 것은 결과다. 다섯 지표 중 CLS는 항상, INP는 상호작용이 있었던 경우에, LCP는 입력 없이 떠나는 경우에 페이지를 떠나는 순간 처음으로 값이 생긴다. 상호작용이 한 번도 없으면 INP는 초기값 -1에서 바뀌지 않아 보고되지 않는다. 주기 export에 실려 미리 나가 있을 수가 없는 값들이니, 떠나는 순간의 전송이 실패하면 이 지표들은 통째로 빠진다.

## 1.30.1의 sendBeacon은 실패를 성공으로 보고한다

사내 코드는 OTel JS 1.30.1과 OTLP exporter 0.57.2를 쓰고 있었다. 이 버전의 브라우저 exporter는 `headers` 옵션이 없으면 `navigator.sendBeacon`으로, 있으면 XHR로 보낸다. 재현 환경에서 다른 origin에 가짜 collector를 띄우고 응답을 바꿔 가며 exporter가 보고하는 결과를 봤다.

| exporter                           | collector가 200 | 400         | CORS 거부                  |
| ---------------------------------- | --------------- | ----------- | -------------------------- |
| 0.57.2, `headers` 없음(sendBeacon) | success         | **success** | **success**(서버 수신 0건) |
| 0.57.2, `headers: {}`(XHR)         | success         | failed      | failed                     |
| 0.223.0(keepalive fetch)           | success         | failed      | 5번 시도 후 failed(약 8초) |

> 각 조합을 두 번 돌려 같은 결과를 얻었다. 측정 코드는 `cwv-otel-grafana/experiments/e2-exporter`에 남겼다.

sendBeacon은 요청을 브라우저 큐에 넣었는지만 알려 준다. 0.57.2는 그 값을 그대로 성공으로 보고해서, CORS에 막혀 서버가 한 건도 받지 못해도 exporter에서는 아무 일도 없었던 것처럼 보인다. 사내에서는 평소에는 XHR로 보내 실패가 보이게 하고, 페이지가 숨겨졌을 때만 sendBeacon으로 바꿨다. 떠나는 순간의 전송만큼은 결과를 알 수 없는 채로 둔 것이다.

## 최신 exporter는 fetch keepalive로 보낸다

최신 버전에서는 이 분기가 사라졌다. 0.204.0부터 `headers`가 있으면 XHR보다 fetch를 먼저 쓰고, 0.211.0에서 XHR 전송이 없어졌고, 0.213.0에서 sendBeacon도 keepalive fetch로 바뀌었다[^1]. keepalive fetch는 문서가 사라진 뒤에도 요청을 끝까지 보내 주는 fetch라서 sendBeacon이 하던 일을 그대로 하면서, 응답 상태를 돌려준다. 앞의 표에서 0.223.0이 400을 바로 failed로 보고한 것이 그 결과다. CORS 거부는 네트워크 오류로 보여 다섯 번 다시 시도한 뒤에 failed가 된다.

브라우저는 진행 중인 keepalive 요청의 본문을 모두 합쳐 64KiB까지만 허용하고, Chrome은 렌더러 프로세스당 동시에 9개까지만 받는다. 0.223.0의 fetch 전송은 이 한도에 걸리지 않도록 진행 중인 keepalive 요청의 누적 크기(헤더를 위한 여유를 남긴 60KiB)와 개수(9개)를 직접 세다가, 넘으면 그 요청은 keepalive 없이 보낸다. keepalive 없이 보낸 요청은 문서가 사라지면 함께 취소되므로, 떠나는 순간에 요청이 몰리면 이 한도가 다시 손실 지점이 될 수 있다.

전송 방식만 보면 최신 버전으로 가는 편이 확실히 낫다. 실패가 보이고, 떠나는 순간에도 요청이 끝까지 간다. 남은 것은 언제 flush하느냐였고, 처음 표의 손실도 여기서 생겼다.

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

주석의 WebKit 버그는 Safari가 문서를 떠날 때 `visibilitychange`를 보내지 않던 문제다. MDN의 브라우저 호환성 데이터(BCD)에도 iOS Safari 14.5 이전에는 그랬다고 적혀 있다. 사내 가이드가 hidden과 pagehide 두 곳에서 flush하라고 한 것도 같은 이유였을 것이다. 그런데 `pagehide`는 Window에서 발생하는 이벤트다. 같은 탭에서 링크로 이동할 때 어떤 리스너가 어떤 순서로 불리는지 sessionStorage에 기록해 다음 문서에서 읽어 봤다.

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

> Chromium 156에서 두 번 실행해 같은 결과를 얻었다. `cwv-otel-grafana/experiments/e3-lifecycle`.

`document`에 건 `pagehide` 리스너는 한 번도 불리지 않았다. pagehide는 document를 거치지 않으니 BatchSpanProcessor의 Safari 대비책은 적어도 Chromium에서는 동작하지 않는다. 그리고 순서가 `pagehide`, `visibilitychange` 순이다. pagehide가 올 때 문서는 아직 visible이고, web-vitals는 그다음 `visibilitychange`에서 CLS와 INP를 보고한다. pagehide에서 flush하면 CLS와 INP가 기록되기 전에 한 번, visibilitychange에서 기록된 뒤에 또 한 번, 이렇게 flush가 두 번 일어난다.

## 2.9.0부터 forceFlush는 진행 중인 export를 기다린다

flush가 두 번 일어나는 것 자체는 문제가 아니어야 한다. 두 번째 flush가 CLS와 INP를 담아 보내면 된다. 처음 표의 결과는 두 번째 flush가 아무것도 보내지 못했다는 뜻이고, 원인은 sdk-metrics의 `forceFlush`가 진행 중인 export를 기다린다는 데 있었다[^3].

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

이 대기를 직접 보려고 페이지를 살려 둔 채 두 이벤트를 다른 task로 보내 봤다. CLS와 INP를 담은 두 번째 요청은 첫 요청의 응답이 도착하고 0.4ms 뒤에야 나갔다(두 번 실행해 응답 7346.7ms 뒤 7347.1ms, 7352.3ms 뒤 7352.9ms). 실제로 떠나는 문서는 그 응답이 오기 전에 사라지므로, 두 번째 collect가 일어날 기회가 없다고 보는 것이 처음 표의 결과와 맞는다. 첫 요청은 keepalive fetch라서 문서가 사라진 뒤에도 끝까지 가고, 그래서 TTFB, FCP, LCP만 도착한다.

이 대기는 2.9.0에서 들어왔다(#6655). 사내 코드가 쓰던 sdk-metrics 1.30.1의 `onForceFlush`에는 이 대기가 없다. 같은 flush 코드라도 sdk-metrics를 2.9.0 이상으로 올리면 CLS와 INP를 오류 없이 잃게 된다. 대기가 들어온 이유인 "collect와 export를 동시에 돌리지 않는다"는 서버 프로세스에서는 자연스러운 보장이라고 생각한다. 다만 떠나는 문서에는 응답을 기다릴 시간이 없다. trace는 BatchSpanProcessor의 `forceFlush`가 이렇게 기다리지 않아서 같은 조건에서도 도착한다.

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

대가도 있다. pagehide를 빼면 iOS Safari 14.5 이전처럼 떠날 때 `visibilitychange`가 오지 않는 브라우저에서는 주기 export를 기다리던 값이 빠진다. 다만 그런 브라우저에서는 web-vitals도 hidden을 받지 못해 CLS와 INP를 보고하지 않으므로, 잃는 것은 TTFB, FCP, LCP 중 아직 나가지 않은 값이다. CLS와 INP를 매번 잃는 것보다는 이 손실이 작다고 판단했다. 떠날 때 hidden이 오지 않는 환경이 오래된 Safari만은 아니라는 점은 [3편](/2026/10/cwv-to-tempo-and-mimir)의 앱 웹뷰에서 다시 나온다.

다만 이 수정으로 없앤 것은 pagehide에서 시작한 export와의 경쟁뿐이다. `onForceFlush`는 어떤 export든 진행 중이면 먼저 기다리므로, hidden이 오는 순간 10초 주기 export가 나가 있거나 quiet LCP를 확정하며 부른 flush(3편)가 아직 응답을 받지 못했다면 같은 대기가 걸린다. 그 사이 문서가 사라지면 CLS와 INP는 같은 방식으로 빠진다. 처음 표의 실험은 첫 주기 export 전에 떠나는 방문만 만들었으므로, 이 경우가 얼마나 자주 일어나는지는 재지 않았다. 대기 자체를 없애려면 주기 export를 끄고 hidden에서만 보내면 된다. 하지만 떠날 때 hidden이 오지 않는 앱 웹뷰에서는 주기 export가 값을 내보낼 유일한 기회라서 이 방법을 쓸 수 없었다. 같은 수집을 일반 브라우저가 찾아오는 블로그에 올렸을 때는 이 방법을 골랐고, 그 이야기는 4편에서 다룬다.

## 버전을 올릴 때 다시 세야 하는 것

이 글의 두 변화는 모두 각자의 자리에서 옳은 변경이었다. exporter는 실패를 숨기던 sendBeacon을 버리고 결과를 알려 주는 keepalive fetch로 바꿨고, sdk-metrics는 collect와 export가 겹치지 않도록 flush가 앞선 export를 기다리게 했다. 그런데 두 번째 변경이 들어오자, 떠나기 직전에 두 번 부르던 flush의 의미가 바뀌었다. 바뀐 결과는 오류가 아니라 도착하지 않은 값으로만 나타났다.

떠나는 순간의 전송은 결과를 페이지 안에서 확인할 방법이 사실상 없다. 응답이 오기 전에 문서가 사라지기 때문이다. 그래서 수집 코드나 OTel 버전을 바꿀 때는, 떠나는 방문을 만들어 지표별로 보낸 건수와 저장소에 도착한 건수를 세어 보는 것이 가장 확실한 확인 방법이라고 생각한다. 처음 표도 그렇게 세다가 나왔다. 그 표에서 TTFB, FCP, LCP만 보고 있었다면 이 손실은 꽤 오래 보이지 않았을 가능성이 높다.

[^1]: [opentelemetry-js experimental/v0.223.0 `experimental/CHANGELOG.md`](https://github.com/open-telemetry/opentelemetry-js/blob/experimental/v0.223.0/experimental/CHANGELOG.md)의 #5807, #6317, #6391 항목

[^2]: [opentelemetry-js v2.12.0 `packages/sdk-trace/src/platform/browser/export/BatchSpanProcessor.ts`](https://github.com/open-telemetry/opentelemetry-js/blob/v2.12.0/packages/sdk-trace/src/platform/browser/export/BatchSpanProcessor.ts#L36-L42)

[^3]: [opentelemetry-js v2.12.0 `packages/sdk-metrics/src/export/PeriodicExportingMetricReader.ts`](https://github.com/open-telemetry/opentelemetry-js/blob/v2.12.0/packages/sdk-metrics/src/export/PeriodicExportingMetricReader.ts#L259-L273). 이 동작은 [`CHANGELOG.md`](https://github.com/open-telemetry/opentelemetry-js/blob/v2.12.0/CHANGELOG.md)의 2.9.0 항목(#6655)에서 들어왔다.
