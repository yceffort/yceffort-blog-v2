---
title: 'web-vitals는 <em>언제</em> CWV를 보고하는가'
tags:
  - web-performance
  - browser
published: false
date: 2026-10-17 09:00:00
description: 'web-vitals 6.2.3의 콜백은 값이 생길 때가 아니라 보고할 때가 됐다고 판단할 때 불린다. 그 판단을 하는 bindReporter, 가시성 감시, idle 대기를 소스로 따라가며 보고되지 않는 방문과 여러 번 보고되는 방문을 정리하고, 상호작용 직후 떠나면 INP가 빠지는 문제를 Chromium에서 다시 쟀다.'
series: 'Core Web Vitals 직접 수집하기'
seriesOrder: 1
---

## Table of Contents

## 버튼을 누르고 0.3초 뒤에 떠난 방문

처리에 200ms가 걸리는 버튼을 한 번 누르고, 0.3초 뒤에 다른 페이지로 이동하는 방문을 Playwright로 띄운 Chromium 156에서 10번 만들었다. web-vitals 6.2.3의 다섯 콜백 중 TTFB, FCP, LCP, CLS는 매번 불렸고, INP는 한 번도 불리지 않았다. 떠나는 시점만 1.5초 뒤로 늦추자 INP가 10번 모두 보고됐다(200~208ms). 브라우저는 두 경우 모두 그 클릭을 Event Timing 항목으로 web-vitals에 넘겼다. 다른 점은 web-vitals가 그 항목을 처리하기 전에 페이지가 숨겨졌느냐 하나였다.

값을 받는 쪽에서는 이 두 방문을 구별할 수 없다. INP가 오지 않은 방문은 상호작용이 한 번도 없었던 방문과 똑같이 보인다. 그래서 이 시리즈에서 CWV를 trace와 metric으로 나눠 모으기 전에, web-vitals가 어떤 조건에서 콜백을 부르고 어떤 조건에서 부르지 않는지부터 정리할 필요가 있었다. 콜백을 부를지는 세 곳의 코드가 정한다. 보고할 값을 거르는 `bindReporter`, 페이지가 숨겨졌는지 지켜보는 가시성 감시, 처리를 idle까지 미루는 `whenIdleOrHidden`이다. 위의 INP는 이 세 곳이 어긋나는 순간에 사라진다.

> 기준 버전은 web-vitals [v6.2.3](https://github.com/GoogleChrome/web-vitals/tree/v6.2.3)(2026-10-05 배포, 커밋 `400d019`)이고, 인용한 소스와 줄 번호는 모두 이 태그 기준이다. 6.2.3 이후 main에 들어온 수정은 커밋 [`e773747`](https://github.com/GoogleChrome/web-vitals/tree/e773747b90f0e8763aed02ba5a74358d628a828c)(2026-10-16) 기준으로 따로 표시했다. 이 글을 쓴 2026-10-17에는 이 수정이 담긴 버전이 배포되지 않았다.
>
> 측정은 Playwright 1.64.1의 Chromium 156(new headless)에서 했고, web-vitals는 소스에서 직접 빌드했다(6.2.3과 4.2.1 빌드는 npm 배포본과 바이트 단위로 같았다). 클릭은 Chrome DevTools Protocol(CDP)로 보냈고, 핵심 조건은 DevTools를 붙이지 않은 Chrome Beta 157 창에 macOS 마우스 이벤트로 클릭해 다시 쟀다. 측정 코드는 [blog-experiments/cwv-otel-grafana/experiments/e13-report-timing](https://github.com/yceffort/blog-experiments/tree/b9c014609b3a604d1c169239bf327020813411ed/cwv-otel-grafana/experiments/e13-report-timing)에 있다.

## 콜백은 bindReporter를 거쳐서만 불린다

다섯 지표의 함수(`onTTFB`, `onFCP`, `onLCP`, `onCLS`, `onINP`)는 값을 계산하는 방법이 서로 다르지만, 사용자가 넘긴 콜백을 직접 부르지는 않는다. 각 함수는 `bindReporter`가 돌려준 `report` 함수를 들고 있다가 보고할 때가 됐다고 판단하면 그 함수를 부르고, 콜백을 부를지는 `report` 안에서 한 번 더 걸러진다[^1].

```ts
// web-vitals v6.2.3 src/lib/bindReporter.ts
export const bindReporter = <MetricName extends MetricType['name']>(
  callback: (metric: Extract<MetricType, {name: MetricName}>) => void,
  metric: Extract<MetricType, {name: MetricName}>,
  thresholds: MetricRatingThresholds,
  reportAllChanges?: boolean,
) => {
  let prevValue: number;
  let delta: number;
  return (forceReport?: boolean) => {
    if (metric.value >= 0) {
      if (forceReport || reportAllChanges) {
        delta = metric.value - (prevValue ?? 0);

        // Report the metric if there's a non-zero delta or if no previous
        // value exists (which can happen in the case of the document becoming
        // hidden when the metric value is 0).
        // See: https://github.com/GoogleChrome/web-vitals/issues/14
        if (delta || prevValue === undefined) {
          prevValue = metric.value;
          metric.delta = delta;
          metric.rating = getRating(metric.value, thresholds);
          callback(metric);
        }
      }
    }
  };
};
```

조건은 세 가지다.

- **`metric.value >= 0`**: 지표 객체를 만드는 `initMetric`의 기본값이 -1이다. CLS만 `initMetric('CLS', 0)`으로 0에서 시작하고, 나머지 넷은 -1에서 시작한다. 그래서 잴 대상이 한 번도 생기지 않은 지표는 `report(true)`를 몇 번 불러도 보고되지 않는다. 상호작용이 없는 방문의 INP가 대표적이다.
- **`forceReport || reportAllChanges`**: `reportAllChanges` 없이 쓰면 각 함수가 `report(true)`를 부르는 순간만 보고 시점이 된다. 그러니 "언제 보고되는가"는 다섯 함수가 `report(true)`를 어디서 부르는가와 같은 질문이다.
- **`delta || prevValue === undefined`**: 첫 보고는 값이 0이어도 나간다. 레이아웃 이동이 한 번도 없는 페이지에서도 CLS 0이 보고되는 이유다. 그 뒤로는 값이 바뀌었을 때만 나가므로, 같은 값으로 `report(true)`를 다시 불러도 아무 일도 일어나지 않는다.

하나 더 짚어 둘 것은 콜백이 받는 `metric`이 매번 같은 객체라는 점이다. `report`는 `value`, `delta`, `rating`을 덮어쓴 다음 같은 객체를 넘긴다. README의 배치 예제가 보고를 배열이 아니라 `Set`에 모으는 것도 이 성질 덕분에 맞게 동작한다[^2]. 같은 지표가 두 번 보고돼도 `Set`에는 하나만 남고, flush하는 순간 그 객체는 가장 최근 값을 들고 있다. 반대로 콜백에서 받은 객체를 그대로 붙잡아 두었다가 나중에 읽는 코드는, 콜백이 불렸을 때의 값이 아니라 그 뒤에 덮어쓴 값을 읽게 된다.

이제 남은 질문은 두 가지다. 각 함수가 `report(true)`를 언제 부르는가, 그리고 그 전에 값을 -1에서 바꿀 기회를 얻는가.

## 숨겨진 뒤에 그린 것은 세지 않는다

FCP와 LCP는 페이지가 처음 숨겨진 시각보다 먼저 그려졌을 때만 값을 받는다.

```ts
// web-vitals v6.2.3 src/onFCP.ts
          // Only report if the page wasn't hidden prior to FCP.
          if (entry.startTime < visibilityWatcher.firstHiddenTime) {

// web-vitals v6.2.3 src/onLCP.ts
        // Only report if the page wasn't hidden prior to LCP.
        if (renderTime < visibilityWatcher.firstHiddenTime) {
```

`firstHiddenTime`은 모든 지표가 같이 쓰는 가시성 감시(`getVisibilityWatcher`)가 들고 있다. 처음 불릴 때 값을 정하고, 그 뒤로는 window에 capture로 건 `visibilitychange` 리스너로 갱신한다[^3].

```ts
// web-vitals v6.2.3 src/lib/getVisibilityWatcher.ts
    // Prefer that, but if it's not available and the document is hidden when
    // this code runs, assume it was hidden since navigation start. This isn't
    // a perfect heuristic, but it's the best we can do until the
    // `visibility-state` performance entry becomes available in all browsers.
    firstHiddenTime = firstVisibilityStateHiddenTime ?? initHiddenTime();

    // Listen for visibility changes so we can handle things like bfcache
    // restores and/or prerender without having to examine individual
    // timestamps in detail and also for onHidden function calls.
    addEventListener('visibilitychange', onVisibilityUpdate, true);
```

`firstVisibilityStateHiddenTime`은 브라우저가 가시성이 바뀔 때마다 남기는 `visibility-state` 성능 항목에서 찾은 첫 hidden 시각이다. 이 항목은 Chromium 115부터만 있다[^4]. Firefox와 Safari에서는 `initHiddenTime()`이 대신하는데, 스크립트가 실행되는 순간 숨겨져 있으면 처음부터 숨겨져 있었다고 보고(0) 보이면 아직 숨겨진 적이 없다고 본다(`Infinity`). 그래서 이 브라우저들에서는 web-vitals가 실행되기 전에 잠깐 숨겨졌다가 돌아온 페이지가 숨겨진 적 없는 페이지로 취급된다.

이 규칙에서 몇 가지 조합이 따라 나온다.

백그라운드 탭으로 연 페이지는 `firstHiddenTime`이 0이라 FCP와 LCP가 끝까지 보고되지 않는다. CLS도 같이 빠진다. `onCLS`는 FCP가 보고된 다음에야 레이아웃 이동을 모으기 시작하기 때문이다.

```ts
// web-vitals v6.2.3 src/onCLS.ts
  const visibilityWatcher = getVisibilityWatcher();
  // Start monitoring FCP so we can only report CLS if FCP is also reported.
  // Note: this is done to match the current behavior of CrUX.
  onFCP(
    runOnce(() => {
```

반면 `onINP`와 `onTTFB`는 `firstHiddenTime`을 보지 않는다. 백그라운드에서 연 탭을 나중에 열어 버튼을 누르고 닫으면, 그 방문에는 TTFB와 INP만 있고 FCP, LCP, CLS는 없다. Speculation Rules 등으로 prerender해 둔 페이지는 활성화될 때까지(`whenActivated`) 다섯 함수 모두 아무것도 재지 않으므로, 사용자가 끝내 그 페이지로 가지 않으면 보고가 하나도 없다.

이것들은 버그가 아니라 사용자가 보지 않은 화면의 그리기를 세지 않으려는 의도된 동작이다. 다만 결과적으로 같은 방문 묶음에서도 지표별 건수가 같을 이유가 없다. CLS 건수가 FCP보다 적고, INP가 LCP보다 적은 것만으로는 손실이라고 볼 수 없다.

## 지표마다 report(true)를 부르는 곳이 다르다

다섯 함수가 `report(true)`를 부르는 곳과, 그 전에 방문이 끝나면 보고되지 않는 경우를 소스에서 정리하면 다음과 같다.

| 지표 | `report(true)`를 부르는 곳                 | 보고되지 않는 경우                        |
| ---- | ------------------------------------------ | ----------------------------------------- |
| TTFB | `load` 이벤트 다음 task                    | `load` 전에 떠남                          |
| FCP  | 첫 contentful paint 항목을 받은 즉시       | 그리기 전에 숨겨짐                        |
| LCP  | 첫 `keydown`, `click` 뒤 idle, 또는 hidden | 후보가 그려지기 전에 숨겨짐               |
| CLS  | hidden이 올 때마다                         | FCP가 보고되지 않음, Chromium 외 브라우저 |
| INP  | hidden이 올 때마다                         | 처리된 상호작용이 없음                    |

### TTFB는 이미 정해진 값을 load까지 들고 있다

TTFB의 값은 `navigation` 항목의 `responseStart`이고, 응답의 첫 바이트가 도착한 순간 정해진다. 그런데 `onTTFB`는 그 값을 `load` 이벤트 다음 task까지 들고 있다가 보고한다. 함수 주석은 `navigation` 항목의 다른 속성까지 모두 채워진 다음에 넘기기 위해서라고 설명한다[^5].

```ts
// web-vitals v6.2.3 src/onTTFB.ts
const whenReady = (callback: () => void) => {
  if (document.prerendering) {
    whenActivated(() => whenReady(callback));
  } else if (document.readyState !== 'complete') {
    addEventListener('load', () => whenReady(callback), true);
  } else {
    // Queue a task so the callback runs after `loadEventEnd`.
    setTimeout(callback);
  }
};
```

그래서 `load` 전에 떠난 방문에는 TTFB가 없다. 이미지 하나 때문에 `load`가 3초 뒤에 오는 페이지를 만들고, 1초 뒤와 5초 뒤에 떠나게 했다.

| 떠난 시점 | TTFB | FCP | LCP | CLS |
| --------- | ---- | --- | --- | --- |
| 1초 뒤    | 0/5  | 5/5 | 5/5 | 5/5 |
| 5초 뒤    | 5/5  | 5/5 | 5/5 | 5/5 |

> Chromium 156(new headless), web-vitals 6.2.3. 값은 보고된 방문 수 / 방문 수다.

1초 뒤에 떠난 방문도 `responseStart`는 1~3ms로 이미 정해져 있었지만 TTFB는 한 번도 보고되지 않았다. 같은 방문에서 FCP, LCP, CLS는 모두 보고됐으므로, 이 방문 묶음만 보면 TTFB만 빠진 것처럼 보인다.

`load`가 늦는 방문일수록 TTFB가 빠지기 쉬우므로, 모은 TTFB는 `load`까지 머문 방문 쪽으로 치우친다. 서버 응답이 느려 `load`까지 늦어지는 페이지라면 느린 TTFB가 더 많이 빠질 가능성이 높다고 생각한다. 이 치우침이 실제로 얼마나 되는지는 재지 않았다.

### LCP는 입력이나 hidden이 끝을 알린다

브라우저는 더 큰 요소를 그릴 때마다 LCP 후보를 새로 낸다. web-vitals는 `reportAllChanges` 없이 쓰면 후보를 받을 때마다 값만 바꿔 두고, 끝났다는 신호를 받으면 한 번 보고한다. 그 신호가 window에 capture로 건 `keydown`, `click`, `visibilitychange`다[^6].

```ts
// web-vitals v6.2.3 src/onLCP.ts
      const finalizeEventTypes = ['keydown', 'click', 'visibilitychange'];

      const finalizeLCP = (event: Event) => {
        if (event.isTrusted && !isFinalized) {
          // Wrap the listener in an idle callback so it's run in a separate
          // task to reduce potential INP impact.
          // https://github.com/GoogleChrome/web-vitals/issues/383
          const metricIdToFinalize = metric.id;
          whenIdleOrHidden(() => {
            if (!isFinalized) {
              if (!softNavsEnabled) {
                // Do some clean up since these won't be needed anymore.
                po!.disconnect();
                for (const type of finalizeEventTypes) {
                  removeEventListener(type, finalizeLCP, {capture: true});
                }
              }
              // As this is in a whenIdleOrHidden check, whether we're still
              // on the metric you meant to finalize, and ignore if we've moved
              // on in the meantime.
              if (metricIdToFinalize === metric.id) {
                isFinalized = true;
                report(true);
              }
            }
          });
        }
      };
```

이 코드에서 읽을 수 있는 것이 몇 가지 있다.

- `isTrusted`를 보므로 스크립트로 만든 클릭이나 키 입력은 확정 신호가 되지 않는다.
- 스크롤은 브라우저가 후보 기록을 멈추는 입력이지만, 프로그램으로 만들 수 있어 확정 신호에서 뺐다(같은 파일 242~244행 주석). 스크롤만 하고 떠난 방문의 LCP는 hidden에서 보고된다.
- 확정은 입력 즉시 하지 않고 `whenIdleOrHidden`으로 미룬다. 확정하면서 부르는 사용자 콜백(분석 스크립트 등)이 클릭 핸들러 안에서 돌면 그 클릭의 INP가 늘어난다는 지적(#383)이 있었기 때문이다. 이 대기가 실제로 얼마인지는 뒤에서 다시 나온다.
- `metricIdToFinalize`는 idle을 기다리는 사이에 SPA 이동이나 bfcache 복원으로 새 LCP가 시작됐을 때, 옛 지표를 새 지표의 값으로 확정하지 않으려는 장치다.
- 확정하면서 관찰자를 `disconnect()`하는데, 그 전에 관찰자에 남아 있는 항목을 `takeRecords()`로 꺼내지 않는다. 5.x에는 이 호출이 있었고 6.0.0에서 SPA 이동 지원을 넣으며 빠졌다. 그래서 확정 직전에 관찰자 큐에 남아 있던 후보는 버려질 수 있고, main에서 이 호출을 다시 넣었다(#811). 다만 PR 작성자도 실제 브라우저에서 재현하지는 못했다고 적었고, 나도 재현해 보지 않았다.

### CLS와 INP는 숨겨질 때 보고된다

CLS와 INP는 확정이라는 단계가 없고, 가시성 감시에 등록한 함수에서 `report(true)`를 부른다.

```ts
// web-vitals v6.2.3 src/onCLS.ts
        visibilityWatcher.onHidden(() => {
          handleEntries(
            po.takeRecords() as [LayoutShift | PerformanceSoftNavigation],
          );
          report(true);
        });

// web-vitals v6.2.3 src/onINP.ts
      visibilityWatcher.onHidden(() => {
        handleEntries(
          po.takeRecords() as [
            PerformanceEventTiming | PerformanceSoftNavigation,
          ],
          true, // forceReport after processing all entries
        );
      });
```

`onHidden`으로 등록한 함수는 가시성 감시의 리스너가 hidden을 받을 때 차례로 불린다. CLS는 관찰자에 남은 항목을 `takeRecords()`로 꺼내 반영한 다음 바로 보고한다. INP도 남은 항목을 꺼내지만, 그 항목을 `handleEntries`에 넘기면서 처리가 끝나면 보고하라는 표시(`true`)만 붙인다. 이 차이가 처음 본 INP 손실과 이어진다.

## 숨겨질 때마다 다시 보고된다

LCP, FCP, TTFB는 페이지 로드마다 한 번 보고된다. CLS와 INP는 hidden이 올 때마다 `report(true)`가 불리므로, 페이지가 다시 보였다가 값이 바뀐 다음 숨겨지면 같은 `id`로 한 번 더 보고된다. 두 함수의 문서 주석도 같은 페이지 로드에서 콜백이 여러 번 불릴 수 있다고 적고 있다. 탭을 오가는 동안에도 CLS와 INP는 계속 바뀔 수 있는데, 숨겨진 뒤에는 다시 보일지 알 수 없으니 숨겨질 때마다 그때까지의 값을 보고하는 것이다.

3편의 재현 환경에서 숨김, 보임, 위쪽 요소 삽입, 숨김 순서로 만든 방문은 같은 `id`로 CLS 0과 0.1124를 차례로 보고했다[^7]. Playwright의 Chromium에서는 탭을 바꿔도 hidden이 되지 않아서, 이 실험은 `visibilityState`를 덮어쓰고 `visibilitychange`를 보내 흉내 냈다. web-vitals의 hidden 처리는 `isTrusted`를 보지 않으므로 실제와 같은 코드 경로를 탄다.

그래서 받는 쪽은 `id`당 보고가 여러 번 올 수 있다는 것을 전제로 해야 한다. 보고마다 histogram에 한 번씩 넣으면 탭을 오간 방문이 두 번 세어진다. 이 시리즈의 블로그 수집 코드는 `id`마다 처음 온 보고만 histogram에 넣고, 다시 온 보고는 값이 얼마나 자주 바뀌는지 재는 데만 쓴다. 처음 값 대신 마지막 값을 쓰려면 같은 `id`의 값을 나중에 바꿀 수 있는 저장소가 필요한데, 그 구성은 [3편](/2026/10/cwv-to-tempo-and-mimir)에서 다룬다.

## idle까지 미루는 1초

LCP 확정과 INP 처리는 모두 `whenIdleOrHidden`을 거친다[^8].

```ts
// web-vitals v6.2.3 src/lib/whenIdleOrHidden.ts
export const whenIdleOrHidden = (cb: () => void) => {
  // Cap the requestIdleCallback to 1 sec for very busy apps
  // https://github.com/GoogleChrome/web-vitals/issues/754
  // If not using rIC, then the setTimeout timeout should be 0
  const timeout = 'requestIdleCallback' in globalThis ? 1000 : 0;

  const rIC = globalThis.requestIdleCallback || setTimeout;
  const cIC = globalThis.cancelIdleCallback || clearTimeout;

  // If the document is hidden, run the callback immediately, otherwise
  // race an idle callback with the next `visibilitychange` event.
  if (document.visibilityState === 'hidden') {
    cb();
  } else {
    const wrappedCb = runOnce(cb);

    let idleHandle = -1;
    const onHidden = () => {
      cIC(idleHandle);
      wrappedCb();
    };

    addEventListener('visibilitychange', onHidden, {once: true, capture: true});
    idleHandle = rIC(
      () => {
        removeEventListener('visibilitychange', onHidden, {capture: true});
        wrappedCb();
      },
      {timeout: timeout},
    );
  }
};
```

페이지가 보이는 동안에는 `requestIdleCallback`으로 다음 idle을 기다리고, 그 사이에 `visibilitychange`가 오면 기다리지 않고 바로 실행한다. 이미 숨겨져 있으면 즉시 실행한다. INP가 idle을 기다리는 이유는 `src/onINP.ts`의 주석에 있다. 한 상호작용에 딸린 이벤트 항목이 모두 도착할 때까지 기다리려는 것이다.

`timeout: 1000`은 6.0.0에서 들어왔다. 메인 스레드가 쉬지 않는 앱에서는 idle이 오지 않아 DevTools의 INP가 갱신되지 않는다는 이슈(#754)가 있었고, 그래서 idle을 최대 1초만 기다리게 했다(#755). Next.js 16.3.5가 `useReportWebVitals`용으로 번들하는 4.2.1에는 이 상한이 없다[^9]. `requestIdleCallback`이 없는 Safari에서는 `setTimeout`이 대신하므로 다음 task까지만 미뤄진다(소스로만 확인했고 Safari에서 재지는 않았다).

그러면 바쁘지 않은 페이지에서는 idle이 금방 올까. 버튼 하나만 있는 페이지에서 클릭 핸들러는 아무 일도 하지 않게 두고, 여러 시점에 `requestIdleCallback`을 걸어 불리기까지 걸린 시간을 쟀다.

| idle 콜백을 건 시점                          | 불리기까지                                |
| -------------------------------------------- | ----------------------------------------- |
| 첫 클릭 전, 입력 없이                        | 0~1ms                                     |
| 클릭 핸들러 안                               | 3~16ms                                    |
| 클릭의 Event Timing 항목을 받은 관찰자 콜백  | 3초 timeout까지 불리지 않음(19번 중 19번) |
| 같은 관찰자 콜백, macOS 마우스 이벤트로 클릭 | 10초 timeout까지 불리지 않음(5번 중 5번)  |
| 클릭 1초 뒤, 그 뒤로 입력 없음               | 15초 timeout까지 불리지 않음              |
| 클릭 1초 뒤, 6초 뒤에 마우스 이동            | 마우스를 움직인 직후                      |

> 넷째 줄을 빼면 클릭과 마우스 이동은 CDP로 넣었다. 첫 세 줄은 Playwright 1.64.1로 띄운 Chromium 156(headless, headed)과, Playwright 없이 직접 띄우고 CDP로만 붙은 Chrome Beta 157(headless, headed)에서 같은 결과가 나왔다. 넷째 줄은 DevTools를 붙이지 않고 창을 띄운 Chrome Beta 157.0.8103.6에 macOS의 HID 이벤트 탭으로 마우스 이벤트를 넣어 쟀다. 물리 마우스의 입력이 창 서버로 들어오는 지점이다. 마지막 두 줄은 Chromium 156에서만 쟀다.

클릭 핸들러 안에서 건 idle 콜백은 곧바로 불렸지만, 클릭이 끝난 뒤에 건 idle 콜백은 다음 입력이 들어오기 전까지 불리지 않았다. 처음에는 클릭을 1.3초 간격으로 반복하며 재다가 idle이 약 1.28초 뒤에 온다고 읽었는데, 그것은 다음 클릭이 들어간 시각이었다. 간격을 3.5초로 늘리자 idle도 그만큼 늦게 왔다. web-vitals가 INP 항목을 받는 관찰자 콜백은 클릭 다음 화면이 그려진 뒤에 불리므로 이쪽에 해당한다. CDP로 넣은 입력이라 생긴 현상일 수도 있어서, DevTools를 붙이지 않은 브라우저에 macOS 마우스 이벤트로 클릭해 다시 쟀다. 결과는 같았다. 관찰자 콜백에서 건 idle 콜백은 5번 모두 10초 timeout까지 불리지 않았고, 그동안 페이지에 들어온 입력은 없었다. 클릭 3초 뒤에 마우스를 움직여 페이지에 `pointermove`가 들어간 두 번은 그 직후에 불렸다. 원인이 Chromium 스케줄러 어디에 있는지는 코드로 확인하지 못했다. 마우스를 쓰는 방문자는 클릭한 뒤에도 커서를 움직이는 경우가 많으니 그때 idle이 올 것이고, 다음 탭까지 아무 이벤트가 없는 터치 화면에서는 timeout까지 기다리는 경우가 더 많을 것이라고 생각한다(모바일 기기에서는 재지 않았다).

정리하면 6.2.3에서 INP 항목은 다음 idle이나 1초 timeout 중 먼저 오는 때에 처리된다. 메인 스레드가 바쁜 페이지에서는 1초를 거의 다 채우고, 한가한 페이지도 클릭 뒤에 다른 입력이 없으면 1초를 채운다.

## 그 1초 안에 떠나면 INP가 보고되지 않는다

페이지가 숨겨질 때 window의 capture 단계에는 `visibilitychange` 리스너가 둘 걸려 있다. 하나는 가시성 감시가 처음 만들어질 때 건 리스너(`getVisibilityWatcher.ts` 88행)이고, 다른 하나는 상호작용 항목이 도착했을 때 `whenIdleOrHidden`이 건 리스너(`whenIdleOrHidden.ts` 45행)다. 같은 대상, 같은 단계의 리스너는 건 순서대로 불리므로 가시성 감시가 먼저 불린다. 그 다음 일은 이렇게 흘러간다.

1. 가시성 감시가 INP의 `onHidden` 함수를 부른다. `po.takeRecords()`는 비어 있다. 클릭의 항목은 이미 관찰자 콜백으로 전달돼 idle을 기다리는 중이기 때문이다(아래 측정에서 같은 시점에 다른 관찰자로 꺼내 봐도 매번 비어 있었다).
2. 빈 목록으로 `handleEntries(..., true)`가 불리고, 이미 숨겨진 상태라 바로 실행된다. `InteractionManager`에는 아직 이 클릭이 없으므로 `metric.value`는 -1 그대로이고, `report(true)`는 아무것도 보고하지 않는다. 앞서 처리된 상호작용이 있었다면 그 값이 대신 보고된다.
3. 이어서 `whenIdleOrHidden`의 리스너가 기다리던 처리를 실행한다. 이제 클릭이 반영돼 값이 200이 되지만, 이 처리가 부르는 `report()`에는 `forceReport`가 없어 `bindReporter`에서 걸러진다.

페이지가 다시 보였다가 숨겨지면 그때 보고되지만, 다른 페이지로 이동하거나 탭을 닫은 방문에는 다음 hidden이 없다. 클릭한 뒤 떠나는 시점과 방법을 바꿔 가며 INP가 보고된 방문을 셌다. 클릭 핸들러는 200ms 동안 메인 스레드를 잡는다. 바쁜 페이지는 클릭 뒤 3초 동안 45ms짜리 작업을 쉬지 않고 이어 붙여 idle이 생기지 않게 했다. 조건마다 5번씩 두 차례 돌린 합계다.

| 떠나는 방법                        | 클릭 뒤 떠난 시점 | 6.2.3 한가한 페이지 | 6.2.3 바쁜 페이지 | main 한가한 페이지 | main 바쁜 페이지 |
| ---------------------------------- | ----------------- | ------------------- | ----------------- | ------------------ | ---------------- |
| 클릭 핸들러가 `location.href` 변경 | 0.1초             | 1/10                | 0/10              | 10/10              | 10/10            |
|                                    | 0.3초             | 0/10                | 0/10              | 10/10              | 10/10            |
|                                    | 0.6초             | 1/10                | 0/10              | 10/10              | 10/10            |
|                                    | 1.5초             | 10/10               | 10/10             | 10/10              | 10/10            |
| 링크, 응답 0.3초                   | 0.3초             | 0/10                | 0/10              | 10/10              | 10/10            |
| 링크, 응답 1.5초                   | 1.5초             | 10/10               | 10/10             | 10/10              | 10/10            |
| 링크, 응답 즉시                    | 약 0.005초        | 0/10                | 0/10              | 0/10               | 0/10             |
| 탭 닫기                            | 0.3초             | 0/10                | 0/10              | 10/10              | 10/10            |
|                                    | 1.5초             | 10/10               | 10/10             | 10/10              | 10/10            |

> Playwright 1.64.1의 Chromium 156(new headless)에서 쟀다. 값은 보고된 방문 수 / 방문 수이고, 보고된 INP는 모두 200~208ms였다. 떠난 시점은 클릭 핸들러가 끝난 뒤부터 잰 값이다. 바쁜 페이지에서는 이어 붙인 작업 때문에 항목 전달과 이동이 0.1초 남짓 늦어진다.

6.2.3은 클릭 뒤 0.6초 안에 떠난 방문 100번 중 2번만 INP를 보고했고, 1.5초 뒤에 떠난 방문은 모두 보고했다. 경계를 좁혀 보면 0.9초에 떠난 방문은 한가한 페이지 5번 중 3번, 바쁜 페이지 5번 중 0번이 보고됐고, 1.1초는 모두 보고됐다. 1초 timeout 근처에서 갈리는 것이다. 한가한 페이지에서 timeout 전에 보고된 방문은 이동이 시작된 뒤 문서가 사라지기 전에 처리가 끝난 경우로 보인다. 한 번은 같은 페이지의 관찰자가 건 idle 콜백이 `pagehide` 1ms 전에 불린 것을 확인했지만, 나머지는 원인을 확인하지 못했다. main은 같은 조건에서 모두 보고했다.

한가한 페이지와 바쁜 페이지가 거의 같게 나온 것은 앞 절에서 본 대로 클릭 뒤 다른 입력이 없으면 idle이 오지 않아, 두 페이지 모두 1초 timeout에 기대 처리됐기 때문이다. 이 표의 클릭은 CDP 입력이라, 한가한 페이지에서 이동으로 떠나는 조건 일부를 DevTools를 붙이지 않은 Chrome Beta 157에 macOS 마우스 이벤트로 클릭해 다시 쟀다. 6.2.3은 0.3초에 떠난 방문 5번 중 0번, 1.5초에 떠난 방문 5번 중 5번을 보고했고, main은 0.3초에 떠난 방문 5번을 모두 보고했다(보고된 INP는 224~232ms). 다만 클릭한 뒤 떠나기 전에 마우스를 움직인 방문은 그때 idle이 와서 INP가 처리되므로, 실제 방문에서 한가한 페이지의 손실은 이 표보다 적을 수 있다. 바쁜 페이지는 마우스를 움직여도 메인 스레드가 쉬지 않아 이런 차이가 없을 것이고, 결과도 같은 이슈의 실제 Chrome 측정과 맞는다.

응답이 즉시 오는 링크는 두 버전 모두 보고하지 않았다. 이 경우는 클릭 핸들러가 끝나고 약 5ms 만에 `pagehide`가 왔고, 숨겨지는 순간 관찰자 큐를 `takeRecords()`로 직접 꺼내 봐도 클릭 항목이 없었다(6번 중 6번). web-vitals가 처리할 항목 자체가 문서가 사라지기 전에 만들어지지 않은 것이다. 라이브러리로는 어떻게 할 수 없는 손실이다.

Next.js 16.3.5가 번들하는 4.2.1은 timeout이 없어서 결과가 더 나빴다. 같은 환경에서 클릭 뒤 1.5초와 4.5초에 떠난 방문도 한가한 페이지와 바쁜 페이지 모두 한 번도 보고하지 않았다(각 5번 중 0번). 클릭 뒤 다음 입력이 없으면 idle이 오지 않으므로, 4.2.1은 떠나기 전에 다른 입력이 없었던 마지막 상호작용을 아예 처리하지 못한다.

이 문제는 2026-10-09에 web-vitals 이슈로 올라왔다(#805)[^10]. 이슈 작성자는 Chromium 141에서 hidden을 흉내 낸 재현으로 보고했고, 같은 이슈에서 다른 기여자가 실제 Chrome 155에서 바쁜 페이지의 600ms 클릭 뒤 300ms 만에 떠나면 INP가 빠진다는 것을 확인했다. 그 측정에서 한가한 페이지는 대조군으로 매번 보고됐다고 적혀 있는데, 대조군을 클릭 뒤 언제 숨겼는지는 나와 있지 않아 위의 표와 바로 비교하기는 어렵다. 메인테이너는 처음에 드문 경우일 것으로 봤지만 수정을 받아들였고, 10-15에 main에 병합됐다(#809). 그 수정은 두 가지를 바꿨다.

- `onINP`가 idle을 기다리는 항목을 한 목록(`pendingEntries`)에 모으고, hidden에서 보고하기 전에 그 목록까지 처리한다. 2번 단계에서 클릭이 반영된다.
- 가시성 감시가 `pagehide`에서도 `onHidden` 함수를 부른다. 문서를 떠날 때는 `pagehide`가 `visibilitychange`보다 먼저 오므로, 다음 버전부터 CLS와 INP는 `pagehide` 시점에 보고된다.

두 번째 변경은 [2편](/2026/10/cwv-send-on-page-leave)에서 다룬 전송 순서와도 맞물린다. 2편은 6.2.3이 `visibilitychange`에서 CLS와 INP를 보고한다는 것을 전제로 flush 시점을 골랐다. 그 전제는 이 수정이 배포되면 바뀐다.

수정이 배포되기 전에 6.2.3에서 이 손실을 피하는 방법으로는 INP만 `reportAllChanges: true`로 받는 방법이 있다. 그러면 3번 단계의 `report()`가 `bindReporter`를 통과해, 클릭이 반영된 값으로 콜백이 불린다. 이 콜백은 window의 capture 단계에서 불리므로, `document`에 건 `visibilitychange` 리스너에서 flush하면 그 값이 함께 나간다. 같은 환경에서 클릭 뒤 0.3초 만에 이동, 링크, 탭 닫기로 떠난 방문 30번(한가한 페이지와 바쁜 페이지)이 모두 INP를 보고했다. 대신 상호작용이 처리될 때마다 같은 `id`로 보고가 오므로, 받는 쪽은 `id`마다 마지막 값을 골라 보내야 한다. 앞에서 본 것처럼 콜백이 받는 `metric`은 같은 객체라서, `id`별로 객체를 들고 있다가 hidden에서 보내면 된다. 처음 온 보고만 쓰는 이 시리즈의 블로그 수집 코드에 이 방법을 그대로 넣으면 첫 상호작용 값만 남으므로, 쓰려면 INP만 따로 처리해야 한다.

## 돌아온 페이지의 값은 다른 것을 잰다

bfcache(뒤로, 앞으로 가기에서 페이지를 메모리에서 그대로 되살리는 캐시)로 복원된 페이지는 새 내비게이션으로 취급된다. 다섯 함수 모두 `pageshow`의 `persisted`를 보고 새 `id`의 지표를 만들고, `navigationType`은 `back-forward-cache`가 된다. 그런데 이때 보고되는 값은 처음 로드의 값과 같은 것을 재지 않는다.

```ts
// web-vitals v6.2.3 src/onLCP.ts
        doubleRAF(() => {
          metric.value = performance.now() - event.timeStamp;
          isFinalized = true;
          report(true);
        });
```

LCP와 FCP는 복원된 `pageshow`부터 `requestAnimationFrame` 두 번 뒤까지의 시간이다. 복원된 화면에서 어떤 요소가 그려졌는지는 보지 않는다. TTFB는 0으로 보고된다. INP는 복원 이후의 상호작용만 다시 세는데, 상호작용이 있었지만 `durationThreshold`(기본 40ms) 이상인 것이 하나도 없으면 실제 값 대신 8ms를 보고한다.

```ts
// web-vitals v6.2.3 src/lib/InteractionManager.ts
    // If we have a non-zero interactionCountForNavigation but no
    // candidateInteractionIndex, then it's below the 16ms limit
    // so report a dummy 8ms interaction. This is only needed for
    // soft-navs and bfcache restores as `first-input` handles the
    // rest.
    if (
      interactionCountForNavigation &&
      candidateInteractionIndex === -1 &&
      (navigationType === 'soft-navigation' ||
        navigationType === 'back-forward-cache')
    ) {
      return {
        _latency: 8,
        id: -1,
        entries: [],
      };
    }
```

주석은 16ms 한도 아래라서 8ms를 보고한다고 적지만, 후보 목록이 비는 조건은 관찰자의 `durationThreshold`(기본 40ms) 이상인 상호작용이 하나도 없을 때다. 상호작용 수는 `performance.interactionCount`로 세는데, 이 값은 길이와 상관없이 모든 상호작용을 센다(BCD 8.1.6 기준 Chrome 144, Firefox 144, Safari 26.2부터 지원). 그래서 기본값으로 쓰면 40ms 미만의 상호작용은 길이와 상관없이 8ms로 보고된다. 처음 로드에서는 `first-input` 항목이 기준값과 상관없이 들어와 후보가 되므로 이런 일이 없다.

처음 로드에서 버튼을 한 번 누르고 떠났다가, 뒤로 가기로 복원된 페이지에서 같은 버튼을 다시 누르고 떠나는 방문을 클릭 핸들러의 길이별로 5번씩 만들었다.

| 클릭 핸들러 | 처음 로드의 INP | 복원 뒤 INP | 복원 뒤 FCP, LCP | 복원 뒤 TTFB |
| ----------- | --------------- | ----------- | ---------------- | ------------ |
| 0ms         | 16ms            | **8ms**     | 17.3~37.8ms      | 0            |
| 20ms        | 24ms            | **8ms**     | 15.6~39.3ms      | 0            |
| 60ms        | 64ms            | 64ms        | 19.2~37.3ms      | 0            |

> Chromium 156(new headless, Playwright가 기본으로 넣는 `--disable-back-forward-cache`를 뺐다), web-vitals 6.2.3. 15번 모두 복원됐고, INP와 TTFB는 행마다 5번 모두 같은 값이었다.

처음 로드에서 16ms와 24ms로 보고된 클릭이 복원 뒤에는 둘 다 8ms가 됐고, 40ms 이상인 64ms 클릭만 그대로 보고됐다. 복원 뒤의 FCP와 LCP는 모든 방문에서 서로 1ms 안쪽으로 붙어 있었다. 둘 다 같은 `pageshow`에서 프레임 두 개를 기다린 시간이기 때문이다.

Soft Navigations API를 지원하는 Chromium 151 이상에서 `reportSoftNavs: true`로 SPA 이동을 잴 때도 비슷하다. README는 이동마다 TTFB를 0으로 보고하고, FCP와 LCP는 이동 뒤에 새로 그려진 것만 세며, INP는 이동 뒤의 상호작용만 다시 잰다고 적는다[^11]. 이동이 일어나는 순간 앞 내비게이션의 CLS와 INP는 그 시점 값으로 보고된다.

그래서 `navigationType`이 다른 값을 한 분포에 섞으면, 같은 이름의 지표가 서로 다른 것을 잰 값끼리 섞인다. bfcache 복원이 많은 사이트에서 LCP p75가 갑자기 좋아졌다면, 페이지가 빨라진 것이 아니라 복원 비율이 늘어난 것일 수 있다. 이 시리즈의 수집 코드는 `navigationType`을 라벨로 두고 처음 로드와 나머지를 나눠 본다.

## 보고되지 않은 값은 0도 오류도 아니다

web-vitals의 콜백은 값이 생겼다는 신호가 아니라, 보고할 때가 됐다고 web-vitals가 판단했다는 신호다. 그 판단은 `bindReporter`의 세 조건, 처음 숨겨진 시각, idle 대기가 정하고, 어느 조건에도 걸리지 않으면 콜백은 그냥 불리지 않는다. 숨겨진 뒤에 그려진 LCP나 `load` 전에 떠난 방문의 TTFB처럼 의도대로 빠지는 값이 있고, 클릭 직후 떠난 방문의 INP처럼 6.2.3의 리스너 순서 때문에 빠지는 값이 있다. 받는 쪽에서는 두 경우가 똑같이 "보고 없음"으로 보인다.

그래서 지표별 건수를 비교해 손실을 찾으려면, 먼저 web-vitals가 각 지표를 어떤 방문에서 보고하는지 알고 있어야 한다고 생각한다. 그 기준이 없으면 의도된 결측을 손실로 오해하거나, 진짜 손실을 의도된 결측으로 넘기게 된다. 이 글의 INP 손실은 이슈와 수정까지 나와 있지만 아직 배포되지 않았고, 배포되면 CLS와 INP의 보고 시점이 다시 바뀐다. web-vitals 버전을 올릴 때마다 떠나는 방문을 만들어 지표별 보고를 다시 세어 보는 것이 지금으로서는 가장 확실한 확인 방법일 것이다. 그렇게 보고된 값이 페이지를 떠나는 순간 어떻게 나가는지는 [2편](/2026/10/cwv-send-on-page-leave)에 이어진다.

[^1]: [web-vitals v6.2.3 `src/lib/bindReporter.ts`](https://github.com/GoogleChrome/web-vitals/blob/v6.2.3/src/lib/bindReporter.ts#L32-L58). 지표 객체의 초기값은 [`src/lib/initMetric.ts`](https://github.com/GoogleChrome/web-vitals/blob/v6.2.3/src/lib/initMetric.ts#L23-L31)

[^2]: [web-vitals v6.2.3 README의 "Batch multiple reports together"](https://github.com/GoogleChrome/web-vitals/blob/v6.2.3/README.md#batch-multiple-reports-together)

[^3]: [web-vitals v6.2.3 `src/lib/getVisibilityWatcher.ts`](https://github.com/GoogleChrome/web-vitals/blob/v6.2.3/src/lib/getVisibilityWatcher.ts#L63-L113)

[^4]: [MDN browser-compat-data](https://github.com/mdn/browser-compat-data) 8.1.6의 `api.VisibilityStateEntry`. 같은 데이터에서 `LayoutShift`는 Chrome만 지원하고, `LargestContentfulPaint`는 Firefox 122와 Safari 26.2부터, `PerformanceEventTiming.interactionId`와 `Performance.interactionCount`는 Firefox 144와 Safari 26.2부터 지원한다.

[^5]: [web-vitals v6.2.3 `src/onTTFB.ts`](https://github.com/GoogleChrome/web-vitals/blob/v6.2.3/src/onTTFB.ts#L30-L59)

[^6]: [web-vitals v6.2.3 `src/onLCP.ts`](https://github.com/GoogleChrome/web-vitals/blob/v6.2.3/src/onLCP.ts#L210-L249). 확정 전 `takeRecords()` 수정은 [#811](https://github.com/GoogleChrome/web-vitals/pull/811)

[^7]: [blog-experiments `cwv-otel-grafana/experiments/e12-browser-lifecycle`](https://github.com/yceffort/blog-experiments/tree/a855b62c575f053ea9d4fb1bfe2fa2ead53d7df8/cwv-otel-grafana/experiments/e12-browser-lifecycle)의 탭 복귀 시나리오

[^8]: [web-vitals v6.2.3 `src/lib/whenIdleOrHidden.ts`](https://github.com/GoogleChrome/web-vitals/blob/v6.2.3/src/lib/whenIdleOrHidden.ts#L23-L54). 1초 상한은 [#754](https://github.com/GoogleChrome/web-vitals/issues/754)와 [#755](https://github.com/GoogleChrome/web-vitals/pull/755)

[^9]: [Next.js v16.3.5 `packages/next/package.json`](https://github.com/vercel/next.js/blob/v16.3.5/packages/next/package.json)의 devDependencies `web-vitals: 4.2.1`. 4.2.1의 [`src/lib/whenIdle.ts`](https://github.com/GoogleChrome/web-vitals/blob/v4.2.1/src/lib/whenIdle.ts)는 `requestIdleCallback`을 timeout 없이 부른다.

[^10]: [web-vitals #805](https://github.com/GoogleChrome/web-vitals/issues/805), 수정은 [#809](https://github.com/GoogleChrome/web-vitals/pull/809)

[^11]: [web-vitals v6.2.3 README의 "Report metrics for soft navigations"](https://github.com/GoogleChrome/web-vitals/blob/v6.2.3/README.md#report-metrics-for-soft-navigations)
