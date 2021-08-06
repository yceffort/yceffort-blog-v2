---
title: '웹사이트의 성능지표, Core Web Vital'
tags:
  - web
  - javascript
  - browser
published: true
date: 2021-08-06 20:32:31
description: '조만간 웹사이트 하나씩 분석해 보겠습니다'
---

사용자 경험의 질을 향상시키는 것은, 어떤 사이트이든지 장기적으로 성공하는데 있어서 중요한 열쇠다. 비즈니스 오너, 마케터, 개발자건 상관없이, Web Vital (이하 웹 바이탈)은 사이트의 경험을 정량화 하고, 개선할 수 있는 기회를 찾아볼 수 있도록 지원해준다.

## 개요

웹 바이탈은 웹에서 훌륭한 사용자 경험을 전달하는데 필수적인 '품질'에 대한 통일된 지침을 제공하기 위한 일종의 구글의 이니셔티브다.

구글은 지난 수년동안 성능으르 측정하고 보고할 수 있는 많은 도구를 제공해왔다. 일부 개발자는 이런 툴을 사용하는데 익숙한 방면, 또 다른 개발자들은 이런 도구와 지표가 다양해짐에 따라 점차 대응하기 어려워 한다는 것을 알게 되었다.

사이트 소유자는, 사용자에게 제공되는 경험의 품질을 이해하기 위해 꼭 성능 전문가가 될 필요는 없다. 웹 바이탈은 환경을 단순화하고, 사이트에서 가장 중요한 지표인 핵심 웹 바이탈에 집중할 수 있도록 도와주는 것을 목표로 한다.

## Core web vital

Core web vital (이하 핵심 웹 바이탈)은 모든 웹페이지에 적용되는 웹 바이탈의 하위 집합으로, 모든 사이트 소유자가 측정해야 하며, 또한 모든 구글 도구에 걸쳐서 노출된다. 각 핵심 웹 바이탈은 사용자 경험의 고유한 측면을 나타내고, 즉시 측정가능하며, 사용자 중심 결과의 실제 경험을 반영한다.

이 핵심 웹 바이탈 지표는 계속해서 진화해 왔다. 2020년 현재는 크게 세가지 사용자 경험 측면에 집중한다.

- 로딩
- 상호작용
- 시각적 안정화

그리고 이 세가지는 아래의 지표에 기반한다.

![core-web-vital](https://web-dev.imgix.net/image/tcFciHGuF3MxnTr1y5ue01OGLBn2/iHYrrXKe4QRcb2uu8eV8.svg)

- [Largest Contentful Paint (LCP)](https://web.dev/lcp/): 로딩의 성능을 측정한다. 사용자에게 좋은 경험을 제공하기 위해서는, 적어도 2.5초 이내로 첫페이지 로딩이 이루어져야 한다.
- [First Input Delay (FID)](https://web.dev/fid/): 상호작용성을 측정한다. 좋은 사용자 환경을 제공하기 위해서는, 페이지의 FID가 100ms 미만이어야 한다.
- [Cumulative Layout Shift(CLS)](https://web.dev/cls/): 시각적 안정성을 측정한다. 좋은 사용자 환경을 제공하기 위해서는, 페이지가 0.1초 이하의 CLS를 유지해야 한다.

각 지표에 대해 대부분의 사용자에게 권장되는 목표치를 달성하기 위해서는, 모바일 및 데스크톱 장치에 걸쳐 여러번 반복된 페이지 로딩의 상위 75% 이내의 값을 측정해보는 것이 좋다. 핵심 웹 바이탈을 평가하는 도구에서는, 위 3가지 지표에 대해 75%이상을 충족하도록 권장하고 있다.

> 이런 추천이 어떻게 결정되었는지 이해하기 위해서는 https://web.dev/defining-core-web-vitals-thresholds/ 를 참고해보자.

### 핵심 웹 바이탈을 평가하고 보고하는 도구

구글은 핵심 웹 바이탈이 모든 웹 환경에서 중요한 것이라고 믿는다. 따라서, [다양한 도구](https://web.dev/vitals-tools/) 에서 이러한 것들을 측정할 수 있도록 도와주고 있다.

#### 핵심 웹 바이탈을 측정하기 위한 도구

[Chrome User Experience Report](https://developers.google.com/web/tools/chrome-user-experience-report)는 각 핵심 웹 바이탈에 대한 익명화된 실제 사용자 측정 데이터를 수집한다. 이 데이터를 바탕으로, 사이트 소유자는 페이지에서 수동으로 분석할 필요 없이 성능을 신속하게 평가 할 수 있으며, PageSpeed Insights나 Search Console의 Core Web Vitals 보고서와 같은 도구도 마찬가지로 사용할 수 있다.

|                                                                                                        | LCP | FID | CLS |
| :----------------------------------------------------------------------------------------------------: | :-: | :-: | :-: |
| [Chrome User Experience Report](https://developers.google.com/web/tools/chrome-user-experience-report) |  ✔  |  ✔  |  ✔  |
|             [PageSpeed Insights](https://developers.google.com/speed/pagespeed/insights/)              |  ✔  |  ✔  |  ✔  |
|    [Search Console (Core Web Vitals Report)](https://support.google.com/webmasters/answer/9205520)     |  ✔  |  ✔  |  ✔  |

![page-speed-insight](./images/yceffort-page-speed-insight.png)

![search-console](./images/yceffort-search-console.png)

Chrome User Experience Report에서 제공하는 데이터는, 사이트의 성능을 신속하게 평가할 수 있는 도구를 제공하지만, 상세 페이지 별 분석, 회귀 분석, 모니터링 등의 기능은 제공하지 않는다. 따라서 사이트 자체적인 실제 사용자 모니터링을 설정하는 것이 좋다.

#### 자바스크립트로 핵심 웹 바이탈 측정하기

핵심 웹 바이탈은 standard web api를 활용하여 자바스크립트 내에서 측정할 수 있다.

가장 쉽게 측정할 수 있는 방법은 [web-vital](https://github.com/GoogleChrome/web-vitals) 도구를 활용하는 것이다. 이 자바스크립트 라이브러리는, 위에 나열된 모든 구글 도구에서 보고하는 방법과 정확하게 일치하도록 각 지표를 측정하는 기본적인 웹 api를 사용할 수 있도록 제공한다.

이 라이브러리를 사용하면 각 지표를 측정하는 것이 단일 함수를 호출하는 것 만큼 간단하다.

```javascript
import {getCLS, getFID, getLCP} from 'web-vitals';

function sendToAnalytics(metric) {
  const body = JSON.stringify(metric);
  // Use `navigator.sendBeacon()` if available, falling back to `fetch()`.
  (navigator.sendBeacon && navigator.sendBeacon('/analytics', body)) ||
      fetch('/analytics', {body, method: 'POST', keepalive: true});
}

getCLS(sendToAnalytics);
getFID(sendToAnalytics);
getLCP(sendToAnalytics);
```

이 라이브러리를 사용하여 핵심 웹 바이탈 데이터를 측정하고, 분석 엔드 포인트로 보내도록 사이트를 구성한 다음, 해당 데이터를 집계하고 보고하여 페이지가 적절한 지표를 달성하고 있는지 확인하면 된다.

또, [Web Vitals Chrome Extension](https://github.com/GoogleChrome/web-vitals-extension)을 활용하여 코드를 굳이 작성하지 않더라도 각 핵심 코어 바이탈에 대해 보고하도록 할 수 있다. 이 익스텐션은 라이브러리를 활용하여 지표를 측정하고, 사용자가 웹을 탐색할 때 사용자에게 표시한다.

이 익스텐션은, 자체 사이트 및 경쟁 업체 사이트의 웹 성능을 파악하는데 도움을 줄 수 있다.

|                                                                                                        | LCP | FID | CLS |
| [web-vitals](https://github.com/GoogleChrome/web-vitals)|  ✔  |  ✔  |  ✔  |
| [web vitals Extension](https://github.com/GoogleChrome/web-vitals-extension) |  ✔  |  ✔  |  ✔  |

![web-vitals-extension](./images/yceffort-web-vitals-extension)

또는 기본 웹 api를 활용하여 직접 지표를 측정할 수 있다.

`LCP`

```javascript
new PerformanceObserver((entryList) => {
  for (const entry of entryList.getEntries()) {
    console.log('LCP candidate:', entry.startTime, entry);
  }
}).observe({type: 'largest-contentful-paint', buffered: true});
```

`FID`

```javascript
new PerformanceObserver((entryList) => {
  for (const entry of entryList.getEntries()) {
    const delay = entry.processingStart - entry.startTime;
    console.log('FID candidate:', delay, entry);
  }
}).observe({type: 'first-input', buffered: true});
```

`CLS`

```javascript
let clsValue = 0;
let clsEntries = [];

let sessionValue = 0;
let sessionEntries = [];

new PerformanceObserver((entryList) => {
  for (const entry of entryList.getEntries()) {
    // Only count layout shifts without recent user input.
    if (!entry.hadRecentInput) {
      const firstSessionEntry = sessionEntries[0];
      const lastSessionEntry = sessionEntries[sessionEntries.length - 1];

      // If the entry occurred less than 1 second after the previous entry and
      // less than 5 seconds after the first entry in the session, include the
      // entry in the current session. Otherwise, start a new session.
      if (sessionValue &&
          entry.startTime - lastSessionEntry.startTime < 1000 &&
          entry.startTime - firstSessionEntry.startTime < 5000) {
        sessionValue += entry.value;
        sessionEntries.push(entry);
      } else {
        sessionValue = entry.value;
        sessionEntries = [entry];
      }

      // If the current session value is larger than the current CLS value,
      // update CLS and the entries contributing to it.
      if (sessionValue > clsValue) {
        clsValue = sessionValue;
        clsEntries = sessionEntries;

        // Log the updated value (and its entries) to the console.
        console.log('CLS:', clsValue, clsEntries)
      }
    }
  }
}).observe({type: 'layout-shift', buffered: true});
```

#### 핵심 웹 바이탈을 측정할 수 있는 개발단계의 도구들

모든 핵심 웹 바이탈은 실제 배포가 되어 측정되는 현장 기준이지만, 이 중에는 개발단계에서 측정할 수 있는 방법이 있다. 이 방법을 활용한다면, 개발중에 기능의 성능을 미리 테스트할 수 있다. 또한 성능저하가 발생하기 전에 미리 파악할 수 있도록 도와준다.


|                                                                                                        | LCP | FID | CLS |
| [Chrome DevTools](https://developers.google.com/web/tools/chrome-devtools) |  ✔  |  ✘ [TBT](https://web.dev/tbt/) 활용 |  ✔  |
| [Lighthouse](https://developers.google.com/web/tools/lighthouse) |  ✔  |  ✘ [TBT](https://web.dev/tbt/) 활용 |  ✔  |

이러한 도구는 훌륭하지만, 실제 성능 측정을 대체할 수 있는 것은 아니다.

사이트의 성능은 사용자 디바이스의 기능, 네트워크 상태, 디바이스에서 실행 중인 다른 프로세스, 페이지와 상호작용하는 방식에 따라 크게 달라질 수 있다. 실제로 이 핵심 웹 바이탈 지표는 사용자의 인터랙션에 따라서 점수가 달라질 수가 있다. 

### 점수를 높이기 위한 좋은 방법

지표를 측정했다면, 이제 다음은 이 성능을 최적화 하는 것이다. 아래의 방법을 활용하면 각 지표의 성능을 향상 시킬 수 있다.

- LCP: https://web.dev/optimize-lcp/
- FID: https://web.dev/optimize-fid/
- CLS: https://web.dev/optimize-cls/

## 다른 웹 바이탈

핵심 웹 바이탈은 좋은 사용자 환경을 이해하고, 제공하기 위한 중요한 지표이지만 이외에도 다른 중요한 지표도 있다.

- Time to First Byte (TTFB): https://web.dev/time-to-first-byte/
- First Contentful Paint (FCP): https://web.dev/fcp/
- Total Blocking Time (TBT) https://web.dev/tbt/
- Time to Interactive (TTI): https://web.dev/tti/

