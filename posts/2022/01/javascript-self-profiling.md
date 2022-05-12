---
title: '웹 애플리케이션에서 자바스크립트 프로파일링 해보기'
tags:
  - javascript
  - chrome
  - browser
published: true
date: 2022-01-20 21:54:46
description: '해봤지만 해보지 않았습니다'
---

## Table of Contents

## Introduction

자바스크립트를 프로파일링 할 수 있는 api가 있다. https://wicg.github.io/js-self-profiling/ 이 api를 활용하면 실제 고객의 디바이스에서 자바스크립트 웹 애플리케이션의 성능 프로파일을 가져올 수 있다. 즉, 브라우저 개발자 도구에서 로컬 머신 (컴퓨터)로 애플리케이션을 프로파일링 하는 수준 이상을 해볼 수 있다. 애플리케이션을 프로파일링하는 것을 성능을 파악할 수 있는 좋은 방법이다. 프로파일을 활용해서 시간이 지남에 따라 실행되는 항목 (스택)을 확인하고 코드에서 성능에 문제가 되는 핫스팟을 식별할 수 있도록 도와준다.

브라우저에서 개발자 도구를 사용해 봤다면, 자바스크립트 프로파일리에 익숙할 수 있다. 예를 들어, 크롬 브라우저의 개발자도구에서 성능탭을 보면 프로파일을 기록할 수 있다. 이 프로파일은 시간이 지남에 따라 애플리케이션에서 실행 중인 내용을 보여준다.

![performance-example](./images/performance-example.png)

> 내가 만든거 아님

이 api는 크롬에서 여전히 사용할 수 있는 자바스크립트 프로파일러 탭을 상기시켜준다.

![javascript-profiler-example](./images/javascript-profiler-example.png)

이 js self profiling api는 새로운 api로, 크롬 94+ 버전에서만 사용 가능하다. 자바스크립트에서 방문자를 위해 사용할 수 있는 샘플링 프로파일러를 제공한다.

## Sample Profiling이란 무엇인가

일반적으로 오늘날에 사용되는 성능 프로파일러에는 두가지 유형이 존재한다.

1. 계측 (구조화, 추적) 프로파일러: 애플리케이션이 모든 함수의 입력과 출력에 훅을 추가하여 각 함수에서 소요되는 시간을 알 수 있다.
2. 샘플링 프로파일러: 해당 시간에 호출 스택에서 실행 중인 내용을 기록(샘플링)하기 위해 일정한 주기에 따라 응용프로그램의 실행을 일시적으로 중지시킨다.

여기에서 말하는 js self-profiling api는 브라우저에서 후자의 형태로 동작한다. 브라우저 개발자 도구에서 동작하는 샘플 프로파일로도 마찬가지다.

프로파일러에서 "샘플링" 이라는 것은 브라우저가 기본적으로 일정한 간격으로 스냅샷을 생성하여 현재 실행중인 스택을 점검하는 것을 의미한다. 이것은 샘플링 간격이 너무 좁지 않다는 가정하에서 할 수 있는 가벼운 방법이다. 정기적으로 간격을 두는 샘플링 인터럽트는 실행 중인 스택을 빠르게 일단 검사한다음에 나중에 기록한다. 시간이 지남에 따라서 이렇게 샘플링된 스택은 추적 중에 실행되었던 것을 나타낼 수 있지만, 때때로 샘플링이 잘못 읽혀질 수도 있다.

시간이 지남에 따라서 애플리케이션에서 실행되는 함수 스택의 다이어그램을 한번 상상해보자. 샘플링 프로파일러는 현재 실행중인 스택을 일정한 간격 (이 그림에서는 빨간색 세로줄)에 따라 검사하고 다음과 같이 보고할 것이다.

![sampling-profiler-in-function](https://calendar.perfplanet.com/images/2021/nic/sampled-profiler-stacks.svg)

일반적으로 우리가 아는 프로파일링의 경우, (앞서 말한 전자의 경우) 정확히 언제 모든 함수가 호출되어서 시작하고, 끝나는지 알 수 있도록 애플리케이션을 추적하는데 중점을 둔다. 그러나 이러한 측정방법은 많은 오버헤드가 있고 측정 중인 애플리케이션의 속도를 늦출 수 있는 위험성이 있다. 물론 그러한 무리수 덕택에(?) 함수에서 소비되는 상대적으로 정확한 시간을 측정할 수 있다. 이러한 프로파일링은 방문자의 애플리케이션 속도를 떨어뜨리기 때문에 실제로는 거의 사용되지 않는다. 그러나 샘플링 프로파일러는 이러한 성능에 대한 영향이 훨씬 작으므로 실무에서 더 많이 쓰인다.

> https://www.igvita.com/slides/2012/structural-and-sampling-javascript-profiling-in-chrome.pdf

## 샘플링 프로파일링의 다운사이드

물론 이러한 방법이 장점만 있는 것은 아니다. 오버헤드를 줄이는 데에는 유용할 수 있지만, 캡처된 데이터가 잘못되는 경우도 발생할 수 있다.

예를 들어, 콜 스택에서 샘플 8개가 10ms 간격으로 추출되는 상황을 가정해보자.

![sampling-profiler-in-function](https://calendar.perfplanet.com/images/2021/nic/sampled-profiler-stacks.svg)

프로파일러가 알 수 있는 것은 이것이 최선이기 때문에, 샘플링된 프로파일러가 해당 스택을 빨간 세로선 기준으로 검사하는 경우 스택에서 보낸 시간을 다음과 같이 보고할 것이다.

- A, B, C 1회 호출됨 (10ms)
- A, B 2회 호출됨 (20ms)
- A가 1회 호출됨 (10ms)
- D가 2회 호출됨 (20ms)
- idle (20ms)

80ms 이상 시간 동안 일어난 일들 표현하고 있지만, 이는 사실 정확히 맞는 것은 아니다. 사실은

- A, B, C가 6ms 이상 초과 보고됨
- A, B가 12ms 이상 초과 보고됨
- A가 8ms 이하로 보고됨
- D가 8ms 이상 초과 보고됨
- D, D, D는 보고되지 않음
- idle이 15ms 이하로 보고됨

이 잘못된 리포팅은 몇몇 케이스에서 안 좋은 사례로 남을 수 있다. 대부분의 애플리케이션 스택은 또 이렇게 간단하지 않을 것이기 때문에, 실제 프로덕션 환경에서 이러한 현상이 어떻게 발생하는지 는 알 수 없겠지만, 대략 이런일이 발생할 수 있다는 것은 가정해 볼 수 있을 것이다.

먼저 샘플링된 프로파일러가 10ms 마다 샘플을 추출하는데, 애플리케이션이 대략 16ms 동안 2ms 간격으로 작업을 실행되는 상상을 해보자.

![case1](https://calendar.perfplanet.com/images/2021/nic/sampled-profiler-stacks-bad-case-1.svg)

최악의 경우, 위 그림 처럼 런타임 시간의 12.5% 동안 실행은 되지만 샘플링 프로파일러에서는 하나도 보고가 안될 수도 있다.

![case2](https://calendar.perfplanet.com/images/2021/nic/sampled-profiler-stacks-bad-case-3.svg)

위의 경우에서는, 정확히 샘플링 프로파일러와 동일한 주기로 실행될 수 있지만, 샘플링되는 1ms 짜리 실행만 가능하다. 이 경우에는 , 12.% 동안 실행되지만 리포트에서는 그 시간 내내 100% 함수가 실행되는 것으로 오해할 수 있다.

![case3](https://calendar.perfplanet.com/images/2021/nic/sampled-profiler-stacks-bad-case-2.svg)

위 경우는 또 어떤가? 10ms간격으로 샘플링하지만, 함수는 오직 8ms 동안만 실행된다. 샘플링 프로파일러가 어떻게 조사하느냐에 따라서, 런타임 시간의 80%를 사용했지만 정작 리포팅은 하나도 안될 수도 있다.

이 모든 것들은 아주 극단적으로 나쁜 예들을 모아 놓은 것이지만, 이러한 예를 한번 살펴봄으로써 어떤 종류의 애플리케이션 동작들이 샘플링된 프로파일러에 의해 잘못 표현되는지를 볼 수 있었다. 우리는 이러한 것들을 추적하기전에 감안하고 보아야 한다.

## API

### Document Policy

Javascript Self-Profiling API를 호출하려면, HTML 페이지에 `js-profiling`이라고 하는 [문서 정책](https://w3c.github.io/webappsec-permissions-policy/document-policy.html)이 있어야 한다. 일반적으로 `Document-Policy`라고 하는 HTTP 응답헤더 또는 `<iframe policy="">`를 통해 구현할 수 있다.

```
Document-Policy: js-profiling
```

이 옵션이 되면, 써드파티 스크립트를 포함해서 모든 자바스크립트가 프로파일링을 시작할 수 있다.

### API

JS Self-Profiling API는 [Profiler](https://wicg.github.io/js-self-profiling/#the-profiler-interface) 객체를 new로 선언하여 사용할 수 있다.

샘플 프로파일러 객체가 만들어지면, 나중에 언제든 `.stop()`을 호출하여 프로파일링을 멈추고 추적 내역을 받을 수 있다.

```javascript
// Profiler를 지원하는지 확인
if (typeof window.Profiler === 'function') {
  var profiler = new Profiler({ sampleInterval: 10, maxBufferSize: 10000 })
  profiler.stop().then(function (trace) {
    sendProfile(trace)
  })
}
```

```javascript
if (typeof window.Profiler === 'function') {
  const profiler = new Profiler({ sampleInterval: 10, maxBufferSize: 10000 })
  var trace = await profiler.stop()
  sendProfile(trace)
}
```

여기에 두가지 옵션을 확인할 수 있다.

- `sampleInterval`: 애플리케이션에서 필요로하는 샘플 간격 (밀리 초)다. 시작한 뒤부터, `profiler.sampleInterval`로 접근 가능하다.
- `maxBufferSize`: 샘플 수로 측정할 수 있는, 원하는 샘플 버퍼의 크기다.

시작하자마자 바로 시작되는 것은 아니고, 프로파일러를 위한 준비를 브라우저에서 해야 하므로, 약간의 지연이 걸린다. 일반적으로, 데스크톱과 모바일에서 새 프로파일이 시작되는데 보통 1~2ms가 걸리는 것으로 보인다.

### Sample Interval

`sampleInterval`는 브라우저가 자바스크립트의 호출 스택 샘플을 가져오는 빈도를 결정한다. 측정 오버헤드가 없는 한에서, 가능한 정확히 데이터를 제공할 수 있는 좁은 구간을 선택하는 것이 좋다.

스펙 문서에서는, 사용자가 0 이상의 값을 지정해야 한다고 되어 있으며, user agent를 통해서 이러한 샘플링 속도를 선택할 수 있다.

실제로 Chrome 96이상에서 지원하는 최소 샘플링간경은 다음과 같다.

- 윈도우: 16ms
- 맥, 리눅스, 안드로이드: 10ms

이 말인 즉슨, 아무리 1이나 0을 선택해도 운영체제에 따라 만 최소 10ms내지 16ms만 가능하다는 것이다. `.sampleInterval`을 활용하면 언제든 현재 샘플링 레이트를 확인할 수 있다.

```javascript
const profiler = new Profiler({ sampleInterval: 1, maxBufferSize: 10000 })
console.log(profiler.sampleInterval)
```

이와는 별개로, 크롬에서는 실제 샘플링 간격이 최소 값의 다음 배수로 올라간다. 예를 들어, 안드로이드에서 91~99ms를 지정한다면 100ms가 실제로는 부여된다.

### Buffer

또 다르게 추적에 사용할 수 있는 값은 `maxBufferSize` 다. 이 값은 프로파일러가 자체적으로 중단하기 전에 수집할 수 있는 최대 샘플 크기를 의미한다.

예를 들어, `sampleInterval: 100`, `maxBufferSize: 10`를 지정하는 경우 100ms간 10개의 샘플을 얻을 수 있게 되는 것이다. 만약 이 버퍼가 다 차게 되면, `samplebufferfull` 이벤트가 발생하게 되고 더이상 샘플을 수집하지 않게 된다.

```javascript
if (typeof window.Profiler === 'function') {
  const profiler = new Profiler({ sampleInterval: 10, maxBufferSize: 10000 })

  function collectAndSendProfile() {
    if (profiler.stopped) return

    sendProfile(await profiler.stop())
  }

  profiler.addEventListener('samplebufferfull', collectAndSendProfile)

  // do work, or listen for some other event, then:
  // collectAndSendProfile();
}
```

## 누구를 프로파일 할까

모든 방문자에 대해 샘플 프로파일러를 활성화 하면 될까? 아마도 그건 무리일 것이다. 물론 오버헤드는 무시할 만큼 작아보일 수도 있지만, 모든 방문객에게 이 데이터를 추출하고 수집하는데 부담을 주는 것은 좋지 못하다.

이상적으로는, 샘플 프로파일러도 표본으로 (sample) 추출하는 것이 좋다.

예를 들어 방문자의 10%, 1%, 0.1%에 대해 이 기능을 키는 것을 고려할 수 있다. 모든 사용자에게 키지 말아야할 이유는 다름과 같다.

- 최소 수준인 것을 감안하더라도, 샘플링을 활성화 하는 것은 비용이 발생하므로 모든 방문자를 지연 시키는 것은 좋지 못하다.
- 샘플링 프로파일러 추적에 의해 발생하는 데이터의 양은 상당하기 때문에, 이 데이터를 모두 서버에서 처리한다면 좋지 못하다.
- 현재 기준으로 이 api를 지원하는 브라우저는 크롬 뿐이므로 브라우저 편향적인 데이터를 수집하게 된다.

위와 같은 요소를 고려해봤을 때, 특정 페이지 로드 샘플 혹은 특정 방문자 샘플에 대해 프로파일러를 하는 것이 이상적이다.

https://caniuse.com/mdn-api_profiler

## 언제 프로파일 할까

언제 프로파일이 시작되어야 할까? 여기에는 특정 이벤트, 사용자 인러택션, 전체 페이지 로드 그 자체 등 여러가지 세션 중에 프로파일링을 활용할 수 있는 다양한 방법이 있다.

### 특정 작업

애플리케이션은 방문자를 위해 규칙적으로 실행되는 몇가지 복잡한 작업을 가지고 있을 것이다.

이러한 작업을 기준으로 측정한다면, 코드가 실제로 어떻게 흘러가고 수행되는지 모를 때 유용할 수 있다. 이는 호출하는데 얼마나 많은 비용이 드는지 모르는 써드파티 스크립트를 호출 할 때 유용하다.

이러한 작업을 위해서는, Profiler를 단순히 작업의 시작과 끝에 작동과 중지를 하면 된다.

캡쳐한 이 데이터는 프로파일링하는 코드를 알 수 있을 뿐만 아니라, 작업이 다른 코드와 경쟁적으로 일어나고 있는지도 파악할 수 있다.

```javascript
function loadExpensiveThirdParty() {
  const profiler = new Profiler({ sampleInterval: 10, maxBufferSize: 1000 })

  loadThirdParty(async function onThirdPartyComplete() {
    var trace = await profiler.stop()
    sendProfile(trace)
  })
}
```

### 유저 인터랙션

유저 인터랙션은 [First Input Delay](https://web.dev/fid/)와 같은 메트릭이 중요할 때 사용하는 것이 좋다.

사용자 인터랙션을 측정하기 위해, 프로파일러를 시작하는 타이밍과 관련하여 몇가지 방법을 생각해 볼 수 있다.

- 일단 한개는 항상 실행시킨다. 그리고 사용자가 인터랙션을 한다면, 이벤트 전후의 짧은 시간으로 이벤트를 잘라낸다.
  - 만약 `EventTiming`을 사용하고 활성화된 Profiler가 있다면, 이벤트의 `startTime`에서 `processingEnd`까지 측정하여 이벤트 실행전, 중, 결과로 실행된 내용을 파악할 수 있다.
- 마우스가 이동하거나 clickable한 대상으로 이동하기 시작하면 프로파일러 켜기
- 사용자가 인터랙션을 수행할 것으로 예상되는 이벤트 (마우스 다운)과 같은 이벤트가 발생하면 프로파일러 켜기

만약, 인터랙션이 프로파일러를 시작할 때 까지 기다린다면 앞서 언급한 것 처럼 1~2m정도의 시간이 소요된다.

```javascript
let profiler = new Profiler({ sampleInterval: interval, maxBufferSize: 10000 })

const observer = new PerformanceObserver(function (list) {
  const perfEntries = list.getEntries().forEach((entry) => {
    if (profiler && !profiler.stopped && entry.name === 'click') {
      profiler.stop().then(function (trace) {
        const filteredSamples = trace.samples.filter(function (sample) {
          return (
            sample.timestamp >= entry.startTime &&
            sample.timestamp <= entry.processingEnd
          )
        })

        // do something with the filteredSamples and the event

        // start a new profiler
        profiler = new Profiler({
          sampleInterval: interval,
          maxBufferSize: 10000,
        })
      })
    }
  })
}).observe({ type: 'event', buffered: true })
```

### 페이지 로드

만약 페이지 로드 프로세스 전체를 프로파일링 하려면, 문서의 `<head>`에 다른 스크립트보다 먼저 인라인 스크립트를 삽입하여 프로파일러를 시작하는 것이 좋다.

그렇게 하면 추적을 처리하고 전송하기 전에 미리 페이지의 `onload` 이벤트와 딜레이를 기다릴 수 있다.

또한 `pageHide` 나 `visibilitychange` 이벤트에 리스너를 달아서 페이지가 완전히 로드되기전에 페이지를 떠나는지 확인하는 후 프로파일링을 전송할 수 있다.

> `unload` 이벤트에서는 약간의 문제가 있다.

긴 작업이나 EventTiming 이벤트와 같이 페이지 로드 프로세스에서 지표나 이벤트를 측정하는 경우, 이벤트가 어떻게 실행되었는지 이해하기 위해 샘플 프로파일러를 사용하면 유용할 수 있다.

## 프로파일 살펴보기

`Profiler.stop()`의 Promise 콜백에서 리턴되는 trace 객체는 [여기](https://github.com/WICG/js-self-profiling/blob/main/README.md#appendix-profile-format)에 설명되어 있으며, 주요 내용은 아래와 같다.

- `frames`: 프레임의 배열, 즉 스택의 일부 일 수 있응 개별함수들을 포함한다.
  - `innerHTML`과 같은 DOM 함수도 볼 수 있으며, 여기에는 심지어 `Profiler` 자체도 포함될 수 있다.
  - 만약 이름이 없는 경우, `<script>` 이거나 외부 자바스크립트 파일의 루트에서 실행될 자바스크립트일 가능성이 높다.
- `resources`: trace에 프레임이 있는 함수가 포함된 모든 리소스의 배열이 포함된다.
  - 페이지 그 자체가 배열의 첫번째 인 경우가 많으며, 다른 외부 자바스크립트 파일 또는 페이지가 뒤이어 나타난다.
- `samples`: 실제 프로파일러 샘플이며, 발생한 시점에 해당하는 타임스탬프가 있고, `stackId`가 해당 시간된 스택을 가리킨다.
  - 만약 `stackId`가 없다면 해당 시간에는 아무것도 실행되지 않은 것이다.
- `stacks`: 스택위에서 실행중인 프레임의 배열이 포함되어 있다.
  - 각 스택은 `parentId`를 가질 수 있는데, 이를 호출한 함수에 대해 트리의 다음 노드에 매핑된다.

```json
{
  "frames": [
    { "name": "Profiler" }, // the Profiler itself
    { "column": 0, "line": 100, "name": "", "resourceId": 0 }, // un-named function in root HTML page
    { "name": "set innerHTML" }, // DOM function
    { "column": 10, "line": 10, "name": "A", "resourceId": 1 } // A() in app.js
    { "column": 20, "line": 20, "name": "B", "resourceId": 1 } // B() in app.js
  ],
  "resources": [
    "https://example.com/page",
    "https://example.com/app.js",
  ],
  "samples": [
      { "stackId": 0, "timestamp": 161.99500000476837 }, // Profiler
      { "stackId": 2, "timestamp": 182.43499994277954 }, // app.js:A()
      { "timestamp": 197.43499994277954 }, // nothing running
      { "timestamp": 213.32999992370605 }, // nothing running
      { "stackId": 3, "timestamp": 228.59999990463257 }, // app.js:A()->B()
  ],
  "stacks": [
    { "frameId": 0 }, // Profiler
    { "frameId": 2 }, // set innerHTML
    { "frameId": 3 }, // A()
    { "frameId": 4, "parentId": 2 } // A()->B()
  ]
}
```

시간이 지남에 따라 무엇이 실행되었는지 확인하기 위해 샘플 배열을 살펴보자.

```json
"samples": [
  ...
  { "stackId": 3, "timestamp": 228.59999990463257 }, // app.js:A()->B()
  ...
]
```

만약 샘플이 `stackId`를 가지고 있지 않다면, 아무것도 실행되지 않은 것이다.

만약 포함된 경우, `stacks`에서 해당 아이디를 참조할 수 있다.

```json
"stacks": [
  ...
  2: { "frameId": 3 }, // A()
  3: { "frameId": 4, "parentId": 2 } // A()->B()
]
```

`stackId` 3은 `frameId` 4 임을 알 수 있는데, 이는 `parentId` 2를 가진다.

`parentId`를 재귀적으로 체이닝 하다보면, 전체 스택을 볼 수 있다. 이 경우, 이 스택에는 두개의 프레임만 존재한다.

```
frameId:4
frameId:3
```

이 `frameId`로, `frames`을 살펴보면

```json
"frames": [
...
  3: { "column": 10, "line": 10, "name": "A", "resourceId": 1 } // A() in app.js
  4: { "column": 20, "line": 20, "name": "B", "resourceId": 1 } // B() in app.js
],
```

따라서 위의 `228.59999990463257`에 있는 샘플 스택은 다음과 같다.

```
B()
A()
```

이 말은, `A()`가 `B()`를 호출했다는 것이다.

## Beaconing

샘플링 프로파일의 추적이 중지되었다면, 이제 그 데이터를 바탕으로 어떻게든 유의미한 값을 가져와야 할 것이다.

https://nicj.net/beaconing-in-practice/

추적된 데이터의 크기에 따라, 먼저 로컬 (브라우저)에서 처리하거나, 추가 분석을 위해 로우 데이터를 백엔드 서버로 전송하는 등의 작업을 할 수 있다.

해당 데이터를 처리하기 위해 다른 위치로 전송하는 경우, 보다 실행에 용이한 상태로 만들기 위해 몇가지 추적과 관련된 증거를 남겨둘 수 있다. 예를 들어

- 페이지 로드에 걸린 시간 또는 Core Web Vital과 같은 성능 지표
  - 이러한 성능 지표 데이터가 있다면, 유저 경험이 좋은지 나쁜지 이해하는데 도움이 될 수 있다
- `Long Tasks` `EventTiming` 이벤트와 같은 성능 이벤트
  - 이러한 이벤트와 샘플 데이터간의 상관 관계를 분석하여 사용자에게 '나쁜' 영향을 미친 이벤트 동안 어떤 일이 발생했는지 알 수 있음
- User Agent, 디바이스 정보, 페이지 너비와 같은 유저 관련 정보
  - 데이터를 케이스 별로 분할하고, '나쁜' 유저 경험이 있는 패턴을 발견한 경우, 어떤 케이스인지 그 범위를 좁히는데 유용하다.

이러한 샘플링된 프로파일은 이 작업이 수행된 상황을 이해할 수 있을 때 가장 유용하므로, 이 데이터가 '좋은' 유저 경험인지 '나쁜' 유저 경험인지 판단할 수 있는 데이터를 가지고 있어야 한다.

## 압축

CPU 에 약간 투자할 여유가 있다면, 업로드하기전에 데이터 크기를 줄일 수 있는 몇가지 방법이 있다.

한 가지 방법은 [Compression Stream API](https://wicg.github.io/compression/)를 이용하는 것으로, 문자열을 gzip으로 압축된 데이터 스트림으로 변경할 수 있다. 한가지 단점은 비동기식이기 때문에 압축된 프로필 데이터를 업로드 하기 전에 먼저 압축된 바이트가 포함되어 있는 콜백을 기다려야 한다는 것이다.

`application/x-ww-form-urlcoded` 인코딩을 통해 데이터를 전송하기 위해서는, URL 인코딩된 `JSON.stringify()`보다 그 결과가 커질 수 있다는 것을 명심해둬야 한다. 예를 들어 `JSON.stringify`로는 25kb인데 반해, `application/x-ww-form-urlcoded` 는 36kb로 증가한다.

이러한 사태를 방지하기 위해 [JSURL](https://github.com/Sage/jsurl)과 같은 라이브러리를 대신 써보는 것도 검토해봄직하다. 이 라이브러리는 `JSON`과 비슷해 보이지만, `application/x-www-form-urlencoded` 보다는 크기가 작다.

문자열 데이터에 적용할 수 있는 압축방법은 다양하므로, 원하는 방법을 적용해 보는 것이 좋다.

## 팁

### minified javascript

만약 application에 minified 된 javascript가 있다면 프로파일에는 minified된 함수 명이 리포트 될 것이다. 이를 해결하기 위해서는 소스맵 등이 필요할 것이다.

### 기명함수와 익명함수

익명 함수가 있다면 이로 인해 많은 귀찮은 것들이 발생한다.

```javascript
{
  "frames": [
    ...
    { "column": 0, "line": 10, "name": "", "resourceId": 0 }, // un-named function in root HTML page
    { "column": 0, "line": 52, "name": "", "resourceId": 0 }, // another un-named function in root HTML page
    ...
  ],
```

이러한 현상을 방지하기 위해서는

```html
<script>
  // start some work
</script>
```

대신

```html
<script>
  ;(function initializeThirdPartyInHTML() {
    // start some work
  })()
</script>
```

와 같은 즉시 실행 기명 함수를 사용하는 것이 좋다.

```javascript
{
  "frames": [
    ...
    { "column": 0, "line": 10, "name": "initializeThirdPartyInHtml", "resourceId": 0 }, // now with 100% more name!
    { "column": 0, "line": 52, "name": "doOtherWorkInHtml", "resourceId": 0 },
    ...
  ],
```
