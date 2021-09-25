---
title: '뒤로가기, 앞으로가기의 캐시 aka bfcache'
tags:
  - web
  - browser
  - javascript
published: true
date: 2020-11-26 13:09:01
description: '항상 브라우저에 감사하십시오 frontend developers.'
---

뒤로가기/앞으로가기 캐시 (이해 bfcache)는 브라우저에서 일어나는 최적화로, 앞으로가기나 뒤로가기가 발생했을 때 화면을 즉시 보여주는 역할을 한다. 이는 사용자의 브라우저 사용성을 향상시키는데, 특히 느린 네트워크/디바이스에서 빛을 발한다.

## 브라우저 호환성

bfcache는 [파이어폭스](https://developer.mozilla.org/en-US/docs/Mozilla/Firefox/Releases/1.5/Using_Firefox_1.5_caching)와 [사파리](https://webkit.org/blog/427/webkit-page-cache-i-the-basics/)에서 몇년전부터 지원하고 있었다. 크롬 역시 마찬가지다

## bfcache란

bfcache는 인메모리 캐시로, 자바스크립트 힙까지 포함해 페이지 전체를 완전히 캐시로 저장해버리는 것을 의미한다. 전체 페이지가 메모리 안에 있기 때문에, 사용자가 이전페이지로 돌아가고자 했을 때 빠르게 전체 페이지를 보여줄 수 있다.

### bfcache가 비활성화 되어 있다면

이전 페이지 로딩을 위해서 새로운 요청을 시도할 것이며, 반복적인 방문에 따라서 웹페이지가 얼마나 최적화 되어 있냐에 따라서 브라우저는 재다운로드, 재 parsing, 재 실행등을 일부 실행하거나 혹은 이를 다 다시 처음부터 시도할 것이다.

### bfcache가 활성화 되어 있다면

전체 페이지가 메모리에 저장되어 있기 때문에, 네트워크 요청을 할 필요 없이 이전 페이지 로딩이 즉시 이루어진다.

크롬의 사용 데이터에 따르면 데스크톱의 탐색중 10%, 모바일 탐색 중 20%가 뒤로가기/또는 앞으로가기에서 이루어진다. bfcache를 이용하게 되면 웹페이지를 로드하는데 소요되는 데이터, 시간 등을 아낄 수 있다.

## 어떻게 동작하는가?

우리가 흔히 알고 있는 HTTP cache와는 동작이 다르다. bfcache는 자바스크립트 힙을 포함해 전체 페이지를 통채로 스냅샷을 떠서 메모리에 올려 버린다. 이에 반해 HTTP 캐시는 이전 요청에서 이루어진 응답에 대해서만 캐싱할 뿐이다. 페이지 로딩에 필요한 모든 요청을 HTTP 캐시로 만족시키는 것은 매우 드물기 때문에, bfcache 복원을 사용한 페이지 방문은 bfcache를 사용하지 않은 '잘 최적화된' 캐시보다 항상 빠르다.

그러나 페이지 스냅샷을 메모리에 올린다는 것은, 현재 실행중인 코드를 보존하려고 할 때 복잡해진다. 예를 들어, 페이지가 bfcache가 되어 있는 동안에 `setTimeout`호출이 있으면 어떻게 해야할까?

정답은 브라우저가 보류중인 timer, 또는 promise의 실행을 일시 중지하고 (기본적으로 자바스크립트 태스크 큐에 있는 모든 작업) 페이지가 bfcache로 부터 복원이 되었을 때 다시 실행하는 것이다.

이는 매우 합리적인 방법으로 보이지만, 때로는 굉장히 복잡한 결과나 이해할 수 없는 행동을 만들어 낼 수도 있다. 만약에 브라우저가 `IndexedDB transaction` 의 일환인 작업을 중지한다고 한다면, 다른 탭에서도 접근할 수 있는 indexedDB의 특성상 다른 탭에서 페이지를 열었을 때 영향을 미칠 수 있다. 그래서 브라우저는, IndexedDB 트랜잭션 또는 다른 페이지에 영향을 줄 수 있는 API 호출 중에는 페이지를 캐싱하려 하지 않는다.

## bfcache의 작업을 api로 살펴보기

bfcache는 브라우저가 자동으로 하는 최적화이지만, 여전히 개발자들이 이 동작을 잘 이해 한다면 페이지를 최적화 하거나 성능을 측정하고 조정하는데 도움을 얻을 수 있다.

bfcache를 관찰할 수 있는 가장 좋은 이벤트는 `pageshow`와 `pagehide`다. [새로운 페이지 라이프 사이클 이벤트](https://developers.google.com/web/updates/2018/07/page-lifecycle-api)인 `freeze` 와 `resume`도 bfcache를 확인하는데 도움을 얻을 수 있다. 예를 들어, CPU 사용량을 최소화 하기 위하여 백그라운드 탭을 프리징할 때에 이 이벤트를 쓸 수 있다. 하지만 이는 오로지 크로미윰 브라우저에서만 확인가능하다.

### bfcache를 복원하는 순간을 확인하기

`pagehide`와 `pageshow`는 쌍으로 일어난다. `pageshow`는 페이지가 정상적으로 로딩 되거나, bfcache 로부터 페이지 복원 될 때 일어난다. `pagehide`는 마찬가지로 페이지가 정상적으로 언로드 되거나, bfcache로 들어가는 순간에 일어난다.

`pagehide`는 `persisted`속성을 가지고 있는데, `false`가 리턴되면 page가 bfcache되지 않음을 의미하는 것이다. 그렇다고 `true` 라고 해서 bfcache를 보장하는 것은 아니다. 단지 브라우저가 페이지를 캐싱 시도했다는 것을 의미하며, 무언가 다른 이유로 인해서 캐싱이 안될 수도 있다.

```javascript
window.addEventListener('pagehide', function (event) {
  if (event.persisted === true) {
    console.log('bfcache가 될 수도 있음')
  } else {
    console.log(
      '정상적으로 unload되며 이전 페이지 상태는 bfcache가 안들어가기 때문에 다 버려짐.',
    )
  }
})
```

비슷하게 `freeze`에도 같은 속성이 있으며, 같은 이유로 캐싱을 보장하지 않는다.

## bfcache로 페이지 최적화 하기

모든 페이지가 bfcache로 처리되는 것은 아니며, 심지어 캐싱이 되었다 하더라도 영원히 남아 있는 것은 아니다. 개발자들이 캐시 히트 레이트를 극대화 하기 위해 bfcache를 가능하게/혹은 불가능하게 만드는 것을 이해하는 것이 중요하다.

### 절대로 `unload` 이벤트를 사용하지 말 것

모든 브라우저에서 bfcache로 최적화하는데 가장 중요한 것은 절대절대로 `unload`이벤트를 임의로 사용해서는 안된다는 것이다. 이 `unload`이벤트는 bfcache 이전에 발생하고, 인터넷의 많은 페이지가 `unload`이벤트가 발행 후에는 페이지가 더 이상 존재하지 않는다는 (합리적인) 가정하에 동작하기 때문에, `unload`의 이벤트는 문제를 야기 할 수 있다. 많은 개발자들이 `unload` 이벤트가 더 이상 사용자가 페이지 네비게이션을 하지 않을 때 발생한다고 믿고 있는데, 이는 사실이 아니다.

> Many developers treat the unload event as a guaranteed callback and use it as an end-of-session signal to save state and send analytics data, but doing this is extremely unreliable, especially on mobile! The unload event does not fire in many typical unload situations, including closing a tab from the tab switcher on mobile or closing the browser app from the app switcher.

파이어폭스는 `unload`에 리스너가 달려 있을 경우, bfcache에 적합하지 않은 페이지로 처리한다. 사파리는 `unload`이벤트 리스너와 함께 페이지 케싱을 시도하는데, 잠재적인 버그를 줄이고자 유저가 네비게이션을 실행해버리면 `unload` 이벤트를 실행시키지 않는다. 크롬은 현재 [전체 페이지의 65%에 `unload`이벤트를 달았는데](https://www.chromestatus.com/metrics/feature/popularity#DocumentUnloadRegistered) 사파리와 마찬가지로 이를 실행하지 않는다.

`unload`이벤트 대신에 `pagehide`이벤트를 사용하는 것이 좋다.

### 조건이 있을 때만 `beforeunload`이벤트를 추가해라

`beforeunload`는 크롬과 사파리에 영향을 받지 않지만, 파이어 폭스의 경우 bfcache를 무력화 할 수 있으므로 사용해서는 안된다.

`unload`이벤트와는 다르게, 합리적으로 `beforeunload`를 사용할 수 있는 케이스가 존재한다. 예를 들어, 사용자가 데이터를 저장하지 않고 페이지를 떠나려고 하는 경우, `beforeunload`를 사용하여 사용자에게 경고를 하고, 사용이 끝난 즉시 지워버리는 것이 좋다.

🙅‍♂️ 하지 말것

```javascript
// 이벤트가 계속 남아있게 된다.
window.addEventListener('beforeunload', (event) => {
  if (pageHasUnsavedChanges()) {
    event.preventDefault()
    return (event.returnValue = 'Are you sure you want to exit?')
  }
})
```

🙆 해도 되는 것

```javascript
function beforeUnloadListener(event) {
  event.preventDefault()
  return (event.returnValue = 'Are you sure you want to exit?')
}

onPageHasUnsavedChanges(() => {
  window.addEventListener('beforeunload', beforeUnloadListener)
})

// 더 이상 사용이 필요 없으면 바로 지운다.
onAllChangesSaved(() => {
  window.removeEventListener('beforeunload', beforeUnloadListener)
})
```

### window.opener references의 사용을 피할 것

일부 브라우저 (크롬 86 포함) 페이지를 `window.open`또는 `target=_blank`에 `rel="noopner"`를 명시하지 않고 페이지를 열었을 경우 열린 페이지는 페이지를 이 페이지를 열어준 페이지에 대해서 참조를 사용하게 된다.

또한 보안상의 이유로 인해, `window.opener`의 null이 아닌 참조를 가지고 있는 페이지는 bfcache를 안전하게 사용할 수 없다. 이는 bfcache에 접근을 시도하는 페이지에 대해 위협이 되기 때문이다.

따라서 `window.opener`를 쓸 때는 반드시 `rel="noopener"`를 써야 한다. 만약 열린 윈도우에 대해 제어가 필요하다면 `window.postMessage`를 사용하는게 좋다. 그렇지 않고 윈도우 객체를 직접 참조하게되면, 열린 페이지 또한 열려있는 페이지 모두 bfcache를 누리지 못하게 된다.

### 사용자가 다른 페이지로 가기전에 모든 connection을 close해라.

위에서 언급했던 것처럼, bfcache에 들어가기전에 모든 예약된 자바스크립트 태스크는 중단되고, cache에서 나올때 다시 시작된다. 만약 스케쥴된 자바스크립트 태스크가 단순히 DOM Api에 접근하거나, 현재 페이지와 별개로 작동하는 API라고 한다면, 페이지를 일시정지해서 bfcache로 들어가는 것이 크게 문제가 되지 않는다.

만약 IndexedDB, Web Locks, WebSockets과 같이 다른 페이지에서도 접근할 수 있는 데이터와 관련된 API 라고한다면, 다른 탭의 실행에도 영향을 미칠 수 있기 때문에 문제가 될 수 있다. 따라서 다음 시나리오 상에서는 대부분의 브라우저가 bfcache를 시도하지 않는다.

- 페이지에 끝나지 않은 indexedDB transaction이 있는 경우
- fetch나 XMLHttpRequest가 진행 중인 경우
- WebSocket, WebRTC 연결이 살아 있는 경우

만약 페이지에서 위의 경우에 해당한다면, `pagehide`나 `freeze`이벤트에서 이러한 연결을 모두 끊어 버리는 것이 좋다. 이는 다른 탭에 영향을 주는 위험이 없이 안전하게 cache를 하는 방법이다.

그리고, bfcache로 부터 페이지가 살아난다면, 이러한 API를 다시 열어두면 된다. `pageshow` `resume` 이벤트에서 다시 연결해두면 된다.

> 유저가 페이지르 떠나기전에 해당 API가 사용 중이지 않다면 bfcache는 사용가능하다. 그러나 Embedded Plugins, Workers, Broadcast Channel 등 [일부 API](https://source.chromium.org/chromium/chromium/src/+/master:content/browser/frame_host/back_forward_cache_impl.cc;l=124;drc=e790fb2272990696f1d16a465832692f25506925?originalUrl=https:%2F%2Fcs.chromium.org%2F)들은 사용하게 되면 bfcache가 불가능하게 된다. 크롬이 bfcache 초기 출시 당시 의도적으로 보수적으로 접근하고 있지만, 장기적인 목표로는 최대한 많은 API에서 동작하게 끔 하려고 한다.

### 페이지가 캐싱 가능한지 테스트

page가 unloading시에 캐싱이 되는지 안되는지 확실히 결정할수는 없지만, 뒤로가기나 앞으로가기시에 bfcache가 올바르게 되고 있는지는 확인이 가능하다. 크롬의 경우 bfcache가 최대 3분까지 남아 있으므로, Puppetter나 Webdriver와 같은 테스트 도구로 `pageShow`이벤트의 `persisted`의 속성이 true로 남아있는지 확인하기에 충분하다.

물론 일반적인 상황에서는 캐시가 가능한 길게 남아있지만, 시스템의 메모리가 부족한 아쉬운 상황에서는 언제든 캐시가 날아갈 수 있다. 실패 테스트가 반드시 캐시가 안된다고는 단언할 수 없으므로, 실패의 기준과 테스트 설정을 유심히 해야할 필요가 있다.

> 크롬의 bfcache는 모바일에서만 가능하므로, 데스크톱에서 사용하기 위해서는 [#back-forward-cache 설정을 켜야 한다.](https://www.chromium.org/developers/how-tos/run-chromium-with-flags)

### bfcache를 제거하는 법

최상단 페이지 응답에 아래와 같이 설정해두면 bfcache를 제거할 수 있다.

```
Cache-Control: no-store
```

`no-cache`와 `no-store`등은 bfcache에 영향을 미치지 않는다.

이는 bfcache를 무력화시키는 확실한 방법이지만, [성능과 캐싱을 위해서 각자가 원하는 방법을 적용할 수 있도록 하자는 제안도 존재한다.](https://github.com/whatwg/html/issues/5744) (예를 들어 로그아웃과 같이 명시적인 시점에 bfcache 날린다던지)

## bfcache 가 분석 및 성능 측정에 미치는 영향

만약 분석 도구를 활용하여 사이트의 방문을 추적해본적이 있다면, 크롬이 bfcache를 사용함에 따라서 전체 페이지뷰가 감소한다는 것을 인지했을 수도 있다. 실제로 bfcache를 구현한 브라우저는 다른 브라우저에 비해 페이지뷰를 과소보고 하는 경우가 있는데, 이는 대부분의 트래킹 라이브러리들이 bfcache의 복원을 새로운 페이지뷰로 간주하지 않기 때문이다.

이를 방지하기 위해서는 아래와 같은 코드 추가를 고려해볼 수도 있다.

```javascript
// Send a pageview when the page is first loaded.
gtag('event', 'page_view')

window.addEventListener('pageshow', function (event) {
  if (event.persisted === true) {
    // Send another pageview if the page is restored from bfcache.
    gtag('event', 'page_view')
  }
})
```

### 성능 측정

bfcache는 특히 실제 수집된 성능 지표, 그 중에서도 페이지 로드 시간을 측정하는 지표에 부정적인 영향을 미칠 수 있다. bfcache 네비게이션은 실제 새로운 페이지를 로딩하는게 아니고 기존 페이지를 단순히 복원하는 것이기 때문에, bfcache를 사용하게 되면 수집된 페이지 로드의 총 숫자가 감소한다. 중요한 것은 bfcache로 복원으로 대체되는 페이지 로드가 페이지 속도 측정 중에서 가장 빠른 페이지로드로 인식될 수 있다는 것이다. 그 결과 데이터 집합에서 빠른 페이지 로드가 줄어들어, 사용자가 경험하는 실제 성능이 향상되었음에도 불구하고 그래프 상으로는 속도가 일정하지 않은 것 처럼 보일 수도 있다.

이 문제를 해결하는 데 몇가지 방법이 있는데 그 중 하나는, 모든 페이지 로드 메트릭에 `navigate` `reload` `back_forward` `prerender`와 같은 주석을 달아두는 것이다. 이러한 접근 방식은 TTFB(Time to First Byte)와 같은 사용자 중심 페이지 로드 메트릭에 권장된다. Core Web Vitals와 같은 사용자 중심 메트릭의 경우, 사용자가 경험하는 것을 정확하게 나타내는 값을 보고하는 것이 더 좋다.

### Core Web Vital에 미치는 영향

[Core Web Vital](https://web.dev/vitals/)이란 로딩 속도, 상호작용성, 시각적 안정성 등 다양한 면에 결쳐 웹페이지의 사용자 경험을 측정한다. bfcache 복원으로 사용자는 기존 페이지로드보다 더 빠르게 탐색하게 되므로, Core Web Vital에 이를 반영하는 것이 중요하다. 당연하게도, 사용자는 무슨 기법을 썼든지 간에 아무튼 속도가 빠른 것에만 신경을 쓰기 때문에.

곧 [Chrome User Experience Report](https://developers.google.com/web/tools/chrome-user-experience-report)와 같은 도구에서 bfcache 복원을 별도의 페이지 방문으로 처리하도록 업데이트 될 예정이다.

bfcache 복원에 따른 성능을 측정하기 위한 web performance api는 아직 존재하지 않지만, 기존 API로 대략 유추 해볼 수는 있다.

- [Largest Contentful Paint(LCP)](https://web.dev/lcp/): `pageshow` 이벤트의 타임스템프와 다음 프레임이 페인트 되는 시점의 타임스탬프를 비교하여 사용할 수 있다. (bfcache의 경우 LCP와 FCP의 값은 같다.)
- [First Input Delay(FID)](https://web.dev/fid/): `pageshow`이벤트에 이벤트 리스너를 다시 달아서 bfcache 복원 후 FID의 지연을 보고 할 수 있다.
- [Cumulative Layout Shift(CLS)](https://web.dev/fid/): 기존 performance observer를 계속 사용할 수 있으며, 현재 CLS 값을 0으로 재설정하면 된다.

## 더 읽어보기

- [Firefox Caching](https://developer.mozilla.org/en-US/Firefox/Releases/1.5/Using_Firefox_1.5_caching)
- [Page Cache](https://webkit.org/blog/427/webkit-page-cache-i-the-basics/)
- [브라우저에 따른 bfcache](https://docs.google.com/document/d/1JtDCN9A_1UBlDuwkjn1HWxdhQ1H2un9K4kyPLgBqJUc/edit?usp=sharing)
- [bfcache 테스터](https://back-forward-cache-tester.glitch.me/?persistent_logs=1)

출처: https://web.dev/bfcache/#optimize-your-pages-for-bfcache
