---
title: '자바스크립트 성능과 번들 사이즈'
tags:
  - javascript
  - browser
published: true
date: 2021-02-27 23:40:05
description: '자바스크립트 성능에 중요한 건 번들크기 만은 아니다. 근데 개발 하느라 이것도 잘 못챙기고 있는듯.'
---

## Table of Contents

## 시작하며

자바스크립트 커뮤니티에서 요즘 가장 중요하게 생각하는 것중 하나는 번들 사이즈 인것 같다. 얼마나 많은 의존성을 가지고 있는가? 번들 사이즈를 더 작게 만들수는 없나? 레이지 로드를 잘 활용하고 있나? 사람들이 번들 사이즈에 대해 집착하는 이유 중 하나는 아무래도 눈에 잘 띈다는 점 때문일 거다. 물론, 번들 사이즈가 중요하지 않다는 것은 아니다. 하지만 번들 사이즈 외에도 중요한 것은 많이 있다.

- 파싱/컴파일에 걸리는 시간
- 실행 시간
- 파워 사용량
- 메모리 사용량
- 디스크 사용량

자바스크립트의 의존성은 위 모든 지표에 영향을 미친다. 위 지표는 그러나 번들 사이즈에 비해서는 덜 논의되는 편이다. 아무래도 번들사이즈에 비해 측정하기가 어렵기 때문일 것이다.

## 번들 사이즈

자바스크립트 코드의 크기를 논의 할 때, 우리는 명확히 할 필요가 있다. minified는 한 크기인가? gzip도? 트리쉐이킹은 했나? gzip 세팅은 제일 크게 했는가? 혹시 [브로틀리](https://yceffort.kr/2021/01/brotli-better-html-compression)를 사용했는가?

사소한걸 피곤하게 따지는 것 같지만, 사실 크기를 논의 할 때, 특히 압축과 비압축사이에서는 굉장히 중요하다. 압축된 크기는 사용자 브라우저에 얼마나 빠르게 전달되는지에 영향을 미칠 것이고, 비압축된 사이즈는 사용자의 브라우저에서 얼마나 빠르게 파싱되고, 컴파일되고, 실행되는지에 영향을 미칠 것이다. 

### Bundlephobia

자바스크립트 라이브러리 크기를 분석하는데 있어서 유용한 툴은 바로 [Bundlephobia](https://bundlephobia.com/)다. 라이브러리의 의존성을 볼 수도 있고, minified된 사이즈와 압축된 사이즈까지 볼수도 있고 다운로드에 걸리는 시간도 볼 수 있다.

![bundlephobia](./images/bundlephobia.png)

그러나 bundlephobia를 보는데 있어서 주의를 해야할 것이 있다.

- 여기에는 트리쉐이킹이 반영되있지 않다는 것이다. 라이브러리에서 특정 모듈만을 사용한다면, 다른 모듈들은 트리쉐이킹되어 줄어들 것이다. 
- 또한 서브 디렉토리의 의존성에 대해서는 알수가 없다. 예를 들어 `preact`를 가져오는데 한 비용은 알수가 있다. 그러나 `preact/compat`에 대해서는 알 방법이 없다. `compat.js`가 정말 큰 파일이라도 그것을 알 방법이 없다.
- 만약 폴리필이 포함되는 경우 (`Object.assign()`나 `Buffer API`등 번들러가 주입하는 경우) 여기서 반드시 표시되지 않는다. 

위의 요소들을 점검하기 위해서는, 번들러를 실행해서 결과물을 확인해보면 된다. 번들러는 서로 다 다르며, 설정과 여러가지 요소에 따라서 크기가 달라질 수 있다. 

### Webpack Bundle Analyzer

[Webpack Bundle Analyzer](https://github.com/webpack-contrib/webpack-bundle-analyzer)는 웹팩 결과물로 나온 모든 청크들을 잘 보여주고, 어떤 모듈이 청크들을 구성하는지 확인할 수 있다.

![Webpack Bundle Analyzer](https://cloud.githubusercontent.com/assets/302213/20628702/93f72404-b338-11e6-92d4-9a365550a701.gif)

여기서 중요한 것은 `Parsed`와 `Gzipped`다. Bundlephobia와는 다르게 실제로 번들러를 거쳤을 때 의 크기를 볼 수가 있다.

### Rollup Plugin Analyzer

[Rollup Plugin Analyzer](https://github.com/doesdev/rollup-plugin-analyzer)는 번들의 크기를 빌드하는 과정에서 콘솔로도 볼 수 있다. 그러나, minified나 gzipped한 크기는 볼수가 없다. 

그 외에도 아래와 같은 도구가 있다.

- [bundlesize](https://github.com/siddharthkp/bundlesize)
- [Bundle Buddy](https://www.npmjs.com/package/bundle-buddy)
- [Sourcemap Explorer](https://github.com/danvk/source-map-explorer)
- [Webpack Analyse](https://github.com/webpack/analyse)

## 번들 크기 말고 다른 것

앞에서도 언급했듯이, 번들사이즈가 전부는 아니다. 이 외에도 살펴볼 만한 다양한 것들이 많다.

### Runtime CPU Cost

첫번째 이자 가장 중요한 것중 하나는 런타임 시의 비용이다. 이는 여러가지 요소로 나눠서 생각해 볼 수 있다.

- Parsing
- Compilation
- Execution

이 세가지 요소는 기본적으로 엔드 투 엔드 비용으로, `require('something')` 이나 `import 'something'`를 호출 할 때 발생한다. 번들사이즈와도 연관이 있지만, 반드시 일치하는 것은 아니다. 

```javascript
const start = Date.now()
while (Date.now() - start < 5000) {}
```

위의 이상한 코드를 살펴보자. Bunldephobia에서는 위 코드가 높은 점수를 받겠지만 (코드의 크기가 작기 때문에) 메인스레드를 무려 5초나 잡아먹는다. 위의 예제 처럼, 작은 라이브러리가 메인스레드에 악영향을 미치는 경우가 더러있다. DOM의 모든 요소를 순회하거나, 로컬 스토리지에 큰 배열을 포문을 돌거나 하는 등. 직접 모든 의존성을 하나하나 살펴 보지 않는 한, 내부에서 무엇을 하는지 알 수 없다.

Parsing과 Compilation은 모두 측정하기 어려운 항목이다. 브라우저는 바이트 코드 캐싱에 대한 최적화 기능을 가지고 있기 때문에 이 것들을 알기가 쉽지 않다. 예를 들어, 브라우저가 두세번째 페이지 로딩 부터는 파싱과 컴파일 단계를 거치지 않는다. 또는 자바스크립트가 서비스워커에서 캐싱을 했을 수도 있다. 따라서 실제로 브라우저가 미리 모듈을 캐시한다면, 모듈을 파싱하고 컴파일하는데 비용이 적게 든다고 생각할 수 있다. 

따라서 100% 제대로 확인할 수 있는 방법은, 브라우저 캐시를 완전히 지우고 첫 페이지를 로딩하는 것이다. 일반적으로 이러한 작업은 private 모드나 주로 사용하지 않는 브라우저에서 이 작업을 수행한다. 또한 브라우저 확장 기능을 꺼둬야 한다. 

또한 크롬의 기능을 활용하여 CPU 쓰로틀링을 4x 또는 6x를 설정해두는 것도 좋은 방법이다. 이는 하이엔드 개발자의 컴퓨터 보다 훨씬 더 실제 사용자를 더 잘 대표할 수 있다.

네트워크 속도의 경우, 네트워크 속도 조절 기능을 사용할 수 있다. (3G)

이를 모두 요약하자면, 다음과 같은 단계를 거친다.

1. 사생활 보호 모드로 브라우저를 킨다
2. `about:blank` 로 진입한다. (브라우저 홈 화면의 `unload` 이벤트를 측정하지 않기 위해)
3. 크롬의 DevTool을 연다
4. Performance Tab을 연다
5. Cpu와 네트워크 쓰로틀링을 켠다.
6. Record 버튼을 누른다
7. URL을 입력하고 엔터를 친다.
8. 페이지 로딩이 끝나면 레코딩을 중지한다.

![performance](./images/performance.png)

이제 최초 페이지 로딩시에 자바스크립트 코드가 parse, compile, execution에 걸리는 시간을 측정할 수 있다. 

이에 덧붙여 [User Timing API](https://developer.mozilla.org/en-US/docs/Web/API/User_Timing_API)를 활용해서 웹 애플리케이션의 일부를 사용자에게 의미 있는 이름으로 표시한다. 루트 애플리케이션의 초기렌더링, 블로킹 XHR 호출 등 비용이 많이 들 것으로 우려되는 부분에 초점을 맞춘다. 만약 이러한 측정 작업으로 인한 오버헤드가 우려되는 경우, 프로덕션에서 이러한 기능을 제거하는 방법도 있다. [쿼리 파라미터를 사용해서](https://github.com/nolanlawson/pinafore/blob/ba3b76f769455908eca9f6f59584d18e2bd19f0e/src/routes/_utils/marks.js) 측정 기능을 끌 수도 있고, [terser의 pure_funcs](https://terser.org/docs/api-reference.html#compress-options)를 활용해서 제거할 수도 있다.

```javascript
const enabled = process.browser && performance.mark && (
  process.env.NODE_ENV !== 'production' ||
  (typeof location !== 'undefined' && location.search.includes('marks=true'))
)

const perf = process.browser && performance

export function mark (name) {
  if (enabled) {
    perf.mark(`start ${name}`)
  }
}

export function stop (name) {
  if (enabled) {
    perf.mark(`end ${name}`)
    perf.measure(name, `start ${name}`, `end ${name}`)
  }
}
```

또 다른 유용한 도구는 [mark loader](https://github.com/statianzo/mark-loader)다. 이 플러그인은 디펜던시 런타임 비용을 볼수 있오록 모듈을 측정 api로 감싸는 웹팩 플러그인이다. 

런타임 성능을 측정 할 때 한 가지 주의해야 할 점은, 축소된 코드와 그렇지 않은 코드 사이에 다를 수 있따는 것이다. 사용되지 않는 함수들이 제거 될 수 있고, 코드가 더 작아지고 최적화 될 수 있으며, `env.NODE_ENV === 'development'`로 인해 프로덕션 모드에서 코드가 정제될 수도 있다. 이러한 상황을 잘 해결할 수 있는 방법은, `performance.mark`와 `performance.measure`를 활용하는 것이다. 

### Poser Usage

굳이 환경보호론자가 아니더라도 전력 사용을 최소화하는 것이 중요하다는 것은 누구나 알 것이다. 사람들은 점점 전원 콘센트가 꽂히지 않은 모바일 기기에서 웹을 많이 검색하고 있다. 그리고 고객들은 잘못된 웹사이트 때문에 전력이 바닥나는 것을 원치 않을 것이다. 

전력 사용량은 CPU 사용량의 부분집합으로 취급되곤 한다. 몇 가지 예외를 제외하면, 대부분의 경우 웹사이트가 과도한 전력을 사용하는 이유는 메인스레드에서 과도하게 CPU를 사용하기 때문이다.

따라서 앞에서 설명한 자바스크립트의 parse/compile/execute 시간을 개선한다면, 전력 소비량도 줄일 수도 있다. 특히 수명이 긴 웹 애플리케이션의 의 경우, 대부분의 전력 소모가 첫 페이지 로드 이후에 발생한다. 그렇게 되면 사용자가 유휴 웹 페이지만 보고 있어도, 노트북 팬이 돌아가거나, 휴대폰이 뜨거워지는 것을 알아차릴 수 있다.

이러한 상황에서 선택할 수 있는 도구는 앞서 말한 Chrome DevTools의 Performance 탭이다. 일반적으로, 타이머 또는 애니메이션으로 인해 CPU 사용량이 증가한다. 예를 들어 잘못 코딩된 커스텀 스크롤바, Intersection Observer Polyfill 또는 애니메이션 로딩 스피너는, 매 requestAnimationFrame 이나 setInterval 루프에서 반복되고 있을 수 있다.

이러한 종류의 전원 누수는 또한 최적화되지 않은 CSS 애니메이션으로 인해 발생할 수 있다. 즉, 자바스크립트의 문제가 아닐 수도 있다. (이 경우 Chrome UI에서 보라색으로 표시된다.) CSS 애니메이션을 오래 실행하는 경우, GPU 가속 CSS 속성을 사용해야 한다.

Chrome Performance Monitor Tab은 Performance Tab과 다르다. 수동으로 추적을 시작/중지할 필요 없이 웹사이트가 작동하는 방식을 보여주는 일종의 하트비트 모니터다. 비활성 웹 페이지에서 CPU 사용량이 일정하지 않으면 전원 사용에 문제가 있을 수 있다.

![performance monitor](./images/performance-monitor.png)

> [performance monitor](https://developers.google.com/web/updates/2017/11/devtools-release-notes#perf-monitor)

### Memory Cost

메모리 사용량 분석은 어려웠지만, 최근에는 많은 도구들이 나오면서 개서되었다.

한가지 중요한 것은, 메모리 사용량과 메모리 누수는 별개의 문제라는 것이다. 별도의 메모리 누수가 없다 하더라도, 메모리 사용량이 높을 수도 있다. 반면 작은 규모의 웹사이트가, 유출로 인하여 메모리 사용량이 커질 수도 있다.

메모리 사용량을 분석하는 api는 [performance.measureUserAgentSpecificMemory](https://www.chromestatus.com/feature/5685965186138112)가 있다. 이 API는 다음과 같은 이점이 있다.

1. 가비지 콜렉팅 된 이후에 resolve 되는 Promise를 반환한다.
2. 자바스크립트 VM 크기 뿐만 아니라, 웹 워커와 iframe 등을 포함하는 DOM 메모리도 포함한다.
3. site isolation로 인해 프로세스가 분리된 cross-origin iframe도 세분화한다. 따라서 embedded나 광고가 메모리를 얼마나 먹는지 판단할 수 있다.

```javascript
{
  bytes: 60_000_000,
  breakdown: [
    {
      bytes: 40_000_000,
      attribution: [
        {
          url: "https://foo.com",
          scope: "Window",
        },
      ]
      types: ["JS"]
    },
    {
      bytes: 0,
      attribution: [],
      types: []
    },
    {
      bytes: 20_000_000,
      attribution: [
        {
          url: "https://foo.com/iframe",
          container: {
            id: "iframe-id-attribute",
            src: "redirect.html?target=iframe.html",
          },
        },
      ],
      types: ["JS"]
    },
  ]
}
```

위에 있는 `bytes`는 사용중인 메모리 양을 나타내는 지표다. 

이 API를 사용하는 것은 그럼에도 여전히 까다로울 수 있다. 일단 Chrome 89+에서만 사용할 수 있다. (오래된 버전의 경우 실험용 기능을 체크하고 쓸 수 있다) 그러나 더 문제가 되는 것은, 이를 남용할 가능성 때문에 이 API의 호출은 cross-origin 의 격리된 컨텍스트로 제한되었다는 것이다. 따라서 일부 특수 헤더를 설정해야 하며, cross-origin 리소스 (외부 CSS, 자바스크립트, 이미지 등)에 의존하는 경우 일부 특수 헤더도 설정해야 한다.

이 API를 자동화된 테스트에만 사용할 계획이라면, `--disable-web-security`플래그를 사용하여 크롬을 실행할 수 있다. (물론 어느 정도 위험성을 감수해야 한다.) 그리고 또한가지 참고해야 할 것은, 메모리 측정은 헤드리스 모드에서는 작동하지 않는다는 것이다. 

물론, 이 API 는 세분화해서 데이터를 제공하지 않는다. 예를 들어 리액트와 lodash가 각각 몇 바이트를 차지 하는지는 정확히 알 수 없다는 뜻이다. 이를 위한 가장 확실한 방법은 A/B 테스트다. 이는 메모리 측정을 위해 기존에 사용햇던 방법보다 훨씬더 나은 방법이다.

### Disk Usage

디스크 사용량은 장치에 따라 사용 가능한 저장 용량에 따라 브라우저 할당량 제한에 걸릴수도 있으므로 웹 애플리케이션 사용에 있어서 중요하게 고려해야 한다. 과도한 스토리지 사용은 서비스워커 캐시에 너무 많은 대용량 이미지를 채우는 등 여러가지 형태로 나타날 수 있으며, 자바스크립트 또한 마찬가지다.

자바스크립트 모듈의 디스크 사용량이 번들 크기와 직접적인 상관관계가 있다고 생각할 수 있지만, 꼭 그런것 만은 아니다. 예를 들어 [emoji-picker-element](https://github.com/nolanlawson/emoji-picker-element)의 경우, 이모지 데이터를 indexeddb에서 꽤나 무겁게 사용하고 있기 때문에, 데이터베이스가 디스크 사용을 어떻게 사용하고 있는지 인식해야 한다.

![application-storage](./images/application-storage.png)

크롬 DevTools에 있는 Application Tab에서 현재 웹사이트가 사용하고 있는 전체 용량을 알 수가 있다. 처음 보기엔 괜찮아 보이지만, IndexedDB 의 경우 브라우저 마다 구현 방식이 다르기 때문에 브라우저 마다 차지하는 용량이 다르게 나타날 수 있다. 이를 해결할 수 있는 방법 중하나는 Puppeteer와 비슷한 [Playwright](https://github.com/microsoft/playwright)에서 아래 코드를 실행하는 것이다.

```javascript
function getIdbFolder (browserType, userDataDir) {
  switch (browserType) {
    case 'chromium':
      return path.join(userDataDir, `Default/IndexedDB/http_localhost_${port}.indexeddb.leveldb`)
    case 'firefox':
      return path.join(userDataDir, `storage/default/http+++localhost+${port}/idb`)
    case 'webkit':
      return path.join(userDataDir, `databases/indexeddb/v1/http_localhost_${port}`)
  }
}
```

초기화된 빈 브라우저를 시작할 수 있으므로, 브라우저를 먼저 시작하고 `/temp`에 스토리지를 기록한다음, 인덱싱된 스토리지를 측정하는 것이 가능하다.

## 결론

성능이란 것은 다방면적인 측면을 고려 해야 한다. 번들 사이즈만 줄여서 해결된다면 좋겠지만, 여러가지 측면을 고려해야 한다. 이 때문에 이것이 굉장히 부담스럽게 느껴질 수도 있다. 그래서 [core web vital](https://web.dev/vitals/)이나 번들 사이즈에 집중해서 문제를 해결하는 것이 꼭 나쁜 것 만은 아니다. 웹 애플리케이션 성능 측정에 여러가지를 고민해야 한다고 말하면, 사람들은 이에 압도되서 아무것도 안 할수도 있다. 그렇지만, 이런 것도 있다는 것을 알아둔다면 최적화에 도움이 되지 않을까.


> 참고: https://nolanlawson.com/2021/02/23/javascript-performance-beyond-bundle-size/