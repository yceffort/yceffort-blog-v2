---
title: '자바스크립트 함수의 성능 측정하기'
tags:
  - javascript
published: true
date: 2020-12-02 20:44:19
description: '사실 실전에서 해본적은 거의 없음 😇'
---

## Table of Contents

## `Performance.now`

Performance API는 `performance.now()`를 통해서 [DOMHighResTimeStamp](https://developer.mozilla.org/en-US/docs/Web/API/DOMHighResTimeStamp)에 접근할 수 있게 해준다. `performance.now()`는 페이지를 로드한 이후로 지난 ms를 보여준다. 최대 정밀도는 `5µs`정도다.

```javascript
const t0 = performance.now()
for (let i = 0; i < array.length; i++) {
  // some code.......
}
const t1 = performance.now()
console.log(t1 - t0, 'milliseconds')
```

`Chrome`

```bash
0.6350000001020817 "milliseconds"
```

`Firefox`

```bash
1 milliseconds
```

Chrome 과 Firefox 의 결과에 조금 차이가 있는 것을 볼 수 있는데, 이는 Firefox가 60버전 이후로 performance API의 정밀도를 2ms 정도로 조정했기 때문이다.

Performance API는 이외에도 다양한 기능을 제공하는데, [여기](https://blog.logrocket.com/how-to-practically-use-performance-api-to-measure-performance/)에tj 확인 가능하다.

### `Date.now`를 써도 되지 않을까?

물론 이것도 가능하지만, 약간의 차이가 있다.

`Date.now`는 마찬가지로 ms를 리턴하는데, 이는 시스템의 시간에서 Unix epoch(1970-01-01T00:00:00Z)의 차이를 리턴한다. 이는 부정확할 뿐만 아니라, 항상 증가한다고도 볼 수 없다.

> System time을 기반으로한 Date를 기준으로 실제 사용자를 모니터링하는 것은 적절치 않다. 대부분의 시스템은 정기적으로 시간을 동기화 하는 데몬을 실행한다. 그리고 그 시계는 15분 내지 20분 마다 몇 ms 씩 조정되는 것이 일반적이다. 따라서 그 속도에서 측정된 10 초간격의 1% 정도가 부정확할 것이다.

> Perhaps less often considered is that Date, based on system time, isn't ideal for real user monitoring either. Most systems run a daemon that regularly synchronizes the time. It is common for the clock to be tweaked a few milliseconds every 15-20 minutes. At that rate about 1% of 10 second intervals measured would be inaccurate.

출처: https://developers.google.com/web/updates/2012/08/When-milliseconds-are-not-enough-performance-now

## `Performance.mark` and `Performance.measure`

`Performance.now` 외에도 코드의 여러 지점에서 시간을 특정하고, 이를 [Webpagetest](https://felixgerschau.com/custom-metrics-webpagetest/)와 같은 성능 테스트 도구세어 사용자 지정 메트릭으로 사용할 수 있는 몇가지 다른 함수들이 존재한다.

### `Performance.mark`

이름에서 느껴지는 것 처럼, 코드 내에서 마킹을 할 수 있는 용도다.이 마크는 performance buffer에서 timestamp를 생성하여 나중에 코드의 특정 부분을 실행하는데 걸린 시간을 측정하는데 사용 가능하다.

마킹을 생성하기 위해서는, string을 파라미터로 함수를 호출해야 하며, 이 string은 나중에 식별자 용도로 사용된다. 마찬가지로 최대 정밀도는 `5µs`정도다.

```javascript
performance.mark('name')
```

- detail: null
- name: "name"
- entryType: "mark"
- startTime: 268528.33999999985
- duration: 0

### `Performance.measure`

이 함수는 1~3개의 arguments를 받는다. 첫번째 인수는 `name`이고, 나머지는 측정하고 싶은 마킹 영역을 넣으면 된다.

네비게이션 시작부터 측정

```javascript
performance.measure('measure name')
```

네비게이션 시작부터 특정 마킹 까지

```javascript
performance.measure('measure name', undefined, 'mark-2')
```

특정 마킹 부터 바킹까지

```javascript
performance.measure('measure name', 'mark-1', 'mark-2')
```

마킹 부터 지금까지

```javascript
performance.measure('measure name', 'mark-1')
```

## 측정 값 수집

### `performance entry buffer`로 부터 데이터 수집

이전 부터 계속 측정 결과가 `performance entry buffer` 에 수집된다고 언급했는데, 이제는 여기에 접근하여 값을 가져오는 방법을 알아보고자 한다.

이를 위해 performance API는 3종류의 api를 제공한다.

- `performance.getEntries()`: `performance entry buffer`에 저장된 모든 것을 보여준다.
- `performance.getEntriesByName('name')`
- `performance.getEntriesByType('type')`: 특정 타입에 대해서만 보여준다. `measure`, `mark`만 가능

모든 예제를 종합하자면, 대략 아래와 같은 코드가 만들어 질 것이다.

```javascript
performance.mark('mark-1')
// 성능을 측정할 코드...........
performance.mark('mark-2')
performance.measure('test', 'mark-1', 'mark-2')
console.log(performance.getEntriesByName('test')[0].duration)
```

## `console.time`

단순히 `console.time`을 호출하고, 측정 종료 시점에 `console.timeEnd`를 호출하면 된다.

```javascript
console.time('test')
for (let i = 0; i < array.length; i++) {
  // some code
}
console.timeEnd('test')
```

`chrome`

```bash
test: 0.766845703125ms
```

`firefox`

```bash
test: 2ms - timer ended
```

다른 API 대비 사용하기 간단하고, 수동으로 비교를 하지 않아도 알아서 비교를 해준다는 장점이 있다.

## 시간 정확도

당연한 이야기 이지만, 여러 브라우저에서 성능을 측정하다보면 결과가 다르다는 것을 눈치 챌 수 있다. 이는 브라우저가 [타이밍 공격](https://en.wikipedia.org/wiki/Timing_attack)과 [핑거프린팅](https://pixelprivacy.com/resources/browser-fingerprinting/) 등의 공격기법으로 부터 유저를 보호하기 위해서다. 이 시간이 너무나도 정확하다면, 해커는 사용자를 간단하게 식별할 수 있을 것이다.

앞서 언급한 이유 때문에, 60버전이후의 Firefox에서는 이러한 정확도를 최대 2ms정도로 감소 시켰다.

## 유념해야 할것

### 분할해서 살펴볼 것

단순히 코드의 어떤 부분이 느린지 엉뚱하게 추측하지 말고, 위에서 언급한 기능들을 사용하여 각각 나눠서 정밀하게 측정하자. 느린부분을 찾기 위해, 느린 코드 블록 주위에 `console.time`을 배치하자. 그 다음, 각부분의 성능을 측정하자. 만약 어떤 부분이 다른 부분보다 느리다는 것을 알아넀다면, 계속 나아가서 병목현상을 일으키는 부분을 찾을 때 까지 더 깊이 들어가자.

### 입력 값에 주의를

실제 애플리케이션에서는, 함수의 입력 값에 따라 결과가 많이 달라질 수 있다. 단순히 함수의 랜덤 값으로 테스트 할 것이 아니라, 실제로 사용되는 예제를 바탕으로 측정하는 것이 좋다.

### 함수를 여러번 실행하자.

배열을 순회하는 함수 내에서, 각각의 원소값을 계산하고 그 결과를 배열로 리턴하는 함수가 있다고 가정해보자. `forEach`와 `for`중에 무엇이 더 성능에 우위가 있을지 알아보고 싶을 것이다.

```javascript
function testForEach(x) {
  console.time('test-forEach')
  const res = []
  x.forEach((value, index) => {
    res.push((value / 1.2) * 0.1)
  })

  console.timeEnd('test-forEach')
  return res
}

function testFor(x) {
  console.time('test-for')
  const res = []
  for (let i = 0; i < x.length; i++) {
    res.push((x[i] / 1.2) * 0.1)
  }

  console.timeEnd('test-for')
  return res
}
```

```javascript
const x = new Array(100000).fill(Math.random())
testForEach(x)
testFor(x)
```

파이어 폭스에서 실행한다면 대략 이런 결과가 나올 것이다.

```bash
test-forEach: 4ms - 타이머 종료됨
test-for: 2ms - 타이머 종료됨
```

`forEach`가 더 느린가? 🤔 싶지만 여러번 하게 되면

```bash
test-forEach: 4ms
test-forEach: 3ms
test-for: 2ms
test-for: 1ms
```

별반 차이가 없음을 알수 있다.

### 그리고 다양한 브라우저에서

똑같은 짓을 크롬에서 해보자.

```bash
test-forEach: 5.589111328125 ms
test-forEach: 5.730712890625 ms
test-for: 4.765869140625 ms
test-for: 6.64892578125 ms
```

firefox와 chrome 은 서로 다른 자바스크립트 엔진을 가지고 있고, 이는 성능 최적화에도 차이가 있다. 이 경우, 같은 input 기준으로 firefox에서 보다 최적화를 잘하고 있음을 볼 수 있다. 그리고 두 엔진 모두에서 `forEach`보다는 `for`가 나은 것을 볼 수 있다. (유의미한 차이라고 볼 수 있을지는 모르겠지만)

따라서 성능 측정은 한브라우저에서 할 것이 아니라, 가능한 많은 모던 브라우저에서 해봐야 한다.

### CPU 스로틀링

항상 내가 개발하고 있는 컴퓨터는 대부분의 사용자가 사용하는 모바일 환경보다 더 빠르다는 것을 염두해 두어야 한다. 브라우저별로 CPU 성능을 쓰로틀 해주는 기능을 가지고 있으므로, 이를 활용해서 테스트 해야 한다.

- https://developers.google.com/web/updates/2017/07/devtools-release-notes#throttling
