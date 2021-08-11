---
title: '브라우저와 Nodejs의 이벤트 루프는 무엇이 다를까'
tags:
  - web
  - javascript
  - browser
published: true
date: 2021-08-10 22:22:37
description: '인생은 돌고 도는 이벤트 루프'
---

## Table of Contents

## 이벤트 루프는 정확히 무엇인가?

`이벤트 루프` 사실 일반적인 프로그래밍 패턴을 지칭하는 용어다. 프로그래밍의 이벤트나 메시지를 대기하나가 처리하는 일종의 프로그래밍 구조체라고 볼 수 있다. (https://ko.wikipedia.org/wiki/%EC%9D%B4%EB%B2%A4%ED%8A%B8_%EB%A3%A8%ED%94%84) 자바스크립트와 Nodejs의 이벤트 루프도 별반 다르지 않다. 자바스크립트는 애플리케이션이 실행되면 다양한 이벤트를 발생시키고, 이러한 이벤트는 처리를 위해 이벤트 핸들러 형태로 대기열에 존재한다. 이벤트 루프는 대기중인 이벤트 핸들러를 지속적으로 지켜보다가, 이벤트 핸들러가 존재하면 이를 실행한다.

### HTML5 스펙으로 살펴보는 이벤트 루프

[HTML5의 스펙](https://html.spec.whatwg.org/)은 여러 벤더가 브라우저나 자바스크립트 런타임, 또는 기타 관련한 라이브러리를 개발하는데 사용할 수 있는 표준 가이드라인을 제시한다.

대부분의 브라우저와 자바스크립트 런타임은 이러한 가이드라인을 그대로 따르기 때문에 전세계 웹서비스에 더 나은 호환성을 제공한다. 그러나 사실은 이 단일 소스에서 약간씩 벗어나서 흥미로운 (혹은 짜증나는) 결과를 유발하기도 한다.

여기에서는 이러한 흥미로운 결과, 특히 Nodejs와 브라우저와의 차이에 대해서 알아보려고 한다. 개별 브라우저 구현은 언제든 조금씩 변할 수 있으므로, 자세히 알아보지는 않는다.

### 클라이언트 사이드와 서버사이드 자바스크립트

지난 수년간, 자바스크립트는 브라우저에서 실행되는 웹 애플리케이션에서만 사용되어져 왔다. 그리고 이 후 자바스크립트는 nodejs를 사용하여 서버 사이드 애플리케이션을 만드는데에도 사용할 수 있다. 두 곳 모두 자바스크립트를 사용하지만, 클라이언트와 서버사이드에서의 요구사항은 조금씩 다를 수 있다.

브라우저는 일종의 샌득박스 환경이며, 파일 시스템 작업, 네트워크 작업 등 자바스크립트가 수행할 수 있는 작업에 권한 제한이 있다. 그러나 서버사이드 자바스크립트(Nodejs)는 이벤트루프에서 이러한 것들을 모두 실행할 수 있다.

브라우저와 Nodejs 모두 자바스크립트를 사용하여 비동기 이벤트 기반 패턴을 구현한다. 그러나 브라우저의 맥락에서 봤을 때에 "이벤트"란 웹 페이 지 내에서의 상호작용 (클릭, 마우스 이동, 키보드 이벤트 등..)이지만, Nodejs에서의 맥락에서 이벤트란 파일 I/O, 네트워크 I/O 등이다. 이러한 요구 사항의 차이로 인해 크롬과 Node는 자바스크립트 실행을 위해 모두 V8 엔진을 사용하지만, 이벤트 루프 구현에는 차이가 있다.

'이벤트루프'란 결국 프로그래밍 패턴에 불과하기 때문에, V8은 자바스크립트 런타임과 함께 외부 이벤트 루프 구현을 플러그인 해줄 수 있록 해준다. 이러한 유연성을 바탕으로, 크롬 브라우저는 [libevent](https://libevent.org/)를, nodejs는 [libuv](https://blog.insiderattack.net/javascript-event-loop-vs-node-js-event-loop-aea2b1b85f5c#:~:text=and%20NodeJS%20uses-,libuv,-to%20implement%20the)를 각각 이벤트 루프 구현을 위해 사용한다. 그러므로, 자바스크립트와 Nodejs의 이벤트루프는 기본적으로 다른 라이브러리를 사용하여 약간의 차이가 있을 수 있지만, '이벤트루프'라고 하는 일반적인 프로그래밍 패턴을 구현하고 있다는 것에서 비슷하다.

## 브라우저 vs Nodejs 무엇이 다른가?

### 마이크로, 그리고 매크로 태스크

> 간단히말해, 마이크로 태스크와 매크로 태스크는 서로 다른 비동기 태스크 처리기다. 매크로 태스크에 비해 마이크로 태스크의 우선순위가 더 높다. 마이크로 태스크의 예로는 `Promise`가 있다. `setTimeout은 대표적인 매크로 태스크다.

브라우저와 Nodejs에 눈에 띄는 차이점은 **마이크로 태스크와 매크로 태스크의 우선순위를 어떻게 정하느냐** 이다. Nodejs 11 이상에서는 브라우저의 동작과 일치하지만, 이전 버전은 상당히 다르다. 자, 아래 면접 질문으로 나올 것 만 같은 아래 코드르 보자.

> nodejs 11이전 버전에서 무슨일이 있는지 살펴보려면 https://blog.insiderattack.net/new-changes-to-timers-and-microtasks-from-node-v11-0-0-and-above-68d112743eb3

```javascript
Promise.resolve().then(() => console.log('promise1 resolved'))
Promise.resolve().then(() => console.log('promise2 resolved'))
setTimeout(() => {
  console.log('set timeout3')
  Promise.resolve().then(() => console.log('inner promise3 resolved'))
}, 0)
setTimeout(() => console.log('set timeout1'), 0)
setTimeout(() => console.log('set timeout2'), 0)
Promise.resolve().then(() => console.log('promise4 resolved'))
Promise.resolve().then(() => {
  console.log('promise5 resolved')
  Promise.resolve().then(() => console.log('inner promise6 resolved'))
})
Promise.resolve().then(() => console.log('promise7 resolved'))
```

> `queueMicrotask`를 사용하여 마이크로 태스크를 스케쥴링 할 수도 있다.

브라우저 (크롬, 파이어폭스, 사파리. IE는 브라우저가 아니므로 제외) + Nodejs 11 이상

```bash
promise1 resolved
promise2 resolved
promise4 resolved
promise5 resolved
promise7 resolved
inner promise6 resolved
set timeout3
inner promise3 resolved
set timeout1
set timeout2
```

nodejs 11 미만

```bash
promise1 resolved
promise2 resolved
promise4 resolved
promise5 resolved
promise7 resolved
inner promise6 resolved
set timeout3
set timeout1
set timeout2
inner promise3 resolved
```

[HTML5 스펙에 정의된 이벤트 루프 가이드라인](https://html.spec.whatwg.org/multipage/webappapis.html#event-loop-processing-model)에 따르면, 이벤트 루프는 매크로 태스큐에서 하나의 매크로 태스크를 처리하기전에 마이크로 태크스에 있는 모든 것을 처리해야 된다. 이 예제에서는, `set timeout3` 콜백이 실행되면, promise 콜백을 예약한다. HTML5의 스펙에 따라서, 타이머 콜백 큐의 다른 콜백을 처리하기전에, 이벤트 루프가 마이크로태스크 큐가 비어있는지 확인해야 한다. 따라서 새로 추가된 promise callback을 실행하고 처리하여야 한다. 이 작업을 처리하면, 비로소 마이크로 태스크 큐가 비어 이벤트 루프가 남은 `setTimeout1` `setTimeout2`을 실행할 수 있게 된다.

그러나 11 버전 이전의 nodejs에서는, 이벤트 루프의 두 사이 단계에서만 마이크로 태스크열을 비우게 된다. 따라서 `inner promise3`은 모든 `setTimeout3`이 실행되기 전까지 실행될 수가 없게 된다.

### 내부 타이머 동작의 차이

타이머 동작은 nodejs, 브라우저 간 뿐만아니라 브라우저 벤더간, 버전마다 다르다. 여기서 가장 주목할만한 두가지는 timeout이 0일때와, timeout이 중첩되어 있을 때다. 이 러한 두가지 동작의 차이를 알기 위해 nodejs v10.19.0, v11.0.0, chrome, firefox, safari에서 아래의 코드를 실행해보자. 이 코드는 timeout이 0 인 중첩타이머 8개를 스케쥴링하고, 각 콜백이 스케쥴링 된이후 실행되기까지의 걸린 시간을 계산한다.

```javascript
const startHrTime = () => {
  if (typeof window !== 'undefined') return performance.now()
  return process.hrtime()
}

const getHrTimeDiff = (start) => {
  if (typeof window !== 'undefined') return performance.now() - start
  const [ts, tns] = process.hrtime(start)
  return ts * 1e3 + tns / 1e6
}

console.log('start')
const start1 = startHrTime()
const outerTimer = setTimeout(() => {
  const start2 = startHrTime()
  console.log(`timer1: ${getHrTimeDiff(start1)}`)
  setTimeout(() => {
    const start3 = startHrTime()
    console.log(`timer2: ${getHrTimeDiff(start2)}`)
    setTimeout(() => {
      const start4 = startHrTime()
      console.log(`timer3: ${getHrTimeDiff(start3)}`)
      setTimeout(() => {
        const start5 = startHrTime()
        console.log(`timer4: ${getHrTimeDiff(start4)}`)
        setTimeout(() => {
          const start6 = startHrTime()
          console.log(`timer5: ${getHrTimeDiff(start5)}`)
          setTimeout(() => {
            const start7 = startHrTime()
            console.log(`timer6: ${getHrTimeDiff(start6)}`)
            setTimeout(() => {
              const start8 = startHrTime()
              console.log(`timer7: ${getHrTimeDiff(start7)}`)
              setTimeout(() => {
                console.log(`timer8: ${getHrTimeDiff(start8)}`)
              })
            })
          })
        })
      })
    })
  })
})
```

`node 10.1.0`

```bash
timer1: 0.650208
timer2: 1.617334
timer3: 1.456791
timer4: 1.417208
timer5: 1.38725
timer6: 1.379334
timer7: 1.374334
timer8: 1.377042
```

`node 14.17.1`

```bash
timer1: 0.990541
timer2: 1.715584
timer3: 1.872625
timer4: 1.55775
timer5: 1.509125
timer6: 1.48125
timer7: 1.474916
timer8: 1.4655
```

`chrome`

```bash
timer1: 1.5999999940395355
timer2: 1.399999976158142
timer3: 1.5
timer4: 1.4000000059604645
timer5: 5.300000011920929 # 4번째 타이머 부터 4ms 이후에 실행됨
timer6: 5.199999988079071
timer7: 4.9000000059604645
timer8: 5.300000011920929
```

`safari`

```bash
timer1: 1
timer2: 1.0000000000004547
timer3: 1.9999999999995453
timer4: 1
timer5: 1.0000000000004547
timer6: 4.999999999999545
timer7: 4
timer8: 5
```

`firefox`

```bash
timer1: 0
timer2: 0
timer3: 0
timer4: 1
timer5: 5
timer6: 6
timer7: 5
timer8: 5
```

살펴본 결과 아래 몇가지 사실을 알 수 있었다.

- 0으로 설ㅈ어하더라도, Nodejs 타이머는 최소 1ms이후에 실행된다.
- 크롬과 파이어폭스는 처음 4개의 타이머가 1ms 언저리에 실행되었지만, 그 후에는 4ms 이후에 실행되었따.
- 사파리는 크롬/파이어폭스와 비슷하지만, 6번째 타이머부터 4ms 이후에 실행된다.

브라우저에서오는 4ms 의 시간차이는 어디서 만들어진걸까? 이는 앞서 언급했던 [HTML5 스펙](https://html.spec.whatwg.org/multipage/timers-and-user-prompts.html#timers)에 기재되어 있다.

> Timers can be nested; after five such nested timers, however, the interval is forced to be at least four milliseconds.

이 규칙에 따르면, 크롬과 파이어폭스는 기재된 스펙에 맞게 5번째 부터 발생했지만, 사파리는 규칙을 제대로 따르고 있지 않는 것 같다.

브라우저는 잠시 뒤로하고, node는 중첩에 따라 시간 제한을 별도로 두지 않아도 된다는 것을 알 수 있다.

### Nodejs와 Chrome의 최소 타임아웃 시간

NodeJs와 크롬 모두 중첩되지 않은 경우라 할지라도 모든 타이머에 최소 1ms의 시간 지연이 발생한다. 그러나 크롬과 다르게 nodejs는 중첩수준에 상관없이 꾸준히 1ms내외로만 적용된다. 아래 코드를 살펴보면, 모든 타이머에 1ms의 시간이 왜 nodjs에서 적용되는지 알 수 있다.

```javascript
function Timeout(callback, after, args, isRepeat, isRefed) {
  after *= 1 // Coalesce to number or NaN
  if (!(after >= 1 && after <= TIMEOUT_MAX)) {
    if (after > TIMEOUT_MAX) {
      process.emitWarning(
        `${after} does not fit into` +
          ' a 32-bit signed integer.' +
          '\nTimeout duration was set to 1.',
        'TimeoutOverflowWarning',
      )
    }
    after = 1 // Schedule on next tick, follows browser behavior
  }

  // ....redacted
}
```

크롬도 위와 비슷한 작업을 `DOMTimer`에서 한다. 그리고, `maxTimerNestingLevel`에 다다르면 4ms가 적용되는 것도 알 수 있다.

```cpp
DOMTimer::DOMTimer(ExecutionContext* context, PassOwnPtrWillBeRawPtr<ScheduledAction> action, int interval, bool singleShot, int timeoutID)
    : SuspendableTimer(context)
    , m_timeoutID(timeoutID)
    , m_nestingLevel(context->timers()->timerNestingLevel() + 1)
    , m_action(action)
{
    // ... redacted ...
    double intervalMilliseconds = std::max(oneMillisecond, interval * oneMillisecond);
    if (intervalMilliseconds < minimumInterval && m_nestingLevel >= maxTimerNestingLevel)
        intervalMilliseconds = minimumInterval;
    if (singleShot)
        startOneShot(intervalMilliseconds, FROM_HERE);
    else
        startRepeating(intervalMilliseconds, FROM_HERE);
}
```

위에서 알 수 있듯, 자바스크립트 런타임에는 타이머와 중첩된 타이머가 0으로 설정되었을 때 실행되는 방법에 대한 독특한 구현이 있다. 따라서 자바스크립트 애플리케이션이나 라이브러리를 개발 할 때, 호환성을 높이기 위해 런타임 별 동작에 크게 의존하지 않는 것이 좋다.

### `process.nextTick`, `setImmediate`

또다른 브라우저와 nodejs의 차이점은 `process.nextTick`과 `setImmediate`이다.

`process.nextTick`은 NodeJS에만 있는 api이며 브라우저에는 이와 비슷한 동작을 하는 api는 없다. `nextTick`이 nodejs의 libuv 이벤트 루프의 일부는 아니지만, `nextTick`은 이벤트 루프 동안 nodejs가 C++과 JS 경계를 넘어가는 과정에서 실행된다. 그래서, 어떤 측면에서는 이벤트 루프와 관련있다고 볼 수도 있다.

`setImmediate`또한 nodejs 전용 api다. [MDN](https://developer.mozilla.org/en-US/docs/Web/API/Window/setImmediate)과 [caniuse.com](https://caniuse.com/?search=setImmediate)에 따르면, 놀랍게도 IE10, 11, 그리고 초기 엣지 버전에서 사용이 가능한 api다. 그외에 다른 브라우저 에서는 사용이 불가능하다.

둘의 차이를 알기 위해서는, 이벤트 루프의 과정에 대해 알필요가 있다.

![pahse of event loop](https://jinoantony.com/static/624c9768d8888b109a4649298c0cb091/29492/event-loop.png)

1. Timer: `setTimeout`의 시간이 다된 타이머, `setInterval`로 추가된 인터벌 함수가 실행됨
2. Pending Callback: 다음 루프로 지연된 I/O 콜백을 실행
3. Idle handler: 내부적으로 사용되는 libuv 내부 작업을 수행
4. Prepare Handler: 내부적으로 사용되는 I/O를 폴링하기전에 몇가지 사전작업 수행
5. I/O poll: 새 I/O 이벤트를 검색하고, I/O 관련 콜백을 실행
6. Check Handler: `setImmediate`가 실행
7. Close callback: 클로즈 핸들러 실행

`setImmediate()`

```javascript
console.log('Start')
setImmediate(() => console.log('Queued using setImmediate'))
console.log('End')
```

```bash
Start
End
Queued using setImmediate
```

`setImmediate()`는 callback을 인수로 받으며, 이를 이벤트 큐에 추가한다. (immediate queue) 위에서 언급했듯, `setImmediate()`는 `Check Handler`과정에서 수행된다.

`process.nextTick()`

```javascript
console.log('Start')
process.nextTick(() => console.log('Queued using process.nextTick'))
console.log('End')
```

```bash
Start
End
Queued using process.nextTick
```

마찬가지로 callback을 인수로 받지만, `next tick` 큐라고 하는 별도의 큐에 추가한다. 이 `process.nextTick()`로 넘겨받은 callback은 현재 phase가 넘어간 이후에 실행된다. 즉, 이벤트 루프의 각 단계가 넘어갈 때마다 실행된다.

따라서, 차이점을 정리하자면

1. `setTimeout`은 Check handler과정에서 실행되지만, `process.nextTick`은 이벤트 루프의 각 단계 사이에서 실행된다.
2. 1번에 의거하여, `process.nextTick`의 우선순위가 더 높다. (= 먼저 실행된다.)

   ```javascript
   setImmediate(() => console.log('I run immediately'))

   process.nextTick(() => console.log('But I run before that'))
   ```

   ```bash
   But I run before that
   I run immediately
   ```

3. 만약 특정 단계에서 `process.nextTick()`이 호출되면, 이벤트루프를 계속하기전에 모든 콜백이 전달된다. `process.nextTick`이 재귀적으로 호출되면, 이벤트루프가 차단되고 `I/O Starvation`이 생성된다. 아래 예제 코드를 실행해보면, `setImmediate`나 `setTimeout`이 실행되지 않고 `process.nextTick`만 계속 도는 것을 알 수 있다.

   ```javascript
   let count = 0

   const cb = () => {
      console.log(`Processing nextTick cb ${++count}`)
      process.nextTick(cb)
   }

   setImmediate(() => console.log('setImmediate is called'))
   setTimeout(() => console.log('setTimeout executed'), 100)

   process.nextTick(cb)

   console.log('Start')
   ```

   ```bash
   Start
   Processing nextTick cb 1
   Processing nextTick cb 2
   Processing nextTick cb 3
   Processing nextTick cb 4
   Processing nextTick cb 5
   Processing nextTick cb 6
   Processing nextTick cb 7
   Processing nextTick cb 8
   Processing nextTick cb 9
   Processing nextTick cb 10
   # 무한히 안끝나고 nextTick만 계속 돈다
   ```

1. `process.nextTick`과는 다르게, 재귀적으로 `setImmediate`를 호출하면 이벤트루프를 블로킹하지 않는다. 모든 재귀 호출은 다음 이벤트 루프에서 실행된다. 아래 코드를 보면, `setImmediate`가 재귀적으로 호출되지만 이벤트 루프를 블로킹하지 않아 간간히 `setTimeout`이 호출되는 것을 알 수 있다.
   ```javascript
   let count = 0

   const cb = () => {
      console.log(`Processing setImmediate cb ${++count}`)
      setImmediate(cb)
   }

   setImmediate(cb)
   setTimeout(() => console.log('setTimeout executed'), 100)

   console.log('Start')
   ```

   ```bash
   Start
   Processing setImmediate cb 1
   Processing setImmediate cb 2
   Processing setImmediate cb 3
   Processing setImmediate cb 4
   ...
   Processing setImmediate cb 503
   Processing setImmediate cb 504
   setTimeout executed
   Processing setImmediate cb 505
   Processing setImmediate cb 506
   ...
   ```

그럼 각각 언제 써야할까? 문서에 따르면 왠만하면 `setImmediate()`를 사용하라고 되어 있다.

> We recommend developers use setImmediate() in all cases because it's easier to reason about.

그렇다면, `process.nextTick`은 언제 사용하는게 좋을까? 아래 코드를 살펴보자.

```javascript
function readFile(fileName, callback) {
  if (typeof fileName !== 'string') {
    return callback(new TypeError('file name should be string'))
  }

  fs.readFile(fileName, (err, data) => {
    if (err) return callback(err)

    return callback(null, data)
  })
}
```

`readFile()` 함수는 인수가 넘어오는 것에 따라서 동기도 비동기도 될 수 있다. 따라서 이는 예측하지 못한 문제가 발생할 수 있다. 그렇다면 어떻게 100% 비동기로 동작하게 할 수 있을까?

```javascript
function readFile(fileName, callback) {
  if (typeof fileName !== 'string') {
    return process.nextTick(
      callback,
      new TypeError('file name should be string'),
    )
  }

  fs.readFile(fileName, (err, data) => {
    if (err) return callback(err)

    return callback(null, data)
  })
}
```

바로 `process.nextTick`을 사용하면 된다. `filename`이 string이 아니면 `process.nextTick`을 활용하여 적절하게 콜백을 수행할 것이다. 이처럼 `process.nextTick`는 스크립트를 실행한 직후 즉시 콜백을 실행해야 하는 여러 상황에서 유용하다.
