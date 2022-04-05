---
title: React count down에서 배운 event-emitter 와 requestAnimationFrame
tags:
  - react
  - javascript
  - typescript
published: true
date: 2020-01-15 04:32:32
description:
  '# 리액트에서 카운트 다운을 만들며 배운 것들 리액트에서 카운트 다운을 만든다고 가정해보자. 가장 먼저 생각나는대로,
  빠르게 구현한다면 아래와 같은 느낌이 될 것이다.  https://codepen.io/yceffort/pen/BayPyNe  하지만 이
  코드는 한가지 문제를 가지고 있다.  ## setInterval, setTimeout  `setInte...'
category: react
slug: /2020/01/learning-from-react-count-down/
template: post
---

# 리액트에서 카운트 다운을 만들며 배운 것들

리액트에서 카운트 다운을 만든다고 가정해보자. 가장 먼저 생각나는대로, 빠르게 구현한다면 아래와 같은 느낌이 될 것이다.

https://codepen.io/yceffort/pen/BayPyNe

하지만 이 코드는 한가지 문제를 가지고 있다.

## setInterval, setTimeout

`setInterval`은 자바스크립트의 메인 스레드에서 실행된다. 그런데, 자바스크립트는 싱글스레드 기반으로, 동시에 할 수 있는 일은 단 한가지로 제한 되어 있다. 따라서 중간에 interruption이 있거나 모종의 이유로 처음에 선언한 시간을 정확히 지켜서 (여기서는 1000ms) 실행을 보장해주지는 않는다.

아래 코드를 사파리에서 실행해보자.

```javascript
const start = Date.now()

setInterval(() => {
  console.log(`${Date.now() - start}ms`)
}, 1000)
```

```
1001ms
2002ms
3003ms
4003ms
5003ms
6004ms
7004ms
```

자바스크립트 엔진은 오직 싱글 스레드만을 사용하므로, 비동기 이벤드들을 큐에 대기시킨다. 따라서 그 사이에 다른 이벤트 (마우스, 키보드 등) 가 발생하면 이벤트가 지연될 수 있다. 또한 지연없는 `setTimeout` 또는 `setInterval`이 5회 이상 실행될 경우, 4ms 이상의 지연시간이 강제적으로 추가된다. 그리고 이는 HTML5 표준으로 지정되어있다.

- https://developer.mozilla.org/ko/docs/Web/API/WindowTimers/setTimeout
- https://html.spec.whatwg.org/multipage/timers-and-user-prompts.html#timers

> Timers can be nested; after five such nested timers, however, the interval is forced to be at least four milliseconds.

> This API does not guarantee that timers will run exactly on schedule. Delays due to CPU load, other tasks, etc, are to be expected.

또한 CPU가 과부하 상태이거나, 브라우저 탭이 백그라운드 모드이거나, 노트북이 배터리에 의존 하는 등의 경우에도 마찬가지로 시간이 지연된다.

## 브라우저가 리페인팅하는 시간

브라우저가 화면에 무언가를 그리는 데에는 여러 단계를 거친다.

![pixel-pipeline](https://developers.google.com/web/fundamentals/performance/rendering/images/intro/frame-full.jpg?hl=ko)

그러나 이 과정에서 만약 setInterval 등이 호출되면 어떻게 될까?

![set-timeout](https://developers.google.com/web/fundamentals/performance/rendering/images/optimize-javascript-execution/settimeout.jpg?hl=ko)

setTimeout은 자바스크립트 엔진 단에서 실행되므로, 브라우저가 이를 페인팅하는 시간 따위를 신경쓰지 않고 일정 간격으로 계속해서 작동하게 된다. 이는 종종 프레임을 누락시켜 버벅거리는 현상을 사용자에게 노출 시킬 수 있다.

실제로 jquery의 기본 animate 들은 setTimeout을 통해서 사용되고 있다.

## 해결책 1) requestAnimationFrame

[참고](https://developer.mozilla.org/ko/docs/Web/API/Window/requestAnimationFrame)

`window.requestAnimationFrame`은, 브라우저에게 수행하기를 원하는 애니메이션을 알리고, 다음 리페인트가 작동하기 전에 해당 애니메이션을 업데이트 하는 함수를 호출하게 된다.

```javascript
window.requestAnimationFrame(callback)
```

화면에 새로운 애니메이션을 업데이트 할 준비가 될 때마다 호출하는 것이 좋다. 일반적으로 대부분의 브라우저에서는 디스플레이 주사율에 맞춰 콜백을 호출한다. 성능을 위해서, 백그라운드 탭, hidden, iframe 등에서는 실행 되지 않는다.

이 함수는 0이 아닌 고유한 요청 id를 리턴하는데, window.cancelAnimationFrame(requestId)로 해당 요청을 취소할 수 있다.

## 해결책 2) EventEmitter 사용

https://github.com/Gozala/events는 Node.js의 [events](https://nodejs.org/api/events.html)를 브라우저와 같은 다양한 환경에서 사용할 수 있도록 만들어주는 패키지다.

Nodejs는 이벤트를 처리하기 위해서 EventEmitter를 사용한다. 일종의 옵저버 패턴으로, 이벤트를 대기하는 이벤트 리스너들이 (옵저버들이) 이벤트를 기다리다가, 해당 이벤트가 실행되면 이를 처리하는 함수가 실행된다.

이것과 `requestAnimationFrame` 을 적절하게 이용한다면, 보다 나은 카운트다운 컴포넌트를 만들 수 있을 것이다.

```typescript
import EventEmitter from 'events'

class Timer extends EventEmitter {
  // 최초 시작시간
  private time: number = -1
  // requestAnimationFrame의 ID
  private timerId: number = -1

  // Timer를 생성할 때 몇초를 셀지 받는다.
  constructor(private duration: number) {
    super()
  }

  // 타이머가 시작하면, requestAnimationFrame와 함께 step함수를 호출한다.
  start() {
    this.timerId = requestAnimationFrame(this.step)
    return this.timerId
  }

  // 타이머가 끝나면, cancelAnimationFrame를 호출하여 repaint를 막고
  // 타이머를 다시 초기화 시킨다.
  stop() {
    cancelAnimationFrame(this.timerId)
    this.timerId = -1
  }

  // 현재 시간을 받는다.
  private step = (timestamp: number) => {
    // time이 -1이라면 === 맨처음 생성되었다면
    // 받은 시간을 timer의 시간으로 갱신하한다.
    if (this.time === -1) {
      this.time = timestamp
    }

    // 현재시간과 타이머 내장시간의 차이
    const progress = timestamp - this.time

    // progress의 차이가 처음에 받는 시간의 차이보다 크다면
    if (progress < this.duration) {
      // progress를 인자로 하는 progress 이벤트를 시작한다.
      this.emit('progress', progress)
      // 그리고 이는 브라우저의 리페인팅 (== 카운트 다운 갱신)이 필요하므로,
      // requestAnimationFrame를 호출한다.
      this.timerId = requestAnimationFrame(this.step)
    } else {
      // 카운트 다운이 종료되었다면 stop을 호출하고 이벤트를 끝낸다.
      this.stop()
      this.emit('finish')
    }
  }
}
```

```typescript
const [countDown, setCountDown] = useState(duration)

useEffect(() => {
  // 타이머 선언
  const timer = new Timer(duration)

  // progress 이벤트를 정의한다. 받은 시간만큼, 현재 카운트 다운 시간에서 제외한다.
  timer.on('progress', (elapsed: number) => {
    setCountDown(duration - elapsed)
  })

  // finish 이벤트를 정의한다.
  timer.on('finish', onFinished)

  timer.start()

  // useEffect가 끝날때마다 timer를 멈춘다.
  return () => {
    timer.stop()
  }
}, [duration]) // 시간이 변경될 때마다 이 함수를 다시 호출한다.
```

## 구현

https://codesandbox.io/s/react-countdown-ghe1j

## 참고자료

https://developer.mozilla.org/ko/docs/Web/API/Window/requestAnimationFrame

https://developers.google.com/web/fundamentals/performance/rendering?hl=ko

https://developers.google.com/web/fundamentals/performance/rendering/optimize-javascript-execution?hl=ko
