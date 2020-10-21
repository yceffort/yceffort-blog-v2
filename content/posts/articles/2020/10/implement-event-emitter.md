---
title: 'EventEmitter 구현해보기'
tags:
  - nodejs
  - javascript
published: true
date: 2020-10-21 19:30:12
description: '면접 때 잘 대답 못했던 질문 22222'
---

가령 아래와 같은

```javascript
const button = document.querySelector("button");
button.addEventListener("click", (event) => /* do something with the event */)
```

이 코드에서, 버튼 클릭에 대해서 리스너를 달았다. 이 뜻은 이벤트가 발생하는 것 (emitted를 발생하다라고 의역했다.) 에 대해서 구독을 했다는 것이고, 그러한 이벤트가 발생할 경우 콜백을 실행하겠다는 것을 의미한다. 버튼을 클릭할 떄 마다, 이벤트가 발생하게 되고 해당 이벤트와 함께 콜백이 실행된다.

기존 코드베이스에서 작업 할 때, 커스텀 이벤트를 발생시키고 싶을 떄가 있다. 버튼 클릭과 같은 DOM 이벤트 말고, 다른 트리거를 기반으로 이벤트를 내보내고 응답하도록 한다고 가정해보자. 이를 위해서는 커스텀 EventEmitter가 필요하다.

EventEmitter란 정의된 이벤트를 수신하고, 콜백을 실행한다음, 값과 함꼐 해당 이벤트를 내보내는 패턴이다. 이를 `pub, sub` 모델, 혹은 리스너라고 부르기도 한다.

```javascript
let n = 0
const event = new EventEmitter()
event.subscribe('THUNDER_ON_THE_MOUNTAIN', (value) => (n = value))
event.emit('THUNDER_ON_THE_MOUNTAIN', 18)
// n: 18
event.emit('THUNDER_ON_THE_MOUNTAIN', 5)
// n: 5
```

위 예제에서는 `THUNDER_ON_THE_MOUNTAIN`라고불리는 이벤트를 구독하였고, 이 이벤트가 발생할 때 마다 콜백 `(value) => (n = value)`를 실행하였다.이 이벤트를 실행하기 위하여, `.emit()`을 사용하였다.

이는 비동기 코드로 작업 할 때 유용하다. 그리고 현재 모듈과 다른 위치에서 값을 업데이트 해야 한다. (값을 업데이트 할 수 있는 스코프가 되어야 한다.)

이에 대한 적절한 예가 Redux다. Redux는 내부 저장소가 업데이트 된 것을 외부에 공유해 줄 수 있는 방법이 필요하다. 그래야 리액트가 해당 값이 변경되었다는 것을 인지하고 `setState()`를 호출하여 다시 렌더링을 할 수 있다. 이러한 일련의 과정이 `EventEmitter`를 통해 이루어진다. Redux에는 subscribe 함수가 존재하며, 이는 새로운 값을 받을 때 마다 실행되는 콜백을 함수로 받는다. 이를 Redux `<Provider />` 컴포넌트라고 하며, 새로운 값을 받을 때 마다 `setState()`를 호출한다.

## Implementation

위 EventEmitter는 Nodejs에만 존재한다. 한번 실재로 이를 구현해보도록 하자.

```typescript
class EventEmitter {
  public events: Events
  constructor(events?: Events) {
    this.events = events || {}
  }
}
```

### Events

Events 인터페이스를 정의해보자.

```typescript
interface Events {
  [key: string]: Function[]
}

/**
{
  "event": [fn],
  "event_two": [fn]
}
*/
```

### Subscribe

먼저 정의된 이벤트를 구독할 방법이 필요하다.

```typescript
event.subscribe('named event', (value) => value)
```

두개의 파라미터 (이벤트 명, 콜백)을 받을 수 있도록 구현해보자.

```typescript
class EventEmitter {
  public events: Events
  constructor(events?: Events) {
    this.events = events || {}
  }

  public subscribe(name: string, cb: Function) {
    ;(this.events[name] || (this.events[name] = [])).push(cb)
  }
}
```

### Emit

다음으로, `emit`을 활용해서 이벤트를 발생시키는 코드를 구현해야 한다. 파라미터는 몇개가 있을지 모르므로, 유연하게 대처해야 한다.

```typescript
class EventEmitter {
  public events: Events
  constructor(events?: Events) {
    this.events = events || {}
  }

  public subscribe(name: string, cb: Function) {
    ;(this.events[name] || (this.events[name] = [])).push(cb)
  }

  public emit(name: string, ...args: any[]): void {
    ;(this.events[name] || []).forEach((fn) => fn(...args))
  }
}
```

### Unsubscribing

이제는 이벤트 구독을 해제 해보자.

```javascript
subscribe(name: string, cb: Function) {
  (this.events[name] || (this.events[name] = [])).push(cb);

  return {
    unsubscribe: () =>
      this.events[name] && this.events[name].splice(this.events[name].indexOf(cb) >>> 0, 1)
  };
}
```

이제 `subscribe`에서 `unsubscribe`를 리턴한다. 화살표 함수를 활용하여 부모 스코프와 같은 스코프를 사용할 수 있도록 했다. 이 함수에서, 부모에게 전달한 콜백의 인덱스를 찾고자 bitwise operator (`>>>`) 를 사용했다.

이제 아래와 같이 이벤트 구독을 해제 할 수 있다.

```javascript
const subscription = event.subscribe('event', (value) => value)

subscription.unsubscribe()
```

## 결론

```typescript
interface Events {
  [key: string]: Function[]
}

export class EventEmitter {
  public events: Events
  constructor(events?: Events) {
    this.events = events || {}
  }

  public subscribe(name: string, cb: Function) {
    ;(this.events[name] || (this.events[name] = [])).push(cb)

    return {
      unsubscribe: () =>
        this.events[name] &&
        this.events[name].splice(this.events[name].indexOf(cb) >>> 0, 1),
    }
  }

  public emit(name: string, ...args: any[]): void {
    ;(this.events[name] || []).forEach((fn) => fn(...args))
  }
}
```

출처: https://css-tricks.com/understanding-event-emitters/
