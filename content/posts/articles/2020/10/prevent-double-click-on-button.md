---
title: '(함수형으로) 자바스크립트로 HTML 버튼 중복 클릭 방지하기'
tags:
  - javascript
published: true
date: 2020-10-22 23:04:23
description: '어렸을 때 내가 어떻게 했더라?'
---

버튼 중복 클릭을 방지하는 것은 중요하다. 물론 기본적인 중복 방지에 대한 처리는 서버에 되어 있어야 하지만, 그렇다고 마냥 프론트엔드에서 손놓고 있을 수는 없는 일이다.

주니어 풀스택 개발자일때, 중복 클릭에 대해서 프론트엔드에서 부단히도 막아보려고 노력했지만, 사용자들은 갖가지 방법으로 여러번 클릭을 했고, 그 때마다 사용자들은 더욱 더 빠른 속도로 (혹은 창의적인 방법으로) 중복클릭을 해서 괴롭히곤 헀다. 결국 어느정도 감내하고 (?) 서버에서 막는 방향으로 가긴했지만 - 그 때 마다 더 좋은 방법이 없을까 고민하곤 했다. 그 때 나왔던 다양한 중복 클릭 방지 방법을 이야기 해보고, 문제점은 무엇이었으며, 이를 해결할 수 있는 최선의 코드를, 함수형으로 써보면 어떻게 되는지 알아보장

## 방법 1) 글로벌 플래그 사용하기

```javascript
let clicked = false

function payment() {
  if (!clicked) {
    clicked = true
    window.alert('결제가 진행 중입니다.')
    // 결제 프로세스..
  }
}
```

이 방법도 물론 작동은 하겠지만 몇가지 문제가 있다.

- 전역 변수를 선언하는 것은 위험하다. 전역변수는 누구든 접근 가능하기 떄문에, 자신이 알지도 못하는 새에 값이 변경될 수 있다.
- 사용자가 다시 버튼을 누를 수 있는 상황에 대비하여 해당 변수를 다시 초기화 하는 코드를 어딘가에도 넣어야 한다.
- 외부 변수에 의존하기 때문에 테스트 하기가 곤란하다.

## 방법 2) 핸들러를 날려 버리기 (혹은 변경하기)

```javascript
function payment() {
  document.getElementById('payment').onclick = null
  window.alert('결제가 진행 중입니다.')
  // 결제 프로세스..
}
```

이 방법도 일단은 작동하겠지만 문제가 존재한다.

- DOM의 버튼과 매우 강하게 연결 되어 있는 코드 이기 때문에, 재사용이 불가능하다.
- 위 예제와 마찬가지로 다시 버튼 클릭할 수 있는 상황에 대비하여 초기화하는 코드가 필요하다.
- DOM 요소가 있어야만 테스트가 가능하다.

## 방법 3) 버튼을 disable 처리하기

```javascript
function payment() {
  document.getElementById('payment').setAttribute('disabled', 'true')
  window.alert('결제가 진행 중입니다.')
  // 결제 프로세스..
}
```

역시 위와 마찬가지 이슈가 있다.

## 방법 4) 지역 변수 사용하기

```javascript
var payment = ((clicked) => {
  return () => {
    if (!clicked) {
      clicked = true
      window.alert('결제가 진행 중입니다.')
      // 결제 프로세스..
    }
  }
})(false)
```

이번에는 즉시실행함수를 사용하였다. 이는 함수형 접근법이기도 하고, `clicked`를 외부에서 접근할 수 없기도 하다. 1번의 예제에서 전역변수로 되어 있던 것을 지역변수로 바꾸고, 스코프를 잠궈서 접근하지 못하게 했다고 생각하면 된다. 이 코드의 단점은 한번만 클릭이 필요한 모든 함수에 대해서 이와 같이 똑같이 작업을 해야한다는 것이다.

## (아마도) 최선의 방법

- 단 한번만 호출해야 하는 원래 함수는 원래 작업 (결제) 이외의 작업을 해서는 안된다.
- 원본 함수를 수정해서는 안된다.
- 원본 함수를 한번만 호출하는 새로운 함수가 필요하다.
- 기존 기능에 적용할 수 있는 일반적인 솔루션이 필요하다.

여기에 필요한 것이 바로 고차함수 (higher-order function) 이다.

```javascript
const doOnce = (fn) => {
  let done = false
  return (...args) => {
    if (!done) {
      done = true
      fn(...args)
    }
  }
}
```

```typescript
const doOnceTypescript = (fn: Function) => {
  let done = false
  return (...args: any) => {
    if (!done) {
      done = true
      fn(...args)
    }
  }
}
```

이를 적용하면 아래와 같은 모습일 것이다.

```html
<button id="payment" onclick="doOnce(payment)()">결제하기</button>
```
