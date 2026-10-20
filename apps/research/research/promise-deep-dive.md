---
title: 'Promise 딥다이브: 동작 원리와 실행 순서'
marp: true
paginate: true
theme: midnight
tags:
  - javascript
  - async
  - error-handling
date: 2026-08-13
description: '상태 머신, 마이크로태스크, 체이닝, 조합기, 에러 처리 — Promise의 동작 원리 정리와 실행 순서 퀴즈'
published: true
art:
  undraw: completion-progress
---

# Promise 딥다이브

동작 원리와 실행 순서

<!-- _class: invert deck-cover -->

@yceffort

---

## 이 덱의 목표

```js
const user = fetchUser()
console.log(user) // Promise { <pending> }
console.log(user.name) // undefined
```

- Promise는 값이 아니라 **미래의 값을 감싼 객체**다
- 동작 원리를 모르면 실행 순서를 예측할 수 없고, `await`를 붙여 해결해도 이유를 설명할 수 없다

목표: **어떤 Promise 코드든 실행 순서를 예측할 수 있게 되는 것**

---

## 이 덱에서 다루는 것

다섯 부분으로 나눠 다룬다.

1. **배경** — Promise가 해결하려던 문제
2. **상태 머신** — pending, fulfilled, rejected와 "한 번 정해지면 불변"
3. **마이크로태스크** — then 콜백은 정확히 _언제_ 실행되는가
4. **체이닝** — then이 매번 새 Promise를 만든다는 것의 의미
5. **조합기와 실패** — all/race/any, 에러 전파, 실무 패턴

마지막에 **종합 퀴즈 12문제**로 전체를 점검한다.

---

## 기준

- 동작 설명은 **ECMA-262 스펙**과 **HTML 이벤트 루프 명세** 기준
- 모든 실행 순서 예제는 **Node.js 24 / Chrome 139**에서 직접 실행해 확인했다
- 다루지 않는 것: RxJS 같은 스트림 추상화, 제너레이터 기반 코루틴, Web Worker

<!-- 구두 보충: 예제가 "브라우저마다 다르지 않냐"는 질문이 자주 나오는데, Promise의 실행 순서는 스펙에 정의되어 있어 모던 런타임에서는 전부 동일하다. 다른 건 대부분 태스크 큐(타이머) 쪽 구현 차이. -->

---

## Part 0 — 배경: 콜백 방식의 한계

<!-- _class: invert -->

---

## 콜백 방식

```js
getUser(userId, function (err, user) {
  if (err) return handleError(err)
  getOrders(user.id, function (err, orders) {
    if (err) return handleError(err)
    getShipping(orders[0].id, function (err, shipping) {
      if (err) return handleError(err)
      render(user, orders, shipping)
    })
  })
})
```

흔히 "콜백 지옥"이라 부르지만, 들여쓰기는 사실 부차적인 문제다.

---

## 진짜 문제: 제어의 역전

```js
analytics.trackPurchase(order, function () {
  chargeCreditCard(order) // 결제 함수를 콜백으로 넘겼다
})
```

내 함수(`chargeCreditCard`)의 실행 권한을 **남의 라이브러리에 통째로 넘겼다.**

- 콜백을 한 번도 부르지 않으면 결제가 되지 않는다
- 콜백을 두 번 부르면 결제가 두 번 된다
- 에러 전달 방식, 동기/비동기 호출 여부가 전부 **라이브러리 구현에 달려 있다**

콜백 방식에는 이런 동작을 강제할 규약이 없다.

---

## Promise가 보장하는 것

Promise는 "미래의 값"을 담는 객체이면서, 다음 네 가지를 보장한다.

| 콜백 방식의 문제           | Promise의 보장                              |
| -------------------------- | ------------------------------------------- |
| 안 부르거나 여러 번 부름   | 결과는 **정확히 한 번**만 정해진다          |
| 성공/실패 전달 방식 제각각 | `then`/`catch`라는 **표준 인터페이스**      |
| 동기/비동기 뒤섞임         | 콜백은 **항상 비동기로** 호출된다           |
| 이미 끝난 뒤 등록하면 유실 | 언제 등록해도 결과를 받는다 (**시점 무관**) |

이후 파트에서 이 보장들이 어떻게 구현되는지 다룬다.

---

## Part 1 — 상태 머신으로서의 Promise

<!-- _class: invert -->

---

## 세 가지 상태

```text
              resolve(value)
   pending ──────────────────▶ fulfilled (값과 함께)
      │
      │       reject(reason)
      └──────────────────────▶ rejected (이유와 함께)
```

- 시작은 항상 **pending**
- fulfilled 또는 rejected가 되는 것을 **settled**(확정)라고 부른다
- 핵심 규칙: **한 번 settled되면 절대 다시 바뀌지 않는다**

내부적으로 Promise 객체는 상태(`[[PromiseState]]`)와 결과(`[[PromiseResult]]`)를 숨겨진 슬롯에 들고 있다. 외부에서 직접 읽거나 바꿀 방법은 없고, 오직 `resolve`/`reject` 호출과 `then` 등록만 가능하다.

---

## executor는 즉시, 동기로 실행된다

```js
console.log('1')

const p = new Promise((resolve) => {
  console.log('2') // new 하는 순간 바로 실행
  resolve('완료')
})

console.log('3')
```

```text
1 → 2 → 3
```

- `new Promise(executor)`의 executor는 **그 자리에서 동기 실행**된다
- "Promise를 만들면 비동기가 된다"는 흔한 오해 — 비동기인 건 executor가 아니라 **then 콜백 쪽**이다

---

## resolve는 한 번만 유효하다

```js
const p = new Promise((resolve, reject) => {
  resolve('첫 번째')
  resolve('두 번째') // 무시
  reject(new Error('에러!')) // 이것도 무시
})

p.then((v) => console.log(v)) // "첫 번째"
```

- 최초의 `resolve`/`reject` 호출만 유효하고, 이후 호출은 **조용히 무시**된다
- 에러도 안 난다. 그냥 아무 일도 일어나지 않는다
- 이것이 "콜백을 두 번 부르면?" 문제의 해답 — **구조적으로 두 번 확정될 수 없다**

---

## resolve ≠ fulfilled

미묘하지만 중요한 구분. `resolve`에 무엇을 넘기느냐에 따라 결과가 다르다.

```js
new Promise((resolve) => resolve(42))
// → 42로 fulfilled

new Promise((resolve) => resolve(Promise.reject(new Error('실패'))))
// → rejected! fulfilled가 아니다
```

- `resolve(값)` → 그 값으로 fulfilled
- `resolve(다른 Promise)` → 그 Promise를 **따라간다**. 그 Promise가 실패하면 함께 rejected
- 그래서 이름이 "fulfill"이 아니라 "resolve"(결정)다 — **어느 쪽으로 갈지 위임할 수도 있다**

이 "따라가기"는 Part 3에서 다시 다룬다.

---

## then 콜백은 항상 비동기다

```js
const p = Promise.resolve('이미 완료된 Promise')

p.then((v) => console.log('A:', v))
console.log('B')
```

```text
B
A: 이미 완료된 Promise
```

- 이미 fulfilled인 Promise에 `then`을 붙여도 콜백은 **지금 당장 실행되지 않는다**
- 항상 "현재 실행 중인 코드가 전부 끝난 뒤"에 실행된다
- 동기/비동기가 상황에 따라 뒤섞이면 코드의 실행 순서를 예측할 수 없기 때문에, 스펙이 **항상 비동기**로 못박았다

그럼 "현재 코드가 끝난 뒤"란 정확히 언제인가? → Part 2

---

## Part 1 정리

- executor는 `new` 하는 순간 **동기 실행**
- 상태는 pending → fulfilled/rejected 단방향, **한 번 확정되면 불변**
- 두 번째 `resolve`/`reject`는 조용히 무시
- `resolve(Promise)`는 fulfilled가 아니라 **그 Promise를 따라간다**
- `then` 콜백은 이미 끝난 Promise라도 **항상 비동기로** 호출

---

## Part 2 — 마이크로태스크: then은 언제 실행되는가

<!-- _class: invert -->

---

## 이벤트 루프 30초 복습

JavaScript는 싱글 스레드. 한 번에 하나의 코드만 실행한다.

```text
[ 콜 스택 ]  ← 지금 실행 중인 코드
     ▲
     │  스택이 비면 큐에서 하나 꺼내 실행
     │
[ 태스크 큐 (매크로태스크) ]  ← setTimeout, 클릭 이벤트, ...
```

여기까지가 흔히 아는 그림. 그런데 Promise 콜백은 **이 태스크 큐로 가지 않는다.**

---

## 마이크로태스크 큐

Promise 콜백을 위한 **별도의 우선순위 큐**가 있다.

```text
[ 콜 스택 ]
     ▲
     │  ① 스택이 비면 마이크로태스크 큐를 먼저, 전부 비운다
     │
[ 마이크로태스크 큐 ]  ← Promise 콜백, queueMicrotask
     ▲
     │  ② 그 다음에야 태스크를 하나 꺼낸다
     │
[ 태스크 큐 ]  ← setTimeout, 이벤트, ...
```

- 태스크는 **한 번에 하나**씩, 마이크로태스크는 **큐가 빌 때까지 전부**
- 실행 도중 새로 추가된 마이크로태스크도 그 자리에서 이어서 처리된다

---

## 퍼즐 1 — 기본기

```js
console.log('A')

setTimeout(() => console.log('B'), 0)

Promise.resolve().then(() => console.log('C'))

console.log('D')
```

출력 순서는?

<!-- 손들기로 진행: A B C D / A D B C / A D C B. 대부분 setTimeout 0이면 바로 실행된다고 생각하는 지점을 짚는다. -->

---

## 퍼즐 1 — 풀이

```text
A D C B
```

| 단계 | 실행                     | 마이크로태스크 큐 | 태스크 큐 |
| ---- | ------------------------ | ----------------- | --------- |
| 1    | `A` 출력                 |                   |           |
| 2    | setTimeout 등록          |                   | `B`       |
| 3    | then 등록                | `C`               | `B`       |
| 4    | `D` 출력, 스택 비움      | `C`               | `B`       |
| 5    | 마이크로태스크 먼저: `C` |                   | `B`       |
| 6    | 그 다음 태스크: `B`      |                   |           |

> `setTimeout(fn, 0)`의 0은 "즉시"가 아니라 "**현재 코드와 모든 마이크로태스크가 끝난 뒤** 최대한 빨리"다.

---

## 퍼즐 2 — 체인은 한 칸씩

```js
Promise.resolve()
  .then(() => console.log('A'))
  .then(() => console.log('B'))

Promise.resolve()
  .then(() => console.log('1'))
  .then(() => console.log('2'))
```

출력 순서는?

---

## 퍼즐 2 — 풀이

```text
A 1 B 2
```

- `A B 1 2`가 아니다. 체인이 통째로 실행되는 게 아니라, **then 하나가 마이크로태스크 하나**다
- `A`가 실행되어야 그 다음 `.then(B)`이 큐에 들어간다. 그 사이에 이미 대기 중이던 `1`이 먼저 실행된다

```text
큐의 변화: [A, 1] → [1, B] → [B, 2] → [2]
```

> 두 체인이 **한 칸씩 번갈아** 전진한다.

---

## await의 정체

`async/await`는 새로운 메커니즘이 아니라 **then의 문법 설탕**이다.

```js
async function main() {
  console.log('1')
  await someAsyncWork()
  console.log('2') // ← 여기부터 아래 전부가...
}

// ...대략 이렇게 변환된다
function main() {
  console.log('1')
  return someAsyncWork().then(() => {
    console.log('2') // then 콜백이 된다
  })
}
```

`await`에서 멈추는 것은 **이 async 함수 하나뿐**이다. 제어는 즉시 호출부로 돌아가 바깥 코드가 계속 실행되고, 함수의 나머지는 마이크로태스크로 예약된다. **JavaScript 엔진이 멈춰서 기다리는 것이 아니다.**

---

## 퍼즐 3 — await와 실행 순서

```js
async function main() {
  console.log('1')
  await null // Promise가 아니어도 await 가능
  console.log('2')
}

main()
console.log('3')
```

출력 순서는?

---

## 퍼즐 3 — 풀이

```text
1 3 2
```

- `main()` 호출 → `1` 출력 (async 함수도 **await 전까지는 동기 실행**)
- `await null` → `null`을 `Promise.resolve(null)`로 감싸고, 나머지(`2` 출력)를 마이크로태스크로 예약
- 함수 밖으로 나와 `3` 출력
- 스택이 비면 마이크로태스크 실행 → `2`

> "async 함수는 전부 비동기"가 아니다. **await를 만나기 전까지는 일반 함수와 똑같이 동기**로 돈다.

---

## 마이크로태스크의 함정: 굶주림

마이크로태스크는 "빌 때까지 전부" 처리되기 때문에, 스스로를 계속 추가하면 **태스크 큐와 렌더링이 영원히 실행되지 못한다.**

```js
function loop() {
  Promise.resolve().then(loop) // 마이크로태스크가 마이크로태스크를 낳는다
}
loop()

setTimeout(() => console.log('실행되지 않는다'), 0)
```

- `setTimeout`이라면 반복 사이사이에 렌더링과 다른 태스크가 끼어들 수 있지만, 마이크로태스크 재귀는 **화면을 완전히 얼린다**
- 실무에서 만나는 형태: 재귀적으로 자신을 다시 예약하는 잘못된 폴링, 상태 갱신 루프

---

## Part 2 정리

- Promise 콜백은 태스크 큐가 아닌 **마이크로태스크 큐**로 간다
- 스택이 비면: **마이크로태스크 전부** → 태스크 **하나** → 반복
- `then` 하나 = 마이크로태스크 하나. 체인은 **한 칸씩** 전진한다
- `await` = 나머지 코드를 then 콜백으로 바꾸는 문법 설탕. **await 전까지는 동기**, 멈추는 것은 **그 함수 하나뿐**
- 마이크로태스크 재귀는 렌더링을 얼린다

---

## Part 3 — 체이닝의 실체

<!-- _class: invert -->

---

## then은 매번 새 Promise를 만든다

```js
const p1 = fetch('/api/user')
const p2 = p1.then((res) => res.json())
const p3 = p2.then((user) => user.name)

console.log(p1 === p2) // false
console.log(p2 === p3) // false
```

- `then`은 기존 Promise를 변형하는 게 아니라 **새 Promise를 만들어 반환**한다
- 새 Promise의 운명은 **콜백이 무엇을 반환하느냐**로 결정된다
- "체이닝"이란 이 새 Promise들을 계속 이어받는 것

그렇다면 콜백의 반환값에 따라 무슨 일이 벌어지는가?

---

## 콜백 반환값의 네 갈래

새 Promise의 상태는 콜백이 **무엇을 반환하느냐**로 정해진다. 경우는 네 가지뿐이다.

<!-- prettier-ignore -->
```js
const p = Promise.resolve(1)

p.then((v) => v + 1) //             ① 값 반환      → 2로 fulfilled
p.then((v) => {}) //                ② 반환 없음    → undefined로 fulfilled
p.then((v) => { throw new Error('x') }) // ③ throw → 그 에러로 rejected
p.then((v) => fetchUser()) //       ④ Promise 반환 → 그 Promise를 따라간다
```

- ①②: 다음 `then`의 성공 콜백이 그 값을 받는다
- ③: 프로그램이 죽는 게 아니라 **다음 Promise가 rejected가 된다** — 에러가 체인을 타고 catch까지 흐르는 이유
- ④: 그 Promise가 끝날 때까지 기다렸다가 결과를 그대로 이어받는다

---

## 흔한 실수 1 — return을 빼먹는다

```js
fetchUser()
  .then((user) => {
    fetchOrders(user.id) // return이 없다!
  })
  .then((orders) => {
    console.log(orders) // undefined
  })
```

- 콜백이 아무것도 반환하지 않았으므로 다음 Promise는 `undefined`로 fulfilled
- `fetchOrders`는 실행은 되지만 결과가 **유실된다**. 실패해도 catch에 잡히지 않는다

```js
  .then((user) => {
    return fetchOrders(user.id) // Promise를 반환해야 "따라간다"
  })
```

---

## 흔한 실수 2 — then 중첩

```js
// 콜백 중첩과 같은 구조
fetchUser().then((user) => {
  fetchOrders(user.id).then((orders) => {
    fetchShipping(orders[0].id).then((shipping) => { ... })
  })
})
```

Promise를 반환하면 "따라간다"는 성질 덕분에 **평평하게 펼 수 있다**:

```js
fetchUser()
  .then((user) => fetchOrders(user.id))
  .then((orders) => fetchShipping(orders[0].id))
  .then((shipping) => { ... })
  .catch(handleError) // 세 단계 어디서 실패해도 여기로
```

중첩 버전은 에러 처리도 단계마다 따로 해야 한다. 중첩된 then은 대부분 평평한 체인으로 바꿀 수 있다.

---

## catch와 finally는 then의 변형이다

```js
p.catch(onError)
// = p.then(undefined, onError)

p.finally(onDone)
// ≈ 성공/실패 양쪽에 onDone을 끼워넣되, 결과는 그대로 통과
```

- `catch`도 **새 Promise를 반환**한다. 그래서 catch 뒤에 then을 또 붙일 수 있다
- `finally` 콜백은 인자를 받지 않고, 반환값도 (throw만 아니면) **결과에 영향을 주지 않는다**

```js
Promise.resolve(42)
  .finally(() => '이 값은 무시된다')
  .then((v) => console.log(v)) // 42
```

---

## catch의 위치가 의미를 바꾼다

```js
// A: 요청 실패 시 → parse는 실행 안 됨, 최종 fallback 사용
fetchData()
  .then(parse)
  .catch(() => DEFAULT)

// B: 요청 실패 시 → 먼저 DEFAULT로 복구되고, parse(DEFAULT)가 실행됨
fetchData()
  .catch(() => DEFAULT)
  .then(parse)
```

- catch는 "이전 단계까지의 실패"를 잡고, **fulfilled 상태의 새 Promise를 반환**한다 (복구)
- 즉 catch 뒤의 체인은 **아무 일 없었다는 듯 계속 진행**된다
- "어디까지의 실패를, 어느 시점에 복구할 것인가"가 catch 위치의 의미

---

## 더 깊이 — return Promise의 숨은 비용

```js
Promise.resolve()
  .then(() => {
    console.log('1')
    return Promise.resolve('2') // 그냥 '2'를 return하면 결과가 달라진다
  })
  .then((v) => console.log(v))

Promise.resolve()
  .then(() => console.log('3'))
  .then(() => console.log('4'))
  .then(() => console.log('5'))
```

```text
1 3 4 5 2   ← '2'가 '4' 다음이 아니라 '5' 다음에 나온다
```

콜백이 **Promise를 반환하면** "따라가기" 절차(then 호출 예약 + 결과 전달)에 마이크로태스크 **2틱**이 추가로 든다. `return '2'`였다면 `1 3 2 4 5`.

<!-- 구두 보충: 이걸 외울 필요는 없다는 것을 강조. "Promise를 반환하면 즉시 이어지는 게 아니라 흡수 절차를 거친다" 정도만 가져가면 된다. async 함수의 return await도 같은 원리로 틱이 추가된다. -->

---

## Part 3 정리

- `then`/`catch`/`finally`는 모두 **새 Promise를 반환**한다
- 다음 Promise의 운명 = 콜백의 반환값 (값 / throw / Promise 따라가기)
- **return을 빼먹으면** 결과와 에러가 모두 유실된다
- 중첩 대신 **평평한 체인** — 에러 처리가 한 곳으로 모인다
- catch는 실패를 잡고 **복구**한다. 위치가 곧 복구 범위

---

## Part 4 — 조합기: 여러 Promise 다루기

<!-- _class: invert -->

---

## 네 가지 조합기

여러 Promise를 하나로 묶는 정적 메서드 4종. **성공/실패 판정 기준**이 다르다.

| 메서드               | 성공 조건          | 실패 조건      | 결과                          |
| -------------------- | ------------------ | -------------- | ----------------------------- |
| `Promise.all`        | **전부** 성공      | 하나라도 실패  | 성공값 배열 (순서 보장)       |
| `Promise.allSettled` | 항상 성공          | 없음           | `{status, value/reason}` 배열 |
| `Promise.race`       | **첫 확정**이 성공 | 첫 확정이 실패 | 가장 빠른 것의 결과           |
| `Promise.any`        | 하나라도 성공      | **전부** 실패  | 첫 성공값                     |

---

## Promise.all — 전부 아니면 무효

```js
const [user, orders, config] = await Promise.all([
  fetchUser(),
  fetchOrders(),
  fetchConfig(),
])
```

- 결과 배열의 순서는 **완료 순서가 아니라 입력 순서**
- 하나라도 실패하면 **그 즉시** 전체가 rejected (fail-fast)
- 주의: 실패해도 **나머지 Promise가 취소되는 건 아니다**. 결과만 버려질 뿐, 요청은 계속 돈다

```js
// 실패를 허용하려면 allSettled
const results = await Promise.allSettled([...])
const ok = results.filter((r) => r.status === 'fulfilled')
```

---

## race와 any — 헷갈리는 한 끗

```js
// race: 성공이든 실패든 "가장 빠른 결과"를 따라간다
Promise.race([fetchData(), timeout(5000)])
// → 5초 안에 응답 없으면 timeout의 reject가 이긴다 (타임아웃 패턴)

// any: "가장 빠른 성공"을 기다린다. 실패는 무시
Promise.any([cdnA.fetch(), cdnB.fetch(), cdnC.fetch()])
// → 셋 중 하나만 성공하면 OK (미러/폴백 패턴)
```

- `race`는 첫 **확정**(settled), `any`는 첫 **성공**(fulfilled)
- `any`는 전부 실패했을 때만 rejected — 모든 실패 이유를 담은 `AggregateError`를 던진다

---

## 순차 vs 병렬 — await의 대표적 함정

```js
// 순차: 앞이 끝나야 뒤가 시작. 총 2초
const user = await fetchUser() // 1초
const config = await fetchConfig() // 1초, user와 무관한데도 기다림

// 병렬: 두 요청을 먼저 시작해 두고 기다린다. 총 1초
const [user2, config2] = await Promise.all([fetchUser(), fetchConfig()])
```

- `await`는 그 async 함수의 진행을 그 줄에서 멈춘다. 의존성 없는 작업을 await로 줄줄이 쓰면 **불필요한 직렬화**
- 반대로 뒤 작업이 앞 결과를 **필요로 하면** 순차가 정답이다
- 판단 기준: **"이 요청이 앞 요청의 결과를 쓰는가?"**

---

## 실무 패턴 — 동시성 제한

대량 작업을 전부 동시에 시작하지 않고, "동시에 최대 N개"를 유지하는 풀 패턴:

<!-- prettier-ignore -->
```js
async function pool(tasks, limit) {
  const results = []
  const running = new Set()
  for (const task of tasks) {
    const p = task().finally(() => running.delete(p))
    running.add(p)
    results.push(p)
    if (running.size >= limit) await Promise.race(running) // 자리가 날 때까지
  }
  return Promise.all(results)
}

await pool(urls.map((url) => () => upload(url)), 5) // 동시 5개
```

`race`로 "가장 먼저 끝나는 것"을 기다렸다가 다음 작업을 투입한다. p-limit 같은 라이브러리의 핵심이 이 구조다.

---

## Part 4 정리

- `all` 전부 / `allSettled` 무조건 / `race` 첫 확정 / `any` 첫 성공
- `all`의 결과 순서는 **입력 순서**, 실패는 fail-fast (단, 취소는 아님)
- 의존성 없는 작업을 await로 직렬화하지 말 것 — **먼저 시작해 두고 기다리기**
- 대량 작업은 `race`를 활용한 **동시성 제한 풀**

---

## Part 5 — 에러 처리

<!-- _class: invert -->

---

## 에러는 체인을 타고 흐른다

```js
fetchUser()
  .then((user) => fetchOrders(user.id)) // ① 여기서 실패하면
  .then((orders) => render(orders)) // ② 건너뛰고
  .then(() => logSuccess()) // ③ 건너뛰고
  .catch((err) => showError(err)) // ④ 여기서 잡힌다
```

- rejected 상태는 **성공 콜백들을 건너뛰며** 아래로 전파된다
- 가장 가까운 catch(또는 then의 두 번째 인자)가 잡을 때까지
- try/catch의 비동기 버전이라고 생각하면 정확하다 — 실제로 `async/await`에서는 진짜 `try/catch`로 잡는다

```js
try {
  const orders = await fetchOrders(user.id)
} catch (err) {
  showError(err) // 같은 동작
}
```

---

## 아무도 안 잡으면: unhandled rejection

```js
Promise.reject(new Error('fail'))
// 브라우저 콘솔: Uncaught (in promise) Error: fail
```

- catch 없는 rejection은 **unhandledrejection** 이벤트를 발생시킨다
- 판정 시점: rejection 발생 직후가 아니라, **마이크로태스크 큐를 비운 뒤에도 핸들러가 없을 때**
- 그래서 같은 흐름 안에서 catch를 붙이면 경고가 없다:

```js
const p = Promise.reject(new Error('x'))
p.catch(handleError) // 같은 틱에 붙였으므로 경고 없음
```

- Node.js에서는 처리되지 않은 rejection이 **프로세스를 종료**시킨다 (v15+ 기본값). "나중에 잡을 거니까"는 통하지 않는다

<!-- 구두 보충: window.addEventListener('unhandledrejection', ...)로 전역 로깅을 걸 수 있고, Sentry 같은 도구가 기본으로 하는 일이 바로 이것. -->

---

## async 함수의 rejection 유실

```js
// 이벤트 핸들러, useEffect 등에서 흔한 패턴
button.addEventListener('click', async () => {
  await submitOrder() // 실패하면? 아무도 모른다
})
```

- async 함수는 Promise를 반환하지만, 이 반환값을 **아무도 받지 않는다**
- `submitOrder`가 실패하면 그대로 unhandled rejection
- 호출부가 결과를 받지 않는 async 함수는 **내부에서 스스로 에러를 처리**해야 한다

```js
button.addEventListener('click', async () => {
  try {
    await submitOrder()
  } catch (err) {
    showToast('주문에 실패했습니다')
  }
})
```

---

## 취소 — AbortController

Promise에는 **취소 기능이 없다.** 한 번 시작된 작업은 스스로 멈추지 않는다.
취소가 필요하면 취소 신호를 별도 채널로 전달하는 `AbortController`를 쓴다:

```js
const controller = new AbortController()

fetch('/api/search?q=keyword', {signal: controller.signal})
  .then((res) => res.json())
  .catch((err) => {
    if (err.name === 'AbortError') return // 취소는 에러가 아니다
    throw err
  })

// 사용자가 검색어를 바꾸면
controller.abort()
```

- 검색 자동완성처럼 **이전 요청이 무의미해지는** 상황의 표준 패턴
- 취소하지 않으면 다음 슬라이드의 race condition이 발생할 수 있다

---

## race condition — 응답 순서 역전

```js
input.addEventListener('input', async (e) => {
  const results = await search(e.target.value)
  render(results) // 문제의 줄
})
```

"ab" 검색 직후 "abc"를 검색했는데 **"ab"의 응답이 더 늦게 도착하면?** 화면에는 "ab"의 결과가 남는다.

```js
let latest = 0
input.addEventListener('input', async (e) => {
  const id = ++latest
  const results = await search(e.target.value)
  if (id !== latest) return // 최신 요청이 아니면 버린다
  render(results)
})
```

요청마다 번호표를 붙이고 **응답이 왔을 때 아직 최신인지** 확인한다. AbortController로 이전 요청을 끊는 방식과 병행하면 더 좋다.

---

## Part 5 정리

- rejection은 성공 콜백을 건너뛰며 **가장 가까운 catch까지 전파**
- catch 없는 rejection = unhandled rejection. **Node에서는 프로세스가 죽는다**
- 반환값을 아무도 안 받는 async 함수는 **내부에서 try/catch**
- Promise는 취소가 없다 — **AbortController**로 신호를 보낸다
- 연속 요청은 **응답 시점에 최신 여부를 검사** (번호표 or abort)

---

## 종합 퀴즈

<!-- _class: invert -->

열두 문제. 문제마다 대응하는 Part를 표시했다.

---

## 퀴즈 1 — 실행 순서 (Part 2)

```js
console.log('start')

setTimeout(() => console.log('timeout'), 0)

Promise.resolve()
  .then(() => console.log('then1'))
  .then(() => console.log('then2'))

console.log('end')
```

출력 순서는?

---

## 퀴즈 1 — 정답

```text
start → end → then1 → then2 → timeout
```

- 동기 코드 먼저: `start`, `end`
- 스택이 비면 **마이크로태스크 전부**: `then1`, 그로 인해 예약된 `then2`까지
- 마지막에 태스크: `timeout`

> 핵심: 마이크로태스크는 "전부", 태스크는 "하나씩". `then2`가 `timeout`보다 먼저다.

---

## 퀴즈 2 — executor의 동작 (Part 1)

```js
const p = new Promise((resolve) => {
  console.log('A')
  resolve('B')
  console.log('C')
  resolve('D')
})

p.then((v) => console.log(v))
console.log('E')
```

출력 순서는?

---

## 퀴즈 2 — 정답

```text
A → C → E → B
```

- executor는 **동기 실행**: `A` 출력, `resolve('B')`로 확정, `C`도 계속 출력
- `resolve('D')`는 이미 확정된 뒤이므로 **무시**
- `then` 콜백은 항상 비동기 → `E`가 먼저, 그 다음 `B`

> resolve를 호출해도 executor의 나머지 코드는 계속 실행된다. resolve는 return이 아니다.

---

## 퀴즈 3 — 체이닝의 반환값 (Part 3)

```js
Promise.resolve(10)
  .then((v) => {
    v * 2
  })
  .then((v) => console.log('결과:', v))

Promise.reject(new Error('실패'))
  .catch(() => 99)
  .then((v) => console.log('복구:', v))
```

각각 무엇이 출력될까?

---

## 퀴즈 3 — 정답

```text
결과: undefined
복구: 99
```

- 첫 번째: `return`이 없으므로 다음 Promise는 `undefined`로 fulfilled. `20`은 유실
- 두 번째: catch가 `99`를 반환 → **fulfilled로 복구** → 다음 then은 성공 콜백 실행

> catch는 "잡고 끝"이 아니라 "잡고 **복구**"다. catch 뒤의 체인은 정상 진행된다.

---

## 퀴즈 4 — Promise.all의 실패 (Part 4)

```js
const p1 = Promise.resolve('첫째')
const p2 = Promise.reject(new Error('둘째 실패'))
const p3 = new Promise((r) => setTimeout(() => r('셋째'), 1000))

Promise.all([p1, p2, p3])
  .then((values) => console.log('성공:', values))
  .catch((err) => console.log('실패:', err.message))
```

무엇이, 대략 언제 출력될까? 그리고 `p3`의 타이머는 어떻게 될까?

---

## 퀴즈 4 — 정답

```text
실패: 둘째 실패   (1초를 기다리지 않고 즉시)
```

- `all`은 fail-fast — `p2`의 실패가 확인되는 즉시 전체가 rejected
- `p3`를 1초 기다리지 않는다
- 하지만 `p3`의 타이머는 **취소되지 않고 계속 돈다**. 결과만 버려질 뿐

> "빨리 실패를 알려주는 것"과 "나머지 작업을 멈추는 것"은 다르다. 멈추려면 AbortController가 필요하다.

---

## 퀴즈 5 — 몇 초 걸릴까 (Part 4)

```js
const delay = (ms, v) => new Promise((r) => setTimeout(() => r(v), ms))

async function run() {
  const p1 = delay(1000, 'a')
  const p2 = delay(1000, 'b')

  const a = await p1
  const b = await p2
  console.log(a, b)
}
```

`run()`은 완료까지 약 몇 초 걸릴까? ① 약 1초 ② 약 2초

---

## 퀴즈 5 — 정답: ① 약 1초

- 두 타이머는 `delay()`를 **호출한 시점**에 이미 시작됐다
- `await p1`으로 1초 기다리는 동안 `p2`의 타이머도 **같이 흐른다**
- `await p2`에 도달했을 땐 이미 (거의) 끝나 있다

```js
// 이랬다면 2초 — 호출 자체가 뒤로 밀리므로
const a = await delay(1000, 'a')
const b = await delay(1000, 'b')
```

> 소요 시간은 await의 위치가 아니라 **작업을 시작한 시점**이 결정한다.

---

## 퀴즈 6 — try/catch와 비동기 에러 (Part 5)

```js
async function handleClick() {
  try {
    setTimeout(() => {
      throw new Error('fail')
    }, 100)
  } catch (err) {
    console.log('잡았다:', err.message)
  }
}
```

`'잡았다'`가 출력될까?

---

## 퀴즈 6 — 정답: 출력되지 않는다

- `setTimeout` **등록**은 성공적으로 끝났고, try 블록은 이미 통과했다
- 100ms 뒤 콜백이 실행될 때의 호출 스택에는 try/catch가 **존재하지 않는다**
- 이 에러는 uncaught error가 되어 전역으로 터진다

```js
// try/catch가 잡으려면 "await하는 Promise"로 만들어야 한다
try {
  await new Promise((_, reject) =>
    setTimeout(() => reject(new Error('fail')), 100),
  )
} catch (err) {
  console.log('잡았다:', err.message) // 잡힌다
}
```

> try/catch는 **같은 호출 스택**의 에러만 잡는다. 비동기 에러를 잡으려면 Promise로 연결해야 한다.

---

## 퀴즈 7 — async 함수 두 개 (Part 2)

```js
async function foo() {
  console.log('A')
  await bar()
  console.log('B')
}

async function bar() {
  console.log('C')
}

foo()
console.log('D')
```

출력 순서는?

---

## 퀴즈 7 — 정답

```text
A → C → D → B
```

- `foo()` 호출 → `A` 출력. `bar()` 호출도 **동기로 진행** → `C` 출력
- `await`는 `bar()`가 반환한 Promise에 도달한 뒤, `foo`의 나머지(`B`)를 마이크로태스크로 예약하고 호출부로 복귀
- `D` 출력 후 스택이 비면 `B`

> async 함수 **호출** 자체는 동기다. 실행이 넘어가는 지점은 호출이 아니라 **await**다.

---

## 퀴즈 8 — resolve에 Promise 넘기기 (Part 1)

```js
const inner = Promise.reject(new Error('inner fail'))

const outer = new Promise((resolve) => {
  resolve(inner) // reject가 아니라 resolve를 호출했다
})

outer
  .then(() => console.log('fulfilled'))
  .catch((e) => console.log('rejected:', e.message))
```

무엇이 출력될까?

---

## 퀴즈 8 — 정답: `rejected: inner fail`

- `resolve(Promise)`는 fulfilled 확정이 아니라 그 Promise를 **따라가겠다는 위임**이다
- `inner`가 rejected이므로 `outer`도 rejected가 된다
- `resolve`를 호출했다는 사실이 최종 상태를 보장하지 않는다

---

## 퀴즈 9 — catch의 위치 (Part 3)

```js
Promise.resolve()
  .catch((e) => console.log('catch:', e.message))
  .then(() => {
    throw new Error('fail')
  })
  .then(() => console.log('done'))
```

무엇이 출력될까?

---

## 퀴즈 9 — 정답: 아무것도 출력되지 않는다

- catch는 **이전 단계까지의 실패**만 잡는다. 앞의 Promise가 fulfilled이므로 catch 콜백은 실행되지 않고 통과한다
- 뒤의 then에서 던진 에러는 잡아줄 catch가 없다 → **unhandled rejection**
- `done`도 출력되지 않는다 (rejected는 성공 콜백을 건너뛴다)

> 이런 이유로 catch는 체인의 **마지막**에 두는 것이 기본이다.

---

## 퀴즈 10 — finally의 통과 규칙 (Part 3)

```js
Promise.resolve(1)
  .finally(() => 2)
  .then((v) => console.log('A:', v))

Promise.reject(new Error('fail'))
  .finally(() => console.log('B'))
  .then((v) => console.log('C:', v))
  .catch((e) => console.log('D:', e.message))
```

무엇이, 어떤 순서로 출력될까?

---

## 퀴즈 10 — 정답

```text
B → A: 1 → D: fail
```

- finally의 반환값 `2`는 무시되고 원래 값 `1`이 통과한다 → `A: 1`
- rejection도 finally를 그대로 통과해 catch까지 간다 → `D: fail` (`C`는 건너뜀)
- `B`가 가장 먼저인 이유: finally 콜백도 **마이크로태스크 한 칸**이고, 두 체인이 한 칸씩 번갈아 실행되기 때문 (첫 칸: 위 체인의 finally, 둘째 칸: `B`)

---

## 퀴즈 11 — race vs any (Part 4)

```js
const slowOk = new Promise((r) => setTimeout(() => r('성공'), 100))
const fastFail = new Promise((_, rj) =>
  setTimeout(() => rj(new Error('실패')), 50),
)

Promise.race([slowOk, fastFail]) // ?
Promise.any([slowOk, fastFail]) // ?
```

`race`와 `any`는 각각 어느 쪽으로 확정될까?

---

## 퀴즈 11 — 정답: race는 rejected, any는 fulfilled

| 조합기 | 결과                          | 이유                                            |
| ------ | ----------------------------- | ----------------------------------------------- |
| `race` | 50ms 뒤 `실패`로 **rejected** | 첫 **확정**을 따라간다. 실패가 먼저 확정됐다    |
| `any`  | 100ms 뒤 `성공`으로 fulfilled | 첫 **성공**을 기다린다. 그 전의 실패는 무시한다 |

- 타임아웃 패턴에 `race`를 쓰는 이유이자, 폴백 패턴에 `any`를 쓰는 이유

---

## 퀴즈 12 — 태스크와 마이크로태스크 중첩 (Part 2)

```js
setTimeout(() => console.log('A'), 0)

Promise.resolve().then(() => {
  console.log('B')
  setTimeout(() => console.log('C'), 0)
})

Promise.resolve().then(() => console.log('D'))

console.log('E')
```

출력 순서는?

---

## 퀴즈 12 — 정답

```text
E → B → D → A → C
```

- 동기 코드: `E`
- 마이크로태스크 전부: `B`, `D` — `B` 안에서 등록한 `C`는 태스크 큐의 `A` 뒤에 줄을 선다
- 태스크 하나씩: `A`, 그 다음 `C`

> 마이크로태스크 안에서 등록한 태스크는 **이미 대기 중인 태스크 뒤**로 간다.

---

## 전체 요약 — 다섯 문장

1. Promise는 결과를 **정확히 한 번, 항상 비동기로, 표준 인터페이스로** 전달한다
2. 콜백은 태스크가 아닌 **마이크로태스크** — 스택이 비면 전부, 태스크보다 먼저
3. `then`은 새 Promise를 만들고, 그 상태는 **콜백의 반환값**이 정한다
4. 의존성 없는 작업은 **먼저 시작해 두고** `all`로 기다린다
5. 잡히지 않은 rejection은 프로세스 종료나 미탐지 오류로 이어진다 — **경계마다 catch를 둔다**

---

## 참고 자료

- [ECMA-262 — Promise Objects](https://tc39.es/ecma262/#sec-promise-objects) — 상태, 반응(reaction), 잡(job)의 공식 정의
- [HTML Standard — Event loops](https://html.spec.whatwg.org/multipage/webappapis.html#event-loops) — 태스크/마이크로태스크 처리 모델
- [MDN — Using Promises](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Using_promises)
- [V8 블로그 — Faster async functions and promises](https://v8.dev/blog/fast-async) — await의 내부 최적화 이야기
- Jake Archibald — [In The Loop (JSConf.Asia)](https://www.youtube.com/watch?v=cCOL7MC4Pl0) — 이벤트 루프 시각화 발표

---

# 감사합니다

<!-- _class: invert -->

@yceffort
