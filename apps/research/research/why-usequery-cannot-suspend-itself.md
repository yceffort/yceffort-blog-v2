---
title: 'useQuery는 왜 스스로 suspend하지 못할까'
marp: true
paginate: true
theme: midnight
tags:
  - react
  - data-fetching
  - async
date: 2026-10-10
description: '처음 마운트되는 중에 suspend된 컴포넌트는 아무것도 남기지 못한다. 이 사실 하나로 useSuspenseQuery의 렌더 중 요청부터 experimental_prefetchInRender의 도입과 제거까지 실험으로 따라간다.'
published: true
art:
  undraw: throw-away
---

# useQuery는 왜 스스로 suspend하지 못할까

렌더가 버려진다는 사실 하나로 따라가기

<!-- _class: invert deck-cover -->

@yceffort

<!--
대상: useQuery와 useSuspenseQuery를 써 봤고, Suspense가 promise를 받아 컴포넌트를 다시 렌더한다는 정도를 아는 사람.
목표는 하나다. 다음 장의 네 질문이 같은 사실에서 나온다는 것을 실험으로 확인하는 것.
로그는 전부 Node 24 + jsdom에서 React 19.2.8, @tanstack/react-query 5.101.4와 5.104.1로 직접 실행한 결과다. ms 값은 실행마다 몇 ms씩 달라진다.
-->

---

## 코드를 따라 읽다 보면 생기는 질문 네 개

1. suspend가 풀린 뒤의 재시도는 리렌더일까, 리마운트일까
2. useSuspenseQuery는 렌더 중에 요청을 보낸다. 렌더는 순수해야 하지 않나
3. useQuery에는 왜 렌더 중에 요청하는 옵션이 없을까
4. 그런 옵션(`experimental_prefetchInRender`)이 있었는데 왜 사라졌을까

하나씩 따로 공부해야 할 것처럼 보이지만, 넷 다 같은 사실 하나에서 답이 나온다.

<!-- 질문이 하나 풀릴 때마다 다음 질문이 생기는 경험을 했다면, 그 질문들이 사실은 같은 뿌리였을 가능성이 높다는 이야기로 연다. -->

---

## 먼저 단어 네 개

| 단어    | 이 덱에서의 뜻                                                                                    |
| ------- | ------------------------------------------------------------------------------------------------- |
| 렌더    | React가 컴포넌트 함수를 불러 **무엇을 그릴지 계산**하는 것. 이때는 아직 화면이 바뀌지 않는다      |
| 커밋    | 렌더한 결과를 **실제 DOM에 반영**하는 것. 여기서부터 사용자 눈에 보인다                           |
| 마운트  | 컴포넌트가 **처음으로** 커밋되는 것                                                               |
| suspend | 렌더 도중 promise를 던져 "데이터가 아직이라 못 그린다"고 알리는 것. 그동안 fallback이 대신 보인다 |

렌더는 계산일 뿐이라서, React는 렌더한 결과를 **커밋하지 않고 버릴 수도 있다.** 이 덱 전체가 이 한 가지 성질에서 출발한다.

---

## 그 사실

> 처음 마운트되는 중에 suspend된 컴포넌트는 **아무것도 남기지 못한다.**
>
> 그래서 남아야 하는 것은 **컴포넌트 밖에** 둔다.

풀어 쓰면, 한 번도 커밋되지 못한 컴포넌트는 React가 기억해 주지 않는다. `useState`나 `useRef`에 넣어 둔 값도 렌더 결과와 함께 버려진다.

| 항목                  | 버전                                                     |
| --------------------- | -------------------------------------------------------- |
| React                 | 19.2.8                                                   |
| @tanstack/react-query | 5.101.4 (플래그가 있던 마지막 버전), 5.104.1 (현재 최신) |
| 실행 환경             | Node 24 + jsdom, 요청은 50ms 걸리는 가짜 함수            |

---

## 실험 1: 렌더될 때마다 번호를 매기는 컴포넌트

```jsx
const cache = new Map() // 컴포넌트 밖에 있는 캐시

function read(key) {
  if (!cache.has(key)) cache.set(key, startFetch(key)) // 50ms짜리 요청 시작
  const entry = cache.get(key)
  if (entry.status === 'pending') throw entry.promise // 아직이면 suspend
  return entry.value
}

function Profile({id}) {
  const [inst] = useState(() => ++created) // 초기화될 때마다 #1, #2, #3
  useLayoutEffect(() => log(`커밋 (state #${inst})`), [])
  return <p>{read(id)}</p>
}
```

- `useState` 초기화 함수는 컴포넌트가 **새로 만들어질 때만** 돈다. 번호가 오르면 새로 만들어졌다는 뜻이다
- `useLayoutEffect`는 커밋 직후에 돈다. "커밋" 로그가 찍히면 화면에 나간 것이다

<!-- startFetch는 덱에서 줄인 이름이다. 실제 실험 코드는 {status: 'pending'} 객체를 만들고 sleep(50).then()으로 status와 value를 채운다. 동작은 같다. -->

---

## 실험 1 결과: 렌더 세 번, 초기화도 세 번

```text
   3ms  useState 초기화 → #1   render
   5ms  fallback 커밋
   7ms  useState 초기화 → #2   render
  56ms  useState 초기화 → #3   render
 310ms  커밋 (state #3)
```

- 커밋되기 전까지 Profile은 **세 번** 렌더됐고, 세 번 모두 `useState`가 **새로** 초기화됐다
- 화면에 남은 것은 #3뿐이다. #1과 #2가 들고 있던 값은 그대로 사라졌다
- #1은 첫 시도, #3은 데이터가 온 뒤의 재시도다. #2는 fallback을 띄운 직후 React가 한 번 더 해 본 렌더다. 렌더는 생각보다 자주 일어나고, 언제든 버려질 수 있다

<!--
#2의 정체: React 19부터는 suspend가 일어나면 fallback을 먼저 커밋하고, 같은 트리의 다른 컴포넌트들이 요청을 미리 시작할 수 있도록 suspend된 트리를 한 번 더 렌더한다(pre-warm). 로그에서 fallback 커밋 바로 뒤에 #2가 나오는 것이 그 흔적이다.
56ms에 데이터가 왔는데 커밋이 310ms인 이유: fallback을 너무 빨리 걷어 화면이 깜빡이지 않도록 React가 fallback을 최소 300ms 보여준다(react-dom 19.2.8의 FALLBACK_THROTTLE_MS = 300). 이 덱의 주제와는 상관없으니 질문이 나오면 그때만 짚는다.
-->

---

## 네 가지 경우를 나란히 놓으면

| 상황                       | 렌더 | `useState`               | 화면                                |
| -------------------------- | ---- | ------------------------ | ----------------------------------- |
| 리렌더 (setState)          | 1번  | 그대로 #3                | 그대로                              |
| 커밋된 뒤 다시 suspend     | 3번  | **그대로 #3**            | 기존 DOM을 `display: none`으로 숨김 |
| **처음 마운트 중 suspend** | 3번  | **매번 새로** #1, #2, #3 | fallback                            |
| 리마운트 (key 변경)        | 1번  | 새로 #4                  | 새로 그림                           |

한 번이라도 커밋된 컴포넌트는 React가 기억해 둔다. 그래서 다시 suspend돼도 숨겨 두었다가 그대로 꺼내 쓴다. 한 번도 커밋되지 않은 컴포넌트는 기억할 것이 없어서, 재시도는 사실상 처음부터 다시 마운트하는 것과 같다. 이것이 질문 1의 답이다.

<!-- "커밋된 뒤 다시 suspend"는 id를 a에서 b로 바꾼 동기 업데이트로 측정했다. 이때 숨겨진 트리의 layout effect는 정리됐다가 다시 보일 때 다시 실행되지만, useState 값은 #3 그대로였다. -->

---

## 그러니 남아야 하는 것은 밖에 둔다

실험 1에서 #3이 데이터를 받을 수 있었던 것은 `cache`가 컴포넌트 밖에 있었기 때문이다. 캐시를 컴포넌트 안으로 옮기면 어떻게 될까.

```jsx
function Profile({id}) {
  const cache = useRef(new Map()).current // 캐시를 안으로
  // ...나머지는 실험 1과 같다
}
```

- 2초 동안 `useState` 초기화가 **94번** 돌았고, 화면은 계속 loading이었다
- 렌더할 때마다 새 Map이 생기니 앞 렌더가 시작한 요청을 모른다. 또 요청하고, 그 요청이 끝나 재시도하면 또 새 Map이 생긴다

react-query도 정확히 같은 제약 위에 서 있다.

---

## react-query에서 무엇이 어디에 사나

도서관에 빗대면 이렇다.

| 도서관         | react-query                        | 사는 곳                                 |
| -------------- | ---------------------------------- | --------------------------------------- |
| 책             | **Query**: 데이터와 진행 중인 요청 | QueryCache (**컴포넌트 밖**)            |
| 그 책의 신청서 | **QueryObserver**                  | 컴포넌트의 `useState` (**컴포넌트 안**) |

<!-- prettier-ignore -->
```ts
// react-query 5.104.1, useBaseQuery.ts:83 (타입 생략)
const [observer] = React.useState(
  () => new Observer(client, defaultedOptions),
)
```

신청서는 컴포넌트와 함께 버려지고, 책은 서가에 남는다. 책은 queryKey마다 한 권, 신청서는 useQuery를 부를 때마다 한 장이다.

<!-- react-query 딥다이브 2부의 도서관 비유(QueryClient=도서관, QueryCache=서가, Query=책, QueryObserver=알림 신청)를 그대로 쓴다. -->

---

## useSuspenseQuery는 렌더 중에 요청한다

```ts
// react-query 5.104.1, useBaseQuery.ts:119
if (shouldSuspend(defaultedOptions, result)) {
  throw fetchOptimistic(defaultedOptions, observer, errorResetBoundary)
}

// suspense.ts:49 (타입 생략)
export const shouldSuspend = (defaultedOptions, result) =>
  defaultedOptions?.suspense && result.isPending
```

- 풀어 쓰면: 데이터가 아직 없으면(`isPending`) 그 자리에서 요청을 시작하고, 그 요청의 promise를 던진다(suspend)
- `fetchOptimistic`은 서가(QueryCache)에서 책(Query)을 찾거나 새로 만든 뒤 요청을 보낸다

---

## 실험 2: useSuspenseQuery가 suspend될 때

```text
   3ms  render Profile
   4ms    new QueryObserver #1
   4ms    queryFn 호출 #1
   7ms  render Profile
   7ms    new QueryObserver #2
  56ms  render Profile
  56ms    new QueryObserver #3
 309ms  커밋 Profile
 310ms    Observer #3 구독 시작
```

| 버려진 것 (컴포넌트 안) | 남은 것 (컴포넌트 밖)               |
| ----------------------- | ----------------------------------- |
| QueryObserver #1, #2    | Query 1개, 요청 1번, 받아 온 데이터 |

신청서는 세 장 썼다가 두 장이 버려졌다. 책은 한 권이고, 책을 가지러 간 것도 한 번이다.

<!-- 측정 방법: QueryObserver.prototype.bindMethods(생성자에서 한 번 불린다)와 onSubscribe를 감싸 Observer에 번호를 붙였고, queryFn은 호출 횟수를 셌다. 렌더 세 번의 구조는 실험 1과 같다. -->

---

## 렌더가 세 번인데 요청은 왜 한 번일까

| 렌더 | QueryCache에서 찾은 `['user']` | 그래서                                        |
| ---- | ------------------------------ | --------------------------------------------- |
| #1   | 없음                           | 새로 만들고 요청을 시작한다                   |
| #2   | 있음, 요청 중                  | 새로 요청하지 않고 진행 중인 promise를 받는다 |
| #3   | 있음, 데이터 도착              | 데이터를 바로 읽는다. suspend하지 않는다      |

```ts
// query-core 5.104.1, queryCache.ts:163 build() 발췌
let query = this.get(queryHash) // 같은 키의 Query가 이미 있으면 그것을 쓴다
if (!query) {
  query = new Query({...})
  this.add(query)
}

// query.ts:608 fetch() 발췌. 이미 요청 중이면
return this.#retryer.promise // 진행 중인 promise를 그대로 돌려준다
```

---

## 그래서 렌더 중 요청은 괜찮은가

엄밀히 말하면 규칙 위반이다. React는 렌더 중에 **부수 효과**(네트워크 요청이나 전역 값 변경처럼 계산 바깥을 바꾸는 일)를 하지 말라고 한다. 렌더가 몇 번 돌지, 버려질지 알 수 없기 때문이다.

그래도 react-query의 렌더 중 요청이 문제를 일으키지 않는 것은 그 두 가지 걱정을 모두 피하기 때문이라고 생각한다.

- **몇 번 돌아도 요청은 한 번이다.** 같은 키면 같은 Query, 같은 요청으로 모인다. StrictMode에서 Observer가 6개 만들어져도 queryFn은 1번이었다
- **버려져도 잃는 것이 없다.** 요청과 데이터는 컴포넌트 밖 QueryCache에 남는다

이것이 질문 2의 답이다.

<!-- React 규칙 원문은 react.dev의 "Components and Hooks must be pure" 문서의 "Side effects must run outside of render". 순수하다고 주장하는 것이 아니라, 규칙을 어겨도 관찰 가능한 차이가 생기지 않도록 설계했다는 점을 강조한다. -->

---

## 라이브러리도 "suspend 뒤에는 리마운트된다"고 적어 두었다

```ts
// react-query 5.104.1, suspense.ts:24 발췌
if (defaultedOptions.suspense) {
  // Handle staleTime to ensure minimum 1000ms in Suspense mode
  const MIN_SUSPENSE_TIME_MS = 1000
  // ...staleTime이 1000보다 작으면 1000으로 올린다
}
```

> This prevents unnecessary refetching when components remount after suspending (바로 아래 줄 주석)

- staleTime 기본값은 0이라, 데이터는 받자마자 **stale**(다시 받아야 할 데이터)로 취급된다
- 실험 2에서 #1이 받아 온 데이터를, 나중에 마운트하는 #3은 stale로 보고 또 요청하려 한다
- 그래서 suspense 모드에서만 staleTime을 최소 1초로 올려 이 중복 요청을 막는다. 이 보정이 **suspense 모드에만** 있다는 점은 실험 4에서 다시 나온다

---

## 이제 useQuery: 요청은 언제 나갈까

```text
   3ms    new QueryObserver #1
   4ms  render Profile (status=pending)
   6ms  커밋 Profile
   6ms    Observer #1 구독 시작
   6ms    queryFn 호출
  60ms  render Profile (status=success)
```

```ts
// query-core 5.104.1, queryObserver.ts:110 발췌 (한국어 주석은 덱에서 붙임)
protected onSubscribe(): void { // 구독이 시작될 때 불린다
  if (this.listeners.size === 1) {
    this.#currentQuery.addObserver(this)
    if (shouldFetchOnMount(this.#currentQuery, this.options)) {
      this.#executeFetch() // 요청은 여기서 나간다
```

**구독**은 Observer가 Query에 "바뀌면 알려 달라"고 등록하는 것이다. useQuery에서 구독은 커밋된 뒤에 시작되고, 요청도 그때 나간다. 신청서를 제출해야 사서가 책을 가지러 가는 셈이다.

<!-- 첫 렌더의 결과는 queryFn이 불리기 전인데도 fetchStatus가 'fetching'으로 나온다. getOptimisticResult가 "곧 요청할 것"을 미리 반영하기 때문이다. 실제 queryFn 호출은 구독 시점이다. -->

---

## useQuery를 쓰는 컴포넌트가 스스로 suspend하면

실험 1의 표에 대입해 본다.

1. 처음 마운트되는 중에 suspend되면 커밋되지 않는다
2. 커밋되지 않으면 Observer는 구독을 시작하지 않는다
3. 구독이 없으면 요청도 나가지 않는다
4. 요청이 없으면 promise는 풀리지 않고, 재시도할 계기도 생기지 않는다

신청서를 제출하기도 전에 찢어 버리니, 아무도 책을 가지러 가지 않는 상황이다. useQuery는 "커밋된 뒤에 요청한다"는 전제로 만든 훅이라 렌더 중 suspend와 맞물리지 않는다. 이것이 질문 3의 답이다.

---

## 그래도 시도는 있었다: `experimental_prefetchInRender`

```jsx
// react-query 5.101.4까지
const queryClient = new QueryClient({
  defaultOptions: {queries: {experimental_prefetchInRender: true}},
})

function Profile() {
  const {promise} = useQuery({queryKey: ['user'], queryFn})
  const user = use(promise) // React 19의 use()로 suspend
  return <p>{user.name}</p>
}
```

- 플래그를 켜면 useQuery 결과에 `promise`가 붙는다
- `use(promise)`는 React 19 API다. promise가 아직이면 suspend하고, 끝났으면 값을 돌려준다
- 메인테이너 TkDodo는 2024년 12월에 useSuspenseQuery라는 별도 훅을 없애고 이 방식으로 가는 것을 고민하고 있다고 했다(토론 #8381)

---

## 실험 3: 렌더 중 요청 블록을 지워 보면

플래그가 렌더 중에 요청을 걸어 주지 않았다면 어떻게 됐을지 보려고, 5.101.4에서 그 블록만 지운 사본으로 위 Profile을 실행했다.

```text
   3ms  render Profile
   4ms    new QueryObserver #1
   7ms  render Profile
   7ms    new QueryObserver #2

1.5초 뒤: 구독한 Observer 없음, queryFn 0회, 화면은 loading
```

앞 장의 1~4가 그대로 일어났다. 아무도 요청을 보내지 않으니 fallback에서 나오지 못한다.

<!-- 렌더가 두 번(첫 렌더와 미리 렌더링)에서 멈춘 것은 promise가 영원히 풀리지 않아 재시도가 일어나지 않기 때문이다. -->

---

## 그래서 플래그가 손으로 해 주던 일 두 가지

<!-- prettier-ignore -->
```ts
// react-query 5.101.4, useBaseQuery.ts:153 발췌 (5.102.0에서 삭제)
if (defaultedOptions.experimental_prefetchInRender && ...) {
  const promise = isNewCacheEntry
    ? fetchOptimistic(defaultedOptions, observer, errorResetBoundary) // (1)
    : query?.promise
  promise?.catch(noop).finally(() => {
    observer.updateResult() // (2)
  })
}
```

- (1) **구독을 기다리지 않고 렌더 중에 직접 요청한다.** 신청서를 제출하기 전에 사서에게 먼저 달려가 부탁해 두는 셈이다
- (2) **요청이 끝나면 promise를 직접 풀어 준다.** React가 기다리는 promise는 곧 버려질 Observer #1이 만든 것이다. #1은 구독을 하지 않으니, 원래라면 아무도 이 promise를 풀어 주지 않는다

<!--
(1) 원문 주석: "Fetch immediately on render in order to ensure `.promise` is resolved even if the component is unmounted"
(2) 원문 주석: "`.updateResult()` will trigger `.#currentThenable` to finalize". updateResult가 Observer 내부의 currentThenable(= useQuery 결과의 promise)에 데이터를 넣어 resolve시킨다(5.101.4 queryObserver.ts의 finalizeThenableIfPossible).
-->

---

## 실험 4: 원본 5.101.4에서는

```text
   4ms    new QueryObserver #1
   4ms    queryFn 호출 #1          ← 렌더 중에 요청
   7ms    new QueryObserver #2
  57ms    new QueryObserver #3
 311ms    Observer #3 구독 시작
 312ms    queryFn 호출 #2          ← 같은 데이터를 또 요청
```

- 화면은 제대로 나온다. 렌더 중 요청 블록 덕분이다
- 그런데 요청이 **두 번** 나갔다. #3이 마운트하며 구독할 때 데이터가 이미 stale이었기 때문이다
- useSuspenseQuery였다면 staleTime 1초 보정이 이 요청을 막았을 것이다. useQuery에는 그 보정이 없다. staleTime을 1000으로 주면 1번으로 줄어드는 것도 확인했다

---

## 실험 5: 컴포넌트를 나누면 문제가 사라진다

```jsx
function Parent() {
  const {promise} = useQuery({queryKey: ['user'], queryFn})
  return (
    <Suspense fallback={<Spinner />}>
      <Child promise={promise} />
    </Suspense>
  )
}
const Child = ({promise}) => <p>{use(promise)}</p>
```

| 5.101.4                | Observer | queryFn | 화면 |
| ---------------------- | -------- | ------- | ---- |
| 원본                   | 1개      | 1회     | 정상 |
| 렌더 중 요청 블록 삭제 | 1개      | 1회     | 정상 |

suspend되는 것은 Child뿐이다. 신청서를 가진 Parent는 커밋되고 구독하니, 렌더 중 요청 블록이 없어도 평소처럼 요청이 나간다.

---

## 메인테이너의 결론 (2025-12-30, 이슈 #9988)

> I think we should generally disallow suspending the same component that calls useQuery and get rid of the experimental flag. I think it's too hacky. What you'd need to do is to pass the promise to a separate component and wrap that in suspense.

- useQuery를 호출한 **바로 그 컴포넌트**를 suspend시키는 것은 막아야 한다
- 실험 플래그는 없애야 한다. 너무 꼼수 같다(too hacky)
- promise를 별도 컴포넌트로 넘기고, 그쪽을 Suspense로 감싸야 한다

없애자고 한 것은 꼼수(실험 3, 4)다. 바깥 배치(실험 5)는 오히려 **권장하는 방법**이었다.

---

## 그리고 제거 (5.102.0)

| 날짜       | 일                                                                        |
| ---------- | ------------------------------------------------------------------------- |
| 2025-12-30 | TkDodo, "같은 컴포넌트를 suspend시키는 것은 막고 플래그는 없애야" (#9988) |
| 2026-08-17 | "이 방식은 오래 버티지 못할 것 같다" (토론 #10733)                        |
| 2026-08-20 | PR #11221 "ref: remove experimental_prefetchInRender" 병합                |
| 2026-08-22 | 5.102.0 배포. 플래그와 결과의 `promise` 필드가 함께 사라짐                |

- 플래그만이 아니라 `promise` 필드까지 지웠다(5.104.1에서 직접 확인). 그래서 **바깥 배치(실험 5)도 함께 사라졌다**
- PR 본문과 changeset에는 지웠다는 사실만 있고 이유는 없다. 바깥 배치를 보여 주던 공식 문서 예제(Suspense 가이드)도 이 PR에서 삭제됐다

<!-- 참고로 useSuspenseQuery는 지금도 promise를 직접 던진다(useBaseQuery.ts:120). React 19의 use()로 바꾸는 PR #11237은 2026-10-10 기준 아직 열려 있다. -->

---

## 헷갈리기 쉬운 점: 사라진 것은 두 가지다

|               | 렌더 중 요청 (꼼수)                                        | 결과의 `promise` 필드              |
| ------------- | ---------------------------------------------------------- | ---------------------------------- |
| 무엇을 위해   | 같은 컴포넌트에서 suspend해도 요청이 나가게                | `use(promise)`로 suspend할 수 있게 |
| 문제가 있었나 | 꼼수 없이는 요청이 안 나가고(실험 3), 있어도 두 번(실험 4) | 바깥 배치에서는 없었다(실험 5)     |
| 지운 이유     | "too hacky" (#9988)                                        | **적혀 있지 않다**                 |

- 실험 3, 4에서 본 문제(버려지는 Observer와 중복 요청)는 **왼쪽 열**, 같은 컴포넌트에서 suspend할 때의 이야기다
- 바깥 배치는 멀쩡했는데도 `promise`와 함께 사라졌다. 이유가 기록에 없으니 다음 장은 근거를 붙인 추측이다

---

## 바깥 배치는 왜 같이 사라졌을까 (추측)

**1. 바깥 배치만 골라 남길 방법이 없다**

- `promise`는 useQuery가 돌려주는 값이라, 같은 컴포넌트에서 바로 `use(promise)`를 부르는 코드가 가장 자연스럽다. 토론 #10733에 달린 답변도 그렇게 썼다
- 꼼수만 지우고 `promise`를 남기면, 그 자연스러운 코드는 실험 3처럼 **영원히 loading에 멈춘다**

**2. 바깥 배치로 하던 일은 기존 훅으로 된다**

- 자식이 promise를 받는 대신 같은 키로 useSuspenseQuery를 부르면 된다(다음 장)
- 조건에 따라 suspend하고 싶다면 queryFn 안에서 조건을 검사해 `null`을 돌려주고, 그 조건을 queryKey에 넣는다(TkDodo, #10733)

<!-- 메인테이너가 직접 밝힌 이유가 아니라는 점을 분명히 말한다. 근거로 쓸 만한 정황이 하나 더 있다: 제거 PR이 query-core의 thenable.ts, retryer.ts, hydration.ts까지 고쳤고, useQueries는 끝내 이 기능을 지원하지 못했다(#9988). 유지 비용도 컸다는 뜻이다. -->

---

## 실험 6: promise 없이 같은 배치를 만들면

```jsx
function Parent() {
  useQuery({queryKey: ['user'], queryFn}) // 부모도 데이터가 필요하다면
  return (
    <Suspense fallback={<Spinner />}>
      <Child /> {/* promise를 넘기지 않는다 */}
    </Suspense>
  )
}
function Child() {
  const {data} = useSuspenseQuery({queryKey: ['user'], queryFn}) // 같은 키
  return <p>{data}</p>
}
```

| 5.104.1                               | queryFn | 화면 |
| ------------------------------------- | ------- | ---- |
| 부모 useQuery + 자식 useSuspenseQuery | 1회     | 정상 |
| 자식 useSuspenseQuery만               | 1회     | 정상 |

같은 키는 서가에서 책 한 권으로 모인다. 신청서가 몇 장이든 책을 가지러 가는 것은 한 번이다.

<!--
Observer는 부모 useQuery가 있을 때 5개(구독은 부모 #1과 자식 #5), 자식만 있을 때 3개(구독은 #3)가 만들어졌다. 자식의 Observer가 버려지는 것은 실험 2의 useSuspenseQuery와 똑같다. 즉 Observer가 버려지는 것 자체는 문제가 아니다. 같은 컴포넌트에서 useQuery로 suspend하면 요청을 맡을 Observer가 하나도 남지 않는다는 것이 문제였다.
-->

---

## 질문 네 개, 다시

| 질문                                | 답                                                                                                                     |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| 1. 재시도는 리렌더인가 리마운트인가 | 한 번도 커밋되지 않았다면 리마운트와 같다. `useState`가 매번 새로 만들어진다                                           |
| 2. 렌더 중 요청이 왜 괜찮은가       | 몇 번 렌더돼도 요청은 한 번이고 결과는 컴포넌트 밖에 남는다. 렌더가 버려져도 잃는 것이 없다                            |
| 3. useQuery에는 왜 그 옵션이 없나   | useQuery는 커밋 뒤 구독할 때 요청한다. 스스로 suspend하면 커밋도, 구독도, 요청도 없다                                  |
| 4. 있던 옵션은 왜 사라졌나          | 같은 컴포넌트 suspend를 메우던 꼼수라서 걷어냈다. 바깥 배치는 멀쩡했지만 `promise`와 함께 사라졌다(이유는 기록에 없다) |

넷 다 "처음 마운트되는 중에 suspend된 컴포넌트는 아무것도 남기지 못한다"에서 나온다.

---

## 확인 문제

<!-- prettier-ignore -->
```jsx
function Providers({children}) {
  const [client] = useState(() => new QueryClient())
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>
}

<Suspense fallback={<Spinner />}>
  <Providers>
    <Profile /> {/* 안에서 useSuspenseQuery를 쓴다 */}
  </Providers>
</Suspense>
```

**Q. 처음 화면을 열면 어떻게 될까?**

1. 요청 1번 뒤 Profile이 보인다
2. 요청이 계속 반복되고 Spinner에서 나오지 못한다

<!-- 힌트 없이 먼저 고르게 한다. 오늘의 문장에 대입하면 풀린다. Providers가 어디에 있는지를 보면 된다. -->

---

## 정답: 2. 요청이 계속 반복된다

| Suspense 경계 위치 | 2초 동안 만들어진 QueryClient | queryFn | 화면    |
| ------------------ | ----------------------------- | ------- | ------- |
| Providers **바깥** | 85~90개                       | 85~90회 | loading |
| Providers **안쪽** | 1개                           | 1회     | 정상    |

- 경계가 바깥에 있으면 Providers도 "처음 마운트 중 suspend된 트리"에 들어간다
- 재시도할 때마다 `useState`가 새로 돌아 QueryClient가 새로 생긴다. 새 QueryClient의 QueryCache는 비어 있으니 앞 요청의 결과를 모르고, 또 요청하고 또 suspend된다. 도서관을 매번 새로 짓는 셈이다
- 공식 SSR 가이드(advanced-ssr.md, 5.104.1)에도 같은 경고가 있다: "Avoid useState when initializing the query client if you don't have a suspense boundary between this and the code that may suspend"

"남아야 하는 것은 밖에 둔다"는 QueryClient 자신에게도 적용된다.

<!-- 바깥 경계 수치는 세 번 실행해 85, 88, 90이 나왔다. 실험 1의 useRef 캐시와 같은 구조의 실패다. -->
