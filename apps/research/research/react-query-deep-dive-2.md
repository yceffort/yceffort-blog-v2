---
title: 'react-query 딥다이브 2부: 내부 동작, 실전 함정, SSR'
marp: true
paginate: true
theme: midnight
tags:
  - data-fetching
  - react
  - caching
date: 2026-08-14
description: 'QueryObserver 구독 모델, structural sharing, HydrationBoundary — 내부 동작에서 실전 함정, SSR까지 react-query 강의 2부'
published: true
art:
  undraw: real-time-sync
---

# react-query 딥다이브 2부

내부 동작, 실전 함정, SSR

<!-- _class: invert deck-cover -->

@yceffort

---

## 1부 복습 — 여섯 줄

1. 서버 상태는 원본의 사본 — 문제는 저장이 아니라 **동기화 시점**
2. **queryKey**가 캐시의 주소, queryFn이 쓰는 값은 전부 키에
3. 상태는 두 축: status(데이터가 있나) × fetchStatus(요청 중인가)
4. **staleTime**(기본 0)이 요청량을, **gcTime**(기본 5분)이 메모리를 결정
5. stale 캐시는 먼저 보여주고 뒤에서 갱신 — 계기는 mount/focus/reconnect
6. 쓰기 후에는 **invalidateQueries** — stale 표시 + active만 즉시 refetch

지금까지는 외워야 하는 규칙이었다. 2부가 끝나면 **구조에서 유도**할 수 있게 된다.

---

## 퀴즈 ① — 기다려줄까

```jsx
const {data: user} = useQuery({queryKey: ['user'], queryFn: fetchUser})
const {data: orders} = useQuery({
  queryKey: ['orders', user?.id],
  queryFn: () => fetchOrders(user.id),
})
```

첫 렌더에서 두 번째 쿼리에 무슨 일이 일어날까? `user`는 아직 도착하지 않았다.

<!-- 진행: "user를 기다린다"가 다수. 훅에 기다림 같은 건 없다는 점부터 짚는다. -->

---

## 퀴즈 ① — 정답: 기다려주지 않는다

- queryFn이 첫 렌더에 그대로 호출돼 `user.id`에서 **TypeError**가 나고, 쿼리는 error로 끝난다
- 키도 갈라진다. 배열 원소의 `undefined`는 `null`로 직렬화되어 `["orders",null]`이라는 **별도 캐시 항목**이 생긴다 (1부의 "undefined 무시"는 **객체 속성** 한정이다)
- user가 도착하면 `['orders', 7]`로 옮겨가고, 앞의 error 항목은 캐시에 그대로 남는다

```js
enabled: user?.id !== undefined // 조건은 훅을 if로 감싸는 게 아니라 여기에
```

> 종속 쿼리에서 enabled를 빠뜨리면 "가끔 에러"가 아니라 **매번** 에러다. 첫 렌더는 언제나 있기 때문이다.

---

## 퀴즈 ② — 네 값을 불러보자

목록이 이미 화면에 떠 있고, 탭에서 돌아오며 백그라운드 갱신이 도는 중이다.

```jsx
const {status, fetchStatus, isPending, isFetching} = useQuery({
  queryKey: ['todos'],
  queryFn: fetchTodos,
})
```

네 값은 각각 무엇일까?

---

## 퀴즈 ② — 정답: success / fetching / false / true

- 데이터가 있으니 `status`는 success, 요청이 날아가고 있으니 `fetchStatus`는 fetching
- `isPending`은 "보여줄 데이터가 없다"는 뜻이므로 여기서는 **false**
- 두 축이 분리된 이유가 바로 이 상태다. "데이터가 있으면서 동시에 요청 중"은 예외가 아니라 **정상 운영 상태**다

> 스피너를 `isFetching`에 걸면 갱신할 때마다 화면이 통째로 스피너가 된다. 첫 진입은 isPending, 갱신 표시는 isFetching.

---

## 퀴즈 ③ — 401을 받았다

```js
useQuery({
  queryKey: ['me'],
  queryFn: async () => {
    const res = await fetch('/api/me')
    if (!res.ok) return
    return res.json()
  },
})
```

서버가 401을 내려줬다. 이 쿼리의 `status`는 무엇일까?

<!-- 진행: error / success 손들기. 답은 error인데, 이유를 물으면 대부분 틀린다. -->

---

## 퀴즈 ③ — 정답: error, 단 이유가 다르다

- **401이라서가 아니다.** queryFn이 `undefined`를 반환해서다
- 에러 메시지도 `["me"] data is undefined`라 서버 상태와 아무 관련이 없다
- 401이든 500이든 이 코드는 같은 에러를 낸다. 화면은 원인을 구분할 방법이 없다

```js
if (!res.ok) throw new Error(`${res.status}`) // 실패는 throw로 알린다
```

> `return`은 성공이고, 그중 `undefined`만 금지다. "데이터 없음"은 `null`로 표현한다.

---

## 퀴즈 ④ — gcTime을 무한으로

재방문할 때마다 스피너가 뜨는 게 싫어서 이렇게 뒀다.

```js
useQuery({queryKey: ['todos'], queryFn: fetchTodos, gcTime: Infinity})
```

이 쿼리의 네트워크 요청은 줄어들까?

---

## 퀴즈 ④ — 정답: 줄지 않는다

- `staleTime`이 기본 0이면 데이터는 도착 즉시 낡는다. 마운트와 포커스 복귀마다 백그라운드 갱신이 그대로 나간다
- gcTime이 막은 것은 **캐시 삭제**뿐이다. 사용자는 스피너 대신 옛 데이터를 먼저 보지만, 뒤에서 요청은 똑같이 나간다
- 덤으로 이 설정은 안 쓰는 캐시가 **영원히 남는다**는 뜻이기도 하다

> 요청량의 손잡이는 **staleTime**, 메모리의 손잡이는 **gcTime**. 서로 독립인 두 시계다.

---

## 퀴즈 ⑤ — 에러 화면은 언제 뜨나

API 서버가 죽어서 모든 요청이 500으로 떨어진다.

```jsx
const {status, error} = useQuery({queryKey: ['todos'], queryFn: fetchTodos})
if (status === 'error') return <ErrorView error={error} />
```

사용자는 이 에러 화면을 얼마나 빨리 보게 될까?

<!-- 진행: "바로 뜬다"가 다수. 기본 retry 값을 되물으면 표정이 바뀐다. -->

---

## 퀴즈 ⑤ — 정답: 빨라야 7초 뒤

- 기본 `retry: 3`에 지수 백오프가 붙는다 — 1초 → 2초 → 4초, 여기에 요청 시간이 더해진다
- `status: 'error'`는 **재시도가 전부 소진된 뒤**에야 된다. 그동안 status는 계속 pending이라 화면은 스피너다
- 4xx는 세 번 더 물어봐도 답이 같다. 보통은 조건을 건다

```js
retry: (failureCount, error) => error.status >= 500 && failureCount < 3
```

> 서버에서는 기본값이 0회다. SSR 중 3회 재시도는 응답 지연으로 직결되기 때문이다.

---

## 퀴즈 ⑥ — 추가는 됐는데

```jsx
const {mutate} = useMutation({mutationFn: createTodo})
// 목록은 다른 컴포넌트에서
useQuery({queryKey: ['todos'], queryFn: fetchTodos})
```

서버가 201을 내려줬고 mutation은 success다. 목록 쿼리는 저절로 갱신될까?

---

## 퀴즈 ⑥ — 정답: 갱신되지 않는다

- mutation은 **캐시에 대해 아무것도 모른다**. 서버에 요청을 보내고 결과를 알려줄 뿐이다
- `['todos']`와 `createTodo` 사이에 연결고리는 없다 — react-query는 데이터 간 의존성을 모른다
- 읽기와 쓰기를 잇는 다리는 자동으로 놓이지 않는다

```js
useMutation({
  mutationFn: createTodo,
  onSuccess: () => queryClient.invalidateQueries({queryKey: ['todos']}),
})
```

> 쓰기 뒤에 무엇을 무효화할지는 자동으로 정해지지 않는다. 키 설계가 여기서 회수된다.

---

## 이 덱에서 다루는 것

4. **내부 동작** — useQuery 한 줄 뒤에서 벌어지는 일 (QueryClient, Query, QueryObserver), 그리고 useSuspenseQuery
5. **실전 함정 모음** — 리뷰에서 반복해서 만나는 잘못된 패턴 여섯 가지
6. **SSR** — 서버 프리페치, dehydrate, HydrationBoundary. 그리고 서버에서 달라지는 것들

1부처럼 파트 끝마다 **중간 점검 퀴즈**가 있고, 마지막에 **종합 퀴즈 6문제** (1부 내용 포함).

기준은 1부와 같다: **TanStack Query v5.101.4**, 소스 인용은 v5.101.4 태그, 동작 확인은 Node.js 24 + query-core 실측.

---

## Part 4 — useQuery 한 줄의 뒤편

<!-- _class: invert -->

---

## 그 전에 — 도서관 하나

이름부터 외울 필요는 없다. 내부 배치는 **도서관**과 같다.

| 도서관                        | react-query   |
| ----------------------------- | ------------- |
| 도서관 (운영 주체)            | QueryClient   |
| 서가 — 청구기호로 정리된 책장 | QueryCache    |
| 책 한 권                      | Query         |
| 그 책에 걸어둔 알림 신청      | QueryObserver |

- 내가 집에 가도 **책은 서가에 남는다** → 언마운트해도 캐시가 남는 이유
- 같은 책을 셋이 신청해도 사서는 **한 번만** 꺼내온다 → 요청 중복 제거
- 아무도 찾지 않는 책은 정해둔 기간 뒤 폐기 → gcTime

1부에서 외웠던 규칙들이, 여기서부터는 **구조를 보면 당연해진다**.

---

## 등장인물 네 명

```text
[ useQuery ]  ──만들고 구독──▶  [ QueryObserver ]   훅 호출당 1개
                                      │ 구독
                                      ▼
[ QueryClient ] ──보관──▶ [ QueryCache ] ──보관──▶ [ Query ]   키당 1개
     지휘자                 Map(해시 → Query)         상태+데이터 본체
```

| 객체          | 정체                                        | 수명                                   |
| ------------- | ------------------------------------------- | -------------------------------------- |
| QueryClient   | 전체 지휘자. invalidate 등 명령의 진입점    | 앱과 함께                              |
| QueryCache    | `Map<키 해시, Query>` 그 자체               | 앱과 함께                              |
| Query         | 키 하나의 데이터·상태·요청을 들고 있는 본체 | gcTime까지                             |
| QueryObserver | Query를 구독하는 관찰자                     | 훅 호출과 함께 (useQuery 두 번이면 둘) |

---

## Query — 컴포넌트 밖에서 산다

1부의 규칙들이 유도되는 첫 지점: **데이터의 주인은 컴포넌트가 아니라 Query 객체**다.

```text
[ Query ('["todos"]') ]
   ├─ state: { data, status, fetchStatus, dataUpdatedAt, ... }
   ├─ 진행 중인 요청 (retryer)     ← 재시도도 여기서
   ├─ GC 타이머                    ← gcTime도 여기서
   └─ observers: [ 관찰자 목록 ]   ← 구독자가 0이면 GC 시작
```

- 캐시(`QueryCache`)는 키 해시로 Query를 찾는 Map이다 — 소스(`queryCache.ts`): `this.#queries.get(query.queryHash)`
- 컴포넌트가 언마운트돼도 Query는 남는다 → **돌아오면 즉시 표시**가 가능한 이유
- 요청 상태도 Query가 가진다 → 같은 키의 요청이 **자동으로 하나로 합쳐지는** 이유 (1부의 중복 제거)

---

## 잠깐 — 옵저버 패턴이란

유튜브 채널 구독과 같다. 내가 계속 새로고침하지 않고, **구독해 두면 채널이 알려준다**.

```js
class Channel {
  #subscribers = new Set()
  subscribe(fn) {
    this.#subscribers.add(fn)
    return () => this.#subscribers.delete(fn) // 구독 해제 함수를 돌려준다
  }
  upload(video) {
    this.#subscribers.forEach((fn) => fn(video)) // 바뀌면 구독자 전원에게 알린다
  }
}
const off = channel.subscribe((v) => console.log('알림:', v)) // 구독 신청
off() // 해제. 이제 upload가 일어나도 조용하다
```

- 바뀌는 쪽(채널)이 **구독자 목록**을 들고, 바뀌는 순간 목록을 돌며 알린다. 보는 쪽은 기다릴 뿐
- 채널은 구독자가 몇 명인지 안다. 그래서 "아무도 안 본다"를 **판단할 수 있다**

---

## react-query는 이 패턴 그대로다

| 옵저버 패턴      | react-query                                                   |
| ---------------- | ------------------------------------------------------------- |
| 채널 (바뀌는 쪽) | Query                                                         |
| 구독자           | QueryObserver (해제 함수는 언마운트 때 호출)                  |
| 구독자 0명       | 마지막 구독자가 떠나면 `scheduleGc()`. gcTime 타이머의 출발점 |

```ts
// query.ts (v5.101.4) 요지. 앞 장 그림의 observers가 곧 구독자 목록이다
removeObserver(observer) {
  this.observers = this.observers.filter((x) => x !== observer)
  if (!this.observers.length) this.scheduleGc() // 아무도 안 보면 GC 예약
}
#dispatch(action) {
  this.observers.forEach((observer) => observer.onQueryUpdate()) // 전원에게 알림
}
```

1부에서 외웠던 "구독자가 0이면 GC 시작"은 규칙이 아니라 이 패턴의 **자연스러운 결과**다.

---

## QueryObserver — 컴포넌트의 대리인

컴포넌트가 Query를 직접 보지 않는다. 사이에 **관찰자**가 있다.

```text
<TodoList />  ──▶  QueryObserver A ──┐
<Header />    ──▶  QueryObserver B ──┼──구독──▶  Query ('["todos"]')
<Sidebar />   ──▶  QueryObserver C ──┘
```

QueryObserver가 하는 일:

- 자기 컴포넌트의 **옵션**(staleTime, enabled, select...)을 들고 있다 — 같은 Query를 봐도 컴포넌트마다 옵션이 다를 수 있다
- Query의 상태 변화를 받아 **자기 컴포넌트만** 다시 그리게 한다
- 마운트 시 "다시 요청이 필요한가"를 판정한다 (1부의 `shouldFetchOn` 판정을 이 객체의 `shouldFetchOnWindowFocus`/`shouldFetchOnReconnect`가 호출한다)

---

## useQuery는 데이터를 가져오지 않는다

소스를 보기 전에 문장으로. useQuery가 하는 일은 셋뿐이다.

```text
① 만든다   — 이 컴포넌트 몫의 QueryObserver를 하나 만든다
② 구독한다 — 그 Observer를 통해 Query의 변화를 듣는다
③ 받는다   — 지금 상태를 {data, status, ...} 모양으로 돌려받는다
```

- useQuery는 **캐시에 붙는 코드**지 요청을 보내는 코드가 아니다
- 요청, 재시도, GC는 전부 Query가 한다 — 도서관 비유로는 사서의 일
- 같은 키로 열 번 호출해도 요청이 한 번인 이유: 신청서만 열 장 걸렸을 뿐이다

다음 장에서 이 셋을 실제 소스로 확인한다. 코드는 낯설어도 하는 일은 위의 셋 그대로다.

---

## useQuery = 만들고, 구독하고, 반환한다

앞 장의 셋을 `useBaseQuery.ts`(v5.101.4)에서 그대로 확인할 수 있다:

```js
// ① Observer를 만든다 (훅 호출당 하나, 리렌더 간 유지)
const [observer] = React.useState(() => new QueryObserver(client, options))

// ② useSyncExternalStore — React 바깥의 값을 구독하는 React 18 공식 API
React.useSyncExternalStore(
  (cb) => observer.subscribe(notifyManager.batchCalls(cb)),
  () => observer.getCurrentResult(),
)

// ③ 결과를 그대로 주지 않고 trackResult로 한 번 감싸 반환한다
return observer.trackResult(result)
```

- 여기서 "바깥의 값"이 곧 Query다. 로직은 useQuery가 아니라 프레임워크 무관한 **query-core**에 있고 React는 구독만 한다 (Vue/Svelte/Solid 어댑터가 존재하는 이유)
- 외울 이름은 둘. ②의 `batchCalls`와 ③의 `trackResult`가 다음 장들의 주인공이다

---

## 리렌더 최적화 ① — tracked properties

Query가 바뀔 때마다 구독한 컴포넌트를 전부 다시 그리면 낭비다. Observer는 **컴포넌트가 실제로 읽은 필드**만 추적한다.

```jsx
const {data} = useQuery({queryKey: ['todos'], queryFn: fetchTodos})
// data만 구조분해했다
// → 백그라운드 갱신으로 isFetching이 true/false로 바뀌어도
//   data가 그대로면 이 컴포넌트는 리렌더되지 않는다
```

- 소스(`queryObserver.ts`): `trackResult()`가 결과 객체를 **Proxy로 감싸**, 필드를 읽는 순간 그 이름을 구독 목록에 등록한다
- 읽지 않은 필드의 변화는 알림 대상이 아니다. 단, **아무 필드도 읽지 않았다면** 안전을 위해 모든 변화에 알림이 온다
- 반대로 `isFetching`을 읽으면 갱신 인디케이터가 가능해지는 대신, 요청 시작/종료마다 두 번의 리렌더를 지불한다 — **읽는 것에만 비용을 낸다**

---

## 리렌더 최적화 ② — structural sharing

**구조 공유**: 새 응답을 옛 데이터와 깊이 비교해, 바뀐 가지만 새로 만들고 나머지는 **옛 참조를 그대로 재사용**한다. JSON 파싱은 매번 새 객체를 뱉지만 캐시에 들어가는 것은 그 결과가 아니다. Node 실측:

```js
// 내용이 완전히 같은 응답을 두 번 받으면: data1 === data2 (참조 유지)

// id 2의 done만 바뀐 응답이라면, 바뀐 가지만 새 객체가 된다
prev.list[0] === next.list[0] // true  (안 바뀐 항목: 참조 유지)
prev.list[1] === next.list[1] // false (바뀐 항목만 교체)
prev.meta === next.meta // true
```

- 소스: `replaceEqualDeep()` — 새 데이터를 옛 데이터와 깊이 비교해 같은 부분은 옛 참조를 재사용
- tracked properties와 합쳐지면: **내용이 같으면 `data` 참조가 같고, 참조가 같으면 리렌더가 없다**
- `useMemo`/`useEffect`의 deps에 `data`를 넣어도 안전한 이유

---

## notifyManager — 알림은 모아서

응답 하나에 Query의 상태는 여러 번 바뀐다. 그대로 흘려보내면 리렌더가 곱셈으로 불어난다.

```text
응답 도착 ─▶ Query 상태가 연달아 바뀜 (data 반영 → fetching 종료)
                  │   그대로 흘리면: 변경 2 × 구독자 3 = 최대 6번 리렌더
                  ▼
        [ notifyManager ]  ← 알림을 모았다가 한 틱 뒤에 한 번에
                  │
                  ▼
        React 18의 automatic batching ─▶ 리렌더 1번
```

- 소스: `notifyManager.ts` — 변경 알림을 모아뒀다가 다음 틱에 몰아서 flush한다
- 리렌더를 실제로 합치는 주체는 **React 18의 automatic batching**이다. notifyManager는 알림들이 같은 틱에 떨어지게 만들어 그 조건을 갖춰준다
- 외울 것은 한 줄: **알림은 뭉쳐서 오고, React가 한 번에 그린다**

<!-- 구두 보충: flush는 setTimeout(0), 즉 프레임이 아니라 매크로태스크 단위다. -->

---

## select — 컴포넌트 전용 가공

Observer가 useQuery 호출마다 있다는 사실의 활용: **같은 캐시, 다른 모양**.

```jsx
// 캐시에는 전체 목록 하나만 있다
const {data: count} = useQuery({
  queryKey: ['todos'],
  queryFn: fetchTodos,
  select: (todos) => todos.filter((t) => !t.done).length,
})
// count는 숫자. 이 컴포넌트는 "미완료 개수가 바뀔 때만" 리렌더
```

- select는 **캐시를 바꾸지 않는다**. Observer가 반환 직전에 가공할 뿐
- 결과가 이전과 같으면(참조/값) 리렌더 없음 — 목록 순서만 바뀌어도 개수가 같으면 조용하다
- 주의: select에 **인라인으로 무거운 계산**을 넣으면 리렌더마다 재실행된다. 무거우면 `useCallback`이나 컴포넌트 밖 함수로

---

## useSuspenseQuery — 분기 없는 세계

useQuery가 Observer를 **구독**하는 훅이라면, 같은 Observer를 다른 방식으로 소비하는 훅이 하나 더 있다.

```jsx
function TodoList() {
  const {data} = useSuspenseQuery({queryKey: ['todos'], queryFn: fetchTodos})
  return <ul>{data.map(...)}</ul> // data는 항상 있다. 타입도 Todo[] (undefined 없음)
}

// 로딩과 에러는 바깥에서 선언한다
<ErrorBoundary fallback={<ErrorView />}>
  <Suspense fallback={<Spinner />}><TodoList /></Suspense>
</ErrorBoundary>
```

- `status === 'pending'` 분기가 사라진다. 데이터가 아직 없으면 컴포넌트가 **Promise를 throw**하고, React가 그것을 받아 가장 가까운 `<Suspense>`의 fallback을 대신 그린다. Promise가 풀리면 그 컴포넌트만 다시 렌더한다
- 에러도 같은 방식으로 throw되어 ErrorBoundary가 받는다. 대가로 `enabled`, `placeholderData`, `throwOnError`는 **타입에서 빠졌다** — "데이터는 항상 있고 에러는 항상 던져진다"는 모델과 모순이기 때문이다

---

## Suspense의 함정 — 요청이 직렬이 된다

```jsx
function Dashboard() {
  const {data: user} = useSuspenseQuery({queryKey: ['user'], ...})
  // ↑ 여기서 suspend. 아래 줄은 user가 도착해야 실행된다
  const {data: stats} = useSuspenseQuery({queryKey: ['stats'], ...})
}
```

```text
useQuery 2개:          [ user ──────]
                       [ stats ─────]        총 1초

useSuspenseQuery 2개:  [ user ──────][ stats ─────]  총 2초
```

- suspend는 **함수 실행을 그 줄에서 중단**하는 것이다. 두 번째 훅은 첫 응답 전엔 호출조차 안 된다 (Promise 덱의 "순차 await" 함정과 같은 구조)
- 서로 독립인 쿼리 여럿은 **`useSuspenseQueries`**로 병렬 선언한다

---

## Part 4 정리

- 데이터의 주인은 컴포넌트가 아니라 **Query** — 언마운트 후 생존, 요청 합류, GC가 전부 여기서 나온다
- 컴포넌트는 **QueryObserver**를 통해 구독한다 — Observer는 훅 호출당 하나, 옵션은 호출마다, 데이터는 키마다
- useQuery는 **useSyncExternalStore 구독**일 뿐, 로직은 프레임워크 무관한 query-core에 있다
- **tracked properties**: 읽은 필드만 리렌더 사유가 된다
- **structural sharing**: 내용이 같으면 참조도 같다 → deps에 data를 넣어도 안전
- **select**: 캐시는 하나, 컴포넌트마다 다른 단면 — 가공 결과가 같으면 리렌더 없음
- **useSuspenseQuery**: 같은 Observer, pending이면 Promise를 throw → 분기가 사라진다. 한 컴포넌트에 여럿이면 **직렬**, 독립 쿼리는 `useSuspenseQueries`

---

## 중간 점검 ① — 몇 개나 만들어질까

```jsx
function Header() {
  const {data} = useQuery({queryKey: ['me'], queryFn: fetchMe})
}
function Sidebar() {
  const {data} = useQuery({queryKey: ['me'], queryFn: fetchMe})
  const {data: unread} = useQuery({
    queryKey: ['me'],
    queryFn: fetchMe,
    select: (me) => me.unreadCount,
  })
}
```

두 컴포넌트가 같은 화면에서 함께 마운트된다. Query와 QueryObserver는 각각 몇 개 만들어지고, 네트워크 요청은 몇 번 나갈까?

---

## 중간 점검 ① — 정답: Query 1개, Observer 3개, 요청 1번

- Query는 **키당 1개** — 셋 다 `['me']`이므로 본체는 하나
- Observer는 **useQuery 호출당 1개** — Header 1 + Sidebar 2 = 3개 ("컴포넌트당 1개"가 아니다)
- 동시 요청은 같은 Query에 합류하므로 **1번** (1부의 중복 제거)
- select는 Observer의 일이다 — 캐시는 하나인데 Sidebar의 두 번째 훅만 숫자를 받는 이유

> 데이터와 요청은 Query(키)마다, 옵션과 가공은 Observer(호출)마다.

---

## 중간 점검 ② — 목록이 바뀌었는데 조용하다

```jsx
const {data: count} = useQuery({
  queryKey: ['todos'],
  queryFn: fetchTodos,
  select: (todos) => todos.filter((t) => !t.done).length,
})
```

백그라운드 갱신이 일어났고, 응답에서 **완료된 항목 하나의 제목이 바뀌었다** (미완료 개수는 그대로). 이 컴포넌트는 리렌더될까?

<!-- 진행: "캐시 데이터는 분명히 바뀌었다"를 먼저 합의시키고 손들기 — select가 어느 층에서 일하는지로 반전. -->

---

## 중간 점검 ② — 정답: 리렌더되지 않는다

- 캐시의 data는 바뀌었다 — 제목이 바뀐 항목은 structural sharing을 거쳐도 새 참조가 된다
- 하지만 이 Observer가 컴포넌트에 반환하는 것은 **select를 거친 숫자**이고, 미완료 개수는 그대로다
- 가공 결과가 이전과 같으면 알림이 없다 — "이 컴포넌트의 관심사는 미완료 개수뿐"이라고 select로 선언했기 때문

> 리렌더 방어는 겹겹이다: 읽은 필드만(tracked properties), 내용이 같으면 같은 참조(structural sharing), 가공 결과가 같으면 침묵(select).

---

## 중간 점검 ③ — 1초짜리 요청 셋

```jsx
function Dashboard() {
  const {data: user} = useSuspenseQuery({queryKey: ['user'], ...})
  const {data: stats} = useSuspenseQuery({queryKey: ['stats'], ...})
  const {data: news} = useSuspenseQuery({queryKey: ['news'], ...})
}
```

세 요청은 각각 1초 걸리고 서로 독립이다. 화면이 뜨기까지 몇 초일까? 어떻게 고칠까?

---

## 중간 점검 ③ — 정답: 3초. useSuspenseQueries로 1초

- suspend는 **그 줄에서 함수 실행을 중단**한다 — 두 번째 훅은 user가 도착하기 전엔 호출조차 안 된다
- 1초 × 3 직렬 = 3초 (워터폴)
- 서로 독립인 쿼리는 `useSuspenseQueries`로 한 번에 선언 → 병렬 1초

단, stats가 user의 결과를 필요로 하는 **종속 쿼리라면 직렬이 정답**이다 — 그때는 워터폴이 아니라 의존성이다.

---

## Part 5 — 실전에서 반복되는 함정 여섯

<!-- _class: invert -->

전부 실제 코드 리뷰에서 반복해서 만나는 패턴들이다.

---

## 함정 1 — 흩어진 queryKey

```js
// list.tsx
useQuery({queryKey: ['todos', 'list'], ...})
// mutation.ts
queryClient.invalidateQueries({queryKey: ['todo', 'list']}) // todos가 아니라 todo
```

- 오타·단복수 불일치는 에러가 아니다. **조용히 아무 일도 안 일어난다** (없는 키를 무효화했을 뿐)
- 해결: 키와 fn을 한 곳에 선언하는 **queryOptions 헬퍼**

```ts
// queries/todos.ts — 키의 유일한 출처
export const todoListOptions = queryOptions({
  queryKey: ['todos', 'list'],
  queryFn: fetchTodos,
})

useQuery(todoListOptions)
queryClient.invalidateQueries({queryKey: todoListOptions.queryKey})
```

타입 추론도 따라온다. v5에서 키 관리의 사실상 표준.

<!-- 구두 보충: "사실상 표준"의 근거 — queryOptions는 v5 공식 문서의 TypeScript 가이드가 권하는 패턴이고, 메인테이너 TkDodo의 "The Query Options API" 글이 배경 설명. 팩토리 객체(todoKeys.list() 류)로 더 키우는 팀도 많다. -->

---

## 함정 2 — 파라미터 바꾸고 refetch()

```jsx
// ❌ page는 클로저로, 갱신은 수동으로
const {data, refetch} = useQuery({
  queryKey: ['orders'],
  queryFn: () => fetchOrders(page),
})
useEffect(() => {
  refetch()
}, [page])
```

- `['orders']` 캐시 하나를 페이지마다 **덮어쓴다** — 1페이지로 돌아가도 즉시 표시가 안 된다
- 키와 fn이 어긋난 상태라 중복 제거, invalidate, 그리고 Part 6의 SSR hydrate까지 전부 오작동의 씨앗

<!-- prettier-ignore -->
```jsx
// ✅ 파라미터는 키에. refetch는 필요 없다
useQuery({queryKey: ['orders', page], queryFn: () => fetchOrders(page)})
```

`refetch`의 정당한 용도는 "**같은 키를 명시적 계기로 다시**"(새로고침 버튼)뿐이다.

---

## 함정 3 — data를 useEffect로 복사

```jsx
// ❌ 서버 상태를 클라이언트 상태로 복사
const {data} = useQuery({queryKey: ['todos'], queryFn: fetchTodos})
const [todos, setTodos] = useState([])
useEffect(() => {
  if (data) setTodos(data)
}, [data])
```

- 렌더가 한 박자 늦고, 백그라운드 갱신이 `todos`에 반영 안 되는 순간이 생기고, "어느 쪽이 진실인가" 문제가 시작된다
- 파생값은 **렌더 중 계산**하거나 **select**로:

```jsx
const {data: undone} = useQuery({
  ...todoListOptions,
  select: (todos) => todos.filter((t) => !t.done),
})
```

- 예외는 편집 화면의 로컬 초안뿐 — "서버 값은 초기값"이라고 선을 긋고 복사한다

---

## 함정 4 — invalidate를 기다리지 않기

```jsx
// ❌ 폼 제출 → 목록으로 이동. 그런데 옛 목록이 잠깐 보인다
onSuccess: () => {
  queryClient.invalidateQueries({queryKey: ['todos']})
  navigate('/todos') // invalidate의 refetch는 아직 진행 중
}
```

- `invalidateQueries`는 **Promise를 반환**한다 — active 쿼리들의 refetch가 끝날 때까지
- 갱신 완료를 보장하고 이동해야 한다면 await:

```jsx
onSuccess: async () => {
  await queryClient.invalidateQueries({queryKey: ['todos']})
  navigate('/todos')
}
```

- 반대로 "이동 먼저, 갱신은 백그라운드"가 낫다면 await 없이 — **선택이라는 것을 알고 고르는 것**과 모르는 것의 차이

<!-- 구두 보충: await하면 그만큼 버튼이 pending에 오래 머문다는 트레이드오프를 꼭 언급. 목록 화면이 어차피 스피너/기존 캐시를 보여줄 수 있으면 await 없는 쪽이 체감이 좋은 경우도 많다. -->

---

## 함정 5 — 낙관적 업데이트에서 cancelQueries 생략

1부의 낙관적 업데이트에서 첫 줄이 왜 `cancelQueries`였는가.

```text
t0  포커스 계기 → ['todos'] 백그라운드 갱신 시작 (옛 목록 응답 대기)
t1  사용자가 체크박스 클릭 → setQueryData로 캐시 미리 수정 (화면 반영)
t2  t0의 응답 도착 → 서버의 "옛" 목록이 캐시를 덮어쓴다
    → 체크가 풀린 것처럼 보인다. 잠시 후 무효화가 최신을 가져와 돌아온다 (깜빡임)
```

- 낙관적 수정과 진행 중이던 백그라운드 갱신의 **레이스 컨디션**이다
- `await queryClient.cancelQueries({queryKey: ['todos']})`가 t0의 응답을 버리게 해서 이 창을 닫는다
- 1부 인트로 코드의 레이스 컨디션을 react-query가 없애줬지만, **캐시를 직접 만지는 순간 레이스는 내 책임으로 돌아온다**

---

## 함정 6 — 에러 처리를 컴포넌트마다 복붙

- v5는 useQuery의 `onError` 콜백을 **제거**했다 — 구독하는 useQuery 호출마다 실행돼, 같은 토스트가 구독 수만큼(Part 4 그림이라면 세 번) 뜨는 문제 때문
- 전역 처리는 **QueryCache 레벨**에 한 번만 건다:

```js
const queryClient = new QueryClient({
  queryCache: new QueryCache({
    onError: (error, query) => {
      if (query.state.data !== undefined) {
        // 데이터가 이미 화면에 있는 백그라운드 갱신 실패만 토스트
        toast.error(`갱신 실패: ${error.message}`)
      }
    },
  }),
})
```

- 첫 로딩 실패는 컴포넌트의 `status === 'error'` 분기(또는 ErrorBoundary)가, 백그라운드 실패는 전역 토스트가 — 역할 분담이 선명해진다

---

## Part 5 정리 — 체크리스트

리뷰할 때 이 여섯을 순서대로 본다.

- [ ] queryKey가 **queryOptions 헬퍼**로 한 곳에 모여 있는가
- [ ] `refetch()` 호출이 있다면 — 파라미터가 키에 안 들어간 신호 아닌가
- [ ] `useEffect`로 data를 **useState에 복사**하고 있지 않은가
- [ ] invalidate 후 이동/후속 동작 — **await가 필요한 상황**인가
- [ ] `setQueryData` 앞에 **cancelQueries**가 있는가
- [ ] 에러 토스트가 컴포넌트마다 있는가 — **QueryCache onError** 한 곳이면 되는가

---

## Part 6 — SSR

<!-- _class: invert -->

---

## SSR — 문제 정의

지금까지의 모델은 전부 브라우저 안 이야기였다. 서버 렌더링이 끼면 문제가 생긴다.

```text
[ 서버 ] HTML 렌더                    [ 브라우저 ] hydrate 후
   QueryClient A                         QueryClient B
   캐시: (요청해서 채움)                  캐시: 비어 있음!
        │                                    │
        └── HTML엔 데이터가 박혀 있는데 ──────┘
            클라이언트 캐시는 빈손 → 스피너 → 재요청 (깜빡임)
```

서버가 이미 받은 데이터를 **클라이언트 캐시에 넘겨주는 통로**가 필요하다.

react-query의 답: 서버 캐시를 **직렬화(dehydrate)** 해서 HTML에 실어 보내고, 클라이언트에서 **복원(hydrate)** 한다. (React가 HTML에 이벤트를 붙이는 hydration과는 별개의, 캐시 쪽 용어다)

---

## 표준 패턴 — prefetch, dehydrate, HydrationBoundary

Next.js App Router의 서버 컴포넌트에서:

```jsx
export default async function Page() {
  const queryClient = new QueryClient() // 요청마다 새로 (뒤에서)
  await queryClient.prefetchQuery({queryKey: ['todos'], queryFn: fetchTodos})

  return (
    <HydrationBoundary state={dehydrate(queryClient)}>
      <TodoList /> {/* 'use client' 컴포넌트 */}
    </HydrationBoundary>
  )
}
```

- `prefetchQuery`: 서버 캐시를 채운다. `await`를 붙여야 dehydrate 시점에 데이터가 들어 있다
- `dehydrate(queryClient)`: 그 캐시를 **JSON으로 직렬화**해 HTML에 싣는다
- `HydrationBoundary`: 자식이 렌더되기 전에 그 JSON을 **브라우저 캐시에 부어 넣는다**

---

## 클라이언트는 아무것도 몰라도 된다

```jsx
'use client'
function TodoList() {
  const {data} = useQuery({queryKey: ['todos'], queryFn: fetchTodos})
  // 서버에서 hydrate된 캐시가 이미 있다
  // → 첫 렌더부터 data 있음, 요청 없음, 스피너 없음
}
```

- 컴포넌트 코드는 CSR일 때와 **완전히 같다**. "서버에서 미리 받았는지"는 캐시 상태의 문제일 뿐
- **키가 연결 고리다**: prefetch한 키와 useQuery의 키가 다르면 hydrate는 무의미해지고 조용히 재요청한다 (함정 1의 queryOptions가 이 병을 막는다)
- hydrate된 데이터도 평범한 캐시다 — `dataUpdatedAt`은 서버에서 받은 시각, staleTime이 지났으면 마운트 계기로 백그라운드 갱신. 1부의 생명주기가 그대로 적용된다. 그래서 SSR에서는 **staleTime을 0보다 크게** 두는 것이 정석이다 (방금 받은 데이터를 hydrate 직후 또 요청하는 낭비 방지)

---

## 클라이언트가 둘이라는 것

SSR에서 QueryClient는 **두 개**다. 이름이 같을 뿐 하는 일이 다르다.

| 구분 | 서버의 QueryClient                        | 브라우저의 QueryClient         |
| ---- | ----------------------------------------- | ------------------------------ |
| 정체 | dehydrate할 데이터를 담는 **일회용 그릇** | 앱이 사는 동안의 **진짜 캐시** |
| 개수 | prefetch하는 서버 컴포넌트마다            | 앱 전체에 하나                 |
| 수명 | 응답을 보내면 버려진다                    | 탭을 닫을 때까지               |

- 서버 쪽은 **캐시로 쓸 물건이 아니다**. 담고, 직렬화하고, 버린다. 그게 전부다
- Provider용 클라이언트는 공식 `getQueryClient()`가 한 파일에서 분기한다: `isServer`면 매번 새로, 브라우저면 `browserQueryClient ??= ...`로 하나만 유지

<!-- 구두 보충: 공식 문서는 Provider 클라이언트를 useState로 만들지 말라고 경고한다. suspend할 수 있는 코드와 사이에 Suspense 경계가 없으면 첫 렌더에서 React가 클라이언트를 버리기 때문. -->

---

## QueryClient는 요청마다 새로

```js
// ❌ 모듈 스코프 싱글턴 — 브라우저에선 정석, 서버에선 사고
const queryClient = new QueryClient()

// ✅ 서버 컴포넌트 안에서 생성
export default async function Page() {
  const queryClient = new QueryClient()
  ...
}
```

- **안전**: 서버 프로세스는 모든 사용자가 공유한다. 모듈 스코프 캐시에 A의 `['me']`가 남아 있으면 **B의 렌더에 A의 개인정보**가 실려 나간다
- **잃는 것도 없다**: 어차피 직렬화하고 버릴 그릇이라 재사용해서 얻을 게 없다. 공식 가이드도 한 요청에서 queryClient를 여러 개 만들어 각각 dehydrate해도 된다고 명시한다
- 한 요청 안의 여러 서버 컴포넌트가 하나를 공유하게 하려면 React의 `cache()`로 감싼다 — `cache(() => new QueryClient())`는 **요청 스코프**라 요청 사이로는 새지 않는다

---

## 서버에서는 기본값이 다르다

1부에서 미뤄둔 소스를 회수한다. query-core는 서버 환경을 감지해 기본값을 바꾼다.

```ts
// retryer.ts — 재시도
const retry = config.retry ?? (environmentManager.isServer() ? 0 : 3)

// removable.ts — GC. 구독자들이 다른 gcTime을 주면 최댓값이 이긴다
this.gcTime = Math.max(
  this.gcTime || 0,
  newGcTime ?? (environmentManager.isServer() ? Infinity : 5 * 60 * 1000),
)
```

- **retry 0**: SSR 중 3회 재시도는 곧 응답 지연이다. 서버는 빠르게 실패하고, 클라이언트에서 다시 시도하는 편이 낫다
- **gcTime Infinity**: 서버의 QueryClient는 요청과 함께 버려진다. GC 타이머를 걸어봐야 실행될 일이 없고, 타이머 자체가 누수가 된다

<!-- 구두 보충: isServer 판정이 environmentManager로 감싸져 있는 이유 — 테스트 등에서 setIsServer로 런타임 오버라이드가 가능하도록 v5.101에서 간접화됐다. -->

---

## Part 6 정리

- SSR의 통로: **prefetchQuery → dehydrate → HydrationBoundary** — 클라이언트 코드는 그대로
- hydrate된 데이터도 평범한 캐시 — 1부의 생명주기가 그대로 적용
- 서버의 QueryClient는 **요청마다 새로** — 싱글턴은 사용자 간 데이터 유출
- 서버 기본값: retry 0, gcTime Infinity. 그리고 SSR에서는 **staleTime > 0**으로 hydrate 직후 재요청을 막는 것이 정석

---

## 중간 점검 — prefetch가 무시된다

```jsx
// 서버 컴포넌트
await queryClient.prefetchQuery({queryKey: ['todos'], queryFn: fetchTodos})

// 클라이언트 컴포넌트
const {data} = useQuery({queryKey: ['todos', 'list'], queryFn: fetchTodos})
```

프리페치도, HydrationBoundary도 정상 동작했다. 그런데 첫 화면에 스피너가 뜨고 요청이 다시 나간다. 왜일까?

---

## 중간 점검 — 정답: 키가 다르다

- hydrate는 잘 됐다 — 클라이언트 캐시에 `['todos']` 항목이 들어 있다
- 하지만 useQuery가 보는 주소는 `['todos', 'list']` — **다른 캐시 항목**이므로 빈손에서 시작한다
- 에러도 경고도 없다. 프리픽스 관계여도 소용없다 (프리픽스 매칭은 invalidate의 기능이고, 조회는 정확한 키)
- 예방: 서버와 클라이언트가 **같은 queryOptions 헬퍼**를 import하게 한다 (함정 1의 queryOptions)

> SSR 최적화가 통째로 무시되는데 화면은 멀쩡히 돌아간다 — 가장 발견이 늦는 종류의 버그다.

---

## 종합 퀴즈

<!-- _class: invert -->

여섯 문제. 1부 내용도 포함이다.

---

## 퀴즈 1 — 요청이 나갈까 (1부 Part 2)

```jsx
// A 컴포넌트 (이미 마운트되어 화면에 있음)
useQuery({queryKey: ['user'], queryFn: fetchUser, staleTime: 60_000})

// 10초 뒤, B 컴포넌트가 새로 마운트된다
useQuery({queryKey: ['user'], queryFn: fetchUser, staleTime: 60_000})
```

B가 마운트될 때 네트워크 요청이 나갈까? staleTime이 0이었다면?

<!-- 진행: "나간다/안 나간다" 손들기. "마운트하면 무조건 요청"이라고 아는 사람이 많아 "나간다"가 우세한 편 — shouldFetchOn의 stale 조건으로 반전시킨다. -->

---

## 퀴즈 1 — 정답

**나가지 않는다.** staleTime 0이었다면 **나간다.**

- 데이터 도착 10초 뒤 = 아직 fresh → 마운트 계기가 와도 `shouldFetchOn`은 **stale일 때만** 통과시킨다
- B는 캐시를 즉시 받는다 (요청 0회)
- staleTime 0이면 도착 즉시 stale → 마운트 계기에 백그라운드 갱신. 단, 이때도 B는 캐시부터 즉시 보여준다 (스피너 없음)

> 갱신을 결정하는 것은 "계기 × 신선도"의 곱이다. 계기만으로는 아무 일도 일어나지 않는다.

---

## 퀴즈 2 — 같은 캐시일까 (1부 Part 1)

```js
// ① 과 ②, ③ 과 ④ — 각각 같은 캐시 항목을 볼까?
useQuery({queryKey: ['todos', {page: 1, size: 10}], ...}) // ①
useQuery({queryKey: ['todos', {size: 10, page: 1}], ...}) // ②

useQuery({queryKey: ['users', 42], ...})   // ③
useQuery({queryKey: ['users', '42'], ...}) // ④
```

---

## 퀴즈 2 — 정답: ①②는 같고, ③④는 다르다

- 해시는 객체의 **키를 정렬한 뒤** 직렬화한다 → 객체 속성 순서는 무관 (①=②)
- 타입은 구분한다 → 숫자 42와 문자열 '42'는 **다른 캐시** (③≠④)
- ③④가 실무 단골 사고다: `useParams()`는 문자열을 주고, 직접 만든 키는 숫자를 쓰면 같은 유저가 캐시 두 벌로 갈라진다

> 키를 queryOptions 헬퍼로 모으면 타입까지 한 곳에서 강제된다 (함정 1).

---

## 퀴즈 3 — invalidate의 사정거리 (1부 Part 3)

```js
// 화면에 있는 것: ['todos', 'list']
// 화면에 없는 것(캐시만 생존): ['todos', 'detail', 7], ['stats']

queryClient.invalidateQueries({queryKey: ['todos']})
```

각 캐시 항목에 무슨 일이 일어날까?

---

## 퀴즈 3 — 정답

| 항목                     | 매칭         | 일어나는 일                            |
| ------------------------ | ------------ | -------------------------------------- |
| `['todos', 'list']`      | ✓ (프리픽스) | stale 표시 + **즉시 refetch** (active) |
| `['todos', 'detail', 7]` | ✓ (프리픽스) | stale 표시만 — **다음 마운트 때** 갱신 |
| `['stats']`              | ✗            | 아무 일 없음                           |

- invalidate = "낡았다고 선언" — 갱신 실행 여부는 **화면에 있는가**가 결정한다
- stats도 낡았다면? 그건 invalidate 호출부가 `['stats']`도 지목해야 한다. react-query는 데이터 간 의존성을 모른다

---

## 퀴즈 4 — 리렌더는 몇 번 (2부 Part 4)

```jsx
function TodoCount() {
  const {data} = useQuery({queryKey: ['todos'], queryFn: fetchTodos})
  return <span>{data?.length}</span>
}
```

포커스 계기로 백그라운드 갱신이 일어났고, **응답 내용은 이전과 완전히 같다.** 이 컴포넌트는 몇 번 리렌더될까?

<!-- 진행: 0번/1번/2번 손들기. 0번을 고르는 사람이 거의 없다 — "isFetching이 두 번 바뀌니 2번"이 다수인데, "이 컴포넌트는 isFetching을 읽지 않았다"를 짚으며 반전시킨다. -->

---

## 퀴즈 4 — 정답: 0번

두 겹의 방어가 모두 작동한다.

1. **structural sharing**: 내용이 같으므로 `data`는 **이전과 같은 참조** (실측: `data1 === data2`)
2. **tracked properties**: 이 컴포넌트가 읽은 것은 `data`뿐. 갱신 동안 `isFetching`이 두 번 바뀌었지만 **읽지 않은 필드**라 알림 대상이 아니다

> 만약 `{data, isFetching}`을 구조분해했다면? isFetching의 true/false 전환으로 2번 리렌더된다. 읽는 것에만 비용을 낸다.

---

## 퀴즈 5 — 롤백이 되지 않는다 (1부 Part 3 + 2부 Part 5)

낙관적 업데이트를 구현했다. cancelQueries도 잊지 않았다. 그런데 서버가 실패했을 때 **롤백해도 화면이 원래대로 돌아오지 않는다.**

```js
onMutate: async (toggled) => {
  await queryClient.cancelQueries({queryKey: ['todos']})
  queryClient.setQueryData(['todos'], (old) => toggle(old, toggled.id))
  const snapshot = queryClient.getQueryData(['todos'])
  return {snapshot}
},
onError: (err, toggled, context) => {
  if (context) queryClient.setQueryData(['todos'], context.snapshot)
},
```

무엇이 잘못됐을까?

<!-- 진행: 함정 5 직후라 다들 cancelQueries부터 찾는다 — "이번엔 있다"고 미리 못박고 코드를 줄 단위로 읽게 유도. -->

---

## 퀴즈 5 — 정답: 스냅샷을 뜨는 시점

- 스냅샷을 **캐시를 수정한 뒤에** 떴다 — `snapshot`에 담긴 것은 원본이 아니라 **낙관적으로 수정된 목록**이다
- onError의 롤백은 그 수정본을 다시 쓰는, **아무것도 되돌리지 않는 코드**가 된다
- 순서를 바꾸면 끝: **스냅샷 먼저, 수정은 그 다음**

```js
const snapshot = queryClient.getQueryData(['todos']) // ① 원본 확보
queryClient.setQueryData(['todos'], (old) => toggle(old, toggled.id)) // ② 수정
```

> 낙관적 업데이트의 3박자(스냅샷 → 미리 그리기 → 롤백)는 순서 자체가 계약이다.

---

## 퀴즈 6 — 서버에서 생긴 일 (2부 Part 6)

SSR 프리페치 코드. 배포 후 "가끔 첫 화면이 다른 사람의 장바구니"라는 제보가 왔다.

```js
const queryClient = new QueryClient() // 모듈 스코프

export default async function Page() {
  await queryClient.prefetchQuery({queryKey: ['cart'], queryFn: fetchCart})
  return (
    <HydrationBoundary state={dehydrate(queryClient)}>...</HydrationBoundary>
  )
}
```

무엇이 잘못됐을까?

---

## 퀴즈 6 — 정답: 요청 간 캐시 공유

- 모듈 스코프의 QueryClient는 서버 프로세스의 **모든 요청이 공유**한다
- A의 요청이 `['cart']`를 채운 뒤 B의 요청이 오면, dehydrate에 **A의 장바구니가 실려 나간다**
- 수정: `new QueryClient()`를 **핸들러 안으로** — 요청마다 새 캐시

> 브라우저에서 "앱 전체에 하나"가 정석인 바로 그 패턴이, 서버에서는 개인정보 유출이 된다. 실행 환경이 캐시의 공유 범위를 결정한다.

---

## 전체 요약 — 여섯 문장

1. react-query는 fetch 라이브러리가 아니라 **서버 상태의 캐시 관리자**다
2. **queryKey가 모든 것의 축**이다 — 캐시 주소, 중복 제거, 무효화, hydrate가 전부 키로 연결된다
3. **staleTime이 요청량을, gcTime이 메모리를** 결정한다 — 갱신은 "계기 × stale"일 때만 (폴링은 예외)
4. 데이터의 주인은 **Query 객체**고 컴포넌트는 Observer로 구독한다 — 참조가 같으면, 읽지 않았으면 리렌더도 없다
5. 쓰기 후에는 무효화, 캐시를 직접 만지면 **cancelQueries부터** — 레이스는 내 책임이 된다
6. 서버에서 QueryClient는 **요청마다 새로** — 환경이 바뀌면 기본값도 정석도 바뀐다

---

## 참고 자료

- [TanStack Query 공식 문서](https://tanstack.com/query/v5/docs/framework/react/overview) — v5 기준
- [TanStack/query v5.101.4 소스](https://github.com/TanStack/query/tree/v5.101.4) — 이 덱의 인용 기준 태그
- [Practical React Query](https://tkdodo.eu/blog/practical-react-query) — 메인테이너 TkDodo의 실전 시리즈
- [Advanced Server Rendering](https://tanstack.com/query/v5/docs/framework/react/guides/advanced-ssr) — App Router 통합 공식 가이드
- [Inside React Query](https://tkdodo.eu/blog/inside-react-query) — 내부 구조(Part 4)를 더 깊게

---

# 감사합니다

<!-- _class: invert -->

@yceffort
