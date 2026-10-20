---
title: 'react-query 딥다이브 1부: 사용법과 캐시의 생명주기'
marp: true
paginate: true
theme: midnight
tags:
  - data-fetching
  - caching
  - react
  - async
date: 2026-08-14
description: '서버 상태라는 문제, useQuery의 상태 모델, staleTime과 gcTime — 캐시 항목의 일생을 따라가는 react-query 강의 1부'
published: true
art:
  undraw: database-tables
---

# react-query 딥다이브 1부

사용법과 캐시의 생명주기

<!-- _class: invert deck-cover -->

@yceffort

---

## 이 코드에는 문제가 몇 개 있을까

```jsx
function UserProfile({userId}) {
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setLoading(true)
    fetch(`/api/users/${userId}`)
      .then((res) => res.json())
      .then((data) => setUser(data))
      .finally(() => setLoading(false))
  }, [userId])

  if (loading) return <Spinner />
  return <div>{user.name}</div>
}
```

리뷰에서 한 번쯤 봤을 법한 코드다. 겉보기엔 멀쩡하다.

<!-- 손들기로 진행: 1개? 2개? 3개 이상? 대부분 에러 처리 누락 정도만 찾는다. -->

---

## 최소 네 개

1. **에러 처리가 없다** — 요청이 실패하면 `user`는 영원히 `null`, 스피너만 사라진다
2. **레이스 컨디션** — `userId`가 1 → 2로 바뀌었는데 1의 응답이 늦게 도착하면, 화면에는 1의 데이터가 남는다
3. **응답이 버려진다** — 언마운트 뒤 도착한 데이터는 쓸 곳 없이 사라진다 (React 17까지는 "setState on unmounted" 경고가 뜨던 지점)
4. **캐시가 없다** — 다른 화면에 갔다 돌아올 때마다, 같은 데이터를 쓰는 컴포넌트마다 매번 다시 요청한다

전부 고치면 이 컴포넌트는 수십 줄이 더 필요하다. 그리고 다음 컴포넌트에서 **또 반복**된다.

react-query는 이 반복을 없애기 위해 존재한다.

---

## 이 시리즈에서 다루는 것

**1부 (오늘)**

0. **배경** — 서버 상태는 왜 useState로 관리하면 안 되는가
1. **useQuery 기본기** — queryKey, queryFn, 상태 모델
2. **캐시의 생명주기** — fresh, stale, inactive와 staleTime/gcTime
3. **useMutation과 무효화** — 쓰기 작업 후 캐시를 갱신하는 법

각 파트 끝에는 이해를 확인하는 **중간 점검 퀴즈**가 있다.

**2부**

4. **내부 동작** — QueryClient, Query, QueryObserver의 구독 모델
5. **Suspense와 SSR** — useSuspenseQuery, HydrationBoundary
6. **실전 함정 모음** + **종합 퀴즈**

---

## 기준

- 라이브러리: **TanStack Query v5.101.4** (react-query의 현재 이름, v4부터 개명)
- 동작 설명 중 코드 인용은 전부 `TanStack/query` **v5.101.4 태그** 소스 기준
- 캐시 동작 예제는 **Node.js 24에서 query-core로 직접 실행해 확인**했다
- 다루지 않는 것: useInfiniteQuery(무한 스크롤), 오프라인 지원(networkMode 상세), devtools 사용법

<!-- 구두 보충: v3(react-query 패키지 시절) 자료가 검색에 많이 걸린다. cacheTime이라는 단어가 보이면 v4 이하 자료다(v5에서 gcTime으로 개명). 이 덱과 내용이 다르면 버전부터 의심할 것. -->

---

## Part 0 — 서버 상태라는 문제

<!-- _class: invert -->

---

## 클라이언트 상태 vs 서버 상태

| 구분      | 클라이언트 상태           | 서버 상태                       |
| --------- | ------------------------- | ------------------------------- |
| 예시      | 모달 열림, 입력값, 탭     | 유저 정보, 상품 목록, 주문 내역 |
| 소유자    | 내 앱                     | **서버** (내 것이 아니다)       |
| 최신 보장 | 항상 최신 (내가 바꾸니까) | **모르는 사이 낡는다**          |
| 접근 방식 | 동기                      | 비동기 (지연, 실패 가능)        |
| 공유      | 컴포넌트 트리 안          | 여러 화면, 여러 사용자          |

- `useState`/`useReducer`는 왼쪽 열을 위한 도구다
- 오른쪽 열은 성격이 다른 문제인데, 같은 도구로 풀려다 보니 Part 시작의 그 코드가 나온다

---

## 서버 상태의 본질: 캐시

서버 상태를 화면에 그리는 순간, 그것은 이미 **원본이 아니라 사본**이다.

```text
[ 서버 DB (원본) ] ──── 요청/응답 ────▶ [ 내 화면 (사본) ]
                                          ▲
                              언제 다시 동기화할 것인가?
```

그래서 서버 상태 관리의 진짜 질문은 "어떻게 저장하나"가 아니라:

- 사본을 **언제까지 신선하다고 볼 것인가**
- 낡은 사본을 **언제, 어떤 계기로 다시 동기화할 것인가**
- 안 쓰는 사본을 **언제 버릴 것인가**

react-query는 fetch 라이브러리가 아니라 **이 질문들에 답하는 캐시 관리자**다. 요청 자체는 여전히 fetch/axios로 내가 한다.

---

## react-query가 대신 해주는 것

Part 시작의 문제 네 개를 다시 보면:

| 수동 fetch의 문제     | react-query의 처리                              |
| --------------------- | ----------------------------------------------- |
| 에러 처리 누락        | `status: 'error'`로 상태화, 재시도 내장         |
| 레이스 컨디션         | 이전 응답은 이전 키의 캐시로 격리, 화면 못 덮음 |
| 언마운트 뒤 응답 유실 | 캐시에 저장 — 다음 방문 때 재사용               |
| 캐시 없음, 중복 요청  | 키 기반 캐시 + 동시 요청 **중복 제거**          |

여기에 백그라운드 갱신, 창 포커스 시 동기화, 로딩/에러 상태 표준화까지 따라온다.

이 목록을 외울 필요는 없다. 이 덱이 끝나면 각각이 **캐시 구조에서 당연히 나오는 결과**로 보이게 된다.

---

## Part 1 — useQuery 기본기

<!-- _class: invert -->

---

## 최소 설정

```jsx
import {QueryClient, QueryClientProvider} from '@tanstack/react-query'

const queryClient = new QueryClient()

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <UserProfile userId={1} />
    </QueryClientProvider>
  )
}
```

- `QueryClient`가 캐시(QueryCache)를 들고 있는 **본체**다. 앱 전체에 하나
- Provider는 이 캐시를 컨텍스트로 내려보내는 통로
- 브라우저 앱에서는 모듈 스코프에 하나 만들면 된다 (SSR에서는 다르다 — 2부)

---

## useQuery 첫 예제

```jsx
import {useQuery} from '@tanstack/react-query'

function UserProfile({userId}) {
  const {data, status, error} = useQuery({
    queryKey: ['users', userId],
    queryFn: () => fetchUser(userId),
  })

  if (status === 'pending') return <Spinner />
  if (status === 'error') return <ErrorView error={error} />
  return <div>{data.name}</div>
}
```

- 인트로의 코드가 이걸로 바뀐다. 문제 네 개가 전부 사라진다
- 필수 옵션은 둘: **queryKey**(캐시의 주소)와 **queryFn**(데이터 가져오는 법)
- 이 둘의 규약만 정확히 알면 useQuery의 절반은 끝난다

---

## queryKey — 캐시의 주소

```js
useQuery({queryKey: ['users'], ...})              // 유저 목록
useQuery({queryKey: ['users', userId], ...})      // 유저 상세
useQuery({queryKey: ['users', userId, 'orders'], ...}) // 유저의 주문
```

- 배열이다. 캐시는 이 키를 해시한 문자열을 주소로 항목을 저장한다
- **키가 같으면 같은 캐시**를 본다. 다른 컴포넌트라도, 다른 화면이라도
- **키가 다르면 완전히 다른 항목**이다. `['users', 1]`과 `['users', 2]`는 남남
- 계층적으로 설계하면 나중에 "users로 시작하는 전부 무효화" 같은 **프리픽스 매칭**을 쓸 수 있다 (Part 3)

---

## queryKey 해시의 규칙

키는 내부에서 문자열로 해시된다. 규칙 네 가지 (전부 v5.101.4로 직접 확인):

```js
// ① 객체의 키 순서는 무관 — 같은 캐시
hashKey([{page: 1, filter: 'a'}]) === hashKey([{filter: 'a', page: 1}]) // true

// ② 배열의 순서는 유관 — 다른 캐시
hashKey(['todos', 1]) === hashKey([1, 'todos']) // false

// ③ 값이 undefined인 속성은 없는 것과 같다
hashKey([{a: 1, b: undefined}]) === hashKey([{a: 1}]) // true

// ④ 타입은 구분한다
hashKey(['todos', 1]) === hashKey(['todos', '1']) // false
```

④가 실무에서 제일 아프다. `useParams()`의 id는 **문자열**이고, 다른 곳에서 숫자 id로 키를 만들면 **캐시가 둘로 갈라진다.**

---

## queryFn의 규약 ① — 성공과 실패

queryFn은 **Promise를 반환하는 함수**면 무엇이든 된다. 규약은 하나: **실패는 throw로 알린다.**

```js
queryFn: async () => {
  const res = await fetch(`/api/users/${userId}`)
  if (!res.ok) {
    throw new Error(`요청 실패: ${res.status}`) // 이게 없으면?
  }
  return res.json()
}
```

- `fetch`는 404, 500에도 **reject하지 않는다**. `res.ok` 검사 없이 쓰면 서버가 죽어도 `status: 'success'`다
- axios는 4xx/5xx에서 throw하므로 이 검사가 필요 없다
- throw된 에러가 `status: 'error'`와 `error` 객체가 된다

---

## queryFn의 규약 ② — undefined 반환 금지

```js
queryFn: async () => {
  const res = await fetch('/api/config')
  const json = await res.json()
  return json.data // json.data가 없으면?
}
```

```text
Error: ["config"] data is undefined     ← 실제 throw되는 에러
// dev 모드 콘솔에는 안내가 함께 찍힌다:
// Query data cannot be undefined. Please make sure to return
// a value other than undefined from your query function. ...
```

- `undefined`를 반환하면 성공이 아니라 **에러로 처리된다** (실행해 확인. 친절한 안내는 dev 전용이고 프로덕션에는 위의 짧은 에러만 남는다)
- "데이터 없음"을 표현하려면 `null`을 반환할 것
- 캐시 입장에서 `undefined`는 "항목 없음"과 구분이 안 되기 때문에 스펙으로 금지했다

---

## 상태 모델 ① — status

쿼리의 **데이터 관점** 상태는 셋뿐이다.

```text
            성공 ──▶  success  (data 있음)
pending ──┤
            실패 ──▶  error    (error 있음)
```

| status      | 의미                           | 보조 불리언 |
| ----------- | ------------------------------ | ----------- |
| `'pending'` | 아직 **데이터가 한 번도 없다** | `isPending` |
| `'success'` | 데이터가 있다                  | `isSuccess` |
| `'error'`   | 시도가 실패했다                | `isError`   |

주의: `pending`은 "로딩 중"이 아니라 "**보여줄 데이터가 없다**"는 뜻이다. 로딩 여부는 다음 슬라이드의 다른 축이 담당한다.

---

## 상태 모델 ② — fetchStatus, 두 번째 축

**네트워크 활동 관점**의 축이 따로 있다: `fetchStatus` — `'fetching' | 'paused' | 'idle'`

|                     | fetching               | idle                          |
| ------------------- | ---------------------- | ----------------------------- |
| **status: pending** | 첫 로딩 중 (스피너)    | 시작 안 함 (`enabled: false`) |
| **status: success** | **백그라운드 갱신 중** | 평온한 상태                   |
| **status: error**   | 실패 후 재시도 중      | 실패로 끝남                   |

- 두 축이 분리된 이유: react-query에서는 "데이터가 있으면서 동시에 요청 중"인 상태가 **정상 운영 상태**이기 때문 (Part 2의 백그라운드 갱신)
- `paused`는 오프라인일 때. 이 덱에서는 다루지 않는다

---

## isPending, isLoading, isFetching

셋 다 "로딩"처럼 들리지만 전부 다르다.

```text
isPending  = status === 'pending'        (보여줄 데이터가 없다)
isFetching = fetchStatus === 'fetching'  (지금 요청이 날아가고 있다)
isLoading  = isPending && isFetching     (데이터도 없는데 첫 요청 중)
```

실무 기준:

- **첫 진입 스피너** → `isPending` (또는 `isLoading`)
- **갱신 중 인디케이터** (데이터는 이미 떠 있음) → `isFetching`
- 백그라운드 갱신마다 화면 전체를 스피너로 덮고 있다면, `isFetching`을 스피너 조건으로 쓴 것이다

<!-- 구두 보충: v4의 isLoading이 v5에서 isPending으로 개명됐고, v5의 isLoading은 위처럼 재정의됐다. 구버전 코드 마이그레이션 시 주의. -->

---

## 파라미터는 queryKey에 넣는다

```jsx
function OrderList({page, filter}) {
  const {data} = useQuery({
    queryKey: ['orders', {page, filter}], // 의존성이 곧 키
    queryFn: () => fetchOrders(page, filter),
  })
}
```

- `page`가 1 → 2로 바뀌면 **다른 캐시 항목**을 보게 된다. 같은 항목을 수동으로 다시 요청(refetch)하는 게 아니다
- 2페이지가 처음이면 요청하고, 이미 캐시에 있으면 **즉시 그 데이터**부터 보여준다
- 1페이지로 돌아가면? 캐시가 살아 있으므로 **요청 없이 즉시** 뜬다
- `useEffect`의 의존성 배열과 같은 원리다: **queryFn이 참조하는 값은 전부 queryKey에**

이 규칙 하나로 인트로의 레이스 컨디션이 사라진다. 키가 바뀌는 순간 이전 키의 응답은 이전 캐시 항목으로 갈 뿐, 현재 화면을 덮어쓰지 못한다.

---

## enabled — 조건부 실행과 종속 쿼리

```jsx
const {data: user} = useQuery({
  queryKey: ['users', email],
  queryFn: () => fetchUserByEmail(email),
})

const userId = user?.id

const {data: orders} = useQuery({
  queryKey: ['orders', userId],
  queryFn: () => fetchOrders(userId),
  enabled: userId !== undefined, // user가 오기 전엔 시작하지 않는다
})
```

- `enabled: false`인 동안 queryFn은 **호출되지 않는다** (`status: 'pending'` + `fetchStatus: 'idle'`)
- 훅 자체를 if문으로 감싸는 것은 훅 규칙 위반이다. **조건은 enabled로** 표현한다
- 뒤 쿼리가 앞 쿼리의 결과를 필요로 하는 "종속 쿼리"의 표준 패턴

---

## Part 1 정리

- 필수는 둘: **queryKey**(캐시 주소)와 **queryFn**(가져오는 법, 실패는 throw)
- 키 해시: 객체 키 순서 무관, 배열 순서 유관, `undefined` 무시, **타입 구분** (`1` ≠ `'1'`)
- 상태는 두 축: **status**(데이터가 있나) × **fetchStatus**(요청 중인가)
- 첫 스피너는 `isPending`, 갱신 표시는 `isFetching`
- **queryFn이 쓰는 값은 전부 queryKey에** — 파라미터 변경은 refetch가 아니라 캐시 항목 이동
- 조건부 실행은 훅을 감싸지 말고 `enabled`

---

## 중간 점검 ① — 서버가 500을 내려줬다

```js
useQuery({
  queryKey: ['config'],
  queryFn: async () => {
    const res = await fetch('/api/config') // 서버가 500을 반환
    return res.json()
  },
})
```

서버가 500과 함께 `{"message": "Internal Server Error"}`를 응답했다. 이 쿼리의 `status`는 무엇일까?

<!-- 진행: pending / error / success 손들기. error가 다수인 편 — fetch의 reject 조건을 되물으며 반전. -->

---

## 중간 점검 ① — 정답: success

- `fetch`는 404, 500에도 **reject하지 않는다** — reject는 네트워크 단절 수준에서만
- `res.json()`도 성공한다 (에러 응답도 유효한 JSON이므로)
- queryFn이 throw하지 않았으니 `status: 'success'`, `data`는 `{message: 'Internal Server Error'}`
- 화면은 에러 객체를 정상 데이터처럼 그리다 깨지고, 에러 분기는 끝까지 침묵한다

> 실패는 throw로 알린다 — fetch를 쓴다면 `res.ok` 검사는 선택이 아니라 규약의 일부다.

---

## 중간 점검 ② — 두 축으로 읽기

```jsx
const {status, fetchStatus, isLoading} = useQuery({
  queryKey: ['orders', userId],
  queryFn: () => fetchOrders(userId),
  enabled: false,
})
```

이 쿼리의 `status`, `fetchStatus`, `isLoading`은 각각 무엇일까?

---

## 중간 점검 ② — 정답: pending / idle / false

- 데이터가 한 번도 없었으니 `status: 'pending'`
- 요청이 날아가고 있지 않으니 `fetchStatus: 'idle'`
- `isLoading = isPending && isFetching`이므로 **false**
- "pending = 로딩 중"으로 외웠다면 여기서 무너진다 — pending은 "보여줄 데이터가 없다"일 뿐, 로딩 여부는 fetchStatus의 몫

> 스피너를 isPending에만 걸면 enabled: false인 화면은 영원히 스피너다. isLoading이 이 경우를 걸러낸다.

---

## Part 2 — 캐시의 생명주기

<!-- _class: invert -->

여기가 react-query의 심장이다. staleTime과 gcTime을 설명할 수 있으면 이 라이브러리를 아는 것이다.

---

## 캐시 항목의 일생

```text
 요청 ──▶ [ fresh ] ──staleTime 경과──▶ [ stale ]
             신선함                        낡음
                                            │
        구독자(사용하는 컴포넌트)가 0이 되면  ▼
 삭제 ◀──gcTime 경과── [ inactive ] ◀──────┘
```

캐시 항목 하나는 네 단계를 지난다.

1. **fresh** — 신선하다. 이 데이터를 그대로 믿는다
2. **stale** — 낡았다. 여전히 보여주되, 기회가 오면 다시 받아온다
3. **inactive** — 아무 컴포넌트도 안 쓴다. 화면에 없다
4. **삭제** — 메모리에서 제거

단계 전환을 결정하는 두 개의 시계가 **staleTime**과 **gcTime**이다.

---

## staleTime — 신선함의 유효기간

**fresh인 동안은 자동 계기로는 요청이 발생하지 않는다.** 캐시를 그냥 쓴다. (명시적 명령인 invalidate·refetch는 예외 — Part 3)

```js
useQuery({
  queryKey: ['user'],
  queryFn: fetchUser,
  staleTime: 60_000, // 1분 동안 fresh
})
```

```js
// Node 실측 — fetchQuery는 컴포넌트 없이 쿼리를 실행하는 코어 API
await client.fetchQuery(opts) // 요청 1회
await client.fetchQuery(opts) // fresh → queryFn 호출 없음, 캐시 반환
// queryFn 호출 횟수: 1
```

- 같은 키를 쓰는 컴포넌트가 열 개 마운트되어도, fresh라면 **요청 0회**
- staleTime이 지나도 데이터가 사라지는 게 아니다. "**낡음**" 딱지가 붙을 뿐이다

---

## stale이 되면: 보여주고, 뒤에서 갱신한다

stale 캐시를 만나면 react-query는 둘 중 하나를 고르지 않는다. **둘 다 한다.**

```text
[ 컴포넌트 마운트 ]
       │
       ├──▶ ① 캐시의 낡은 데이터를 즉시 보여준다   (빈 화면 없음)
       │
       └──▶ ② 백그라운드에서 다시 요청한다         (isFetching: true)
                     │
                     └──▶ ③ 도착하면 조용히 교체    (최신화)
```

- HTTP 캐싱의 **stale-while-revalidate**와 같은 전략
- 사용자는 항상 "일단 뭔가"를 보고, 데이터는 곧 최신이 된다
- Part 1에서 `status: 'success'` + `fetchStatus: 'fetching'` 조합이 정상 운영 상태라고 한 이유

---

## 그럼 갱신은 정확히 언제 일어나는가

stale 딱지는 그 자체로는 아무것도 하지 않는다. **계기**가 와야 한다. 계기는 네 가지다.

| 계기                              | 옵션                   | 기본값 |
| --------------------------------- | ---------------------- | ------ |
| 새 구독자 마운트                  | `refetchOnMount`       | true   |
| 탭이 다시 보임 (visibilitychange) | `refetchOnWindowFocus` | true   |
| 네트워크 재연결                   | `refetchOnReconnect`   | true   |
| 주기적 갱신 (폴링)                | `refetchInterval`      | 없음   |

판정 핵심 한 줄 (`queryObserver.ts`의 `shouldFetchOn` 발췌, 2부에서 상세):

```js
value === 'always' || (value !== false && isStale(query, options))
```

mount/focus/reconnect는 **stale일 때만** 갱신한다 (`'always'` 설정 시 예외). 단, **`refetchInterval`은 이 판정을 거치지 않는다** — 폴링은 fresh여도 돈다.

---

## "포커스만 바꿔도 API가 호출돼요"

주니어가 react-query를 도입하고 가장 먼저 놀라는 지점.

- **staleTime의 기본값은 0**이다 (`query.ts`의 `isStaleByTime(staleTime = 0)`)
- 즉 기본 설정에서 모든 데이터는 **도착하는 순간 이미 stale**
- 그래서 탭을 전환했다 돌아올 때마다(`refetchOnWindowFocus`) 요청이 나간다

이것은 버그가 아니라 **의도된 기본값**이다: "낡은 데이터를 보여줄 바엔 요청을 한 번 더 한다."

```js
// 데이터 성격에 맞게 staleTime을 선언하는 것이 올바른 대응
staleTime: 0 // 주문 상태, 재고 — 항상 최신이어야 함
staleTime: 60_000 // 목록, 프로필 — 1분쯤 낡아도 됨
staleTime: Infinity // 국가 코드, 설정값 — 세션 내 불변
```

`refetchOnWindowFocus: false`로 계기를 끄는 것은 증상 치료다. "이 데이터는 얼마나 낡아도 되는가"에 답하는 staleTime이 원인 치료.

<!-- 진행: "겪어본 사람?" 손들기 → "refetchOnWindowFocus: false로 껐던 사람?" → 그게 왜 증상 치료인지(다른 계기들은 그대로 남는다, 진짜 질문은 신선도)로 연결하면 토론이 잘 붙는 슬라이드. -->

---

## gcTime — 안 쓰는 캐시의 수명

구독자가 0이 된(inactive) 항목은 gcTime이 지나면 **메모리에서 삭제**된다.

```text
[ 목록 화면 ]──이동──▶[ 상세 화면 ]──5분 내 복귀──▶[ 목록 화면 ]
 ['orders'] 구독        구독자 0                    캐시 생존: 즉시 표시
                        (inactive, GC 카운트다운)    + stale이면 백그라운드 갱신
```

- 기본값: **5분** — 소스(`removable.ts`): `newGcTime ?? (environmentManager.isServer() ? Infinity : 5 * 60 * 1000)`
- Node 실측: 구독 해제 후 gcTime 경과 → `getQueryData()`가 `undefined`
- gcTime이 지나 삭제된 뒤 돌아오면? 빈손에서 시작하므로 **스피너부터** 다시 본다
- 구독자가 하나라도 있는 한 GC 대상이 아니다. 구독자들이 서로 다른 gcTime을 주면 `Math.max`로 **가장 긴 값**이 적용된다 (같은 소스의 바로 윗줄)

---

## staleTime vs gcTime — 한 장 비교

| 구분        | staleTime                       | gcTime                              |
| ----------- | ------------------------------- | ----------------------------------- |
| 결정하는 것 | 언제부터 **낡은 취급**할 것인가 | 안 쓰는 캐시를 언제 **버릴** 것인가 |
| 시계 시작   | 데이터 **도착** 시점            | 구독자가 **0이 된** 시점            |
| 지나면      | 계기가 올 때 백그라운드 갱신    | 캐시 삭제 (다음엔 스피너)           |
| 기본값      | **0** (즉시 낡음)               | **5분**                             |
| 관장하는 것 | **요청 횟수** (신선도 정책)     | **메모리** (보관 정책)              |

- 서로 독립적인 두 시계다. "staleTime < gcTime이어야 한다" 같은 제약은 없다
- 다만 staleTime > gcTime이면, 낡기도 전에 캐시가 사라질 수 있으니 보통은 staleTime ≤ gcTime으로 둔다

---

## 시나리오로 확인

`staleTime: 60초, gcTime: 5분`인 주문 목록. 사용자의 7분 30초:

```text
00:00  목록 진입           → 요청, 데이터 도착 (fresh)
00:30  상세 갔다가 복귀     → fresh → 요청 없음, 즉시 표시
01:30  다른 탭 갔다가 복귀  → stale + 포커스 계기
                           → 캐시 즉시 표시 + 백그라운드 갱신
02:00  상세 화면으로 이동   → 구독자 0 (inactive, GC 카운트다운 시작)
07:30  목록으로 복귀        → gcTime(5분, 07:00 만료) 경과, 캐시 삭제됨
                           → 스피너 + 새 요청
```

- 사용자가 빈 화면을 본 것은 처음과 마지막뿐
- 요청은 5번의 화면 전환 동안 3번만 나갔다

---

## 요청 중복 제거 — 캐시 구조의 보너스

같은 키를 쓰는 컴포넌트 셋이 **동시에** 마운트되면?

```js
// Node 실측: 동시에 세 번 요청해도
await Promise.all([
  client.fetchQuery(opts),
  client.fetchQuery(opts),
  client.fetchQuery(opts),
])
// queryFn 호출 횟수: 1
```

- 캐시 항목 하나에 요청 상태가 있으므로, **이미 진행 중인 요청이 있으면 거기에 합류**한다
- staleTime이 0이어도 그렇다 (중복 제거는 신선도와 무관한, 동시성 차원의 동작)
- 헤더, 사이드바, 본문이 제각각 `['me']` 키를 구독해도 네트워크에는 **한 번**

컴포넌트는 "이 데이터가 필요하다"만 선언하고, 요청을 누가 언제 보낼지는 캐시가 조율한다.

---

## retry — 실패했을 때

실패한 queryFn은 자동으로 재시도된다.

- 기본 **3회**, 지수 백오프(1초 → 2초 → 4초, 최대 30초)
- `status: 'error'`는 **모든 재시도가 소진된 뒤**에야 된다 — "에러가 늦게 뜨는" 이유
- 소스(`retryer.ts` v5.101.4)의 기본값에는 조건이 하나 숨어 있다:

```ts
const retry = config.retry ?? (environmentManager.isServer() ? 0 : 3)
```

- **서버에서는 0회**다. SSR 중 3회 재시도는 응답 지연으로 직결되기 때문 (2부에서 다시)
- 4xx는 재시도해도 결과가 같으므로 함수형으로 걸러내는 것이 실무 패턴:

```js
// 주의: error.status는 queryFn에서 throw할 때 직접 실어야 존재한다
retry: (failureCount, error) => error.status >= 500 && failureCount < 3
```

---

## placeholderData vs initialData

첫 로딩의 스피너를 없애는 두 옵션. **캐시에 들어가느냐**가 가른다 (Node 실측).

| 구분        | placeholderData                   | initialData                      |
| ----------- | --------------------------------- | -------------------------------- |
| 캐시 기록   | **안 됨** (관측용 가짜)           | **됨** (진짜 데이터 취급)        |
| status      | `success`                         | `success`                        |
| 요청        | 나감 (데이터 없는 pending이므로)  | staleTime 내면 **안 나감**       |
| 구분 플래그 | `isPlaceholderData: true`         | 없음 (진짜와 구분 불가)          |
| 용도        | 스켈레톤 대체, 목록→상세 미리보기 | 이미 확보한 **진짜** 데이터 주입 |

- 기준: 그 데이터가 **진짜인가?** 진짜(다른 쿼리 응답의 일부, SSR로 받은 값)면 initialData, 그럴싸한 모양만 있으면 placeholderData
- initialData에 가짜를 넣으면 staleTime 동안 **가짜가 진짜 행세**를 한다

---

## Part 2 정리

- 캐시 항목의 일생: **fresh → stale → inactive → 삭제**
- **staleTime**(기본 0): fresh인 동안 자동 계기로는 요청이 없다. 요청량을 결정하는 다이얼
- stale이면 **캐시를 먼저 보여주고 백그라운드에서 갱신** (stale-while-revalidate)
- 갱신의 계기는 mount / focus / reconnect — 단, **stale일 때만** (폴링 `refetchInterval`은 예외, fresh여도 돈다)
- **gcTime**(기본 5분): 구독자 0이 된 뒤의 보관 기간. 메모리를 결정
- 동시 요청은 **자동 중복 제거**, 실패는 3회 재시도(서버는 0회), 에러는 재시도 소진 후에야 표시

---

## 중간 점검 ① — 뒤집힌 두 시계

`staleTime: 5분, gcTime: 1분`으로 설정된 목록 쿼리가 있다.

```text
00:00  목록 진입          → 요청, 데이터 도착 (fresh)
00:30  상세 화면으로 이동  → 구독자 0
02:30  목록으로 복귀       → ?
```

02:30 복귀 시점에 무슨 일이 일어날까? staleTime 기준으로 데이터는 아직 fresh다(5분 안 지남).

<!-- 진행: "즉시 표시 / 스피너" 손들기. "fresh니까 즉시"가 다수 — 두 시계의 독립성으로 반전. -->

---

## 중간 점검 ① — 정답: 스피너 + 새 요청

- 두 시계는 **서로 독립**이다 — staleTime은 신선도, gcTime은 보관
- 구독자가 0이 된 00:30부터 gcTime 1분 카운트다운 → **01:30에 캐시 삭제**
- "아직 fresh였다"는 사실은 도움이 안 된다. 신선하고 말고 할 **캐시 자체가 없다**
- 02:30 복귀는 빈손 시작 → 스피너 + 새 요청

> staleTime > gcTime이 금지는 아니지만, 낡기도 전에 버려지는 설정이다. 보통 staleTime ≤ gcTime으로 둔다.

---

## 중간 점검 ② — 폴링과 신선도

```js
useQuery({
  queryKey: ['stock', itemId],
  queryFn: fetchStock,
  staleTime: Infinity, // 절대 낡지 않는다
  refetchInterval: 5_000, // 5초 폴링
})
```

staleTime이 Infinity면 데이터는 영원히 fresh다. 5초 폴링은 돌까, 멈출까?

---

## 중간 점검 ② — 정답: 돈다

- mount/focus/reconnect 계기는 `shouldFetchOn` 판정을 거친다 — **stale일 때만** 갱신
- **`refetchInterval`은 이 판정을 거치지 않는다** (실측 확인) — fresh여도 인터벌마다 요청이 나간다
- 그래서 이 조합은 모순이 아니라 유효한 선언이다: "자동 계기로는 갱신하지 말고, 5초마다는 무조건 받아라"

> 네 가지 계기 중 폴링만 신선도와 무관하다. "갱신 = 계기 × stale" 공식의 유일한 예외.

---

## Part 3 — useMutation과 캐시 무효화

<!-- _class: invert -->

---

## 읽기가 아닌 것: useMutation

생성/수정/삭제는 useQuery를 쓰지 않는다.

```jsx
const mutation = useMutation({
  mutationFn: (newTodo) => api.post('/todos', newTodo),
})

<button
  disabled={mutation.isPending}
  onClick={() => mutation.mutate({title: '우유 사기'})}
>
  {mutation.isPending ? '추가 중...' : '추가'}
</button>
```

- useQuery는 마운트되면 **알아서** 실행된다. mutation은 **`mutate()`를 불러야** 실행된다
- 캐시도, 재시도도, 중복 제거도 기본으로는 없다 — 쓰기 작업은 두 번 실행되면 사고이기 때문에 전부 반대가 기본값이다

---

## mutation은 캐시를 건드리지 않는다

```jsx
const mutation = useMutation({mutationFn: addTodo})
mutation.mutate(newTodo)
// 서버에는 저장됐다. 그런데 화면의 목록에는 안 보인다.
```

- `['todos']` 캐시는 mutation이 일어난 걸 **모른다**. 화면은 여전히 옛 목록
- react-query는 mutationFn 안의 POST가 어떤 쿼리를 낡게 만드는지 **알 방법이 없다**
- 그 연결은 내가 선언해야 한다. 도구가 **invalidateQueries**

```jsx
const queryClient = useQueryClient()
const mutation = useMutation({
  mutationFn: addTodo,
  onSuccess: () => {
    queryClient.invalidateQueries({queryKey: ['todos']})
  },
})
```

---

## invalidateQueries가 실제로 하는 일

이름과 달리 캐시를 지우지 않는다. **두 가지**를 한다 (Node 실측):

1. 매칭되는 캐시를 **즉시 stale로** 만든다 — staleTime이 60초 남았어도 무시
2. 그중 **화면에 있는(active) 쿼리는 즉시 refetch**한다

```js
// 실측: staleTime 60초짜리 active 쿼리
await client.invalidateQueries({queryKey: ['active']})
// → queryFn이 즉시 다시 호출됐다 (호출 수 1 → 2)

// inactive 쿼리는 stale 표시만 되고 refetch는 없다
// → 다음에 화면에 등장할 때 갱신된다
```

- 화면에 있는 데이터는 바로 최신화, 화면에 없는 데이터는 **필요해질 때** 최신화
- 낭비 없이 전부 갱신되는, 생명주기 모델과 정확히 맞물린 설계다

---

## 프리픽스 매칭 — 키 설계가 여기서 회수된다

`invalidateQueries`는 키를 **앞에서부터** 매칭한다 (Node 실측):

```js
// 캐시에 세 항목이 있을 때
;['todos', 'list'] // ①
;['todos', 'detail', 1] // ②
;['users'] // ③

client.invalidateQueries({queryKey: ['todos']})
// → ①② stale, ③은 그대로
```

- 할 일을 추가하면 목록도, 상세도, 통계도 낡는다 — `['todos']` 한 방이면 전부 커버
- Part 1에서 "키를 계층적으로 설계하라"고 한 이유
- 정확히 한 항목만 집으려면 `exact: true`

---

## 콜백의 위치 — useMutation vs mutate

콜백은 두 곳에 걸 수 있고, 역할이 다르다.

```jsx
const mutation = useMutation({
  mutationFn: addTodo,
  // ① 정의부: 이 mutation의 본질적 후처리 (항상 실행됨)
  onSuccess: () => queryClient.invalidateQueries({queryKey: ['todos']}),
})

mutation.mutate(newTodo, {
  // ② 호출부: 이 호출 한정의 UI 후처리
  onSuccess: () => navigate('/todos'),
})
```

- 실행 순서는 ① → ②
- 주의: ②는 **컴포넌트가 언마운트되면 실행되지 않는다**. 캐시 무효화처럼 반드시 일어나야 하는 일은 ①에, 화면 전환·토스트처럼 그 화면에서만 의미 있는 일은 ②에

---

## 낙관적 업데이트 — 응답을 기다리지 않기

체크박스 토글마다 서버 응답까지 기다리면 UI가 굼뜨다. **성공을 가정하고 먼저 그린다.**

```text
[ 클릭 ] ──▶ 캐시를 즉시 수정 (화면 반영) ──▶ 서버 요청
                                                │
                             성공 ──▶ 무효화로 서버 진실과 동기화
                             실패 ──▶ 수정 전 스냅샷으로 롤백
```

세 단계가 전부다:

1. **onMutate**: 진행 중인 refetch를 멈추고(`cancelQueries`), 현재 캐시를 **스냅샷**으로 떠 둔 뒤, `setQueryData`로 캐시를 미리 수정
2. **onError**: 스냅샷으로 **롤백**
3. **onSettled**: 성공이든 실패든 **무효화**(invalidateQueries)로 서버 진실과 최종 동기화

---

## 낙관적 업데이트 — 코드

<!-- prettier-ignore -->
```jsx
useMutation({
  mutationFn: toggleTodo,
  onMutate: async (toggled) => {
    await queryClient.cancelQueries({queryKey: ['todos']}) // 진행 중 refetch 차단
    const snapshot = queryClient.getQueryData(['todos'])   // 롤백용 스냅샷
    queryClient.setQueryData(['todos'], (old) =>
      old.map((t) => (t.id === toggled.id ? {...t, done: !t.done} : t)),
    )
    return {snapshot} // context로 전달된다
  },
  onError: (err, toggled, context) => {
    if (context) queryClient.setQueryData(['todos'], context.snapshot) // 롤백
  },
  onSettled: () =>
    queryClient.invalidateQueries({queryKey: ['todos']}), // 최종 동기화
})
```

`setQueryData`의 갱신 함수는 **불변으로** 새 배열을 만들어야 한다. 캐시를 직접 mutate하면 리렌더가 누락된다.

---

## Part 3 정리

- 쓰기는 **useMutation** — 자동 실행도, 캐시도, 재시도도 없는 것이 기본값 (쓰기는 두 번이면 사고)
- mutation과 쿼리 캐시의 연결은 자동이 아니다 — **onSuccess에서 invalidateQueries**
- invalidate = **stale 표시 + active만 즉시 refetch**. inactive는 다음 등장 때
- 키는 **프리픽스로 매칭**된다 — 계층적 키 설계가 여기서 회수된다
- 필수 후처리는 **useMutation 쪽** 콜백에 (mutate 쪽은 언마운트 시 증발)
- 낙관적 업데이트 3박자: **스냅샷 → 미리 그리기 → 실패 시 롤백**, 마무리는 무효화

---

## 중간 점검 ① — 무효화가 증발한다

```jsx
const mutation = useMutation({mutationFn: addTodo})

mutation.mutate(newTodo, {
  onSuccess: () => {
    queryClient.invalidateQueries({queryKey: ['todos']})
    navigate('/todos')
  },
})
```

저장 버튼을 누르자마자 사용자가 뒤로가기로 이 화면을 떠났다. 서버 저장은 성공했다. `['todos']` 무효화는 실행될까?

---

## 중간 점검 ① — 정답: 실행되지 않는다

- `mutate()` **호출부의 콜백은 컴포넌트가 언마운트되면 실행되지 않는다**
- 무효화가 증발했으니 목록 화면은 stale 표시조차 안 된 옛 캐시를 그대로 보여준다
- 반드시 일어나야 하는 후처리는 **useMutation 정의부에**, 그 화면에서만 의미 있는 후처리는 호출부에

<!-- prettier-ignore -->
```jsx
const mutation = useMutation({
  mutationFn: addTodo,
  onSuccess: () =>
    queryClient.invalidateQueries({queryKey: ['todos']}), // 항상 실행
})
mutation.mutate(newTodo, {onSuccess: () => navigate('/todos')}) // 이 화면 한정
```

---

## 중간 점검 ② — 캐시는 바뀌었는데 화면이 그대로

```js
queryClient.setQueryData(['todos'], (old) => {
  old.push(newTodo) // 캐시 배열에 직접 추가
  return old
})
```

`getQueryData`로 확인하면 새 항목이 분명히 들어 있다. 그런데 화면의 목록에는 나타나지 않는다. 왜일까?

---

## 중간 점검 ② — 정답: 참조가 그대로다

- `old.push()`는 배열을 직접 수정한다 — 반환된 것은 **이전과 같은 참조**
- react-query는 **참조 비교로 변화를 감지**한다. 같은 참조 = "바뀐 게 없다" → 구독자에게 알림이 가지 않는다
- 갱신 함수는 불변으로 새 배열을 만들어야 한다:

```js
queryClient.setQueryData(['todos'], (old) => [...old, newTodo]) // 새 배열
```

> "참조가 같으면 같은 데이터"는 react-query의 리렌더 최적화가 깔고 있는 전제다 — 왜 그런지는 2부 Part 4(structural sharing)에서 회수한다.

---

## 1부 전체 요약

1. 서버 상태는 내 것이 아닌 **원본의 사본**이다 — 문제는 저장이 아니라 **동기화 시점**
2. **queryKey가 캐시의 주소** — queryFn이 쓰는 값은 전부 키에 넣는다
3. 상태는 두 축 — status(데이터가 있나) × fetchStatus(요청 중인가)
4. **staleTime이 요청량을, gcTime이 메모리를** 결정한다 — 기본값은 0과 5분
5. stale 캐시는 **먼저 보여주고 뒤에서 갱신**된다 — 계기는 mount/focus/reconnect
6. 쓰기 후에는 **invalidateQueries** — stale 표시 + 화면에 있는 것만 즉시 갱신

---

## 2부 예고

1부는 "어떻게 쓰는가"였다. 2부는 "어떻게 돌아가는가".

- useQuery를 호출하면 내부에서 무슨 일이 벌어지는가 — QueryClient, QueryCache, Query, QueryObserver
- 데이터가 그대로면 리렌더도 없다 — structural sharing과 tracked properties, 그리고 useSuspenseQuery
- 실전에서 자주 틀리는 패턴 여섯 — 구조를 알면 왜 틀렸는지 보인다
- **SSR** — 서버 프리페치와 HydrationBoundary, 그리고 **종합 퀴즈**

---

## 참고 자료

- [TanStack Query 공식 문서](https://tanstack.com/query/v5/docs/framework/react/overview) — v5 기준
- [TanStack/query v5.101.4 소스](https://github.com/TanStack/query/tree/v5.101.4) — 이 덱의 인용 기준 태그
- [Practical React Query](https://tkdodo.eu/blog/practical-react-query) — 메인테이너 TkDodo의 실전 시리즈
- [Caching Examples](https://tanstack.com/query/v5/docs/framework/react/guides/caching) — 공식 문서의 생명주기 예제

---

# 1부 끝

<!-- _class: invert -->

2부에서 내부로 들어간다.

@yceffort
