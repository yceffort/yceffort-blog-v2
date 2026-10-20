---
title: useEffect — 안 쓰는 게 최선이다
marp: true
paginate: true
theme: yceffort
tags:
  - react
  - async
date: 2026-04-18
description: '주니어를 졸업하려는 개발자를 위한 useEffect 딥다이브 — 안티패턴, race condition, stale closure, event vs effect, tearing, Suspense까지'
published: true
art:
  undraw: buggy-code
---

# useEffect — 안 쓰는 게 최선이다

안티패턴, race condition, stale closure — 그리고 그 너머

<!-- _class: invert deck-cover -->

@yceffort

---

## 이 강의에서 다루는 것

1. useEffect의 원래 목적
2. **안 써야 하는 경우** — 안티패턴
3. **Race Condition** — 기본 + optimistic update
4. **Stale Closure** — Reactive vs Latest 값
5. **Event vs Effect** — useEffectEvent의 철학
6. **실행 타이밍 & Tearing**
7. **Suspense · `use()` · 커스텀 훅 설계**

---

## 이 강의의 메시지

> `useEffect`는 **탈출구**(escape hatch).

React의 기본 모델은 "state/props를 선언하면 화면이 따라온다"는 선언적 세계. `useEffect`는 이 세계를 벗어나 **명령형으로 외부(DOM, 네트워크, 타이머…)를 건드려야 할 때** 쓰는 비상구다.

필요 없는데 쓰고 있다면 **원인 불명의 버그**가 숨어 있을 가능성이 높다.

그게 이 강의가 다루는 모든 것들이다.

---

## Part 0 — useEffect란 무엇인가

---

## 공식 정의

> **Effects let you synchronize a component with some external system.**

"외부 시스템":

- 브라우저 API (DOM, timer, observer)
- 네트워크 (fetch, WebSocket)
- 서드파티 라이브러리 (차트, 지도)

> 한 줄: **React 바깥과 맞추는 것.**

---

## 실행 타이밍 (기본)

```text
1. render  — 컴포넌트 함수 실행, 가상 트리 계산 (순수해야 함)
2. commit  — React가 실제 DOM에 반영
3. paint   — 브라우저가 화면에 그림
4. effect  — useEffect 콜백 실행
```

- **render는 순수해야 해서 사이드이펙트 금지.** 외부 동기화는 commit 뒤(= effect)로 미뤄진다.
- Effect 안에서 state를 바꾸면 **리렌더가 한 번 더** 일어난다. (세부 타이밍은 Part 5)

---

## 동기화 사이클

```jsx
useEffect(() => {
  const sub = subscribe() // 시작
  return () => sub.unsubscribe() // 정리
}, [deps])
```

- 처음 마운트
- deps가 바뀔 때마다 **(정리 → 시작)**
- 언마운트 시 정리

**시작과 정리가 짝을 이룬다** — 이게 Effect의 정신.

---

## Part 1 — 안 쓰는 게 최선이다

---

## 안티패턴 4가지

| 상황               | 대안                     |
| ------------------ | ------------------------ |
| props → state 복사 | props 그대로 사용        |
| 파생 state 계산    | 렌더 중 계산 / `useMemo` |
| 이벤트 반응        | 이벤트 핸들러            |
| 자식 → 부모 동기화 | state lifting            |

공통 원칙:

- **계산 가능한 값은 state가 아니다**
- **"사용자가 X했기 때문에"라면 Effect가 아니라 핸들러**

---

## 대표 사례: 파생 state

```jsx
// ❌ effect로 갱신 — 렌더가 한 번 더
function TodoList({todos, filter}) {
  const [visible, setVisible] = useState([])
  useEffect(() => {
    setVisible(todos.filter((t) => t.status === filter))
  }, [todos, filter])
  return <List items={visible} />
}

// ✅ 렌더 중 계산
function TodoList({todos, filter}) {
  const visible = todos.filter((t) => t.status === filter)
  return <List items={visible} />
}
```

---

## 대표 사례: 이벤트 반응

```jsx
// ❌ isBought state → effect로 알림
useEffect(() => {
  if (isBought) {
    showNotification(`${product.name} 구매 완료!`)
  }
}, [isBought])

// ✅ 핸들러에서 바로
const handleBuy = () => {
  showNotification(`${product.name} 구매 완료!`)
}
```

**"왜 이 알림이 떴는가"** 를 코드만 봐도 알 수 있어야 한다.

---

## 판별 플로우차트

```text
렌더 중 계산 가능?     → 렌더 중 (Effect ❌)
사용자 액션 반응?       → 이벤트 핸들러 (Effect ❌)
prop으로 state 동기화?  → state lifting (Effect ❌)
React 바깥과 동기화?    → useEffect ✅
```

이 플로우를 통과해야 비로소 Effect의 정당한 사용이다.

---

## 그럼에도 써야 하는 경우

| 상황                           | Effect? | 비고                                     |
| ------------------------------ | ------- | ---------------------------------------- |
| 브라우저 API (title, observer) | ✅      | cleanup 필수                             |
| 서드파티 라이브러리            | ✅      | 인스턴스 생성/파괴                       |
| 외부 스토어 구독               | △       | `useSyncExternalStore` 우선 (Part 5)     |
| 타이머                         | ✅      | stale closure 주의 (Part 3)              |
| 네트워크                       | △       | Server Component / `use()` 우선 (Part 6) |

---

## Part 2 — Race Condition

---

## 상황 재현

```jsx
useEffect(() => {
  fetch(`/search?q=${query}`)
    .then((r) => r.json())
    .then(setResults)
}, [query])
```

`a` → `ab` 빠르게 입력했을 때:

```text
시간 ──────────────────────────────▶
 a   ├──── fetch "a"  (3초) ──────┤ 응답 도착
 ab    ├── fetch "ab" (1초) ─┤ 응답 도착
화면                         "ab결과" → "a결과" ✗
```

두 요청이 **동시에 살아있고**, 늦게 도착한 `a` 응답이 `ab`의 결과를 덮어쓴다.

---

## 기본 해결: AbortController

```jsx
useEffect(() => {
  const controller = new AbortController()
  fetch(`/search?q=${query}`, {signal: controller.signal})
    .then((r) => r.json())
    .then(setResults)
    .catch((err) => {
      if (err.name !== 'AbortError') throw err
    })
  return () => controller.abort() // deps 바뀌면 이전 요청 취소
}, [query])
```

- `signal`은 fetch에게 넘기는 **취소 리모컨**. `controller.abort()` → fetch가 `AbortError`로 reject.
- cleanup에서 abort → 이전 `query`의 요청이 멈추므로 **덮어쓸 응답 자체가 도착하지 않는다**.
- 부가 효과: 서버 부하 감소, 모바일 데이터 절약.

---

## 한 발 더: 데이터 페칭 라이브러리

```jsx
// TanStack Query
const {data} = useQuery({
  queryKey: ['search', query],
  queryFn: ({signal}) =>
    fetch(`/search?q=${query}`, {signal}).then((r) => r.json()),
})
```

내부적으로는 `queryKey`마다 **AbortController + promise identity check**를 자동으로 걸어준다. 앞의 손으로 짠 패턴을 key 단위로 정규화한 것.

→ 요청 취소, 중복 제거, 캐싱, 재시도까지 한 번에.

---

## 더 미묘한 케이스 1 — Optimistic update

```jsx
// 좋아요: 즉시 UI 반영 → 서버 검증 → 실패 시 롤백
async function toggleLike(postId) {
  setLiked(true) // ① 낙관적 반영
  try {
    await api.like(postId) // ② 서버 요청
  } catch {
    setLiked(false) // ③ 롤백
  }
}
```

빠르게 두 번 눌렀을 때 **②가 엉켜서** 최종 상태가 서버와 어긋날 수 있다.

---

## Optimistic update 해결: 요청에 번호 붙이기

```jsx
// 모듈 스코프 — 렌더마다 초기화되지 않는다 (컴포넌트 안이면 useRef로)
let latestId = 0

async function toggleLike(postId) {
  const myId = ++latestId // 내 요청의 번호표
  setLiked(true)
  try {
    await api.like(postId)
  } catch {
    if (myId === latestId) setLiked(false) // 내가 마지막일 때만 롤백
  }
}
```

아이디어는 AbortController와 같다. **"내가 최신 요청인가"를 확인한 뒤 UI에 반영.** AbortController는 요청 자체를 중단시키고, 이 패턴은 결과가 돌아와도 무시한다.

> race condition은 fetch만의 문제가 아니다. **비동기 결과가 UI 상태를 덮어쓰는 모든 지점**에서 발생한다.

---

## Part 3 — Stale Closure

---

## 먼저 — 클로저 1분 복습

**클로저**: 함수가 선언될 때 주변 스코프의 변수를 **함께 기억**하는 JS의 기본 메커니즘.

```js
function makeCounter() {
  let count = 0
  return () => ++count // ← 바깥의 count를 기억
}
const tick = makeCounter()
tick() // 1
tick() // 2  ← 같은 count를 공유
```

`tick`은 `makeCounter`가 이미 끝났는데도 **그때의 `count` 변수**를 움켜쥐고 산다.

> 함수 = 코드 + 주변 변수의 스냅샷.

---

## Stale closure — 오래된 스냅샷

React 컴포넌트는 state가 바뀔 때마다 **함수 자체가 다시 실행**된다. 즉, 렌더할 때마다 새 클로저가 만들어진다.

하지만 **이전 렌더에서 만들어진 함수**(setInterval 콜백, 이벤트 핸들러, Effect 안의 함수 등)가 아직 살아 있다면, 그 함수는 **과거 렌더의 값**을 여전히 움켜쥐고 있다.

> **Stale closure** = 과거 렌더의 state/props를 캡처한 채 지금도 돌아가는 함수.

모든 React 버그의 흔한 원인 중 하나. 아래 데모가 대표 사례다.

---

## 데모: 카운트가 안 올라간다

```jsx
function Counter() {
  const [count, setCount] = useState(0)
  useEffect(() => {
    const id = setInterval(() => {
      setCount(count + 1) // 이 count는 "첫 렌더의 0"
    }, 1000)
    return () => clearInterval(id)
  }, [])
  return <div>{count}</div>
}
```

화면은 `0` → `1` 에서 멈춘다. 왜?

---

## 한 단계씩 — 왜 멈추는가

1. **첫 렌더**: `Counter()` 실행 → 이때 `count = 0` → `setInterval`에 콜백 `() => setCount(count + 1)` 등록. 이 콜백은 **JS 클로저로 "그때의 count = 0"을 캡처**한다.
2. 1초 뒤 콜백 실행 → `setCount(0 + 1)` → 리렌더.
3. **두 번째 렌더**: `Counter()`가 다시 실행되고 새로운 `count = 1`이 생기지만, **deps가 `[]`라 Effect는 재실행되지 않는다** → 첫 렌더의 interval 콜백이 그대로 살아 있음.
4. 다음 tick에서도 `setCount(0 + 1)` → 화면 `1`에서 고정.

```text
첫 렌더 → count=0 ─┐
두 번째 렌더 → count=1 (새) │  첫 렌더의 interval 콜백은
세 번째 렌더 → count=1 ──┘  여전히 "count=0"을 움켜쥠
```

---

## 핵심: 매 렌더는 스냅샷

> Dan Abramov: **Each render has its own props, state, effects, and event handlers.**
> ([_A Complete Guide to useEffect_](https://overreacted.io/a-complete-guide-to-useeffect/))

렌더는 `Counter()` 함수가 **다시 실행**되는 것. 매 렌더마다 새로운 `count`, 새로운 클로저가 만들어지지만, 이전 렌더가 만들어둔 함수(interval 콜백, 이벤트 핸들러 등)는 **그 시점의 값을 움켜쥔 채 살아남는다**.

이게 stale closure — 오래된 스냅샷을 보는 함수.

---

## 해결 2종

```jsx
// 1. functional update — "최신 state"를 인자로 받음 (기본)
useEffect(() => {
  const id = setInterval(() => setCount((prev) => prev + 1), 1000)
  return () => clearInterval(id)
}, [])

// 2. ref — 값을 "렌더 바깥 저장소"에 두고 최신 값 읽기
useEffect(() => {
  countRef.current = count
})
useEffect(() => {
  const id = setInterval(() => setCount(countRef.current + 1), 1000)
  return () => clearInterval(id)
}, [])
```

1번이 기본. ref(2번)는 매 렌더가 **같은 객체**를 공유하므로 `.current`로 항상 최신 값을 읽을 수 있다 — state 여러 개를 함께 참조해야 할 때 유용.

---

## 여기서 멈추지 않는다

대부분의 가이드는 "해결 3종" 에서 끝난다. 하지만 진짜 질문은:

> **어떤 값은 deps에 넣고, 어떤 값은 넣지 말아야 하는가?**

이게 Part 4의 주제다.

---

## Part 4 — Reactive vs Latest

---

## 두 종류의 값

Effect 안에서 참조하는 값은 두 부류:

**Reactive value** — 바뀌면 effect를 **다시 실행해야** 하는 값

- 예: `roomId` — 방이 바뀌면 WebSocket을 끊고 **재연결**해야 함

**Latest value** — 값은 최신으로 읽되 **재실행은 안 해야** 하는 값

- 예: `isMuted` — 메시지가 올 때 음소거 여부에 따라 알림음을 낼지 결정. 하지만 **음소거를 토글했다고 연결을 다시 맺을 이유는 없음**.

---

## 기존 방식으로는 분리 불가

```jsx
useEffect(() => {
  const conn = connect(roomId)
  conn.on('message', (msg) => {
    if (!isMuted) playSound() // isMuted도 참조
  })
  return () => conn.disconnect()
}, [roomId, isMuted]) // ← isMuted 넣으면 토글할 때마다 재연결
// isMuted 빼면 → stale closure (과거의 isMuted를 봄)
```

ref로 우회는 가능하지만 수동이고 실수 쉽다.

---

## useEffectEvent

```jsx
// 2026-04 현재 experimental. 실제 import:
// import {experimental_useEffectEvent as useEffectEvent} from 'react'
import {useEffectEvent} from 'react'

const onMessage = useEffectEvent((msg) => {
  if (!isMuted) playSound() // 호출될 때마다 최신 isMuted를 읽음
})

useEffect(() => {
  const conn = connect(roomId)
  conn.on('message', onMessage)
  return () => conn.disconnect()
}, [roomId]) // ✅ isMuted는 deps에 없지만 최신값이 반영됨
```

`useEffectEvent`는 **"reactive가 아닌 값을 읽는 창구"**. 내부 구현은 ref와 비슷하지만 ESLint deps 규칙이 이를 인식해서 "deps 빠졌다"는 경고를 내지 않으므로, 수동 ref보다 안전하다.

---

## 판단 기준

> **"이 값이 바뀌면, 내 effect는 처음부터 다시 시작돼야 하는가?"**

- **YES** → reactive → deps에 포함
- **NO** → latest → `useEffectEvent`로 읽기

참고: [Separating Events from Effects — React 공식](https://react.dev/learn/separating-events-from-effects)

---

## Setup / Cleanup은 쌍을 이뤄야 한다

Effect는 "한 번 실행되는 사이드이펙트"가 아니라 **"유지되는 연결"**. setup이 시작한 것은 cleanup이 정확히 되돌려야 한다.

```jsx
// ❌ setup은 구독을 만드는데 cleanup이 없음 → 리소스 누적
useEffect(() => {
  conn.on('msg', handler)
  // return이 없음
}, [])

// ✅ setup 1회 = cleanup 1회, 완벽한 대칭
useEffect(() => {
  conn.on('msg', handler)
  return () => conn.off('msg', handler)
}, [])
```

대칭이 깨지는 가장 흔한 실수: 구독은 등록했는데 해제는 안 하기, 타이머는 시작했는데 clear는 안 하기.

---

## Strict Mode가 강제하는 규율

```text
dev에서만: mount → unmount → mount (의도적으로 두 번)
```

개발 모드에서 React는 일부러 Effect를 한 번 더 돌려서 **setup/cleanup 쌍이 제대로 맞춰져 있는지** 드러낸다. 여기서 잡히는 버그:

- cleanup 없는 구독 → 두 번째 mount에서 리스너가 이중으로 붙음
- 네트워크 요청의 race condition → 같은 요청이 두 번 날아가 가시화됨
- setup이 모듈 레벨의 상태를 건드리는데 cleanup이 복구 안 함 → 두 번째 mount가 깨짐

프로덕션은 한 번이지만 개발 중 이중 실행은 **대칭이 무너지는 지점을 드러내는 장치**다.

---

## Part 5 — 실행 타이밍 & Tearing

---

## 세 가지 Effect 훅

```text
render → [insertion effect] → DOM mutation → [layout effect] → paint → [effect]
```

| 훅                   | 타이밍                   | 용도                       |
| -------------------- | ------------------------ | -------------------------- |
| `useInsertionEffect` | DOM mutation **전**      | CSS-in-JS 스타일 주입      |
| `useLayoutEffect`    | DOM mutation 후 paint 전 | 레이아웃 측정 → state 반영 |
| `useEffect`          | paint **후**             | 대부분의 사이드이펙트      |

셋 다 commit 단계에 속하지만, **DOM 변경 전 / 변경 후 paint 전 / paint 후**로 세분화된 것.

---

## useLayoutEffect — 언제 필요한가

```jsx
function Tooltip({target}) {
  const [pos, setPos] = useState({top: 0, left: 0})
  const ref = useRef(null)

  useLayoutEffect(() => {
    const rect = target.getBoundingClientRect()
    const h = ref.current.offsetHeight
    setPos({top: rect.top - h, left: rect.left})
  }, [target])

  return (
    <div ref={ref} style={pos}>
      ...
    </div>
  )
}
```

`useEffect`면 → 잘못된 위치로 paint → 재계산 → **깜빡임**. layout effect는 paint 전 동기 실행.

---

## useInsertionEffect — CSS-in-JS의 해결사

```jsx
// styled-components 같은 라이브러리 내부
useInsertionEffect(() => {
  injectStyleSheet(rules) // commit 전에 DOM에 있어야
}, [rules])
```

왜 있는가:

- `useLayoutEffect`에서 스타일 주입하면 자식들이 **스타일 없이 측정**됨
- `useInsertionEffect`는 **DOM mutation 직전**에 실행 → 측정 시 스타일이 이미 있음

실무에서 직접 쓸 일은 거의 없다. **"왜 존재하는지" 이해**가 목적.

---

## Tearing — 동시성 모드의 숨은 적

**전제**: React 18부터 렌더를 여러 조각으로 쪼개 실행할 수 있다 (concurrent rendering). 즉, **한 번의 렌더 패스 중간에 React가 잠시 멈췄다가 이어 그릴 수 있다**.

```text
1. 외부 스토어 값 = 10
2. 컴포넌트 A 렌더 → 10 읽음
3. React가 양보 (yield) — 다른 급한 일 처리
4. 외부 스토어 값 = 20  ← 이 사이에 변경됨
5. 이어서 컴포넌트 B 렌더 → 20 읽음
6. 같은 화면에 A=10, B=20  ← tearing (찢어짐)
```

`useState + useEffect`로 외부 값을 구독하면 렌더 도중 값이 바뀌는 걸 막을 방법이 없어 발생 가능.

---

## 잠깐 — "외부 세계" vs "외부 스토어"

- **외부 세계** — React 밖 전부 (DOM, fetch, 타이머 등). → `useEffect`의 영역
- **외부 스토어** — 외부 세계 중 ① 동기 스냅샷 ② 변경 알림(pub/sub) 을 모두 만족하는 데이터 원천

| 대상                                | 외부 스토어? |
| ----------------------------------- | ------------ |
| Redux / Zustand / Jotai             | ✅           |
| `navigator.onLine`, `matchMedia(…)` | ✅           |
| fetch 응답, `document.title` 쓰기   | ❌ (일회성)  |
| setInterval                         | ❌           |

**외부 스토어 구독**만 `useSyncExternalStore`. 나머지는 `useEffect`.

---

## useSyncExternalStore가 푸는 방식

```jsx
const value = useSyncExternalStore(
  subscribe, // "값이 바뀌면 알려줘" — 변화 알림 채널
  getSnapshot, // "지금 값이 뭐지?" — 동기적으로 현재값 반환
  getServerSnapshot, // SSR 초기값
)
```

왜 함수가 두 개로 쪼개져 있나:

- **변화를 감지하는 것**(pub/sub)과 **현재 값을 읽는 것**은 다른 일. 분리돼야 React가 렌더 도중 원할 때 "지금 값"을 확인할 수 있다.
- `getSnapshot()`을 **동기적으로** 호출 → 같은 렌더 패스 안에서는 항상 같은 값 → tearing 없음.
- 렌더 도중 값이 바뀌면 React가 자동으로 다시 렌더.

Zustand, Jotai, Redux 최신 버전이 내부적으로 사용.

---

## 실사용 예시

```jsx
// navigator.onLine
const isOnline = useSyncExternalStore(
  (cb) => {
    window.addEventListener('online', cb)
    window.addEventListener('offline', cb)
    return () => {
      window.removeEventListener('online', cb)
      window.removeEventListener('offline', cb)
    }
  },
  () => navigator.onLine,
  () => true,
)
```

> **읽기 전용 외부 값을 구독**하는 모든 케이스 → 이걸로.

---

## Part 6 — Suspense · use() · 커스텀 훅

---

## render-as-you-fetch vs fetch-on-render

```text
기존 (fetch-on-render):
  렌더 → mount → useEffect fetch → 로딩 상태
  [부모] → [자식 mount → fetch] → [손자 mount → fetch]  ← waterfall

모던 (render-as-you-fetch):
  렌더 시작과 동시에 fetch → Suspense로 대기
  [부모 렌더 순간 트리 전체의 fetch 트리거]
```

핵심 차이: **waterfall 제거**.

---

## use() hook

```jsx
function UserProfile({userPromise, postsPromise}) {
  const user = use(userPromise) // pending이면 컴포넌트 일시중단
  const posts = use(postsPromise)
  return <Layout user={user} posts={posts} />
}

;<Suspense fallback={<Spinner />}>
  <UserProfile
    userPromise={fetchUser(id)} // 병렬 시작
    postsPromise={fetchPosts(id)}
  />
</Suspense>
```

**동작 원리**: promise가 pending이면 `use()`가 내부적으로 그 promise를 **throw** → 가장 가까운 `<Suspense>`가 캐치해 fallback 표시 → promise가 resolve되면 React가 컴포넌트를 다시 실행. **useEffect도, useState도 없음**, 로딩/에러는 Suspense / ErrorBoundary가 담당.

---

## 커스텀 훅 — 이펙트를 감춰라

```jsx
// ❌ 사용자가 effect를 관리
function useChat(roomId) {
  return {roomId, setMessages: ...}
}
// 사용처에서 useEffect로 subscribe 직접 호출...

// ✅ Effect를 훅 안에 캡슐화
function useChat(roomId) {
  const [messages, setMessages] = useState([])
  useEffect(() => {
    const conn = connect(roomId)
    conn.on('msg', (m) => setMessages((prev) => [...prev, m]))
    return () => conn.disconnect()
  }, [roomId])
  return messages
}
```

---

## 커스텀 훅 설계 원칙

1. **동기화는 훅 안에서 끝내라** — `useEffect`를 훅 내부에서 실행하지 않고, "호출자가 `useEffect`에 넣을 setup 함수"를 반환하는 식으로 설계하지 말 것. 타이밍과 cleanup이 호출자에게 새어나가 의존성 배열이 꼬인다
2. **객체 props는 destructure** — `useHook({a, b})` 받으면 내부에서 꺼내서 각각 deps에
3. **한 훅 = 한 관심사** — 독립된 동기화는 여러 훅으로
4. **cleanup 경계를 명확히** — 훅이 제공한 리소스는 훅의 cleanup에서 회수

참고: [Reusing Logic with Custom Hooks — React 공식](https://react.dev/learn/reusing-logic-with-custom-hooks)

---

## Part 7 — 체크리스트

---

## useEffect 의사결정 순서

```text
1. React 바깥과 동기화? 아니면 → Effect 아님
2. 외부 스토어 구독? → useSyncExternalStore
3. 데이터 페칭? → Server Component / use() / React Query
4. state/props 참조? → reactive vs latest 구분
                       latest는 useEffectEvent
5. cleanup이 idempotent? Strict Mode로 검증?
6. 레이아웃 측정? → useLayoutEffect
7. 한 Effect = 한 동기화. 독립된 건 분리.
```

---

## 한 줄 요약

> useEffect는 도구다. 진짜 실력은 **쓰지 않아도 되는 구조를 만드는 것**,
> 써야 할 때는 **reactive · latest · cleanup** 을 구분하는 것이다.

<!-- _class: invert -->

---

## Q&A

<!-- _class: invert -->

---

## Q. Effect 안에서 async 함수를 바로 쓸 수 없나요?

`useEffect`는 반환값으로 cleanup을 기대하는데 `async`는 Promise를 반환한다.

```jsx
// ❌ 직접 async 불가
useEffect(async () => { /* ... */ }, [])

// ✅ 내부에 async 선언
useEffect(() => {
  const run = async () => { setData(await fetch(...)) }
  run()
}, [])
```

이렇게 해도 race condition은 그대로 — Part 2 해결책과 함께.

---

## Q. deps 배열 ESLint 경고, 그냥 끄면 안 되나요?

대부분 **stale closure 버그를 심는 행위**. 경고가 뜨는 진짜 원인은 보통 둘 중 하나:

- 함수를 Effect 바깥에서 만들고 있다 → Effect 안으로 / `useCallback`
- 최신 state를 읽고 싶다 → `useEffectEvent`

> 비활성화보다 구조를 고치는 쪽을 먼저.

---

## Q. useSyncExternalStore vs useEffect?

| 구분                  | useEffect                | useSyncExternalStore |
| --------------------- | ------------------------ | -------------------- |
| 외부 스토어 구독      | tearing 위험             | ✅ 권장              |
| 브라우저 API 1회 설정 | ✅ (`document.title`)    | 부적합               |
| 네트워크 fetch        | 가능 (라이브러리가 나음) | 부적합               |
| 동시성 안전           | ❌                       | ✅                   |

**읽기 전용 외부 값 구독** → `useSyncExternalStore`, **부수 효과 + cleanup** → `useEffect`.

---

## Q. React Compiler가 나오면 이 내용도 사라지나요?

Compiler는 `useMemo` / `useCallback` 을 자동화하지만 **Effect 관련은 대부분 유효**:

- 안티패턴 (파생 state, 이벤트 로직) — 컴파일러가 고쳐주지 않음
- race condition — 설계 문제, 컴파일러 영역 밖
- reactive vs latest — `useEffectEvent`가 공식 해결책이지 컴파일러가 아님
- tearing — `useSyncExternalStore`가 답

---

## Q. 독립된 동기화를 한 Effect에 묶어도 되나요?

```jsx
// ❌ title과 WebSocket이 같이 묶임
useEffect(() => {
  document.title = `${user.name} - 마이페이지`
  const ws = new WebSocket(`/chat/${chatId}`)
  return () => ws.close()
}, [user.name, chatId])
```

`user.name`이 바뀌면 **WebSocket까지 재연결**됨.

> **각 Effect는 하나의 독립된 동기화 프로세스.**

---

## 독립 동기화는 Effect 쪼개기

```jsx
useEffect(() => {
  document.title = `${user.name} - 마이페이지`
}, [user.name])

useEffect(() => {
  const ws = new WebSocket(`/chat/${chatId}`)
  return () => ws.close()
}, [chatId])
```

Effect를 나누는 건 **성능 최적화가 아니라 정확성**의 문제.

---

## Q. `use(fetchUser(id))` — 매 렌더마다 새 promise 아닌가요?

맞다. 그대로 쓰면 **렌더마다 새 요청**이 나간다.

```jsx
// ❌ 렌더마다 fetchUser 재호출 → 무한 Suspense
function UserProfile({id}) {
  const user = use(fetchUser(id))
  return <div>{user.name}</div>
}
```

해결은 **promise를 렌더 바깥에서 안정시키는 것** — 다음 슬라이드에서.

---

## `use()` — promise를 안정시키는 두 가지 방법

```jsx
// ✅ Server Component — 요청당 한 번만 실행
async function Page({id}) {
  const userPromise = fetchUser(id)
  return <UserProfile userPromise={userPromise} />
}

// ✅ React.cache — 같은 인자면 같은 promise 반환
const getUser = cache((id) => fetch(`/api/user/${id}`).then((r) => r.json()))
```

> `use()`는 promise를 **기다리는 창구**일 뿐. promise를 **언제 만들지**는 우리가 책임져야 한다.

---

## Q. Server Component가 Effect를 대체한다는 게?

과거:

```jsx
function Page() {
  const [data, setData] = useState(null)
  useEffect(() => {
    fetch('/api/posts')
      .then((r) => r.json())
      .then(setData)
  }, [])
  if (!data) return <Spinner />
  return <Posts data={data} />
}
```

---

## Server Component 패턴

```jsx
// 'use client' 없음 = Server Component.
// 서버에서 한 번 실행되고, 결과 HTML(+직렬화 데이터)만 브라우저로 내려간다.
// 브라우저에서는 다시 렌더되지 않음.
async function Page() {
  const data = await fetchPosts() // 서버에서 바로 await
  return <Posts data={data} />
}
```

- **Effect 없음, useState 없음, 로딩 상태 없음** — 렌더 전에 이미 데이터가 있음
- race condition 원천 차단 — 서버에서 딱 한 번 실행
- 번들 사이즈도 감소 — 서버 전용 코드는 클라이언트로 안 내려감

> 데이터 페칭에 한해서는 **Effect 없는 세계**가 이미 와 있다.

---

## 더 읽을거리 — 필수

- [You Might Not Need an Effect — React 공식](https://react.dev/learn/you-might-not-need-an-effect)
- [A Complete Guide to useEffect — Dan Abramov](https://overreacted.io/a-complete-guide-to-useeffect/)
- [Separating Events from Effects — React 공식](https://react.dev/learn/separating-events-from-effects)

### 심화

- [Hooks, Dependencies and Stale Closures — tkdodo](https://tkdodo.eu/blog/hooks-dependencies-and-stale-closures)
- [Simplifying useEffect — tkdodo](https://tkdodo.eu/blog/simplifying-use-effect)

---

## 더 읽을거리 — 최신 & 도구

### React 19 이후

- [useEffectEvent (React 19.2) — LogRocket](https://blog.logrocket.com/react-useeffectevent/)
- [useSyncExternalStore Demystified — Kent C. Dodds](https://www.epicreact.dev/use-sync-external-store-demystified-for-practical-react-development-w5ac0)

### 도구

- [eslint-plugin-react-you-might-not-need-an-effect](https://www.npmjs.com/package/eslint-plugin-react-you-might-not-need-an-effect)
- [Reusing Logic with Custom Hooks — React 공식](https://react.dev/learn/reusing-logic-with-custom-hooks)

---

# 감사합니다

<!-- _class: invert -->

@yceffort
