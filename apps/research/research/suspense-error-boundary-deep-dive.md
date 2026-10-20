---
title: 'Suspense & ErrorBoundary 딥다이브: 던지는 컴포넌트의 세계'
marp: true
paginate: true
theme: yceffort
tags:
  - react
  - error-handling
  - async
date: 2026-07-29
description: 'throw 하나로 이해하는 선언적 로딩과 에러 처리 — 개념부터 reconciler 내부 동작, 실무 경계 설계, React Query 연계까지'
published: true
art:
  undraw: loading-ui
---

# Suspense & ErrorBoundary 딥다이브

던지는 컴포넌트의 세계 — throw 하나로 이해하는 선언적 로딩과 에러 처리

<!-- _class: invert deck-cover -->

@yceffort

---

## 이 강의에서 다루는 것

1. 왜 이 둘을 **같이** 배우는가
2. **ErrorBoundary** — 개념, 동작, 잡지 못하는 것들
3. **Suspense** — fallback 뒤에서 벌어지는 일
4. **원리 딥다이브** — React 내부에서 throw는 어떻게 처리되는가
5. **실무** — 경계 배치, 깜빡임, waterfall, 리셋, 로깅
6. **React Query**와 조합하기

Part 1~5가 끝날 때마다 **정리와 퀴즈**가 있다.

---

## 이 강의의 메시지

> Suspense와 ErrorBoundary는 별개의 기능이 아니다.
> **컴포넌트가 렌더링을 포기하고 무언가를 throw했을 때**, 그것을 받아주는 하나의 메커니즘의 두 얼굴이다.

- promise를 throw하면 → 가장 가까운 **Suspense**가 받는다
- Error를 throw하면 → 가장 가까운 **ErrorBoundary**가 받는다

이 한 문장을 이해하면, 두 기능의 동작이 전부 같은 그림 위에 놓인다.

---

## Part 0 — 왜 이 둘을 같이 배우는가

---

## 명령형 방식: 우리가 매일 쓰던 코드

```jsx
function UserProfile({id}) {
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    fetchUser(id)
      .then(setUser)
      .catch(setError)
      .finally(() => setLoading(false))
  }, [id])

  if (loading) return <Spinner />
  if (error) return <ErrorMessage error={error} />
  return <Profile user={user} />
}
```

---

## 이 코드의 문제

- **모든 컴포넌트마다 반복**된다 — loading/error 분기가 성공 화면보다 길다
- 로딩 상태의 **조합**이 컴포넌트 수만큼 늘어난다 (스피너 3개가 각자 돈다)
- race condition, 언마운트 후 setState… 부수적인 버그의 온상
- "성공했을 때 어떤 화면인가"가 코드에서 잘 안 보인다

> 로딩과 에러는 **개별 컴포넌트의 관심사가 아니라, 화면 영역의 관심사**다.

<!-- race condition 구두 예시: id가 1→2로 빠르게 바뀌면 늦게 도착한 1의 응답이 2의 화면을 덮는다(응답 순서 역전). Suspense+캐시 세계에서는 "지금 어떤 promise를 보여줄까"가 렌더의 입력으로 정리되어 이 경쟁이 구조적으로 사라진다. -->

---

## 선언적 경계라는 발상

```jsx
<ErrorBoundary fallback={<ErrorMessage />}>
  <Suspense fallback={<Spinner />}>
    <Profile /> {/* 성공한 화면만 그린다 */}
  </Suspense>
</ErrorBoundary>
```

- `Profile`은 **데이터가 이미 있다고 가정하고** 성공 화면만 그린다
- 로딩 중이라면? → 렌더를 **포기하고 promise를 throw** → Suspense가 받는다
- 실패했다면? → 렌더를 **포기하고 Error를 throw** → ErrorBoundary가 받는다

분기가 컴포넌트 밖으로, **트리 구조**로 올라갔다.

---

## 두 경계의 대칭

| 관점        | Suspense                  | ErrorBoundary                  |
| ----------- | ------------------------- | ------------------------------ |
| 받는 것     | promise                   | Error                          |
| 의미        | "아직 준비 안 됨"         | "실패함"                       |
| 보여주는 것 | `fallback` prop           | fallback UI (직접 구현)        |
| 복귀 방법   | promise resolve → 자동    | 리셋 (수동)                    |
| 탐색 방향   | 가장 가까운 조상 Suspense | 가장 가까운 조상 ErrorBoundary |

**같은 구조, 다른 페이로드.** Part 3에서 React 소스로 이 대칭을 직접 확인한다.

---

## Part 1 — ErrorBoundary: 개념과 동작

---

## 렌더 중 에러가 나면 무슨 일이 벌어지는가

```jsx
function Product({item}) {
  // item이 undefined면? → TypeError
  return <div>{item.name}</div>
}
```

ErrorBoundary가 하나도 없다면:

- React는 **트리 전체를 언마운트**한다 (React 16부터)
- 사용자는 **빈 흰 화면**을 본다
- "일부가 깨진 UI를 남겨두는 것이 더 위험하다"는 판단 — 결제 화면에서 금액이 잘못 보이는 것보다 아무것도 안 보이는 게 낫다

> 경계가 없으면, 버튼 하나의 에러가 **앱 전체를 지운다**.

---

## ErrorBoundary 최소 구현

```jsx
class ErrorBoundary extends React.Component {
  state = {error: null}

  static getDerivedStateFromError(error) {
    return {error} // 이 state로 fallback을 그린다
  }

  componentDidCatch(error, info) {
    logError(error, info.componentStack) // 로깅은 여기서
  }

  render() {
    if (this.state.error) return this.props.fallback
    return this.props.children
  }
}
```

자식 트리에서 렌더 중 에러가 throw되면, 이 컴포넌트가 받아서 **자기 영역만** fallback으로 바꾼다.

<!-- 클래스가 처음인 수강생용 30초 글로스: state 필드는 useState 초기값, render()는 함수 컴포넌트의 본문, this.props는 props 인자, static은 인스턴스 없이 React가 직접 부르는 메서드(그래서 this가 없고 순수해야 함). 라이프사이클 = React가 정해진 시점에 대신 불러주는 메서드들. -->

---

## 두 라이프사이클의 역할 분리

| 구분      | `getDerivedStateFromError`     | `componentDidCatch`            |
| --------- | ------------------------------ | ------------------------------ |
| 실행 시점 | **render phase**               | **commit phase**               |
| 하는 일   | fallback을 그릴 state 반환     | 로깅 등 사이드이펙트           |
| 순수해야? | ✅ (static, 사이드이펙트 금지) | ❌ (자유롭게)                  |
| 받는 것   | `error`                        | `error`, `info.componentStack` |

**render phase**는 무엇을 그릴지 계산하는 단계, **commit phase**는 그 결과를 실제 DOM에 반영하는 단계다. render phase는 동시성 모드에서 중단·재실행될 수 있어서, "무엇을 그릴까"(순수)와 "무엇을 기록할까"(사이드이펙트)를 분리해 놓은 것.

---

## 왜 아직도 클래스 컴포넌트인가

표면적으로는 "`getDerivedStateFromError`에 대응하는 훅이 없어서". (2018년 훅 발표 때 "곧 추가될 예정"이라던 그 훅이 8년째 없다)

진짜 이유는 **실행 모델의 불일치**:

- 훅은 **내 함수가 실행되는 동안** 동작하는 API
- 에러 캐치는 정반대 — **내 렌더가 끝난 뒤**, 자식이 던졌을 때 React가 나를 다시 불러줘야 한다
- 클래스에는 렌더 사이에도 살아있는 **인스턴스**와, React가 임의 시점에 호출할 **메서드**가 있다. 함수형은 렌더 = 함수 호출 그 자체라 그 자리가 없다

(Part 3에서 React가 클래스 컴포넌트를 찾아 올라가는 코드로 이를 확인한다)

실무 결론: **클래스를 직접 쓰지 말고 `react-error-boundary`를 쓴다.** 앱에서 클래스가 정당한 거의 유일한 자리가 여기다.

<!-- Q&A 대비: 함수형 ErrorBoundary RFC 논의는 있었지만 우선순위가 낮다 — react-error-boundary가 사실상 공백을 메우고 있고, React 19의 루트 옵션(onCaughtError 등)이 로깅 쪽 수요를 흡수했다. -->

---

## 잡지 못하는 것 1 — 이벤트 핸들러

```jsx
function BuyButton() {
  const handleClick = () => {
    throw new Error('결제 실패') // ErrorBoundary에 안 잡힌다!
  }
  return <button onClick={handleClick}>구매</button>
}
```

- ErrorBoundary는 **렌더링 중** 던져진 에러만 잡는다
- 이벤트 핸들러는 렌더링이 끝난 뒤, 브라우저 이벤트에 반응해 실행된다 — React의 렌더 try/catch 바깥이다
- 화면이 깨질 일이 없으니 fallback으로 바꿀 이유도 없다는 설계
- 안 잡힌 에러는 **전역(window)으로 흘러가** 콘솔에 찍힐 뿐, 이미 커밋된 화면은 그대로다 — 렌더 에러처럼 흰 화면이 되지는 않는다

핸들러 안에서는 **직접 try/catch** 해야 한다. (경계로 보내는 방법은 Part 4에서)

---

## 잡지 못하는 것 2 — 비동기 콜백

```jsx
useEffect(() => {
  setTimeout(() => {
    throw new Error('타이머 안') // ❌ 안 잡힌다
  }, 1000)
  fetchUser(id).then(() => {
    throw new Error('then 안') // ❌ 안 잡힌다
  })
}, [id])
```

`setTimeout`, promise 콜백이 실행될 시점에 렌더링은 이미 끝나 있다. **호출 스택이 다르다.**

단, 헷갈리지 말 것:

```jsx
useEffect(() => {
  throw new Error('effect 본문') // ✅ 이건 잡힌다
}, [])
```

effect 본문의 동기적 throw는 React가 커밋 과정에서 처리하므로 **경계에 잡힌다**.

---

## 잡지 못하는 것 3, 4

**자기 자신의 에러**

```jsx
class ErrorBoundary extends React.Component {
  render() {
    if (this.state.error) return <Fallback data={undefined.foo} /> // 💥
    ...
  }
}
```

fallback 자체가 던지면 **그 위의** 경계로 올라간다. 경계는 자식만 보호한다.

**SSR 에러** — 서버 렌더 중 에러는 클래스 ErrorBoundary의 라이프사이클로 처리되지 않는다. 스트리밍 SSR에서는 다른 메커니즘(가장 가까운 Suspense까지 지우고 클라이언트에서 재시도)이 동작한다. (Part 3 후반)

---

## 실무에서는: react-error-boundary

```jsx
import {ErrorBoundary} from 'react-error-boundary'

function Fallback({error, resetErrorBoundary}) {
  return (
    <div role="alert">
      <p>문제가 생겼습니다: {error.message}</p>
      <button onClick={resetErrorBoundary}>다시 시도</button>
    </div>
  )
}

;<ErrorBoundary FallbackComponent={Fallback} onError={logError}>
  <Profile />
</ErrorBoundary>
```

클래스 보일러플레이트 없이 fallback, 로깅, **리셋**(Part 4)까지 제공한다. 사실상 표준.

<!-- 배경 보충(구두): 만든 사람이 React 코어 팀 출신(Brian Vaughn) — 사실상 이 문제의 레퍼런스 구현. 런타임 의존성 없이 1KB 남짓, 주간 다운로드 천만 회 이상. 수치는 발표 시점에 npm/bundlephobia로 재확인 (2026-07 기준 주간 약 1,450만 회, 의존성 없음은 v6 기준). -->

---

## Part 1 정리

- 경계 없는 렌더 에러 = **트리 전체 언마운트** (흰 화면)
- `getDerivedStateFromError` → 무엇을 그릴까 (render phase, 순수)
- `componentDidCatch` → 무엇을 기록할까 (commit phase, 사이드이펙트)
- 잡는 것: **렌더링, 라이프사이클, 생성자, effect 본문**의 에러
- 못 잡는 것: **이벤트 핸들러, 비동기 콜백, 자기 자신, SSR**
- 직접 만들지 말고 `react-error-boundary`

---

## 퀴즈 1 — ErrorBoundary에 잡힐까?

각 코드의 에러가 부모 ErrorBoundary에 잡히면 O, 아니면 X.

```jsx
function A() {
  return <div>{data.name}</div> // (1) 렌더 — data는 undefined
}

const handleClick = () => {
  throw new Error() // (2) 버튼의 클릭 핸들러
}

useEffect(() => {
  throw new Error() // (3) effect 본문
}, [])

useEffect(() => {
  fetch('/api').then(() => {
    throw new Error() // (4) promise 콜백
  })
}, [])
```

<!-- 정답 공개 전에 손들기로 진행. (3)이 갈리는 문제 — effect 본문과 비동기 콜백의 차이를 짚는 게 목적. -->

---

## 퀴즈 1 — 정답

| 코드 | 정답  | 이유                                                |
| ---- | ----- | --------------------------------------------------- |
| (1)  | **O** | 렌더 중 throw — 경계의 본래 목적                    |
| (2)  | **X** | 이벤트 핸들러 — 렌더 바깥의 호출 스택               |
| (3)  | **O** | effect **본문**의 동기 throw는 커밋 과정에서 잡힌다 |
| (4)  | **X** | promise 콜백 — 렌더도 커밋도 아닌 나중 시점         |

> 판별 기준은 "어디에 썼나"가 아니라 **"React가 실행해주는 흐름 안에서 던져졌나"**.

---

## Part 2 — Suspense: 개념과 동작

---

## Suspense 기본형

```jsx
<Suspense fallback={<Skeleton />}>
  <Profile />
</Suspense>
```

- `Profile`(또는 그 아래 누군가)이 "아직 준비 안 됨"을 선언하면 `fallback`을 대신 보여준다
- 준비가 끝나면 **자동으로** 실제 콘텐츠로 교체된다
- 로딩 상태를 **어디에 보여줄지**를 트리 구조로 선언하는 컴포넌트

그런데 — "아직 준비 안 됨"을 **누가, 어떻게** 선언하는가?

---

## 무엇이 suspend를 일으키는가

| 소스                                                                      | suspend? |
| ------------------------------------------------------------------------- | -------- |
| `React.lazy()` 코드 스플리팅                                              | ✅       |
| `use(promise)` (React 19)                                                 | ✅       |
| Suspense 지원 프레임워크·라이브러리 (Next.js RSC, TanStack Query, Relay…) | ✅       |
| 컴포넌트 안에서 직접 `fetch`                                              | ❌       |
| `useEffect` 안의 데이터 페칭                                              | ❌       |

**fetch를 쓴다고 저절로 suspend되지 않는다.** suspend는 특정한 프로토콜을 따르는 코드만 일으킨다. (표는 **클라이언트 컴포넌트 기준** — async 서버 컴포넌트의 `await`은 서버 렌더러가 같은 신호로 바꿔준다.) 그 프로토콜이 —

---

## suspend의 정체: promise를 throw한다

```jsx
function Profile() {
  const user = resource.read()
  // read()의 내부:
  //   데이터 있음   → 값 반환
  //   로딩 중      → throw promise   ← 이것이 suspend
  //   실패         → throw error     ← ErrorBoundary로
  return <h1>{user.name}</h1>
}
```

- JavaScript의 `throw`는 Error 전용이 아니다 — **아무 값이나 던질 수 있다**
- React는 렌더 중 잡힌 값이 **thenable**(then 메서드를 가진 객체)이면 "에러"가 아니라 "**기다려 달라는 신호**"로 해석한다

이 한 가지 트릭이 Suspense의 전부다.

---

## 가짜 Suspense 리소스 — 직접 만들어보기

```jsx
function createResource(promise) {
  let state = {status: 'pending', result: null}
  promise.then(
    (value) => (state = {status: 'success', result: value}),
    (error) => (state = {status: 'error', result: error}),
  )
  return {
    read() {
      if (state.status === 'pending') throw promise // Suspense가 받는다
      if (state.status === 'error') throw state.result // ErrorBoundary가 받는다
      return state.result
    },
  }
}
```

이 몇 줄로 두 경계에 모두 연결됐다. 라이브러리들이 하는 일도 본질적으로 이것(+ 캐싱)이다.

<!-- 실제 프로덕션에서 이 패턴을 직접 쓰라는 게 아니라는 점을 구두로 강조. 메커니즘 이해용. promise identity·캐시·키 변경 시 재페칭 관리가 곧 라이브러리의 일이라, 실전은 Suspense 지원 페처(React Query 등)나 라우터 loader 도입이 사실상 전제다. -->

---

## promise가 resolve되면?

```text
1. Profile 렌더 → throw promise
2. React: 가장 가까운 Suspense를 찾아 fallback을 화면에 표시
3. React: 잡아둔 promise에 "끝나면 알려줘"라고 .then을 걸어둠
4. 데이터 도착 (promise resolve)
5. React: Profile을 처음부터 다시 렌더링
6. 이번엔 read()가 값을 반환 → fallback 대신 진짜 콘텐츠 표시
```

핵심: **멈춘 곳에서 이어지는 게 아니라, 처음부터 다시 실행된다.**

자바스크립트 함수는 중간에 일시정지했다가 재개할 수 없다. 그래서 React가 할 수 있는 일은 컴포넌트 함수를 한 번 더 호출하는 것뿐이다. 이때 리소스가 결과를 기억해두지 않았다면 read()는 또 promise를 던지고, 화면은 fallback에서 벗어나지 못한다. 결과를 캐시하는 리소스/라이브러리가 필요한 이유가 이것이다.

---

## fallback이 보이는 동안, 아래 트리는?

두 가지 상황을 구분해야 한다.

**① 최초 마운트 중 suspend** — 아직 아무것도 커밋된 적 없음. DOM도 state도 없는 상태에서 fallback만 커밋된다.

**② 이미 보이던 트리가 다시 suspend** (동기 업데이트)

```html
<div style="display: none !important;">기존 콘텐츠</div>
<div>fallback</div>
```

- 기존 콘텐츠를 **언마운트하지 않는다** — `display: none`으로 숨긴다
- 컴포넌트 **state는 그대로 유지**된다
- 준비되면 다시 보여주기만 하면 된다

이 "숨기기" 메커니즘이 내부의 Offscreen(현 Activity) 처리다.

---

## use() — React 19의 공식 입구

```jsx
import {use, Suspense} from 'react'

function Profile({userPromise}) {
  const user = use(userPromise) // pending이면 suspend
  return <h1>{user.name}</h1>
}

// 부모 (예: 서버 컴포넌트 — 리렌더가 없어 promise는 요청당 한 번)
;<Suspense fallback={<Skeleton />}>
  <Profile userPromise={fetchUser(id)} />
</Suspense>
```

- 앞의 `createResource` 패턴을 React가 정식 API로 만든 것
- 단, **promise의 identity가 안정적이어야 한다** — 렌더마다 새 promise를 만들면 재시도할 때마다 다시 suspend되어 영원히 끝나지 않는다 ("uncached promise" 경고)
- 그래서 promise는 **렌더 바깥에서** 만들어 내려보내거나, 캐시를 통해 얻는다

<!-- "클라이언트 props는 직렬화 가능해야 한다던데 promise를 넘겨도 되나?" — promise는 특별 취급이다. RSC 프로토콜이 resolve 값을 스트리밍으로 이어 보내 클라이언트의 use()가 이어받는다. 상세는 별도 덱('use client' 딥다이브)에서. -->

---

## React.lazy — 이미 쓰고 있던 suspend

```jsx
const Settings = lazy(() => import('./Settings'))

;<ErrorBoundary FallbackComponent={ChunkError}>
  <Suspense fallback={<PageSkeleton />}>
    <Settings />
  </Suspense>
</ErrorBoundary>
```

- lazy는 **모듈을 기다리는 promise를 throw**한다 — 기다리는 게 데이터가 아니라 코드일 뿐, 프로토콜은 `use()`와 같다
- 로드 **실패**(배포 직후 청크 404, 오프라인)는 **Error throw** → ErrorBoundary가 받는다
- 한 기능이 두 경계를 모두 쓰는 가장 흔한 실전 사례 — **코드 스플리팅에도 에러 경계가 필요한 이유**다

<!-- 배포 후 옛 청크가 사라져 ChunkLoadError가 나는 시나리오를 구두로: 에러 fallback에서 새로고침을 유도하는 것이 흔한 대응. next/dynamic도 내부는 lazy + Suspense 래핑으로 같은 프로토콜이고, 청크 404도 같은 원리로 가장 가까운 에러 경계(error.tsx 포함)가 받는다. -->

---

## Part 2 정리

- suspend = **promise를 throw하는 것.** fetch가 아니라 이 프로토콜이 Suspense를 작동시킨다
- React는 thenable을 "기다려 달라"로, Error를 "실패"로 해석한다
- resolve 후에는 **재개가 아니라 재시도** — 그래서 캐시가 필수
- 이미 보이던 트리는 fallback 동안 **`display: none`으로 숨겨질 뿐, state가 유지**된다
- `use()`는 이 프로토콜의 공식 입구. promise identity를 안정시킬 것

---

## 퀴즈 2 — 입력값은 살아남을까?

```jsx
function Page() {
  const [submitted, setSubmitted] = useState(null)
  const inputRef = useRef(null)
  return (
    <Suspense fallback={<Spinner />}>
      <input ref={inputRef} /> {/* uncontrolled — 값은 DOM에만 있다 */}
      <button onClick={() => setSubmitted(inputRef.current.value)}>검색</button>
      {submitted && <SearchResult query={submitted} />}
    </Suspense>
  )
}
```

**Q. `"react"`를 입력하고 검색을 눌렀다. suspend로 Spinner가 보였다가 결과가 도착했다 — input에는 무엇이 남아 있을까?**

1. 빈 문자열 — 언마운트됐다 리마운트됐다면 DOM 값은 사라진다
2. `"react"` — 숨겨졌을 뿐이라면 값이 그대로 남는다

<!-- input이 uncontrolled(값이 React state가 아니라 DOM에만 있음)임을 먼저 짚을 것 — controlled였다면 리마운트돼도 value로 복원되어 실험이 성립하지 않는다. 힌트 없이 먼저 투표받기. -->

---

## 퀴즈 2 — 정답: ② "react"가 유지된다

- 한 번 커밋된 트리가 다시 suspend하면, React는 **언마운트하지 않고 `display: none`으로 숨긴다**
- uncontrolled input의 값은 React state가 아니라 **DOM에만** 있다 — 값이 남아있다는 것 자체가 DOM이 살아있었다는 증거
- fallback이 걷히면 숨겨졌던 DOM이 다시 나타난다

**그런데 UX 관점의 진짜 문제**: 검색을 누를 때마다 방금 입력한 검색어까지 통째로 Spinner에 **가려진다** —

이것이 Part 3에서 다룰 **useTransition이 존재하는 이유**다.

---

## Part 3 — 원리 딥다이브: React 내부에서는

---

## 이 파트의 지도

이제 React 소스에서 이 흐름을 직접 따라간다. (이하 **react@19.2.0** 태그 기준, `packages/react-reconciler/src/`)

```mermaid
flowchart LR
    A["render 중 throw 발생"] --> B["workLoop의 try/catch (ReactFiberWorkLoop.js)"]
    B --> C{"handleThrow: 값 분류"}
    C -->|thenable| D["Suspense 경로 (ReactFiberThrow.js)"]
    C -->|Error| E["ErrorBoundary 경로 (ReactFiberThrow.js)"]
```

목표는 함수 이름 암기가 아니라, "**두 경계가 정말 한 곳에서 갈라진다**"를 눈으로 확인하는 것.

다 소화하지 못해도 괜찮다 — **Part 4의 실무는 이 파트 없이도 따라갈 수 있다.** 챙길 것은 마지막의 대칭 구조 한 장이다.

---

## 입구: workLoop의 try/catch

React의 렌더는 fiber(컴포넌트마다 하나씩 만드는 내부 작업 객체) 트리를 순회하는 루프다. 이 루프 전체가 try/catch로 감싸여 있다.

```js
// ReactFiberWorkLoop.js (단순화)
do {
  try {
    workLoopConcurrent() // 컴포넌트 함수들을 실행
    break
  } catch (thrownValue) {
    handleThrow(root, thrownValue) // 모든 throw가 여기로
  }
} while (true)
```

> 코드를 건너뛰어도 이것만 — 무엇을 던지든 **일단 전부 여기서 잡히고**, `handleThrow`가 "대기"인지 "실패"인지 분류표를 붙인다. 판별 기준은 다음 장에서.

<!-- 분류명: use()의 센티널이면 SuspendedOnImmediate, thenable 직접 throw면 SuspendedOnDeprecatedThrowPromise, 아니면 SuspendedOnError. 직접 throw 경로의 분류명에 'Deprecated'가 박혀 있다 — promise 직접 throw는 레거시 경로이고 use()가 공식 경로라는 것이 소스 레벨에 새겨져 있는 셈. -->

---

## 분류: thenable인가?

```js
// 판별의 본질 (단순화)
if (
  thrownValue !== null &&
  typeof thrownValue === 'object' &&
  typeof thrownValue.then === 'function'
) {
  // "기다려 달라"는 신호 → Suspense 경로
} else {
  // 진짜 에러 → ErrorBoundary 경로
}
```

- 덕 타이핑이다 — `then` 메서드가 있으면 thenable
- React 19의 `use()`는 `SuspenseException`이라는 센티널 값(정상 값과 구분하는 표식용 고유 객체)을 던지고 진짜 thenable은 따로 보관하지만, "**대기인지 실패인지 분류한다**"는 본질은 같다

> 코드를 건너뛰어도 이것만 — 판별 기준은 단 하나, **`.then`이 있는가**. Part 0의 대칭표가 여기서 코드가 된다.

---

## Suspense 경로: 경계 찾기

분류가 끝나면 `ReactFiberThrow.js`의 `throwException`이 처리를 맡는다.

```js
// throwException (단순화)
const suspenseBoundary = getSuspenseHandler()
if (suspenseBoundary !== null) {
  suspenseBoundary.flags |= ShouldCapture // "네가 fallback을 그려라"
  attachPingListener(root, wakeable, rootRenderLanes)
}
```

- 렌더가 Suspense 컴포넌트를 지날 때마다 **핸들러 스택**에 쌓아둔다 — `getSuspenseHandler()`는 그 스택의 맨 위, 즉 **가장 가까운 조상 Suspense**를 돌려준다
- 그 경계에 `ShouldCapture` 표시 → 다음 렌더 패스에서 fallback을 그린다

> 코드를 건너뛰어도 이것만 — **가장 가까운 Suspense를 찾아 "네가 fallback을 그려라" 표시**를 해 둔다.

---

## Suspense 경로: ping과 재시도

```js
// attachPingListener (단순화)
const ping = pingSuspendedRoot.bind(null, root, wakeable, lanes)
wakeable.then(ping, ping)
```

- 던져진 promise(wakeable)에 `.then(ping, ping)`을 걸어둔다
- resolve되면 `pingSuspendedRoot` → 스케줄러에 **해당 렌더를 다시 예약**
- 새 렌더에서 컴포넌트 함수가 처음부터 다시 실행되고, 이번엔 값이 준비돼 있으니 통과

> 코드를 건너뛰어도 이것만 — promise가 끝나면 **처음부터 다시 렌더**한다(재개가 아니라 재시도). reject여도 다시 렌더해야, 이번엔 Error가 던져져 ErrorBoundary로 간다.

---

## ErrorBoundary 경로: 같은 곳에서 갈라진다

`throwException`의 나머지 절반. thenable이 아니면 —

```js
// throwException (단순화)
let workInProgress = returnFiber
do {
  const ctor = workInProgress.type
  if (
    workInProgress.tag === ClassComponent &&
    typeof ctor.getDerivedStateFromError === 'function' // 또는 componentDidCatch
  ) {
    workInProgress.flags |= ShouldCapture // getDerivedStateFromError 실행 예약
    return
  }
  workInProgress = workInProgress.return // 부모로 한 칸씩
} while (workInProgress !== null)
```

> 코드를 건너뛰어도 이것만 — `return` 포인터로 **부모 방향으로 한 칸씩 올라가며** 에러 라이프사이클을 가진 클래스를 찾고, 루트까지 못 찾으면 트리 전체 언마운트.

---

## 대칭 구조 — 한 장으로

```mermaid
flowchart LR
    T["컴포넌트가 throw"] --> W["workLoop try/catch"] --> H{"thenable?"}
    H -->|promise| S["가장 가까운 Suspense + ping 예약"]
    H -->|Error| E["가장 가까운 ErrorBoundary + getDerivedStateFromError 예약"]
    S --> SF["fallback 커밋"] --> SR["resolve → 재시도 (자동)"]
    E --> EF["fallback 커밋"] --> ER["리셋 → 재시도 (수동)"]
```

> 아래에서 무언가를 요청하면, 위 어딘가의 핸들러가 응답한다.

<!-- 이 설계의 사고방식은 Dan Abramov의 'Algebraic Effects for the Rest of Us'에 잘 담겨 있다 — 링크는 더 읽을거리(심화)에 있음. -->

---

## 왜 state가 유지되는가

Part 2 퀴즈의 내부 답.

- fallback을 보여줄 때 React는 기존 자식 트리를 **삭제하지 않는다**
- Suspense는 자식을 숨김 모드로 전환할 뿐 — 각 컴포넌트의 fiber와 hook state(`memoizedState`)가 전부 살아있다
- DOM도 삭제 대신 `display: none` — 커밋 단계에서 스타일만 바꾼다
- 단, layout effect는 예외 — 숨겨질 때 **cleanup되고** 다시 보일 때 재실행된다. 일반(passive) effect는 유지된다 — 구독·타이머는 계속 돌고, deps가 그대로면 재실행도 없다

> state는 fiber에 산다. fiber가 살아있는 한 state도 살아있다.

(React 19.2에서 이 숨김 메커니즘은 `<Activity />`로 공식 API가 됐다 — 같은 인프라다.)

---

## useTransition과 Suspense — 이미 보인 화면을 지키기

퀴즈 2의 UX 문제(검색어까지 Spinner에 가려짐)의 해법.

```jsx
const [isPending, startTransition] = useTransition()
function handleChange(e) {
  setQuery(e.target.value) // 급한 업데이트: input은 즉시 반영
  startTransition(() => {
    setSearchQuery(e.target.value) // 느긋한 업데이트
  })
}
```

- `startTransition`으로 감싼 업데이트가 **transition**이다. React에 "급하지 않다"고 표시해 두는 것
- 이 표시가 Suspense의 동작을 바꾼다. transition이 일으킨 렌더가 suspend하면, React는 **보이던 화면을 fallback으로 되돌리지 않고 그대로 둔다**. 준비되면 교체하고, 기다리는 중이라는 표시는 `isPending`으로 하면 된다
- 퀴즈 2라면? `setSubmitted(...)`를 `startTransition`으로 감싸면 Spinner 후퇴가 사라진다
- 값 단위 버전인 `useDeferredValue(query)`도 있다 — **원인이 값 하나면** 더 간단하다

> **fallback은 "처음 나타나는 콘텐츠"의 자리다** — 이미 보여준 것을 도로 감추는 용도가 아니다.

---

## 곁가지: transition이 열어둔 문, tearing

transition 렌더는 화면을 유지한 채 뒤에서, 중간중간 브라우저에 양보(yield)하며 **끊어서** 진행된다. 그 틈에 값이 바뀌면?

```text
1. 렌더 시작: <A>가 store를 읽음 → 1
2. React가 잠시 양보 → 그 틈에 이벤트가 store를 2로 변경
3. 렌더 재개: <B>가 같은 store를 읽음 → 2
4. 커밋: 한 화면에 A는 1, B는 2 (tearing)
```

- `useState`·props는 안전하다. React가 렌더 한 번 동안 **일관된 스냅샷을 보장**한다
- 문제는 **React 바깥의 가변 store**를 렌더 중 직접 읽을 때다 (모듈 변수, 직접 만든 store)
- 해법이 **useSyncExternalStore**다. 렌더 중 store가 바뀐 것을 감지하면 동기 재렌더로 화면을 맞춘다
- Redux(v8+)·Zustand는 내부에서 이미 이것을 쓴다. store를 직접 만들 때만 기억하면 된다

<!-- 곁가지임을 밝히고 시작: transition을 쓰는 순간 렌더가 "끊길 수 있는 것"이 되면서 생기는 부작용이라는 연결만 전달. uSES는 일관성을 위해 감지 시 동기(블로킹) 재렌더로 후퇴한다 = 동시성 이득 일부를 포기하는 트레이드오프라는 점을 구두로. 상세는 reactwg 'What is tearing?' 참고. -->

---

## fallback 깜빡임과 300ms 스로틀

중첩된 Suspense가 차례로 준비되면, fallback → 콘텐츠 → 또 fallback… 화면이 연쇄적으로 튄다.

React의 내장 방어: **FALLBACK_THROTTLE_MS = 300** (ReactFiberWorkLoop.js)

- fallback을 보여준 직후 새 콘텐츠가 준비되어도, **300ms 단위로 묶어서** 공개한다
- 순식간에 여러 번 바뀌는 대신, 한 박자로 정리해 보여주는 것

주의: 이건 **"공개(reveal)"의 스로틀**이다. "로딩이 짧으면 fallback을 아예 안 보여주는" 지연 기능이 아니다 — 그 문제의 해법은 Part 4에서.

<!-- 300ms는 공개 API가 아닌 내부 상수(react@19.2.0 기준)라 버전에 따라 바뀔 수 있음을 구두로 언급할 것. -->

---

## SSR 스트리밍 + selective hydration — 지도만

Suspense는 서버에서 한 번 더 일한다. (`renderToPipeableStream`, Next.js가 쓰는 것)

1. 서버: 준비 안 된 경계는 **fallback인 채로 먼저 HTML 전송** — 나머지 준비되는 대로 스트리밍
2. 서버에서 경계 안이 **에러나면**: 그 경계까지만 지우고 fallback 전송 → 클라이언트에서 재시도 (Part 1의 "SSR 에러" 답)
3. 클라이언트: 경계 단위로 **나눠서 hydration** — 사용자가 클릭한 영역을 먼저 hydrate (selective hydration)

hydration은 서버가 보낸 정적 HTML에 이벤트 핸들러를 붙여 **동작하는 앱으로 만드는 과정**이다. 경계를 어디에 두느냐가 **스트리밍 단위이자 hydration 단위**가 된다는 것만 기억하자. 상세는 별도 덱('use client' 딥다이브)에서.

---

## Part 3 정리

- 모든 throw는 **workLoop의 try/catch → handleThrow** 한 곳으로 모인다
- **thenable이면 Suspense 경로, Error면 ErrorBoundary 경로** — 분기점은 하나
- Suspense: 핸들러 스택에서 가장 가까운 경계 + promise에 **ping** → 재시도
- ErrorBoundary: `return` 포인터로 조상을 거슬러 탐색, 못 찾으면 전체 언마운트
- fallback 동안 fiber는 살아있다 → **state 유지**
- transition 중에는 이미 보인 콘텐츠를 fallback으로 되돌리지 않는다
- 중첩 fallback 공개는 300ms 스로틀

---

## 퀴즈 3 — React는 어떻게 반응할까?

렌더 중 각 값이 throw됐다. React의 반응을 골라 연결하라.

**던져진 것**

1. `fetch()`가 반환한 promise
2. `new Error('boom')`
3. `'문자열'`
4. promise — 단, 조상에 Suspense가 하나도 없다

**보기**

- (a) 가장 가까운 ErrorBoundary가 잡는다
- (b) 가장 가까운 Suspense가 fallback을 보여준다
- (c) 루트까지 아무도 못 잡으면 트리 전체 언마운트
- (d) 커밋을 보류하고 promise가 끝날 때까지 화면을 미룬다

<!-- 3번이 함정 — thenable 판별은 instanceof Error가 아니라 .then 유무라는 것. 문자열은 Error는 아니지만 thenable도 아니므로 에러 경로. -->

---

## 퀴즈 3 — 정답

| 던져진 것            | 정답      | 이유                                                               |
| -------------------- | --------- | ------------------------------------------------------------------ |
| 1. promise           | **(b)**   | thenable → Suspense 경로                                           |
| 2. `new Error()`     | **(a→c)** | 에러 경로. 경계가 없으면 전체 언마운트                             |
| 3. `'문자열'`        | **(a→c)** | `then`이 없으니 thenable이 아니다 — **Error가 아니어도 에러 경로** |
| 4. 경계 없는 promise | **(d)**   | 루트가 대기를 떠안는다 — 초기 렌더 커밋 자체가 미뤄진다            |

> 분류 기준은 단 하나: **`.then`이 있는가.** Error 클래스인지는 보지 않는다.

실무 결론: `use()`·`useSuspenseQuery` 위에는 **항상** 명시적 `<Suspense>`를 두자 — 경계가 없으면 (d)처럼 앱 전체가 하나의 암묵적 경계가 된다.

---

## Part 4 — 실무에서 알아야 할 것들

---

## 경계 배치 전략 — 통짜 vs 조각

```jsx
// A. 페이지 전체를 하나로
<Suspense fallback={<PageSkeleton />}>
  <Header /> <Feed /> <Sidebar />
</Suspense>

// B. 영역별로
<Header />
<Suspense fallback={<FeedSkeleton />}><Feed /></Suspense>
<Suspense fallback={<SidebarSkeleton />}><Sidebar /></Suspense>
```

- **A**: 화면이 한 번에 뜬다. 가장 느린 데이터가 전체를 붙잡는다
- **B**: 준비된 것부터 보여준다. 화면이 조각조각 나타난다

정답은 없고 **디자인의 문제**다. 판단 기준 →

---

## 배치 판단 기준

1. **디자이너의 스켈레톤 단위를 따르라** — 로딩 UI가 설계된 단위가 곧 경계 단위. 코드 사정으로 경계를 긋는 게 아니라 **사용자가 인식하는 화면 영역**으로 긋는다
2. **함께 떠야 의미 있는 것은 한 경계에** — 상품 가격과 구매 버튼이 따로 뜨면 오히려 어색하다
3. **핵심 콘텐츠 vs 부가 콘텐츠 분리** — 본문은 기다리더라도, 추천 위젯이 본문을 붙잡게 하지 마라
4. ErrorBoundary도 같은 논리 — **"이 영역이 죽어도 나머지는 살아야 하는가?"** 위젯 하나의 에러가 페이지를 지우면 안 된다

> 모든 컴포넌트를 감싸는 것도, 루트 하나만 두는 것도 답이 아니다. **화면 설계 단위 = 경계 단위.**

---

## fallback 깜빡임 — 200ms 로딩에 스피너가 번쩍

짧은 로딩에 스피너가 번쩍하는 건 없느니만 못하다. React에 "짧으면 fallback 생략" 기능은 **없다** (Part 3의 스로틀은 공개 쪽 이야기). 대책 셋:

1. **스피너 대신 스켈레톤** — 콘텐츠와 같은 자리·크기라 전환이 덜 튄다
2. **CSS로 fallback 등장을 지연**

```css
.spinner {
  animation: fade-in 0ms 200ms both; /* 200ms 뒤에야 나타남 */
}
@keyframes fade-in {
  from {
    opacity: 0; /* delay 동안 both(backwards fill)가 이 값을 적용 */
  }
}
```

3. **navigation·갱신에는 useTransition** — 애초에 fallback으로 되돌리지 않는 게 최선

---

## waterfall — 중첩이 만드는 순차 로딩

```jsx
function Post({id}) {
  const post = use(getPost(id)) // 렌더가 페칭을 시작 — 1초
  return (
    <>
      <h1>{post.title}</h1>
      {/* post 도착 후에야 렌더 시작 */}
      <Comments postId={post.id} />
    </>
  )
}
// Comments 안에서 또 use(getComments(postId)) → +1초 = 총 2초
```

- suspend는 렌더를 멈추므로, **아래 컴포넌트의 페칭은 시작조차 못 한다**
- 부모 1초 + 자식 1초 = 순차 2초. 경계를 잘게 쪼갤수록 이 함정에 빠지기 쉽다

(전제: `getPost`류는 같은 인자에 **같은 promise를 돌려주는 캐시된 fetcher**다 — 아니면 Part 2의 uncached promise 문제가 그대로 난다)

---

## waterfall 해법 — 페칭을 렌더에서 분리

**render-as-you-fetch**: 렌더가 페칭을 시작하는 게 아니라, **이미 시작된 페칭을 렌더가 소비**하게 한다.

```jsx
// 라우터 loader·서버 컴포넌트 등 렌더 밖에서 미리 출발시킨 promise를 받는다
function Page({postPromise, commentsPromise}) {
  return (
    <Suspense fallback={<PostSkeleton />}>
      <Post postPromise={postPromise} />
      <Suspense fallback={<CommentsSkeleton />}>
        <Comments commentsPromise={commentsPromise} />
      </Suspense>
    </Suspense>
  )
}
```

병렬 1초. 서버 컴포넌트·라우터 loader·React Query prefetch가 전부 이 원리다.

<!-- React Query라면: 라우트 진입 콜백이나 클릭 핸들러에서 queryClient.prefetchQuery({queryKey, queryFn})를 먼저 호출해 두고, 컴포넌트의 useSuspenseQuery가 같은 queryKey로 이미 출발한 요청을 소비한다. 부모·자식이 각자 useSuspenseQuery만 부르면 waterfall 그대로라는 점을 구두 강조. -->

---

## ErrorBoundary 리셋 — "다시 시도" 만들기

에러는 resolve가 없다. **누군가 state를 되돌려줘야** 재시도된다.

```jsx
<ErrorBoundary
  FallbackComponent={Fallback} // resetErrorBoundary를 받는다
  onReset={() => refetch()} // 리셋 시 원인도 해소
  resetKeys={[userId]} // 이 값이 바뀌면 자동 리셋
>
  <Profile userId={userId} />
</ErrorBoundary>
```

- `resetErrorBoundary()` 호출 → 경계 state 초기화 → 자식 재렌더
- **주의**: 원인(실패한 요청, 잘못된 캐시)을 그대로 두고 리셋만 하면 **즉시 다시 에러**다. `onReset`에서 원인을 함께 해소할 것 — React Query와의 연결은 Part 5에서
- `resetKeys`: "다른 사용자를 보러 갔으면 이전 에러는 무의미" — 상태 변화에 따른 자동 복구

---

## key로 리셋하는 트릭

라이브러리 없이도, React의 기본기로 같은 효과를 낼 수 있다.

```jsx
<ErrorBoundary key={userId}>
  <Profile userId={userId} />
</ErrorBoundary>
```

- `key`가 바뀌면 React는 **다른 컴포넌트로 간주하고 언마운트 후 새로 마운트**한다
- 경계의 에러 state도 함께 사라진다 — 사실상 리셋
- Suspense에도 같은 트릭이 통한다: `key`를 바꾸면 유지되던 자식 state를 **의도적으로 버리고** 처음부터 다시 시작

단순하지만 강력하고 — **남용하면 state를 다 날리는** 양날의 검.

---

## 에러 로깅 — 놓치는 에러 없이

**경계에서**: `onError` (= `componentDidCatch`)에서 Sentry 등으로 전송.

```jsx
<ErrorBoundary onError={(error, info) => {
  Sentry.captureException(error, {extra: {componentStack: info.componentStack}})
}}>
```

**루트에서**: 경계 유무와 무관하게 한 곳에서 잡는다.

```jsx
createRoot(container, {
  onUncaughtError: (error, info) => {}, // 어떤 경계에도 안 잡힌 에러 (React 19 신규)
  onCaughtError: (error, info) => {}, // 경계에 잡힌 에러 (React 19 신규)
  onRecoverableError: (error, info) => {}, // hydration 불일치 등 자동 복구 (React 18부터)
})
```

- 경계가 잡았는데 **콘솔에 에러가 찍히는 건 정상** — 그 기본 로그의 주인이 `onCaughtError`다
- 둘 다 못 받는 에러(핸들러·비동기 — 퀴즈 1의 X들)는 `window` **전역 핸들러** 몫 — Sentry 기본 설정이 잡는다

<!-- Next에서는 createRoot 루트 옵션에 직접 접근할 수 없다. 서버·RSC 에러는 instrumentation.ts의 onRequestError, 클라이언트는 전역 핸들러 + Sentry Next SDK가 접점. -->

---

## 이벤트 핸들러 에러 — 경계로 보내는 법

Part 1의 미해결 문제. 핸들러·비동기 에러도 fallback UI로 처리하고 싶다면 —

```jsx
import {useErrorBoundary} from 'react-error-boundary'

function BuyButton() {
  const {showBoundary} = useErrorBoundary()
  const handleClick = async () => {
    try {
      await purchase()
    } catch (error) {
      showBoundary(error) // 렌더 흐름에 태워 경계로 던진다
    }
  }
  return <button onClick={handleClick}>구매</button>
}
```

원리는 단순하다: 에러를 state에 넣고, **다음 렌더에서 throw**한다. 렌더 중의 throw가 되므로 경계가 잡을 수 있다.

<!-- showBoundary는 react-error-boundary의 컨텍스트 전용이라 Next의 error.tsx 경계에는 통하지 않는다. error.tsx로 보내려면 이 슬라이드의 원리 그대로(에러를 state에 넣고 다음 렌더에서 throw) 직접 구현한다. -->

---

## Part 4 정리

- 경계 배치 = **화면 설계 단위.** 스켈레톤 단위, "이 영역이 죽어도 나머지는 살아야 하나"
- 깜빡임: 스켈레톤, CSS 등장 지연, **useTransition**
- waterfall: suspend는 아래 페칭을 막는다 → **페칭을 렌더보다 먼저 출발**시켜라
- 리셋: `resetErrorBoundary` + `onReset`으로 **원인까지 해소**, `resetKeys`/`key`로 자동 리셋
- 로깅: 경계 `onError` + 루트 `onUncaughtError`/`onCaughtError`
- 핸들러 에러는 `showBoundary`로 렌더 흐름에 태운다

---

## 퀴즈 4 — 경계를 어디에 둘까

쇼핑몰 상품 페이지. 요구사항:

- **상품 정보**(이미지·가격·구매 버튼): 이 페이지의 존재 이유. 가격과 구매 버튼이 따로 뜨면 안 됨
- **리뷰 목록**: 무거운 데이터. 늦게 떠도 되고, 실패해도 페이지는 살아야 함
- **추천 상품**: 서드파티 API. 자주 죽는다. 죽으면 그냥 안 보여도 됨

```jsx
<ProductInfo />   // 이미지·가격·구매 버튼
<Reviews />
<Recommended />
```

**Q. Suspense와 ErrorBoundary를 어떻게 감싸겠는가?** (1분 설계)

<!-- 정답은 하나가 아님을 미리 말해두기. 각자 그려보게 한 뒤 다음 장과 비교. -->

---

## 퀴즈 4 — 설계 예시

```jsx
<ErrorBoundary FallbackComponent={PageError}>
  <Suspense fallback={<ProductSkeleton />}>
    <ProductInfo />
  </Suspense>
  <ErrorBoundary FallbackComponent={ReviewsError}>
    <Suspense fallback={<ReviewsSkeleton />}>
      <Reviews />
    </Suspense>
  </ErrorBoundary>
  <ErrorBoundary fallback={null}>
    <Suspense fallback={<RecommendedSkeleton />}>
      <Recommended />
    </Suspense>
  </ErrorBoundary>
</ErrorBoundary>
```

상품 정보는 한 경계로 **함께**, 리뷰는 **죽어도 페이지는 살게**, 추천은 `fallback={null}`로 **조용히**. 상품 정보에 전용 경계가 없는 것도 의도다 — **핵심이 죽으면 페이지 전체가 에러인 게 맞다**.

<!-- fallback={null}로 접히면 layout shift가 생긴다. 자리 유지가 중요한 레이아웃이면 min-height 예약이나 빈 자리 대체 UI를 두는 트레이드오프가 있다고 구두 보충. -->

---

## Part 5 — React Query와 함께 쓰기

---

## useSuspenseQuery — 페칭을 suspend로

```jsx
import {useSuspenseQuery} from '@tanstack/react-query'

function Profile({id}) {
  const {data} = useSuspenseQuery({
    queryKey: ['user', id],
    queryFn: () => fetchUser(id),
  })
  return <h1>{data.name}</h1> // data는 항상 있다 — 타입도 undefined 아님
}
```

- pending이면 내부에서 **promise를 throw** — Part 2에서 만든 `createResource`의 프로덕션 버전 (TanStack Query v5부터, React 18 가능)
- `isLoading` 분기가 사라지고, `data`의 타입에서 `undefined`가 사라진다. 로딩 UI는 부모의 `<Suspense>`가 담당
- 갱신 중 표시는? fallback은 다시 뜨지 않는다 — `isFetching`으로 **인라인** 표시. 초기 로드는 경계, 갱신은 인라인

<!-- Q&A 대비: React 18에서 가능 — Suspense/ErrorBoundary/lazy/useTransition/useSuspenseQuery(v5). 불가 — use(), Activity, onCaughtError류 루트 옵션(React 19). v4는 useQuery의 suspense:true 옵션 시절. 서버 페칭이 기본인 App Router에서는 상호작용 후 갱신·실시간·클라 전용 영역에 쓰고, SSR 결합은 서버 prefetchQuery + HydrationBoundary가 공식 패턴(TanStack SSR 가이드). -->

---

## throwOnError — 에러를 경계로

```jsx
const {data, error} = useQuery({
  queryKey: ['user', id],
  queryFn: () => fetchUser(id),
  throwOnError: (error) => error.response?.status >= 500,
})
```

- 기본 `useQuery`는 에러를 **반환값으로** 준다 (`error` 필드) — 경계까지 가지 않는다
- `throwOnError: true`면 렌더 중 **throw** → ErrorBoundary가 잡는다
- 함수를 주면 **선별**할 수 있다: 위 예시는 "500번대(서버 장애)는 경계로, 404 같은 것은 컴포넌트에서 직접" — 실무에서 가장 유용한 형태
- `useSuspenseQuery`는 **보여줄 캐시 데이터가 없을 때** 에러를 throw한다 (있으면 stale 데이터 유지) — `throwOnError` 옵션 자체가 없어 이 동작은 끌 수 없다

---

## 완성형: 경계 3종 세트

```jsx
// @tanstack/react-query + react-error-boundary
<QueryErrorResetBoundary>
  {({reset}) => (
    <ErrorBoundary onReset={reset} FallbackComponent={Fallback}>
      <Suspense fallback={<Skeleton />}>
        <Profile id={id} />
      </Suspense>
    </ErrorBoundary>
  )}
</QueryErrorResetBoundary>
```

- 리셋할 땐 **원인(실패한 쿼리)도 함께** (Part 4) — `reset`을 `onReset`에 연결하면 "다시 시도"가 refetch까지 한다
- 순서에도 이유가 있다 — promise는 ErrorBoundary를 **그냥 통과**하지만(Part 3), ErrorBoundary가 바깥이어야 **fallback 자체의 에러까지** 받는다
- 이 3종 조합이 Suspense 기반 데이터 페칭의 사실상 표준 패턴

<!-- "영역마다 매번 감싸야 하나?" — 반복이 부담이면 셋을 묶은 공통 래퍼(이른바 AsyncBoundary 패턴)를 만들어 팀 컨벤션으로 쓰는 것이 일반적. children 자리의 함수는 render prop 패턴이고, 훅 대안으로 useQueryErrorResetBoundary도 있다. -->

---

## Part 5 정리 — 배운 메커니즘 그대로

| 이 강의에서 배운 것         | React Query에서의 모습       |
| --------------------------- | ---------------------------- |
| promise throw (Part 2)      | `useSuspenseQuery`의 pending |
| Error throw (Part 1)        | `throwOnError`               |
| 재시도엔 캐시 필수 (Part 2) | QueryCache가 해결            |
| 리셋 시 원인 해소 (Part 4)  | `QueryErrorResetBoundary`    |

새로운 개념은 하나도 없다. **라이브러리는 프로토콜의 구현체**일 뿐 — 메커니즘을 알면 어떤 라이브러리든 같은 그림으로 읽힌다.

---

## 퀴즈 5 — 3종 세트, 제대로 이해했나

완성형 코드(`QueryErrorResetBoundary` + `ErrorBoundary` + `Suspense`) 기준으로:

1. `useSuspenseQuery`가 pending이면 → 어느 경계가 받나?
2. `queryFn`이 reject되면 (보여줄 캐시도 없음) → 어느 경계가 받나?
3. `QueryErrorResetBoundary` 없이 `resetErrorBoundary()`만 누르면?
4. `ErrorBoundary`와 `Suspense`의 **중첩 순서를 서로 바꾸면** 무엇이 달라지나?

<!-- 4번이 핵심 — Part 3의 메커니즘에서 답을 유도할 수 있는지 확인하는 문제. 정답 공개 전에 각자 답하게 할 것. -->

---

## 퀴즈 5 — 정답

1. **Suspense** — pending은 promise throw다 (Part 2)
2. **ErrorBoundary** — 캐시가 없으면 Error throw다 (Part 1·5)
3. **즉시 다시 에러** — 실패한 쿼리가 캐시에 그대로라, 재렌더에서 또 throw된다. Part 4의 "원인까지 해소" 그대로
4. **둘 다 그대로 동작한다** — thenable은 핸들러 스택으로, Error는 `return` 포인터로 **각자 따로** 자기 경계를 찾는다 (Part 3). 차이는 에러 시 **대체되는 UI 범위**뿐 (ErrorBoundary가 바깥이면 로딩 경계까지 함께 error fallback으로 교체)

---

## 전체 체크리스트

**설계할 때**

1. 경계 단위 = 스켈레톤 단위 = "죽어도 되는" 단위인가?
2. 최상단에 최후의 ErrorBoundary가 있는가?
3. 페칭이 렌더보다 먼저 출발하는가? (waterfall 점검)

**구현할 때**

4. 핸들러·비동기 에러는 try/catch 또는 `showBoundary`로 처리했는가?
5. "다시 시도"가 원인(쿼리)까지 리셋하는가?
6. 경계에 잡힌 에러도 로깅되는가?

**다듬을 때**

7. 짧은 로딩에 fallback이 번쩍이거나, 보이던 화면이 fallback으로 후퇴하지 않는가? (스켈레톤·`useTransition`)

---

## 한 줄 요약

> 컴포넌트는 **성공한 화면만** 그리고,
> 준비 안 됐으면 promise를, 실패했으면 Error를 **던진다**.
> 위 어딘가의 경계가 받아서, **그 영역만** 로딩과 실패를 대신 그린다.

나머지는 전부 이 문장의 각주다.

---

## Q&A

<!-- _class: invert -->

---

## Q. Next.js의 loading.tsx / error.tsx와는 무슨 관계인가요?

**그게 바로 이 강의 내용입니다.** App Router는 `loading.tsx`/`error.tsx`를 둔 세그먼트마다 자동으로 경계를 감싼다:

```text
<ErrorBoundary fallback={<Error />}>      ← error.tsx
  <Suspense fallback={<Loading />}>       ← loading.tsx
    <Page />
  </Suspense>
</ErrorBoundary>
```

- `error.tsx`가 클라이언트 컴포넌트여야 하는 이유 → ErrorBoundary는 클래스 컴포넌트 기반 (Part 1)
- `error.tsx`가 받는 `reset` prop → Part 4의 리셋 패턴. 단 `reset()`은 재렌더만 하므로, **원인까지 해소**하려면 `router.refresh()`와 함께 쓴다
- `loading.tsx`가 스트리밍 단위가 되는 이유 → Part 3의 SSR 스트리밍

파일 컨벤션은 설탕이고, 알맹이는 오늘 배운 두 경계다.

<!-- 클래스는 렌더 사이에 살아있는 인스턴스가 필요해, 요청당 한 번 실행되고 끝나는 서버 컴포넌트가 될 수 없다 — error.tsx에 'use client'가 필수인 이유. error.tsx를 하나도 안 두면 Next의 기본 global-error가 최후에 받는다(흰 화면은 아니지만 앱 전체가 죽는 건 같다). -->

---

## Q. 그럼 모든 useQuery를 useSuspenseQuery로 바꿔야 하나요?

아니다. 선택 기준:

| 상황                           | 선택                              |
| ------------------------------ | --------------------------------- |
| 화면 영역 전체가 데이터에 의존 | `useSuspenseQuery`                |
| 조건부 페칭 (`enabled`)이 필요 | `useQuery` — suspense 버전엔 없다 |
| 있으면 좋고 없어도 되는 데이터 | `useQuery` + 자체 분기            |
| 병렬 여러 쿼리                 | `useSuspenseQueries`              |

`useSuspenseQuery`를 한 컴포넌트에 연달아 쓰면 첫 suspend에서 **함수 실행이 멈춰** 둘째는 시작도 못 한다 — 그래서 병렬은 `useSuspenseQueries`다.

두 방식은 **한 페이지 안에서도 공존한다** — 신규 화면·독립 위젯부터 경계 단위로 넓혀가면 된다. suspend는 "이 데이터 없이는 못 그린다"는 선언이니, 아닌 데이터까지 suspend시키면 화면만 늦어진다.

---

## 더 읽을거리 — 필수

- [\<Suspense\> — React 공식](https://react.dev/reference/react/Suspense)
- [Catching rendering errors with an error boundary — React 공식](https://react.dev/reference/react/Component#catching-rendering-errors-with-an-error-boundary)
- [use — React 공식](https://react.dev/reference/react/use)
- [react-error-boundary — GitHub](https://github.com/bvaughn/react-error-boundary)

### 심화

- [Algebraic Effects for the Rest of Us — Dan Abramov](https://overreacted.io/algebraic-effects-for-the-rest-of-us/)
- [New Suspense SSR Architecture in React 18 — reactwg](https://github.com/reactwg/react-18/discussions/37)
- [What is tearing? — reactwg](https://github.com/reactwg/react-18/discussions/69)

---

## 더 읽을거리 — 내부 & React Query

### 소스 (react@19.2.0)

- [ReactFiberWorkLoop.js — handleThrow](https://github.com/facebook/react/blob/v19.2.0/packages/react-reconciler/src/ReactFiberWorkLoop.js)
- [ReactFiberThrow.js — throwException](https://github.com/facebook/react/blob/v19.2.0/packages/react-reconciler/src/ReactFiberThrow.js)
- [How Suspense works internally in Concurrent Mode — JSer.dev (react@18.2 기준)](https://jser.dev/react/2022/04/02/suspense-in-concurrent-mode-1-reconciling)
- [How does useTransition() work internally — JSer.dev](https://jser.dev/2023-05-19-how-does-usetransition-work/)

### React Query

- [Suspense — TanStack Query 공식](https://tanstack.com/query/latest/docs/framework/react/guides/suspense)
- [React Query Error Handling — tkdodo](https://tkdodo.eu/blog/react-query-error-handling)

---

# 감사합니다

<!-- _class: invert -->

@yceffort
