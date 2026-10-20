---
title: 리액트 렌더링 최적화 가이드
marp: true
paginate: true
theme: yceffort
tags:
  - react
  - web-performance
date: 2026-04-13
description: '객체 참조, memo, useMemo, useCallback, 그리고 구조적 최적화까지 — 실무에서 바로 쓰는 렌더링 최적화 판단 기준'
published: true
art:
  undraw: performance-comparison
---

# 리액트 렌더링 최적화 가이드

객체 참조, memo, 그리고 "언제 최적화할 것인가"

<!-- _class: invert deck-cover -->

@yceffort

---

## 이 강의에서 다루는 것

1. 리렌더링은 **언제** 일어나는가
2. 객체 props의 함정
3. 의존성 배열과 객체 — 훅/컴포넌트 만드는 사람이 알아야 할 것
4. `React.memo` 제대로 쓰기
5. `useMemo` / `useCallback`
6. memo 없이 구조로 해결하기
7. React DevTools Profiler로 측정하기

---

## React Compiler 이후의 최적화 지도

React Compiler는 `useMemo` / `useCallback` / `React.memo`를 **자동 삽입**해준다. 그럼 뭐가 남는가?

**자동화되는 영역**: Part 3 (`React.memo`) · Part 4 (`useMemo` / `useCallback`)

**컴파일러 이후에도 남는 영역**

- Part 0 — 리렌더링 원리 (디버깅의 토대)
- Part 2 — 훅 deps 설계 (라이브러리 API 설계)
- Part 5 — 구조적 최적화 (컴파일러 무관)
- Part 6 — 측정

> 컴파일러는 "어떻게 메모할까"를 대신해준다. "어디서부터 렌더가 새어 나오는가"는 여전히 우리가 안다.

---

## Part 0 — 리렌더링은 언제 일어나는가

---

## 리렌더링 = 함수를 다시 실행하는 것

함수형 컴포넌트는 **그냥 함수**다.

```jsx
function Greeting({name}) {
  const message = `안녕하세요, ${name}님`
  return <p>{message}</p>
}
```

리렌더링이 일어나면, React는 이 함수를 **처음부터 다시 호출**한다. `message`도 다시 만들어지고, JSX도 다시 만들어진다. 매번.

> 리렌더링이 많다 = 이 함수가 많이 호출된다.
> 리렌더링이 비싸다 = 이 함수 안에서 하는 일이 많다.

이걸 기억하고 있으면 뒤에 나오는 모든 최적화가 **"이 함수를 덜 호출하거나, 호출됐을 때 할 일을 줄이는 것"**으로 귀결된다.

---

## 흔한 오해

> "props가 바뀌어서 리렌더링되는 거 아닌가요?"

**반쯤 맞고 반쯤 틀리다.** 부모가 리렌더되면 props가 바뀐 것처럼 보이지만, 순서가 거꾸로다.

리렌더링의 실제 트리거는 딱 세 가지:

1. **`setState`** 호출
2. **부모 컴포넌트**가 리렌더링됨
3. **Context** 값이 변경됨

부모가 리렌더되면, 자식은 **props가 같든 다르든** 일단 함수가 다시 호출된다. React는 기본적으로 props를 비교하지 않는다 — 비교를 시키려면 `React.memo`를 명시적으로 걸어야 한다.

---

## 직접 보기

```jsx
function Parent() {
  const [count, setCount] = useState(0)
  console.log('Parent render')
  return (
    <>
      <button onClick={() => setCount((c) => c + 1)}>+</button>
      <Child />
    </>
  )
}

function Child() {
  console.log('Child render') // 매번 찍힘
  return <div>나는 props도 없는데?</div>
}
```

버튼 클릭할 때마다 `Child render`가 찍힌다. `Child`는 props가 없는데도.

---

## 왜 이렇게 동작하나

React의 렌더링은 **트리 탐색**이다.

1. `setState` → 해당 컴포넌트부터 시작
2. JSX를 실행하면서 자식 컴포넌트 함수를 **전부 호출**
3. 반환된 ReactElement 트리를 이전 fiber와 비교 (reconciliation)
4. 차이가 있는 부분만 DOM에 반영 (commit)

2번에서 이미 자식 함수가 호출된다. props 비교는 **기본적으로 하지 않는다.** 비교를 시키려면 명시적으로 `React.memo`를 써야 한다.

---

## 이게 왜 중요한가 — 실시간 데이터

증권 HTS, 코인 거래소, 실시간 대시보드 같은 도메인을 생각해보자.

```text
웹소켓으로 호가 데이터 수신 → setState → 리렌더
```

이게 **초당 수십~수백 회** 일어난다. 컴포넌트 트리가 깊고 자식이 많으면, 한 번의 setState가 수백 개의 함수를 다시 호출하는 셈이다.

일반적인 CRUD 앱에서는 리렌더 최적화가 "있으면 좋은 것"이지만, 이런 도메인에서는 **안 하면 프레임이 뚝뚝 끊기는 것**이다.

---

## Part 1 — 객체 props의 함정

---

## 이 코드의 문제는?

```jsx
function UserCard({user}) {
  return (
    <Card
      style={{padding: 16, border: '1px solid #eee'}}
      config={{showAvatar: true, size: 'large'}}
    >
      {user.name}
    </Card>
  )
}
```

`Card`가 `React.memo`로 감싸져 있어도, **매 렌더마다 리렌더된다.**

---

## `Object.is`와 참조 비교

`React.memo`는 props를 **얕은 비교**한다. 얕은 비교의 핵심은 `Object.is`.

```js
// 매 렌더마다 새 객체가 만들어짐
{ padding: 16 } === { padding: 16 } // false
Object.is({ padding: 16 }, { padding: 16 }) // false

// 같은 참조면 통과
const style = { padding: 16 }
Object.is(style, style) // true
```

인라인 객체 `{}`는 함수가 호출될 때마다 **새 참조**다. 내용이 같아도 `Object.is`는 `false`.

---

## 인라인 배열도 마찬가지

```jsx
// ❌ 매 렌더마다 새 배열
<Select options={['apple', 'banana', 'cherry']} />

// ❌ 매 렌더마다 새 배열
<TagList tags={user.hobbies.filter(h => h.active)} />
```

`[]` 리터럴, `.filter()`, `.map()`, `.slice()` — 전부 새 참조를 만든다.

---

## 해결: 상수 추출

값이 변하지 않는 객체/배열은 **컴포넌트 바깥으로** 빼면 된다.

```jsx
// ✅ 모듈 레벨 상수 — 참조가 절대 안 바뀜
const cardStyle = {padding: 16, border: '1px solid #eee'}
const cardConfig = {showAvatar: true, size: 'large'}

function UserCard({user}) {
  return (
    <Card style={cardStyle} config={cardConfig}>
      {user.name}
    </Card>
  )
}
```

비용: 0. 코드도 더 깔끔해진다. 이건 최적화가 아니라 **좋은 습관**이다.

---

## 해결: 동적 값은 `useMemo`

props나 state에 따라 달라지는 객체는 `useMemo`로 참조를 안정시킨다.

```jsx
function UserCard({user, isCompact}) {
  const cardStyle = useMemo(
    () => ({
      padding: isCompact ? 8 : 16,
      border: '1px solid #eee',
    }),
    [isCompact],
  )

  return <Card style={cardStyle}>{user.name}</Card>
}
```

`isCompact`가 안 바뀌면 같은 참조. `Card`가 memo되어 있다면 리렌더 스킵.

---

## Part 2 — 의존성 배열과 객체

---

## deps에 객체를 넣으면 생기는 일

```tsx
interface AnalyticsConfig {
  event: string
  category: string
  label: string
}

function useAnalytics(config: AnalyticsConfig) {
  // ❌ config 객체의 참조가 바뀔 때마다 실행
  useEffect(() => {
    track(config.event, config.category, config.label)
  }, [config])
}

// 사용하는 쪽
useAnalytics({event: 'click', category: 'button', label: 'submit'})
// 매 렌더마다 새 객체 → effect 매번 실행
```

---

## 해결: destructuring으로 primitive만 의존

```tsx
function useAnalytics({event, category, label}: AnalyticsConfig) {
  // ✅ primitive만 deps에
  useEffect(() => {
    track(event, category, label)
  }, [event, category, label])
}
```

객체를 받더라도, **필요한 필드를 꺼내서** primitive 단위로 deps에 넣는다.

---

## 훅/재사용 컴포넌트를 만드는 사람의 책임

핵심 원칙:

> **내 훅을 쓰는 사람이 참조 안정한 객체를 넘겨줄 거라고 가정하지 말 것.**

```jsx
// ❌ 호출자가 안정적인 options를 넘겨주길 기대
function useSearch(options) {
  useEffect(() => {
    fetch(`/api?q=${options.query}&limit=${options.limit}`)
  }, [options]) // 호출자가 매번 새 객체를 넘기면 무한 fetch
}

// ✅ 방어적으로 destructuring
function useSearch({query, limit}) {
  useEffect(() => {
    fetch(`/api?q=${query}&limit=${limit}`)
  }, [query, limit]) // primitive라서 안전
}
```

---

## 원칙 정리

```text
객체를 받는다 → destructure해서 primitive만 deps에
```

이렇게 하면:

- 호출자가 인라인 객체를 넘겨도 안전
- deps 비교가 예측 가능
- 불필요한 effect/memo 재실행 방지

이건 "최적화"가 아니라, **라이브러리/훅 설계의 기본**이다.

---

## Part 3 — `React.memo` 제대로 쓰기

---

## `React.memo`가 하는 일

```jsx
const MemoizedChild = React.memo(function Child({name, age}) {
  console.log('Child render')
  return (
    <div>
      {name}, {age}
    </div>
  )
})
```

부모가 리렌더될 때:

1. 이전 props와 새 props를 **얕은 비교** (`Object.is`로 각 prop)
2. 전부 같으면 → 함수 호출 스킵, 이전 결과 재사용
3. 하나라도 다르면 → 정상 리렌더

---

## memo가 무력화되는 케이스들

```jsx
function Parent() {
  return (
    <MemoizedChild
      // ❌ 인라인 객체 → 매번 새 참조
      style={{color: 'red'}}
      // ❌ 인라인 함수 → 매번 새 참조
      onClick={() => console.log('click')}
      // ❌ JSX children → 매번 새 ReactElement
      header={<Icon name="user" />}
    />
  )
}
```

이 세 가지 중 **하나라도** 있으면, memo는 매번 비교만 하고 결국 리렌더한다. 비교 비용만 추가되는 셈.

---

## memo를 살리려면

```jsx
const style = {color: 'red'} // 상수 추출

function Parent() {
  const handleClick = useCallback(() => {
    console.log('click')
  }, [])

  const header = useMemo(() => <Icon name="user" />, [])

  return <MemoizedChild style={style} onClick={handleClick} header={header} />
}
```

memo + `useCallback` + `useMemo`가 **세트**로 움직여야 효과가 있다.

---

## 책임 소재: 훅 deps vs memo props

|            | 방어하는 곳           | 방법                               |
| ---------- | --------------------- | ---------------------------------- |
| 훅 deps    | **받는 쪽** (훅 내부) | destructuring → primitive만 deps에 |
| memo props | **넘기는 쪽** (부모)  | 상수 추출 / useMemo / useCallback  |

Card가 내부적으로 아무리 잘 만들어져 있어도, 부모가 `style={{ ... }}`을 인라인으로 넘기면 memo는 매번 뚫린다. **memo의 비교는 Card 함수가 실행되기 전에 일어나니까.**

훅은 받는 쪽이 방어할 수 있지만, memo는 넘기는 쪽이 책임져야 한다.

---

## 언제 memo를 걸어야 하나

걸어야 할 때:

- 리스트의 각 아이템 (`<TodoItem />` × 100)
- 부모가 자주 리렌더되는데, 자식의 렌더링 비용이 비쌀 때
- 이미 Profiler로 불필요한 리렌더를 **확인한** 후

안 걸어도 되는 경우:

- props가 매번 바뀌는 컴포넌트 (비교만 낭비)
- 렌더 비용이 극히 낮은 leaf 컴포넌트
- 자식이 없거나 적은 컴포넌트

---

## Part 4 — `useMemo` / `useCallback`

---

## `useMemo`의 두 가지 용도

```jsx
// 용도 1: 비싼 계산 캐싱
const sorted = useMemo(() => items.sort((a, b) => a.score - b.score), [items])

// 용도 2: 참조 안정성
const filters = useMemo(() => ({status: 'active', role}), [role])
```

용도 1은 직관적이다. 용도 2는 **memo된 자식에 넘기거나, deps에 쓰일 때** 의미가 있다.

---

## `useCallback` — 함수 참조 안정성

```jsx
// ❌ 매 렌더마다 새 함수
<MemoizedList onItemClick={(id) => selectItem(id)} />

// ✅ 참조 유지
const handleItemClick = useCallback(
  (id) => selectItem(id),
  [selectItem],
)
<MemoizedList onItemClick={handleItemClick} />
```

`useCallback`은 `useMemo(() => fn, deps)`의 축약형이다. **memo된 자식에 함수를 넘길 때** 짝으로 쓴다.

---

## 그러면 전부 다 감싸야 하나?

```jsx
// 이렇게 하는 사람 많음
function Form() {
  const name = useMemo(() => 'hello', []) // 🤦 문자열에 useMemo
  const len = useMemo(() => name.length, [name]) // 🤦 사칙연산에 useMemo
  const log = useCallback(() => {
    // 🤦 아무 데도 안 넘기는 함수
    console.log(len)
  }, [len])
  // ...
}
```

`useMemo`/`useCallback` 자체도 비용이 있다:

- deps 배열 생성 + 이전 deps와 비교
- 클로저가 이전 값을 참조하고 있으면 GC 지연

---

## 판단 기준

```text
"이걸 안 쓰면 뭐가 깨지는가?"
```

| 상황                                        | 필요한가? |
| ------------------------------------------- | --------- |
| memo된 자식에 넘기는 객체/함수              | ✅ 필요   |
| deps 배열에 쓰이는 객체                     | ✅ 필요   |
| 배열 정렬/필터, 트리 탐색 등 O(n) 이상 연산 | ✅ 필요   |
| 어디에도 안 넘기는 로컬 변수                | ❌ 불필요 |
| primitive 값                                | ❌ 불필요 |
| 매 렌더마다 어차피 deps가 바뀌는 경우       | ❌ 무의미 |

---

## "비용 0인 좋은 습관" vs "코드 복잡도를 올리는 최적화"

이 강의에서 계속 반복되는 기준이 하나 있다.

> **비용이 거의 0이고 코드도 깔끔해지면 — 그건 최적화가 아니라 좋은 습관이다.**

- 인라인 객체를 상수로 빼기 → 비용 0, 코드 더 깨끗 → **그냥 하는 것**
- 훅에서 객체 destructuring → 비용 0, API가 더 견고 → **그냥 하는 것**
- 리스트 아이템에 memo + useCallback → 약간의 코드 추가, 하지만 효과 명확 → **하는 것**
- 모든 변수에 useMemo → 코드 복잡도 ↑, 효과 불확실 → **하지 않는 것**

판단이 어려우면? **하는 쪽으로 기울어져도 괜찮다.** 다만 그건 측정으로 검증할 것.

---

## Part 5 — 구조로 해결하기

---

## memo 없이도 리렌더 범위를 줄일 수 있다

여기서부터는 `memo`, `useMemo`, `useCallback` 없이 **컴포넌트 구조만으로** 불필요한 리렌더를 제거하는 패턴들.

---

## 패턴 1: State 내리기

```jsx
// ❌ 전체 트리가 리렌더
function Page() {
  const [search, setSearch] = useState('')
  return (
    <>
      <input value={search} onChange={(e) => setSearch(e.target.value)} />
      <HeavyList /> {/* search와 무관한데 매번 리렌더 */}
      <HeavyChart /> {/* search와 무관한데 매번 리렌더 */}
    </>
  )
}
```

---

## State 내리기 — After

```jsx
// ✅ state를 사용하는 곳으로 이동
function SearchInput() {
  const [search, setSearch] = useState('')
  return <input value={search} onChange={(e) => setSearch(e.target.value)} />
}

function Page() {
  return (
    <>
      <SearchInput />
      <HeavyList /> {/* 리렌더 안 됨 */}
      <HeavyChart /> {/* 리렌더 안 됨 */}
    </>
  )
}
```

state가 바뀌어도 `SearchInput` 안에서만 리렌더가 일어난다.

---

## 패턴 2: Children as props — Before

```jsx
// ❌ color가 바뀔 때마다 HeavyContent도 리렌더
function ColorPicker() {
  const [color, setColor] = useState('#000')
  return (
    <div style={{backgroundColor: color}}>
      <input
        type="color"
        value={color}
        onChange={(e) => setColor(e.target.value)}
      />
      <HeavyContent />
    </div>
  )
}
```

`color`가 바뀌면 `ColorPicker`가 리렌더 → 자식인 `HeavyContent`도 리렌더.

---

## 패턴 2: Children as props — After

```jsx
function ColorPicker({children}) {
  const [color, setColor] = useState('#000')
  return (
    <div style={{backgroundColor: color}}>
      <input
        type="color"
        value={color}
        onChange={(e) => setColor(e.target.value)}
      />
      {children}
    </div>
  )
}
```

`HeavyContent`를 자식으로 받도록 바꾼다.

---

## 패턴 2: Children as props — 호출부

```jsx
// ✅ App에서 children으로 넘긴다
;<ColorPicker>
  <HeavyContent />
</ColorPicker>
```

`color`가 바뀌면 `ColorPicker`는 리렌더되지만, `<HeavyContent />`는 **App이 만든 ReactElement**다. App은 리렌더되지 않았으므로 **같은 참조** → 스킵.

---

## 패턴 3: 변하는 부분을 분리

```jsx
// ❌ 스크롤 위치 때문에 전체가 매 프레임 리렌더
function ScrollPage() {
  const [scrollY, setScrollY] = useState(0)
  useEffect(() => {
    const h = () => setScrollY(window.scrollY)
    window.addEventListener('scroll', h)
    return () => window.removeEventListener('scroll', h)
  }, [])

  return (
    <div>
      <ProgressBar progress={scrollY / maxScroll} />
      <Article /> {/* 안 바뀌는데 매 프레임 리렌더 */}
      <Comments /> {/* 안 바뀌는데 매 프레임 리렌더 */}
    </div>
  )
}
```

---

## 변하는 부분을 분리 — After

```jsx
function ScrollProgress() {
  const [scrollY, setScrollY] = useState(0)
  useEffect(() => {
    const h = () => setScrollY(window.scrollY)
    window.addEventListener('scroll', h)
    return () => window.removeEventListener('scroll', h)
  }, [])
  return <ProgressBar progress={scrollY / maxScroll} />
}

function ScrollPage() {
  return (
    <>
      <ScrollProgress />
      <Article /> <Comments /> {/* 리렌더 안 됨 */}
    </>
  )
}
```

자주 바뀌는 state를 별도 컴포넌트로 쪼개면 나머지는 영향을 받지 않는다.

---

## 패턴 4: 구독 기반 상태로 범위 좁히기

```jsx
// ❌ context 하나에 다 넣으면, 뭐 하나만 바뀌어도 모든 consumer 리렌더
<AppContext.Provider value={{user, theme, notifications, cart}}>
```

```jsx
// ✅ zustand/jotai — selector가 반환한 값이 바뀔 때만 리렌더
const userName = useStore((s) => s.user.name)
```

`cart`가 바뀌어도 `userName`만 구독하는 컴포넌트는 영향 없음.

React 빌트인으로는 **`useSyncExternalStore`**가 있다. 외부 스토어(웹소켓, 브라우저 API, 커스텀 이벤트버스 등)를 리렌더와 연결하면서 selector로 범위를 좁히는 표준 API.

> context value를 memo로 감싸는 것보다, 애초에 **필요한 조각만 구독하게** 만드는 게 근본적이다.

---

## 패턴 5: 우선순위로 처리 — `useTransition` / `useDeferredValue`

```jsx
function SearchPage() {
  const [query, setQuery] = useState('')
  const [, startTransition] = useTransition()
  return (
    <>
      <input
        onChange={(e) => startTransition(() => setQuery(e.target.value))}
      />
      <HeavyList query={query} />
    </>
  )
}
```

무거운 렌더를 **낮은 우선순위**로 표시. 더 급한 렌더(input 반응)가 들어오면 React가 리스트 렌더를 중단·재시작한다. `useDeferredValue`는 값 자체에 지연을 걸어 props 전달에 쓰기 좋다.

> "리렌더를 없애는 것"이 아니라 "UX를 막지 않게 밀어두는 것."

---

## 패턴 6: 경계 밖으로 — Server Components

```jsx
// app/page.tsx — 기본값은 서버 컴포넌트
export default async function Page() {
  const posts = await db.posts.findMany()
  return (
    <>
      <PostList posts={posts} /> {/* 서버에서 렌더, 클라 번들 0 */}
      <InteractiveFilter /> {/* 'use client' — 여기만 클라이언트 */}
    </>
  )
}
```

Server Component는 **애초에 클라이언트 리렌더 대상이 아니다.** 번들에도 포함되지 않고, hydration도 없다.

`'use client'` 경계를 **최대한 좁게** 잡는 것이 최상위 구조적 최적화다.

> 리렌더 최적화의 최종 형태: **렌더할 컴포넌트 자체를 줄이는 것.**

---

## 구조적 최적화 정리

| 패턴              | 핵심                                  | memo 필요? |
| ----------------- | ------------------------------------- | ---------- |
| State 내리기      | state를 사용하는 곳으로 이동          | 불필요     |
| Children as props | 부모가 children을 전달                | 불필요     |
| 컴포넌트 분리     | 변하는 부분과 안 변하는 부분을 쪼개기 | 불필요     |
| 구독 기반 상태    | selector로 구독 범위를 좁힘           | 불필요     |
| 우선순위 조정     | `useTransition` / `useDeferredValue`  | 불필요     |
| 서버 경계         | `'use client'` 바깥으로 밀기          | 불필요     |

> memo를 쓰기 전에 먼저 구조를 의심하라.

---

## Part 6 — React DevTools Profiler

---

## Profiler 켜기

1. React DevTools 설치 (Chrome / Firefox 확장)
2. DevTools → **Profiler** 탭
3. ⚙️ 설정에서 **"Record why each component rendered"** 체크
4. 녹화 시작 (⏺) → 동작 수행 → 녹화 중지 (⏹)

---

## Flamegraph 읽는 법

Flamegraph에서 각 바는 하나의 컴포넌트.

- **회색**: 이번 커밋에서 리렌더 안 됨
- **파란~노란색**: 리렌더됨. 노란색에 가까울수록 오래 걸림
- **바의 너비**: 렌더링 소요 시간 (자식 포함)

주의: **회색 바가 아닌 것들이 최적화 대상 후보**다.

---

## "Why did this render?"

컴포넌트를 클릭하면 오른쪽 패널에 렌더 이유가 나온다.

흔한 이유들:

- `The parent component rendered` — 부모 리렌더 연쇄
- `Props changed: (style)` — 특정 prop 참조 변경
- `Hooks changed` — state 또는 context 변경

> "Props changed: (style)" 이 보이면 → 인라인 객체 의심 → Part 1로 돌아가기

---

## Ranked 차트

Ranked 탭은 렌더링 시간 순으로 정렬해준다.

- 가장 오래 걸린 컴포넌트가 위에
- 리렌더 횟수 × 소요 시간 = 실제 체감 영향

Flamegraph로 "어디서" 발생하는지, Ranked로 "뭐가 가장 비싼지" 파악한다.

---

## 수치 기준 — "얼마나 빨라야 충분한가"

- **16ms** — 60fps 프레임 버짓. 한 번의 렌더가 이걸 넘으면 프레임 드롭.
- **50ms** — Long Task 기준. 사용자가 "끊긴다"고 느끼기 시작하는 지점.
- **INP 200ms** — Core Web Vitals의 상호작용 응답성 "Good" 임계값. 클릭/입력 후 다음 페인트까지.

Profiler의 숫자를 이 기준과 비교한다. "막연히 느리다"가 아니라 **"이 인터랙션이 320ms인데 목표가 200ms"**로 바꿔야 우선순위가 정해진다.

---

## 측정 없는 최적화는 최적화가 아니다

최적화를 적용하기 **전에** Profiler로 녹화하고, 적용한 **후에** 다시 녹화해서 비교한다.

- 회색 바가 늘어났는가? → 불필요한 리렌더가 줄었다
- Ranked에서 순위가 내려갔는가? → 렌더 비용이 줄었다
- 체감 차이가 없다면? → 그 최적화는 불필요했다

> "느린 것 같아서"가 아니라 **"Profiler에서 봤더니"**로 시작할 것.

---

## Escape Hatch — 그래도 안 되면 직접 DOM을 건드린다

memo, useMemo, 구조 개선을 전부 했는데도 프레임이 부족하다면? React를 우회하는 것도 방법이다.

```jsx
function Price({ticker}) {
  const ref = useRef(null)

  useEffect(() => {
    const ws = new WebSocket(`wss://api/price/${ticker}`)
    ws.onmessage = (e) => {
      // ✅ setState 없이 직접 DOM 업데이트
      ref.current.textContent = e.data
    }
    return () => ws.close()
  }, [ticker])

  return <span ref={ref} />
}
```

`setState`를 안 부르니까 리렌더가 0번. React가 모르는 사이에 DOM만 바뀐다.

---

## 직접 DOM 업데이트 — 언제 쓰나

증권 호가창, 실시간 차트 틱, 타이머, 애니메이션 카운터 등 **초당 수십 회 이상 값이 바뀌는 곳**에서 유효하다.

주의할 점:

- React가 해당 DOM을 모르므로, **다른 state로 인한 리렌더 시 값이 덮어씌워질 수 있음**
- 복잡한 UI 로직이 필요하면 이 패턴은 한계가 있음
- `ref.current.textContent` 수준의 단순한 업데이트에 한정할 것

> React의 장점을 포기하는 거니까 **최후의 수단**이다. 하지만 필요한 곳에서는 확실히 효과가 있다.

---

## 마무리 — 최적화 체크리스트

---

## 렌더링 최적화 판단 순서

```text
1. 진짜 느린가?
   → Profiler로 측정. 느리지 않으면 여기서 멈춘다.

2. 구조로 해결 가능한가?
   → state 내리기, 컴포넌트 분리, children as props

3. 참조 안정성 문제인가?
   → 인라인 객체 → 상수 추출 또는 useMemo
   → 인라인 함수 → useCallback
   → memo가 무력화되고 있지 않은지 확인

4. memo를 걸어야 하는가?
   → 리스트 아이템, 비싼 서브트리에 한정

5. 다시 측정한다.
```

---

## 비용 0인 좋은 습관 — 항상 하는 것

이건 측정 전에도 그냥 하는 것들이다.

- 변하지 않는 객체/배열은 **컴포넌트 바깥 상수**로
- 훅에서 객체를 받으면 **destructuring해서 primitive만 deps에**
- state는 **필요한 곳에 가까이**
- 변하는 부분과 안 변하는 부분은 **컴포넌트를 쪼개서 분리**

> 이건 최적화가 아니라 **기본**이다. 비용이 0이고 코드가 더 나아진다.

---

## 한 줄 요약

> memo, useMemo, useCallback은 도구다. 진짜 실력은 **쓰지 않아도 되는 구조를 만드는 것**이다.

<!-- _class: invert -->

---

## Q&A

<!-- _class: invert -->

---

## Q. memo를 모든 컴포넌트에 걸면 안 되나요?

걸 수는 있다. 비교 비용은 대부분 무시할 수준이니까.

하지만 그게 답이 아닌 이유:

- 인라인 객체/함수를 넘기면 어차피 매번 뚫림 → **memo만 걸어선 의미 없음**
- 모든 props를 안정시키려면 부모에 useMemo/useCallback이 줄줄이 붙음
- 결국 "모든 곳에 memo" → "모든 곳에 useMemo/useCallback" → **코드 복잡도만 올라감**

memo는 **필요한 곳에, props 안정성이 보장된 상태에서** 거는 것이다.

---

## Q. useCallback 안에서 state를 쓰면 deps에 넣어야 하는데, 그러면 매번 바뀌지 않나요?

```jsx
// ❌ count가 바뀔 때마다 새 함수
const increment = useCallback(() => {
  setCount(count + 1)
}, [count])

// ✅ updater function으로 deps에서 state 제거
const increment = useCallback(() => {
  setCount((prev) => prev + 1)
}, [])
```

`useState`의 setter(`setCount`)는 React가 항상 같은 참조를 보장한다. 그래서 deps에 안 넣어도 된다. 문제는 `count` 값을 클로저로 캡처하는 것인데, `setState(prev => ...)` 패턴을 쓰면 현재 state를 React가 넘겨주므로 `count`를 deps에서 뺄 수 있다.

---

## Q. children as props가 왜 리렌더를 막는지 직관적으로 이해가 안 돼요

Part 0을 떠올려보자. **리렌더링 = 함수를 다시 실행하는 것.**

```jsx
function App() {
  // ← 이 함수가 실행돼야
  return (
    <ColorPicker>
      <HeavyContent /> // ← 이 JSX가 새로 만들어짐
    </ColorPicker>
  )
}
```

`color` state가 바뀌면 `ColorPicker` 함수만 다시 실행된다. `App` 함수는 다시 실행되지 않는다. 그러면 `<HeavyContent />`는 이전 렌더에서 만들어진 그대로다. **함수가 실행 안 됐으니 새 JSX도 안 만들어진 것.**

---

## Q. Context를 쓰면 무조건 전체가 리렌더되나요?

전체는 아니고, 해당 context를 **`useContext`로 구독하는 컴포넌트만** 리렌더된다.

문제는 context value가 객체일 때:

```jsx
// ❌ Provider가 리렌더되면 매번 새 객체
<AuthContext.Provider value={{ user, login, logout }}>
```

Provider의 부모가 리렌더 → value 객체가 새 참조 → 모든 consumer 리렌더.

```jsx
// ✅ value를 memo
const value = useMemo(() => ({ user, login, logout }), [user, login, logout])
<AuthContext.Provider value={value}>
```

그리고 자주 바뀌는 값과 안 바뀌는 값은 **context 자체를 분리**하는 게 낫다.

---

## Q. React Compiler 나오면 이 내용 다 필요 없어지는 거 아닌가요?

React Compiler는 useMemo/useCallback을 **자동으로 삽입**해준다. Part 3~4의 수동 메모이제이션은 대부분 불필요해질 수 있다.

하지만:

- **Part 0** (리렌더링의 원리) — 여전히 알아야 함
- **Part 2** (훅 deps에 객체 넣지 않기) — 컴파일러가 해결해주지 않음
- **Part 5** (구조적 최적화) — 컴파일러와 무관, 여전히 유효
- **Part 6** (Profiler) — 측정은 항상 필요

도구가 자동화해주는 부분과, 설계로 해결해야 하는 부분은 다르다.

---

## Q. 직접 DOM 업데이트하면 React 상태와 꼬이지 않나요?

꼬일 수 있다. 핵심은 **React가 관리하는 영역과 분리**하는 것.

```jsx
// ✅ React는 span의 children을 모름 — 충돌 없음
<span ref={priceRef} />

// ❌ React도 이 값을 관리하고 있음 — 충돌
<span ref={priceRef}>{price}</span>
```

`ref`로 업데이트하는 요소는 **React가 내용을 제어하지 않는 빈 요소**여야 한다. React가 children을 렌더하는 요소에 직접 DOM 조작을 하면 다음 리렌더 시 React가 덮어쓴다.

---

## Q. zustand/jotai 같은 상태관리가 memo를 대체하나요?

상당 부분 대체한다. 이들은 **selector 기반 구독**이라 관심 있는 조각만 리렌더에 연결된다.

```jsx
// context + memo 조합: Provider 리렌더 시 value 안정화가 필요
const value = useMemo(() => ({user, cart}), [user, cart])

// zustand: selector가 반환한 값이 바뀔 때만 리렌더
const user = useStore((s) => s.user)
```

context + memo는 "전역 상태를 한 덩어리로 다루면서 참조를 안정시키는" 접근이고, 구독 기반은 "애초에 필요한 조각만 듣는" 접근이다. 후자가 더 근본적이지만, 라이브러리 의존성과 학습 비용이 추가된다.

---

## Q. Server Components 쓰면 이 내용 다 의미 없어지나요?

의미 없어지지 않는다. 오히려 **무엇이 클라이언트로 넘어가는지 의식하게** 만든다.

- `'use client'` 경계 **안쪽**에서는 이 강의의 모든 원칙이 그대로 적용된다.
- RSC는 "렌더할 컴포넌트 자체를 줄이는" 구조적 최적화의 최상위 형태일 뿐, 클라이언트에서 일어나는 리렌더 문제를 없애지는 않는다.
- "이 컴포넌트가 정말 클라이언트여야 하는가?"를 매번 묻게 되니, 최적화 사고 자체가 강화된다.

---

## Q. useMemo의 캐시가 날아가는 경우가 있다던데?

React 공식 문서에 명시되어 있다:

> React will throw away the cached value if it needs to free memory for offscreen components.

즉, `useMemo`는 **성능 힌트**이지 **보장**이 아니다. 캐시가 날아가도 앱이 정상 동작해야 한다.

실무적으로는 거의 일어나지 않지만, `useMemo`를 **정확성을 위한 수단**(같은 참조 보장)으로 쓰면 안 된다는 의미다. 정확성이 필요하면 `useRef` + 직접 비교를 쓰자.

---

# 감사합니다

<!-- _class: invert -->

@yceffort
