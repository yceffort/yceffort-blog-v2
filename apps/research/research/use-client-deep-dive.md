---
title: "'use client' 디렉티브 딥다이브: 클라이언트 경계의 끝까지"
marp: true
paginate: true
theme: default
tags:
  - react
  - nextjs
  - bundler
date: 2026-05-15
description: "'use client' 한 줄이 만드는 모듈 경계, 빌드 타임 변환, Flight 직렬화"
published: true
post: https://yceffort.kr/2026/05/use-client-deep-dive
art:
  undraw: responsive
---

<!-- _class: deck-cover -->

# `'use client'` 디렉티브 딥다이브 🚪

## 클라이언트 경계의 끝까지

> 디렉티브 3부작 — Part 1
>
> 한 줄이 만드는 빌드 타임 변환과 Flight 직렬화

---

## 오늘 다룰 것

1. `'use client'`는 **모듈 그래프의 진입점 마커**다
2. 빌드 타임에 같은 모듈이 layer별로 다르게 컴파일된다
3. RSC server는 본문 대신 **참조(reference)** 만 직렬화한다
4. 클라이언트는 `$L` 토큰을 보고 chunk를 lazy load한다
5. SSR과 RSC는 같은 컴포넌트를 다른 트리로 빌드한다

---

## 시작은 한 줄

```tsx
'use client'

import {useState} from 'react'

export function Counter() {
  const [count, setCount] = useState(0)
  return <button onClick={() => setCount(count + 1)}>{count}</button>
}
```

이 한 줄이 어떤 일을 하는가? — "클라이언트에서 동작" 표시?

> ❌ 정확히는 **"이 모듈이 클라이언트 번들 그래프의 진입점이다"** 라는 마커다.

---

## 흔한 오해 ❌

> "`'use client'` 적은 컴포넌트만 클라이언트가 되고, import한 컴포넌트는 따로 표시해야 한다"

**정확히 반대다.**

`'use client'`는 **모듈 그래프의 단방향 경계의 시작점**이다. 한 모듈에 적으면 import하는 모든 모듈이 자동으로 클라이언트 번들에 포함된다.

---

## 단방향 경계

```text
app/page.tsx           ← Server Component (default)
  └─ import Layout     ← Server Component
       └─ import Counter   ← 'use client' (경계!)
            └─ import { format } from './utils'
                 └─ import lodash    ← 자동으로 클라이언트
```

- `'use client'` 모듈은 서버 모듈을 **import 못 함**
- 단, `children` prop으로 Server Component를 **받는** 것은 가능

---

## children prop의 비밀

```tsx
// page.tsx (Server)
import {ClientShell} from './ClientShell'
import {ServerContent} from './ServerContent'

export default function Page() {
  return (
    <ClientShell>
      <ServerContent /> {/* OK — props로 전달 */}
    </ClientShell>
  )
}
```

- `ClientShell`은 `ServerContent`를 **import 안 함**
- Server `Page`가 두 컴포넌트를 합치고 children으로 넘김
- import는 단방향, **prop은 양방향**

---

## 빌드 타임 변환

`'use client'`는 런타임 효과가 없다. 진짜 일은 **빌드 타임**에 일어난다.

같은 파일이 webpack의 여러 layer에서 **다른 형태**로 컴파일된다.

| Layer      | 결과물             |
| ---------- | ------------------ |
| RSC server | 본문이 사라진 stub |
| Client     | 원본 그대로        |
| SSR        | 원본 그대로        |

---

## RSC 서버 번들: 본문 제거

```tsx
// 원본
'use client'
export function Counter({initial}) {
  const [count, setCount] = useState(initial)
  return <button onClick={...}>{count}</button>
}
```

→ 빌드 후 RSC 번들에서는:

```js
import {registerClientReference} from 'react-server-dom-webpack/server'

export const Counter = registerClientReference(
  function () {
    throw new Error('Cannot call Counter() from server.')
  },
  'components/Counter.tsx',
  'Counter',
)
```

---

## 세 가지 핵심

1. **함수 본문이 사라졌다** — `useState`, JSX, 이벤트 핸들러는 RSC 번들에 한 글자도 안 들어간다
2. **stub에 메타데이터가 박힌다** — `$$typeof`, `$$id`, `$$async`
3. **stub을 호출하면 에러** — 서버에서 함수처럼 부르면 명확한 에러를 던진다

---

## registerClientReference의 본체

```js
// react-server-dom-webpack/.../ReactFlightWebpackReferences.js
export function registerClientReference(proxyImplementation, id, exportName) {
  return Object.defineProperties(proxyImplementation, {
    $$typeof: {value: CLIENT_REFERENCE_TAG},
    $$id: {value: id + '#' + exportName},
    $$async: {value: false},
  })
}
```

`Object.defineProperties`로 stub에 메타데이터를 박는 게 끝이다.

---

## CJS 모듈은 Proxy로

ESM은 export 단위로 stub을 만들지만, CJS는 모듈 통째로 Proxy화한다.

```js
export function createClientModuleProxy(moduleId) {
  const cache = Object.create(null)
  return new Proxy(
    {},
    {
      get(_, name) {
        return (cache[name] ??= registerClientReference(
          () => {
            throw new Error('...')
          },
          moduleId,
          name,
        ))
      },
    },
  )
}
```

어떤 export를 접근하든 그 자리에서 client reference를 만들어낸다.

---

## flight-client-entry-plugin

webpack 빌드 단계에서 Next.js가 추가하는 플러그인.

- `'use client'` 모듈을 발견하면 **자동으로 클라이언트 entry 추가**
- chunk를 만들고 `client-reference-manifest.json`을 emit
- 이 매니페스트가 `$$id` → chunk URL 매핑의 진실의 원천

---

## 클라이언트 매니페스트의 모양

```json
{
  "components/Counter.tsx#Counter": {
    "id": "5234",
    "chunks": ["app/static/chunks/123.js"],
    "name": "Counter",
    "async": false
  }
}
```

- `id`: webpack 모듈 ID
- `chunks`: 로드할 chunk URL들
- `name`: 모듈 안의 export 이름
- 클라이언트가 이걸 보고 `__webpack_require__`로 모듈을 resolve

---

## 직렬화: serializeClientReference

RSC 서버가 client reference를 만나면, 매니페스트에서 메타데이터를 lookup해 Flight payload에 메타 행을 끼워 넣는다.

```js
// 단순화
function serializeClientReference(request, parent, key, ref) {
  const meta = request.clientManifest[ref.$$id]
  const importId = outlineModel(request, meta)
  return parent[0] === REACT_ELEMENT_TYPE && key === '1'
    ? '$L' + importId.toString(16) // element type 자리
    : '$' + importId.toString(16) // 그 외
}
```

위치에 따라 `$L<hex>` 또는 `$<hex>` 토큰이 나온다.

---

## $L vs $ 토큰

| 위치                               | 토큰      | 의미                                        |
| ---------------------------------- | --------- | ------------------------------------------- |
| element의 `type` 슬롯              | `$L<hex>` | **lazy** — 클라이언트가 React.lazy로 만든다 |
| 그 외 (props, children 안의 값 등) | `$<hex>`  | 일반 import metadata 참조                   |

`$L`이 핵심이다. 이게 React가 컴포넌트로서 lazy하게 들고 있을 표시.

---

## Flight payload 안의 모양

```text
1:I["app/static/chunks/123.js","Counter"]
2:["$","div",null,{"children":["$","$L1",null,{"initial":0}]}]
```

- `1:`는 import metadata 행
- `2:`의 `$L1`이 1번 모듈을 lazy 컴포넌트로 가리킨다
- 클라이언트는 이 stream을 받아 트리를 복원

---

## 클라이언트 측: $L → React.lazy

```js
function parseModelString(response, parentObject, key, value) {
  if (value[0] === '$') {
    if (value[1] === 'L') {
      const id = parseInt(value.slice(2), 16)
      return createLazyChunkWrapper(getOutlinedModel(response, id))
    }
    // ...
  }
}
```

`$L`을 만나면 `React.lazy` 노드를 만들어 그 자리에 끼운다.

---

## chunk 로딩

```js
function preloadModule(metadata) {
  const chunks = metadata[1]
  const promises = []
  for (let i = 0; i < chunks.length; i++) {
    const chunkId = chunks[i]
    if (chunkCache.has(chunkId)) continue // 한 번 받으면 끝
    chunkCache.set(chunkId, __webpack_chunk_load__(chunkId))
    promises.push(chunkCache.get(chunkId))
  }
  return Promise.all(promises)
}
```

`chunkCache`로 dedup. 같은 chunk를 두 번 fetch하지 않는다.

---

## 두 개의 트리: SSR vs RSC

`'use client'`는 SSR도 막지 않는다. 같은 컴포넌트가 **세 곳**에서 빌드된다.

```text
                ┌─► RSC server bundle ──(stub)──► Flight Payload
                │                                       │
   같은 모듈 ───┼─► SSR bundle ────────────────► SSR이 HTML 생성
                │                                       ▲
                └─► Client bundle ────────(hydrate)─────┘
```

---

## ssrModuleMapping

매니페스트에는 RSC payload를 SSR이 소비할 때 쓸 모듈 ID 매핑도 들어있다.

```json
{
  "ssrModuleMapping": {
    "5234": { "id": "ssr-build-7821", ... }
  }
}
```

같은 컴포넌트가 SSR build에서는 다른 ID를 갖기 때문에 변환 테이블이 필요하다.

---

## props 직렬화

`'use client'` 컴포넌트의 props는 매번 RSC payload에 들어간다.

**가능:** 원시값, 일반 객체, 배열, `Date`, `Map`, `Set`, `BigInt`, `TypedArray`, React element
**불가능:** 클래스 인스턴스, 일반 함수 (Server Action은 OK), `Symbol`, `URL`

---

## children prop의 비밀

```tsx
<ClientShell>
  <ServerContent />
</ClientShell>
```

- `<ServerContent />`는 RSC가 먼저 렌더링한 결과
- 그 결과(JSX 트리)가 `children` prop으로 직렬화되어 클라이언트에 전달
- 클라이언트는 그걸 그대로 트리에 끼움
- → **Server 컴포넌트의 출력이 Client 컴포넌트의 children 자리에 들어가는 패턴**

---

## 주의 1: leaf로 미루기

```tsx
// ❌ 페이지 전체를 클라이언트로 끌어내림
'use client'
export default function Page() { ... }

// ✅ 인터랙션 필요한 부분만
export default function Page() {
  return (
    <article>
      <Content />
      <LikeButton /> {/* 이것만 'use client' */}
    </article>
  )
}
```

`'use client'` 위치가 클라이언트 번들 크기를 결정한다.

---

## 주의 2: barrel 파일은 위험

```tsx
// components/index.ts
'use client'
export * from './Button'
export * from './Modal'
export * from './Chart' // 큰 라이브러리 의존
```

barrel에 `'use client'`를 적으면 한 컴포넌트만 써도 **전체가 클라이언트 번들에 들어간다**.

해결: barrel을 쪼개거나, `'use client'`를 개별 파일로 옮긴다.

---

## 주의 3: 서버 코드 import 금지

```tsx
'use client'

import {db} from '@/lib/db' // ❌ 서버 전용
import {SECRET} from '@/lib/env' // ❌ 환경 변수 노출 위험
```

빌드 시점에 에러가 나거나, 더 나쁘게는 **클라이언트 번들에 비밀 정보가 노출**된다.

server-only 패키지를 쓰면 빌드 시점에 강제로 막을 수 있다.

---

## 주의 4: props는 매 RSC payload에 들어간다

```tsx
<HeavyClientComponent
  data={largeArray} // 매번 직렬화
  options={config}
/>
```

- 큰 데이터는 props 대신 클라이언트에서 fetch 권장
- 또는 server-side에서 미리 가공해 작은 형태로 넘기기

---

## 주의 5: Context는 Provider가 클라이언트일 때만

```tsx
// ThemeProvider.tsx
'use client'
export const ThemeContext = createContext()
export function ThemeProvider({children}) {
  return <ThemeContext.Provider value={...}>{children}</ThemeContext.Provider>
}
```

Provider 자체가 `'use client'`여야 동작. 서버 컴포넌트에서는 Context 못 읽는다.

---

## 주의 6: Date/Map/Set은 그대로 넘어간다

```tsx
// Server Component
const now = new Date()
const tags = new Set(['a', 'b'])
return <ClientView createdAt={now} tags={tags} />
```

Flight 직렬화기가 `Date`, `Map`, `Set`을 알고 있어 그대로 복원된다. **JSON.stringify의 한계를 넘는다.**

---

## 성능 — 1: dynamic import

```tsx
'use client'
import dynamic from 'next/dynamic'

const HeavyChart = dynamic(() => import('./HeavyChart'), {
  loading: () => <Skeleton />,
})
```

초기 페이로드에서 분리 → 화면에 보일 때만 chunk 로드.

---

## 성능 — 2: Suspense boundary

```tsx
<Suspense fallback={<Skeleton />}>
  <SlowClientComponent />
</Suspense>
```

- streaming 가시성 ↑
- TTFB 단축, 사용자가 빈 화면을 덜 본다

---

## 성능 — 3: chunk 공유

webpack이 자동으로 처리하므로, **억지로 분리하지 말고 공통 dependency가 자연스럽게 한 chunk에 모이게 두자**.

너무 잘게 쪼개면 HTTP 요청 수만 늘어난다.

---

## Turbopack에서는 무엇이 다른가

같은 점:

- `'use client'` 처리 모델 동일
- 같은 매니페스트 형식
- 같은 Flight 직렬화

다른 점:

- 빌드 트랜스폼이 Rust로 구현됨
- dev 매니페스트가 on-demand로 발행
- chunk URL 형식 약간 다름

**결과물은 사실상 같다.** 이 글의 webpack 인용을 Turbopack으로 읽어도 의미는 통한다.

---

## 'use client' vs 'use server'

|               | `'use client'`        | `'use server'`        |
| ------------- | --------------------- | --------------------- |
| 변환 대상     | 모듈                  | 함수                  |
| 박히는 ID     | 모듈 ID + chunk       | SHA1 함수 ID          |
| 직렬화되는 것 | props                 | 인자 + 결과           |
| 경계의 의미   | RSC → Client (참조로) | Client → Server (RPC) |

---

## 전체 아키텍처

```text
'use client' module
       │
       ▼
┌──────────────────────┐
│  RSC layer: stub     │  ←─ 본문 제거, 메타데이터만
└──────────┬───────────┘
           │  serializeClientReference
           ▼
   Flight payload ($L token)
           │
           ▼
┌──────────────────────┐
│  Client: parseModel  │  ←─ React.lazy 노드
└──────────┬───────────┘
           │  preloadModule
           ▼
   chunk load + chunkCache
           │
           ▼
      실제 컴포넌트
```

---

## 핵심 정리

1. **'use client'는 진입점 마커** — 모듈 그래프의 단방향 경계 시작점
2. **레이어별 다른 빌드** — RSC stub / Client 원본 / SSR 원본
3. **참조의 직렬화** — `Object.defineProperties`로 stub에 메타데이터
4. **`$L` 토큰** — Flight stream의 lazy 표시
5. **chunk dedup** — `chunkCache`로 한 번 받으면 끝
6. **props는 매번 직렬화** — payload 다이어트가 성능

---

## 시리즈 — 다음 편

> 디렉티브 3부작
>
> ✅ Part 1: `'use client'` (이번 편)
> ▶️ **Part 2: `'use server'` — 클라이언트 → 서버 RPC**
> ⏭ Part 3: `'use cache'` — 시간의 경계

서버 → 클라이언트 방향(이번 편)을 이해했다면, 다음은 **반대 방향**의 RPC를 본다.

---

## 참고

- 블로그 원문: [`'use client'` 디렉티브 딥다이브](https://yceffort.kr/2026/05/use-client-deep-dive)
- React v19.2 source: `react-server-dom-webpack/`
- Next.js v16.2 source: `packages/next/src/build/webpack/loaders/next-flight-loader/`
- Overreacted: [What Does 'use client' Do?](https://overreacted.io/what-does-use-client-do/)
