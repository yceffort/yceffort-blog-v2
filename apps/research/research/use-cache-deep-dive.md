---
title: "'use cache' 디렉티브 딥다이브: 캐시 경계의 끝까지"
marp: true
paginate: true
theme: default
tags:
  - nextjs
  - caching
  - react
  - web-performance
date: 2026-05-15
description: '"use cache" 한 줄이 만드는 빌드 타임 변환, 캐시 키, cacheLife, 그리고 Cache Components까지'
published: true
post: https://yceffort.kr/2026/05/use-cache-deep-dive
art:
  undraw: cloud-files
---

<!-- _class: deck-cover -->

# `'use cache'` 디렉티브 딥다이브 ⏱

## 시간의 경계

> 디렉티브 3부작 — Part 3 (마지막)
>
> 한 줄 뒤에 숨은 SWC 변환과 cacheHandler

---

## 오늘 다룰 것

1. `'use cache'`는 함수를 **캐시 엔트리 lookup**으로 치환한다
2. SWC가 함수를 다시 쓰고, 클로저는 bound args로 추출
3. 키는 `[buildId, id, args]` — Flight로 직렬화
4. `cacheLife` 7개 프로필, nested 룰 주의
5. `'use cache: private'`은 서버 캐시가 아니다

---

## 시작은 한 줄

```tsx
async function getProducts() {
  'use cache'
  return db.products.findMany()
}
```

이 함수는 더 이상 일반 함수가 아니다.

→ **캐시 핸들러에 키와 함께 위임되는 함수**

---

## 메모이제이션? — 아니다

| 도구                | 스코프            | 키 생성                             | 수명              |
| ------------------- | ----------------- | ----------------------------------- | ----------------- |
| `useMemo(fn, deps)` | 컴포넌트 인스턴스 | 참조 동등성                         | unmount까지       |
| `React.cache(fn)`   | 한 요청           | 참조 동등성 (Map)                   | 응답 끝나면 폐기  |
| `'use cache'`       | 빌드 + 런타임     | **인자 직렬화 + 함수 ID + buildId** | revalidate/expire |

차이의 핵심: **직렬화된 키 + 프로세스를 넘는 저장소.**

---

## 같은 입력 = 같은 출력?

`useMemo`/`React.cache`:

- 같은 객체 `{id: 1}`을 두 번 만들면 **다른 캐시**

`'use cache'`:

- 같은 의미의 `{id: 1}`이면 **같은 캐시 키**
- → 인자가 직렬화되어 해시되기 때문

---

## 새 deploy면 무효화

`'use cache'`는 캐시 키에 **buildId**를 포함한다.

- 코드 바뀌어 새 빌드 → 새 buildId → 모든 엔트리 미스
- "빌드와 빌드 사이를 안정적으로 살아남는 캐시" 모델은 아님
- **요청과 요청 사이**는 살아남는다 (환경에 따라)

---

## 언제 잘 맞는가

7가지 조건:

```text
1. 같은 입력이면 같은 출력
2. 캐시 키 cardinality가 낮음
3. 여러 사용자가 같은 결과를 봄
4. stale data가 잠깐 보여도 사고 아님
5. mutation 후 무효화 경로를 설명할 수 있음
6. 원본 연산이 비싸거나 느리거나 rate limit 있음
7. 권한·개인정보·마스킹 정책과 무관
```

3~4개만 깨져도 안 쓰는 게 맞다.

---

## 엔터프라이즈에서 잘 안 쓰는 이유

엔터프라이즈 화면은 같은 URL이라도 **결과가 다음에 따라 다름**:

- 사용자 권한, 조직, 역할
- feature flag, 계약 조건
- locale, currency, 마스킹 정책
- 방금 한 mutation 결과

→ 키 cardinality 폭발. 잘못 빼면 **권한 누수**.

> 가장 무서운 캐시 버그는 stale이 아니라, 권한 다른 사용자에게 잘못된 결과가 보이는 것.

---

## 그래도 잘 맞는 영역

- 공개 콘텐츠: 문서, FAQ, 약관, 공지
- Reference data: 국가/통화/은행 코드
- Deterministic 비싼 계산: MDX 변환, syntax highlighting
- 여러 컴포넌트가 공유하는 같은 데이터
- static shell 안의 cached island
- Tenant 단위 공통 설정
- Rate-limited 외부 API (`'use cache: remote'`)

---

## 빌드 타임 변환

SWC가 `'use cache'` 함수를 다시 쓴다 (`server_actions.rs`).

```tsx
// 원본
async function getProducts(filter: string) {
  'use cache'
  return db.products.findMany({where: {filter}})
}
```

```tsx
// 변환 후 (개념)
async function $$cache0$$([], filter) {
  return db.products.findMany({where: {filter}})
}

const getProducts = $$reactCache__('default', '<sha1-id>', 0, $$cache0$$)
```

---

## 변환의 4가지 핵심

1. **본문이 hoist** — 모듈 최상위로 끌어올려 별도 함수
2. **첫 인자는 bound args 배열** — 클로저로 잡힌 변수가 명시적 인자로
3. **함수 ID는 SHA1 hash** — 파일·export name·argument mask 등에서 파생
4. **cache_kind가 박힘** — `'default'` / `'private'` / `'remote'`

---

## 클로저는 bound args로 추출

```tsx
// 원본 — userId가 클로저로 잡힘
async function Component({userId}) {
  const getData = async (filter: string) => {
    'use cache'
    return fetch(`/api/users/${userId}?filter=${filter}`)
  }
  return getData('active')
}
```

```tsx
// 변환 후 — userId가 명시적 인자로 추출됨
async function $$cache0$$([userId], filter) {
  /* 본문 */
}
const getData = $$reactCache__('default', '<id>', 1, $$cache0$$, userId)
//                                                 └─ bound count
```

→ `userId`가 자동으로 캐시 키에 들어감.

---

## 함수 ID는 secure hash

- 함수의 위치(파일명) + export/reference name + argument mask
- SHA1로 묶어서 **secure hash** 생성
- 같은 함수라도 파일이 바뀌면 다른 ID

> 단, deploy 단위의 전체 무효화는 함수 ID가 아니라 **buildId**가 담당한다.

---

## 3가지 변형

| 변형                   | 서버 저장                    | 클라이언트 저장 | request API 직접 접근 | 공유 범위        |
| ---------------------- | ---------------------------- | --------------- | --------------------- | ---------------- |
| `'use cache'`          | in-memory 또는 cache handler | router prefetch | 불가                  | 모든 사용자 공유 |
| `'use cache: remote'`  | remote cache handler         | router prefetch | 불가                  | 모든 사용자 공유 |
| `'use cache: private'` | **저장 안 함**               | 브라우저 메모리 | **가능**              | 클라이언트 본인  |

---

## private의 함정

`'use cache: private'`은 **서버 캐시가 아니다**.

> "results are never stored on the server, they're cached only in the browser's memory and do not persist across page reloads" — 공식 문서

- 매 서버 렌더마다 함수가 실행됨
- static shell에서도 제외
- custom cache handler 설정 불가
- v16 시점 experimental

---

## remote의 함정

`'use cache: remote'`는 **공유 캐시**가 핵심 가치.

쓰는 게 좋은 경우:

- Rate-limited API
- Slow backend
- 비싼 연산
- Flaky service

피해야 하는 경우:

- 50ms 미만 빠른 연산
- key가 거의 unique
- **seconds~minutes 단위로 자주 바뀌는 데이터**
- 이미 KV store 앞단이 있음

---

## 런타임: 캐시 래퍼

빌드 타임에 박힌 `$$reactCache__`는 결국 `cache()` 함수를 가리킨다 (`use-cache-wrapper.ts`).

```text
1. 인자 직렬화 → encodeReply (Flight)
2. 캐시 키 구성 → [buildId, id, args, hmrHash?]
3. ResumeDataCache(RDC) 조회
4. (없으면) cacheHandler 조회
5. (없으면) 본문 실행 → 저장
6. SWR: 만료됐으면 백그라운드 재생성
```

---

## 캐시 키

```ts
const cacheKeyParts = hmrRefreshHash
  ? [buildId, id, args, hmrRefreshHash]
  : [buildId, id, args]
```

- `buildId`: deploy마다 자동 무효화
- `id`: 함수 위치·시그니처 hash
- `args`: 인자 배열 (bound + 호출 인자)
- `hmrRefreshHash`: dev에서 HMR로 자동 무효화

---

## 인자 직렬화: encodeReply

`'use server'`와 같은 Flight 직렬화기.

**가능:** 원시값, 일반 객체/배열, `Date`, `Map`, `Set`, `BigInt`, `TypedArray`, `FormData`, React element (pass-through)

**불가능:** 클래스 인스턴스, 일반 함수, `Symbol`, `URL`

> 인자(엄격) vs 리턴값(JSX 허용)의 직렬화기는 **다르다**.

---

## 두 단계 lookup

```ts
// 1단계: RDC (페이지 렌더 단위)
const cached = lookupResumeDataCache(rdc, key)
if (cached) return cached

// 2단계: cacheHandler
const entry = await cacheHandler.get(key)
if (entry && !shouldDiscard(entry)) return entry

// 3단계: 둘 다 미스 → 본문 실행
return generateCacheEntry(...)
```

`'use cache: private'`은 이 흐름을 거의 안 탄다.

---

## Stale-while-revalidate

```ts
const age = Date.now() - entry.timestamp

if (age < entry.revalidate * 1000) {
  return entry  // fresh
}
if (age < entry.expire * 1000) {
  void generateCacheEntry({skipPropagation: true})  // 백그라운드
  return entry  // stale 응답
}
return await generateCacheEntry(...)  // expired — blocking
```

---

## cacheLife — 7개 빌트인

| 프로필    | stale | revalidate | expire |
| --------- | ----- | ---------- | ------ |
| `default` | 5분   | 15분       | 무한   |
| `seconds` | 30초  | 1초        | 1분    |
| `minutes` | 5분   | 1분        | 1시간  |
| `hours`   | 5분   | 1시간      | 1일    |
| `days`    | 5분   | 1일        | 1주    |
| `weeks`   | 5분   | 1주        | 30일   |
| `max`     | 5분   | 30일       | 1년    |

---

## stale은 모두 5분?

**예. 의도적이다.**

- `stale`은 클라이언트 라우터의 prefetch 캐시 유지 시간
- 사용자 네비게이션 패턴에 맞춘 값
- 서버 데이터 갱신 빈도(`revalidate`)와 별개
- 30초 미만은 강제로 30초로 올라감 (prefetch 안전장치)

---

## 인라인 프로필

```tsx
async function getOffer() {
  'use cache'
  cacheLife({
    stale: 60, // 1분
    revalidate: 300, // 5분
    expire: 3600, // 1시간
  })
  return db.offers.findFirst()
}
```

`expire`는 반드시 `revalidate`보다 커야 함.

---

## Nested cacheLife — 명시 outer 우선

```tsx
async function Dashboard() {
  'use cache'
  cacheLife('hours') // 명시 → 1시간

  return <Widget /> // Widget이 'minutes'(5분)이라도 무관
}
```

- 외부 캐시 hit 시 **내부까지 통째로 응답**
- 외부 1시간 동안 내부 lifetime은 관측되지 않음

---

## Nested — 명시 없는 outer는 깎인다

```tsx
async function Dashboard() {
  'use cache'
  // cacheLife 없음 → default (15분)

  return <Widget /> // Widget이 'minutes'(5분) → Dashboard도 5분
}
```

- 명시 없으면 inner의 짧은 lifetime이 propagation
- 반대(내부가 길다고 외부 늘리기)는 일어나지 않음

> 실무 권장: **모든 캐시 함수에 `cacheLife` 명시.**

---

## 짧은 nested cache는 prerender 에러

`revalidate`이 5분 미만이면 dynamic hole이 됨.

- `seconds` 프로필 자동 dynamic
- 외부에 명시 없는 nested에서 짧은 cache → 빌드 에러
- 해결:
  - outer를 명시적으로 길게: `cacheLife('default')`
  - outer를 명시적으로 짧게 + Suspense

---

## cacheTag

```tsx
async function getProduct(id: string) {
  'use cache'
  cacheTag('products', `product-${id}`)
  return db.products.findUnique({where: {id}})
}
```

- 한 엔트리에 여러 태그 가능
- 같은 태그 두 번 = 한 번 (idempotent)
- 1 태그 최대 256자, 1 엔트리 최대 128개

---

## revalidateTag — 두 번째 인자가 핵심!

| 호출                              | 의미                                     |
| --------------------------------- | ---------------------------------------- |
| `revalidateTag(tag, 'max')`       | **권장**, SWR (stale-while-revalidate)   |
| `revalidateTag(tag, {expire: 0})` | webhook용 즉시 만료                      |
| `revalidateTag(tag)`              | **deprecated**, 즉시 만료 + blocking     |
| `updateTag(tag)`                  | Server Action 전용, read-your-own-writes |

---

## 단일 인자는 SWR이 아니다 ⚠️

```tsx
// ❌ 이름과 달리 SWR 아님 — 즉시 만료 + blocking
revalidateTag('posts')

// ✅ SWR
revalidateTag('posts', 'max')

// ✅ webhook용 즉시 만료
revalidateTag('posts', {expire: 0})
```

> 공식 문서: "The single-argument form is deprecated."

---

## updateTag — read-your-own-writes

```tsx
'use server'
export async function createPost(formData: FormData) {
  const post = await db.posts.create({data: formData})
  updateTag('posts') // 즉시 폐기
  updateTag(`post-${post.id}`)
  redirect(`/posts/${post.id}`) // 사용자는 자기 변경을 본다
}
```

Server Action 전용. mutation 직후 SWR로 stale을 보여주면 운영 사고가 나는 화면에 쓴다.

---

## 런타임 API 제약

`'use cache'`는 `cookies()`/`headers()`/`searchParams` **직접 호출 금지**.

```tsx
async function CachedProfile() {
  'use cache'
  const session = (await cookies()).get('session')?.value // ❌ Error
}
```

이유: 캐시는 요청 간 공유 → request-scoped 데이터 들어가면 권한 누수.

---

## 우회 1 — 인자로 추출 (권장)

```tsx
async function ProfilePage() {
  const session = (await cookies()).get('session')?.value
  return <CachedProfile sessionId={session} />
}

async function CachedProfile({sessionId}: {sessionId: string}) {
  'use cache'
  // sessionId가 자동으로 캐시 키에 들어감
  return <div>{await fetchProfile(sessionId)}</div>
}
```

---

## React.cache 격리

```tsx
const store = cache(() => ({current: null}))

function Parent() {
  const shared = store()
  shared.current = 'value'
  return <Child />
}

async function Child() {
  'use cache'
  const shared = store() // null! 외부 store 안 보임
}
```

`'use cache'` 안에서는 별도의 `React.cache` 스코프. **자기 인자만으로 결정 가능**해야 한다는 원칙.

---

## Cache Components & PPR

`cacheComponents: true` 한 줄이 켜는 모델 — 한 페이지가 3 영역으로:

```tsx
export default function Page() {
  return (
    <>
      <header>Static — 빌드 타임 prerender</header>
      <Stats /> {/* Cached — 'use cache' */}
      <Suspense fallback={<Skeleton />}>
        <Notifications /> {/* Dynamic — request 타임 stream */}
      </Suspense>
    </>
  )
}
```

---

## 함정 1: 빌드 hang (50초)

```tsx
async function Cached({promise}) {
  'use cache'
  const data = await promise // request-scoped Promise면 빌드 hang
}
```

prerender 중 캐시 함수가 영원히 resolve 안 되는 Promise를 await하면 50초 timeout.

해결: `cookies()`/`headers()`는 외부에서 **값**을 추출한 뒤 props로 넘기기.

---

## 함정 2: Math.random은 한 번만

```tsx
async function getId() {
  'use cache'
  return crypto.randomUUID() // 캐시 미스 때만 실행 → 모두 같은 값
}
```

요청별로 다른 값 필요하면 캐시 밖으로:

```tsx
import {connection} from 'next/server'

async function DynamicContent() {
  await connection() // dynamic 강제
  return <div>{crypto.randomUUID()}</div>
}
```

---

## 함정 3: 미지원 환경

- **Edge runtime 미지원** — Node.js 전용 모듈에 의존
- **Static export 미지원** — 런타임 cacheHandler 필요
- **클래스 인스턴스, URL 직렬화 불가** — plain object로 변환

---

## unstable_cache → use cache 마이그레이션

```tsx
// 이전
const getCachedUser = unstable_cache(
  async (id) => getUser(id),
  ['my-app-user'],
  {tags: ['users'], revalidate: 60},
)

// 이후
async function getCachedUser(id: string) {
  'use cache'
  cacheTag('users')
  cacheLife({revalidate: 60})
  return getUser(id)
}
```

단, **drop-in replacement는 아니다.** Cache Components 모델로의 전환.

---

## 디버깅

```bash
NEXT_PRIVATE_DEBUG_CACHE=1 next dev
```

- 콘솔에 캐시 hit/miss와 키 출력
- dev에서 캐시 함수의 `console.log`는 `Cache` prefix → replay 표시
- 응답 헤더 `x-nextjs-stale-time`이 클라이언트 stale time

---

## 'use client' / 'use server' / 'use cache'

| 디렉티브       | 변환 대상 | 박히는 ID       | 경계                    |
| -------------- | --------- | --------------- | ----------------------- |
| `'use client'` | 모듈      | 모듈 ID + chunk | RSC → Client (참조)     |
| `'use server'` | 함수      | SHA1 함수 ID    | Client → Server (RPC)   |
| `'use cache'`  | 함수      | SHA1 함수 ID    | Caller → Cache (lookup) |

세 디렉티브 모두 같은 Flight 직렬화기 공유.

---

## 핵심 정리

1. **캐시 lookup으로 치환** — useMemo/React.cache와 다른 키 도메인·수명
2. **SWC 변환** — 본문 hoist + 클로저 bound args 추출 + SHA1 ID
3. **3가지 변형** — `'use cache'` (공유) / `remote` (외부 핸들러) / `private` (브라우저 메모리)
4. **cacheLife는 항상 명시** — nested 룰의 함정 회피
5. **revalidateTag(tag, 'max')만 SWR** — 단일 인자는 deprecated
6. **엔터프라이즈는 좁은 영역만** — 권한 누수 vs hit rate

---

## 시리즈 — 마무리

> 디렉티브 3부작 완결 🎉
>
> ✅ Part 1: `'use client'` — 클라이언트 경계 (rendering layer)
> ✅ Part 2: `'use server'` — 서버 경계 (process)
> ✅ Part 3: `'use cache'` (이번 편) — 캐시 경계 (시간)

세 디렉티브 모두 — **ID + 직렬화 + 메타데이터** 패턴으로 시스템 경계를 넘는다.

---

## 한 문장으로

> `'use cache'`는 **업무 데이터 캐시 도구**가 아니다.
>
> **공유 가능하고 변동성이 낮은 서버 결과**를 static shell 또는 runtime cache에 편입시키는 도구다.

핵심 질문은 **"이 데이터를 캐시할 수 있나?"가 아니라 "이 결과를 공유해도 되는가?"**

---

## 참고

- 블로그 원문: [`'use cache'` 디렉티브 딥다이브](https://yceffort.kr/2026/05/use-cache-deep-dive)
- [Next.js: use cache](https://nextjs.org/docs/app/api-reference/directives/use-cache)
- [Next.js: cacheLife](https://nextjs.org/docs/app/api-reference/functions/cacheLife)
- [Next.js: revalidateTag](https://nextjs.org/docs/app/api-reference/functions/revalidateTag)
- Next.js v16.2.4: `packages/next/src/server/use-cache/`
