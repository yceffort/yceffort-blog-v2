---
title: 'Sentry 딥다이브 2부: 서버에서 이슈까지'
marp: true
paginate: true
theme: midnight
tags:
  - observability
date: 2026-08-06
description: 'Relay 파이프라인부터 Debug ID, 그루핑 해시, 운영 규칙까지 — 서버에 도착한 이벤트가 이슈가 되는 과정'
published: true
art:
  undraw: server-error
---

# Sentry 딥다이브 2부

서버에서 이슈까지 — 에러 하나가 이슈가 되기까지 ②

<!-- _class: invert deck-cover -->

@yceffort

---

## 1부 복습 — 지도

```text
throw
  │ ① 포착      ✅ 1부 — 전역 핸들러(onerror 교체) + 몽키패치
  ▼
이벤트 조립     │ ② 조립      ✅ 1부 — 3-스코프 병합, processors, beforeSend
  ▼
전송           │ ③ 전송      ✅ 1부 — envelope, rate limit, client report
  ▼
Sentry 서버    │ ④ 수신      ← 오늘 여기부터
  ▼
그루핑         │ ⑤ 그루핑    "같은 에러" 판정 → 이슈
```

- 1부 한 줄 요약: SDK는 **unhandled만 줍고**, 이벤트는 스코프 병합 → 필터 → beforeSend를 거쳐 **envelope에 실려 떠났다**
- 오늘: 도착한 이벤트가 이슈가 되기까지 + **원리에서 도출되는 운영 규칙**과 **꿀팁 부록**

---

## 기준 버전

이 덱의 코드는 전부 실제 소스에서 가져왔다.

- SDK: `getsentry/sentry-javascript` **v10.69.0** (2026-07-29)
- 서버: `getsentry/sentry`, `getsentry/relay` master (2026-01 기준) + 공식 문서
- 그루핑 설정: `all-platforms@2026-01-20` enhancement config

<!-- 버전을 박아두는 이유: Sentry SDK는 v8에서 아키텍처가 크게 바뀌었다(Hub 제거). 구버전 블로그 글과 이 덱의 내용이 다르면 버전부터 의심할 것 -->

---

## Part 4 — 수신

Relay 파이프라인과 소스맵 복원

---

## 이벤트가 도착한 뒤의 여정

self-hosted 문서 기준 실제 파이프라인:

```text
SDK
 └→ Relay          (DSN 검증, inbound filter, rate limit, 정규화, PII 스크러빙)
     └→ Kafka
         └→ ingest consumer → preprocess_event
             └→ Symbolicator  (소스맵 복원 ← 여기!)
                 └→ 저장(nodestore) + 그루핑 → ClickHouse(snuba)
                     └→ post_process  (알림, 이슈 후처리)
```

- 이벤트가 도착하면 제일 먼저 **정문 경비실**을 통과한다 — 이 관문 서비스의 이름이 **Relay**다
- 경비실이 하는 일: 신원 확인(DSN 검증), 잡상인 차단(필터), 출입량 제한(rate limit), 반입 금지품 제거(PII 스크러빙). **여기서 걸러지면 뒤 단계는 아예 시작되지 않는다**
- 그래서 순서가 중요하다: **필터 → 소스맵 복원 → 그루핑**

<!-- 구두: Relay는 Rust로 만든 별도 오픈소스 서비스(getsentry/relay)라는 것 정도만. self-hosted나 데이터 주권 이슈로 자체 Relay를 중간에 둘 수도 있다는 건 심화 주제 -->

---

## 서버에도 필터가 있다: inbound data filters

Project Settings의 Inbound Filters — 구현은 Relay에 있다 (`relay-filter/src/`):

- `browser_extensions.rs` — 확장 프로그램 에러
- `legacy_browsers.rs`, `web_crawlers.rs`, `localhost.rs`
- React hydration error, ChunkLoadError 전용 필터도 있음

SDK 필터(1부의 `ignoreErrors`)와의 결정적 차이 두 가지:

1. **"Filtered events do not consume quota"** — 서버 필터는 쿼터를 아낀다
2. **"exclusively applied at ingest time"** — symbolication **전**에 적용된다
   → 복원된 원본 파일명 기준으로는 필터할 수 없다

> 그래서 실무 전략: 확실한 노이즈는 SDK에서(전송량 절약), 대시보드에서 켤 수 있는 것은 서버에서(쿼터 절약), 둘 다 쓴다.

---

## 소스맵 매칭의 구세대: release + URL

전통 방식은 "release가 같고 파일 URL이 같으면 이 소스맵을 써라"였다.

문제점 (공식 문서 원문):

> "when you changed the subpath of where your javascript files are served...
> the corresponding sourcemap would not be able to be found by Sentry."

- CDN 경로 변경, 쿼리스트링, 프록시… URL은 **신뢰할 수 없는 키**였다
- release를 잊으면 매칭 자체가 불가능

그래서 현재 표준은 **Debug ID** 방식이다 (SDK 7.47+).

---

## Debug ID: 파일 내용이 곧 신원

번들러 플러그인(`@sentry/bundler-plugins`)이 빌드 시점에:

```ts
// 청크 내용의 SHA-256에서 만드는 결정적(deterministic) UUID
const debugId = stringToUUID(code) // rollup/vite 기준. webpack은 청크 hash를 씀
```

1. 소스맵 JSON에 `debug_id` 필드 주입
2. 번들 끝에 `//# debugId=<uuid>` 주석 추가
3. 그리고 번들 **앞**에 이런 스니펫을 주입한다:

<!-- prettier-ignore -->
```js
!function(){try{var e=window;
  var n=(new e.Error).stack;
  n&&(e._sentryDebugIds=e._sentryDebugIds||{},
      e._sentryDebugIds[n]="<uuid>",
      e._sentryDebugIdIdentifier="sentry-dbid-<uuid>")
}catch(e){}}();
```

**`new Error().stack`을 맵의 키로 쓴다.** 왜?

---

## new Error().stack 트릭

빌드 시점에는 이 파일이 **어떤 URL로 서빙될지 알 수 없다.**

- 스니펫이 런타임에 실행되면, 그 순간의 스택에는 **배포된 실제 URL**이 들어 있다
  (`at https://cdn.example.com/assets/main-8f3a2.js:1:1`)
- SDK는 이벤트 조립 시(`applyDebugIds`) 이 스택 문자열들을 파싱해서
  `{ 파일 URL → debugId }` 맵을 만들고, 스택 프레임마다 `debug_id`를 붙인다
- 최종적으로 `event.debug_meta.images`에 `{code_file, debug_id}` 목록으로 실려 간다

```ts
// packages/core/src/utils/prepareEvent.ts
frame.debug_id = filenameDebugIdMap[frame.filename]
```

> 서버는 URL이 아니라 debug_id로 소스맵을 찾는다.
> **"creating a release is no longer required"** — 경로가 어떻게 바뀌어도 매칭된다.

---

## Part 4 정리

- 파이프라인: Relay(필터) → Kafka → symbolication → 그루핑 → 저장
- 서버 필터는 쿼터를 안 먹지만, symbolication **전**이라 원본 파일명 기준 필터 불가
- Debug ID = 청크 내용의 해시. `new Error().stack`으로 런타임 URL과 연결
- release 없이도 소스맵이 매칭된다 — 다만 release는 회귀 추적 때문에 여전히 필요 (Part 6)

---

## 호기심 퀴즈 — Part 5 예고

<!-- _class: invert -->

소스맵은 "스택을 읽기 좋게" 만들어주는 도구다.

**그런데 소스맵 업로드가 이슈가 묶이는 방식(그루핑)에도 영향을 줄까?**

답이 "그렇다"라면, 어떤 경로로?

<!-- 준다. 다음 Part의 핵심 — 그루핑은 symbolication 이후의 프레임으로 해시를 만들고, JS는 특히 context line(그 줄의 소스 코드)을 재료로 쓰기 때문. 답을 여기서 주지 말고 Part 5로 넘어가는 동기로 쓸 것 -->

---

## Part 5 — 그루핑

이슈는 어떻게 묶이는가

---

## 그루핑의 우선순위

공식 문서 원문:

> "All versions consider the **fingerprint** first, the **stack trace** next,
> then the **exception**, and then finally the **message**."

실제 서버 설정 (`src/sentry/grouping/strategies/configurations.py`):

```python
strategies=[
    "chained-exception:v1",  # This handles single exceptions, too
    "threads:v1",
    "stacktrace:v1",
    "template:v1", "csp:v1", ...,
    "message:v1",
],
# "The first strategy to produce a result will become the winner."
```

- SDK가 fingerprint를 보냈으면 알고리즘 결과는 **해시에 반영되지 않는다** (계산은 되지만 비기여 마킹)
- 스택이 없으면 exception type+value → message 순으로 폴백

---

## JS 스택은 무엇으로 해시되는가

플랫폼마다 해시에 넣는 프레임 요소가 다르다 (develop 문서 기준):

| 플랫폼     | 해시 재료                                             |
| ---------- | ----------------------------------------------------- |
| JavaScript | **module + filename(소문자 basename) + context line** |
| Python     | module + function + context line                      |
| Native     | 사실상 function 이름만 (강한 정규화)                  |

- 함수명은 **소스맵으로 복원된 context line이 있을 때만 무시**된다 — 소스맵이 없으면 minified 함수명이 그대로 해시에 들어간다
- context line을 쓰는 플랫폼은 서버 코드에 하드코딩되어 있다:

```python
initial_context={
    "contextline_platforms": ("javascript", "node", "python", "php", "ruby"),
}
```

---

## 따라서: 소스맵이 없으면 그루핑이 파괴된다

해시 재료가 어떻게 바뀌는지 따라가 보자.

- 소스맵 O → **원본 파일명 + 원본 context line** → 빌드가 바뀌어도 동일 → 안정적 그루핑
- 소스맵 X → 한 줄짜리 minified 코드는 **120자 초과라 context line이 해시에서 버려지고**
  ("ignored because line is too long"), 남는 재료가 **해시 붙은 파일명**(`main-8f3a2.js`)과
  **minified 함수명**(`a`, `Xu`) → 빌드마다 달라짐 → 같은 버그가 배포마다 새 이슈

> 1부 도입에서 본 "같은 에러인데 이슈 40개"의 1번 원인.
> 소스맵은 "스택을 읽기 위한 것"이기 전에 **그루핑의 전제 조건**이다.

<!-- 부가(구두 설명용): 해시는 app hash(in-app 프레임만)와 system hash(전체 프레임) 두 개가 만들어지고, 하나라도 기존 이슈와 일치하면 그 이슈에 붙는다 (GroupHash 모델) -->

---

## 어떤 프레임이 그루핑에 참여하는가: enhancement rules

Sentry에는 내장 규칙 파일이 있다 (`all-platforms@2026-01-20.txt`, 약 500줄):

```text
## js
path:**/node_modules/** -app

### transpilers and polyfills are just noise, be more aggressive
module:@babel/** -app -group
module:core-js/** -app -group
module:tslib/** -app -group

## (전 플랫폼 공통 섹션) 예외를 던진 지점 자체는 노이즈
category:throw ^-group -group ^-app -app
```

- 문법: `matcher:expression action…` — `+app/-app`(in-app 여부), `+group/-group`(해시 참여 여부), `^`(위 프레임에), `v`(아래 프레임에)
- node_modules는 in-app이 아니고, babel/core-js 프레임은 **해시에서 제외**된다
- 같은 문법으로 프로젝트 설정에 **커스텀 규칙**을 쓸 수 있다

---

## 뭉친 이슈 나누기: fingerprint

공통 에러 핸들러가 한곳에서 throw하면 스택이 전부 핸들러를 가리킨다 → 이슈 하나에 버그 7개.

**SDK 측** — 이벤트에 fingerprint를 실어 보낸다:

```ts
Sentry.captureException(error, {
  fingerprint: ['api-error', error.endpoint, String(error.status)],
})
// 기본 그루핑에 축만 추가하려면:
fingerprint: ['{{ default }}', error.endpoint]
```

**서버 측** — 배포 없이 대시보드에서 규칙으로:

```text
error.type:DatabaseUnavailable         -> system-down
stack.function:"query_database"        -> {{ default }}, {{ transaction }}
```

변수: `{{ default }}`, `{{ error.type }}`, `{{ message }}`, `{{ stack.function }}`, `{{ transaction }}`, `{{ tags.x }}` 등 — SDK/서버 양쪽에서 동일하게 사용 가능.

---

## 쪼개진 이슈 막기: 메시지 위생

그루핑 폴백이 message까지 내려간 상황이라면:

```ts
// 안티패턴: 이슈가 유저 수만큼 생긴다
throw new Error(`User ${userId} not found`)

// 권장: 메시지는 고정, 가변 값은 컨텍스트로
Sentry.captureException(new Error('User not found'), {extra: {userId}})
```

- 서버가 message에서 파라미터를 일부 제거해주지만, 완전하지 않다
- **에러 메시지 = 에러의 종류, 컨텍스트 = 개별 상황의 값** — 이 분리가 원칙

부가: 신규 이슈 생성 시에는 AI 그루핑(Seer)이 임베딩 유사도로 한 번 더 개입한다. 완전 커스텀 fingerprint면 우회되지만, `{{ default }}` 하이브리드면 그 부분에는 여전히 적용된다.

---

## Part 5 정리

- 우선순위: fingerprint → stack trace → exception → message. fingerprint가 있으면 알고리즘은 실행 안 됨
- JS 해시 재료는 module + filename + **context line** — 그래서 소스맵이 그루핑의 전제
- babel/core-js/node_modules 프레임은 내장 enhancement 규칙이 이미 걸러준다
- 뭉침은 fingerprint로 나누고, 쪼개짐은 메시지 위생으로 막는다

---

## 랩업 퀴즈 — Part 5

<!-- _class: invert -->

**Q1.** 소스맵 없이 운영하다가 나중에 업로드를 시작하면, 기존 이슈들은 어떻게 될까?

**Q2.** `throw new Error(\`User ${userId} not found\`)`가 이슈를 수백 개 만들었다. 그루핑의 어느 단계까지 내려갔길래 이렇게 됐을까?

---

## 정답 — Part 5

<!-- _class: invert -->

**A1.** 해시 재료가 minified에서 원본으로 바뀌므로 **신규 이슈로 갈라진다**. 그루핑 설정 변경과 마찬가지로 "이슈 세대 교체"가 일어난다.

**A2.** 스택 그루핑이 실패해 **message 폴백**까지 내려간 상황. 서버가 파라미터를 일부 제거해주지만 완전하지 않다 — 메시지 고정 + `extra`로 해결한다.

---

## Part 6 — 원리에서 나오는 운영 규칙

지금까지의 내부 동작이 그대로 실무 수칙이 된다

---

## 규칙 1. 소스맵 + release는 협상 불가

- Debug ID 덕에 소스맵 매칭에 release가 필수는 아니지만 (Part 4),
  release는 **다른 세 가지** 때문에 여전히 필요하다:
  - "이 에러는 어느 배포부터인가" (회귀 추적)
  - **Resolved in next release** — 문서 원문: "Fix is in `main`, prod is still on `1.8.0` — only a recurrence in `1.9.0+` reopens it"
  - Releases 화면의 신규 이슈 감지

```ts
Sentry.init({
  release: process.env.NEXT_PUBLIC_GIT_SHA,
  environment: process.env.NEXT_PUBLIC_STAGE,
})
```

- 소스맵은 스택 가독성이 아니라 **그루핑 때문에** 필수다 (Part 5)

---

## 규칙 2. 필터는 2중으로, 각자의 자리에

| 계층                      | 실행 위치           | 특징                                 |
| ------------------------- | ------------------- | ------------------------------------ |
| `ignoreErrors`/`denyUrls` | SDK event processor | 전송량 자체를 줄임. 코드 배포 필요   |
| `beforeSend`              | SDK, processors 뒤  | 동적 조건·마스킹. 최종 이벤트를 본다 |
| Inbound filters           | Relay (서버)        | **쿼터 미소모**. 대시보드에서 즉시   |

- SDK에는 `DEFAULT_IGNORE_ERRORS`가 이미 있다 (1부) — 중복 추가로 목록을 오염시키지 말 것
- 서버 필터는 symbolication 전이라 원본 파일명 조건은 불가 (Part 4)
- 판단 기준: 항구적 노이즈는 SDK에, 긴급 차단은 서버에

---

## 규칙 3. 쿼터는 "미리" 지켜야 한다

1부 Part 3과 이번 Part 4에서 본 사실들:

- 쿼터 초과 이벤트는 **수락 자체가 안 되고 복구 불가** (문서: "will not be accepted")
- rate limit 발동 시 SDK는 이벤트를 **로컬에서 버린다** — 나중에 안 온다
- spike protection이 발동해도 마찬가지로 드랍

따라서:

```ts
Sentry.init({
  sampleRate: 1.0, // 에러는 전량 — 샘플링할 이유가 없다
  tracesSampleRate: 0.05, // 쿼터를 먹는 건 대부분 이쪽
  replaysSessionSampleRate: 0.01,
  replaysOnErrorSampleRate: 1.0,
})
```

- 얼마나 버려지고 있는지는 client report(Stats → Client Discards)로 감시할 수 있다 (1부)

---

## 규칙 4. try/catch 뒤에는 손으로 보내라

1부에서 본 원리: SDK는 **unhandled만** 자동으로 잡는다.

```ts
try {
  await submitOrder(payload)
} catch (e) {
  toast.error('주문에 실패했습니다')
  Sentry.captureException(e, {
    tags: {feature: 'checkout'},
    extra: {orderId},
  })
}
```

- fetch의 4xx/5xx는 JS 관점에서 에러가 아니다 — reject되지 않으므로 안 잡힌다
- `captureMessage`에 가변 문자열을 넣으면 message 그루핑으로 폴백되어 이슈가 분열한다 (Part 5) — **Error 객체 + 고정 메시지 + 컨텍스트**가 항상 정답

---

## 규칙 5. 이슈 상태를 원리에 맞게 쓰기

현재 이슈 상태: **New, Ongoing, Escalating, Regressed, Archived, Resolved**

- **Resolve는 "in next release"로** — release 설정(규칙 1)이 되어 있어야 다음 릴리스에서의 재발만 이슈를 다시 연다
- **Archive는 기본이 "until escalating"** — 발생량이 예측 임계치를 넘으면 자동으로 다시 올라온다. 판정은 직전 1주 데이터 기반 forecast (7일 미만 이슈는 시간당 최대치 × 10)
- 그래서 "혹시 몰라서 안 닫는" 습관은 근거가 없다 — **닫아도 시스템이 다시 열어준다**

알림은 이 상태 전이에 걸어야 한다: 새 이슈 / **Escalating** / **Regressed** / 핵심 경로 태그. "발생했다"는 알림 대상이 아니다.

---

## 규칙 6. 주간 트리아지 15분을 루틴으로

도구가 아니라 습관이 Sentry를 살린다. 주 1회, 돌아가면서:

1. 지난 주 **신규 이슈**를 훑는다 — 필터: `is:unresolved firstSeen:-7d`
2. 각 이슈를 셋 중 하나로 보낸다: **고친다(assign) / 안 고친다(archive) / 노이즈(필터 PR)**
3. 이벤트 수 상위 10개를 확인한다 — 방치된 고빈도 이슈가 쿼터를 먹는다 (규칙 3)

- 15분 안에 안 끝난다면 노이즈 필터(규칙 2)가 부족하다는 신호다
- 새 팀원 온보딩 과제로도 좋다 — 서비스의 약한 곳을 가장 빨리 배운다

<!-- 데모: 실제 프로젝트 대시보드를 열고 신규 이슈 하나를 같이 트리아지해보기 -->

---

## 부록 — 알아두면 좋은 것들

내부 동작을 알아야 보이는 꿀팁 다섯 가지

---

## 꿀팁 1. 에러 화면에 event_id를 노출하라

1부에서 본 것처럼 event_id는 **서버 응답을 기다리지 않고 클라이언트에서 즉시** 생성된다. 그래서 에러가 난 그 자리에서 바로 쓸 수 있다:

```tsx
<Sentry.ErrorBoundary
  fallback={({eventId}) => (
    <ErrorPage message={`문의 시 이 코드를 알려주세요: ${eventId}`} />
  )}
>
```

- CS 문의가 들어오면 event_id로 대시보드에서 **해당 이벤트를 바로 검색**
- "재현이 안 되는데요"라는 대화가 "이벤트 열어보니 breadcrumb에 다 있네요"로 바뀐다
- `Sentry.lastEventId()`로 어디서든 직전 이벤트 id를 얻을 수도 있다

---

## 꿀팁 2. 애드블로커에 막히면 tunnel

애드블로커 상당수가 `*.sentry.io` 요청을 차단한다 — 에러가 많이 나는 환경일수록 리포팅이 누락되기 쉽다.

```ts
// @sentry/nextjs — 자기 도메인의 라우트로 우회
export default withSentryConfig(nextConfig, {
  tunnelRoute: '/monitoring',
})
```

- 1부에서 본 원리: tunnel 사용 시 envelope 헤더에 dsn이 들어가므로, 중계 서버(Next.js가 자동 생성)는 그걸 보고 Sentry로 포워딩한다
- 도입 전후로 이벤트 볼륨을 비교해보면 그동안 얼마나 누락됐는지 보인다

---

## 꿀팁 3. error.cause를 쓰면 체인이 펼쳐진다

기본 integration 중 **linkedErrorsIntegration**(1부)이 `error.cause`를 따라가며 원인 에러를 이벤트에 같이 싣는다.

```ts
try {
  await fetchUser(id)
} catch (e) {
  // 원인을 버리지 않고 도메인 에러로 감싼다
  throw new UserLoadError('failed to load user', {cause: e})
}
```

- 대시보드에서 `UserLoadError`와 원인(`TypeError: Failed to fetch` 등)의 **스택이 둘 다** 보인다
- 도메인 에러로 감싸면 그루핑 단위도 "기능" 기준으로 정리된다 — fingerprint 없이 뭉침을 줄이는 가장 값싼 방법

---

## 꿀팁 4. extra의 객체가 `[Object]`로 잘려 있다면

1부의 조립 파이프라인 마지막 단계가 normalize였다 — 기본 **depth 3**에서 자른다.

```ts
Sentry.captureException(e, {
  extra: {order}, // order.items[0].product.name → 잘림
})

// 방법 1: 필요한 값만 평평하게
Sentry.captureException(e, {
  extra: {orderId: order.id, itemCount: order.items.length},
})

// 방법 2: 정말 깊은 구조가 필요하면
Sentry.init({normalizeDepth: 5})
```

- 순환 참조 제거와 페이로드 폭발 방지를 위한 안전장치라, 올리더라도 적당히
- 애초에 "디버깅에 실제로 쓸 값만 골라 넣는" 쪽이 항상 낫다

---

## 꿀팁 5. 에러 헬퍼가 스택 맨 위를 차지한다면 framesToPop

`assert()` 같은 헬퍼를 만들면 모든 에러의 스택 최상단이 헬퍼 자신이 된다 — 그루핑 재료가 오염된다(Part 5).

```ts
function assert(condition: unknown, message: string): asserts condition {
  if (!condition) {
    const error = new Error(message)
    // Sentry 스택 파서가 이 개수만큼 위에서 프레임을 버린다
    ;(error as any).framesToPop = 1
    throw error
  }
}
```

- SDK 스택 파서가 지원하는 관례다 (`parseStackFrames`) — invariant 같은 라이브러리들이 쓰는 방식
- 스택 최상단이 "실제 호출한 곳"이 되어 가독성과 그루핑이 같이 좋아진다

---

## 번외. @sentry/nextjs는 구조가 조금 다르다

Next.js에서는 서버·엣지·클라이언트 **세 런타임에서 SDK가 따로 초기화**된다:

```text
instrumentation.ts          서버·엣지 config 로드 (register())
                            + onRequestError = Sentry.captureRequestError
instrumentation-client.ts   브라우저 SDK 초기화 ← 1부에서 본 세계
```

- **Server Component에서 난 에러**는 브라우저가 아니라 **서버 런타임의 SDK**가 잡는다 — `onRequestError` 훅 경유. 1부(브라우저 계측)가 아니라 Node 계측의 영역이다
- 클라이언트 렌더 에러는 ErrorBoundary(`global-error.tsx`) → 브라우저 SDK로
- 소스맵 업로드와 tunnel은 `withSentryConfig`가 담당한다 (규칙 1, 꿀팁 2)

<!-- 데모: 같은 throw를 server component와 client component에 각각 넣고, 이벤트가 어느 런타임으로 잡히는지 비교하면 효과적 -->

---

## 전체 체크리스트

- [ ] 번들러 플러그인으로 소스맵 업로드 (Debug ID 자동 주입)
- [ ] `release`(커밋 SHA) + `environment` 설정
- [ ] `tracesSampleRate` 조정, `sampleRate`는 1.0 유지
- [ ] SDK 필터: 프로젝트 고유 노이즈만 (기본 목록과 중복 금지)
- [ ] 서버 Inbound filters: 확장 프로그램·크롤러 필터 활성화
- [ ] try/catch 안의 중요 에러에 `captureException` + 고정 메시지 + 컨텍스트
- [ ] 공통 에러 핸들러에 `fingerprint` 설계
- [ ] 알림: 새 이슈 / Escalating / Regressed / 핵심 경로, production 한정
- [ ] Stats → Client Discards 주기적 확인 (조용함 ≠ 에러 없음)
- [ ] Resolve는 "in next release"로, Archive는 "until escalating"으로
- [ ] 주간 15분 트리아지를 팀 캘린더에

---

## 마지막 한 장

이 시리즈에서 따라간 여정:

> `throw` → onerror(프로퍼티 교체) → 3-스코프 병합 → processors → beforeSend
> → envelope(JSON 라인) → Relay(필터) → Symbolicator(Debug ID) → 그루핑(context line) → 이슈

운영 수칙은 외울 것이 아니라 **이 파이프라인에서 도출되는 것**이다.

- 소스맵을 올리는 이유 = context line 해시
- 필터를 2중으로 두는 이유 = 쿼터는 Relay 앞에서만 절약된다
- Archive를 겁내지 않는 이유 = escalating forecast가 지켜본다

파이프라인이 머리에 있으면, 처음 보는 문제도 위치를 특정할 수 있다.

---

# 감사합니다

<!-- _class: invert -->

기준: sentry-javascript v10.69.0 / getsentry/sentry, relay (2026-01)

@yceffort
