---
title: 패키지 가이드 4탄) dependencies, devDependencies, peerDependencies
marp: true
paginate: true
theme: default
tags:
  - npm
date: 2026-04-04
description: '세 가지 의존성 필드의 역사, 정의, 그리고 현대적 의미'
published: true
art:
  undraw: organized-content
---

# 패키지 가이드 4탄) dependencies, devDependencies, peerDependencies

<!-- _class: invert deck-cover -->

@yceffort

---

## 이 발표의 목표

- `dependencies`, `devDependencies`, `peerDependencies`가 **왜 생겼는지** 역사적 배경
- 각 필드의 **정확한 정의**
- 일반적인 서비스 프로젝트에서 **왜 구분이 중요하지 않은지**
- `peerDependencies`의 진짜 의미와 **mocha 플러그인 사례**
- 현대에 와서 **의미가 변질된 이유**
- 패키지 개발자로서 **주의할 점**
- `pnpm`으로 **올바르게 의존성 검사하는 법**

---

## 역사: npm 초기에는 `dependencies` 하나뿐이었다

<style scoped>section { font-size: 22px; }</style>

| 시기 | 버전    | 변화                                                |
| ---- | ------- | --------------------------------------------------- |
| 2010 | npm 0.x | `dependencies`만 존재. 모든 의존성을 한 곳에        |
| 2011 | npm 1.x | `devDependencies` 추가. 개발 도구와 런타임 분리     |
| 2013 | npm 1.2 | `peerDependencies` 추가. 플러그인 문제 해결         |
| 2015 | npm 3   | flat `node_modules` 도입, peerDeps 자동설치 제거    |
| 2020 | npm 7   | peerDeps 자동설치 부활, `peerDependenciesMeta` 추가 |

- 핵심: 처음부터 세 개가 있던 게 아니라, **문제가 생길 때마다 하나씩 추가**된 것

---

## `devDependencies`는 왜 생겼나?

<style scoped>section { font-size: 22px; }</style>

npm 초기에는 테스트 도구(`mocha`), 빌드 도구(`grunt`)도 전부 `dependencies`에 넣었음

```json
{
  "name": "my-library",
  "dependencies": {
    "lodash": "^1.0.0",
    "mocha": "^1.0.0",
    "grunt": "^0.4.0"
  }
}
```

**문제**: `npm install my-library` 하면 `mocha`, `grunt`까지 딸려옴

- 사용자는 `lodash`만 필요한데 테스트/빌드 도구까지 설치됨
- 설치 시간 증가, `node_modules` 비대화
- **해결**: 개발 시에만 필요한 것을 분리할 필드가 필요 → `devDependencies`

---

## `deps` vs `devDeps` 구분이 흐려진 이유

<style scoped>section { font-size: 18px; }</style>

원래 `devDependencies`의 구분은 명확했다: **`node server.js`로 직접 실행**하던 시절, 런타임에 필요 없는 건 빼는 것

```dockerfile
COPY package.json package-lock.json ./
RUN npm ci --omit=dev           # devDependencies 제외 (구: --production)
COPY dist/ ./dist/
CMD ["node", "dist/server.js"]
```

**그런데 webpack(2012)의 등장이 이 경계를 흐렸다:**

- 번들러가 `import` 그래프를 따라가므로, `package.json` 필드 구분과 **무관하게** 모든 의존성이 번들에 포함
- "빌드 시에만 필요한 것"과 "런타임에 필요한 것"의 경계가 번들러 안에서 사라짐
- 프론트엔드 개발자에게 deps/devDeps 구분은 **실질적 차이 없는 관례**가 됨

**현대 배포도 이 흐름을 강화:**

- **Docker 멀티스테이지**: 빌드 스테이지에서 전부 설치 → 결과물만 복사
- **서버리스 / Edge**: 번들된 단일 파일 배포. `node_modules` 자체가 포함 안 됨
- **프레임워크 관리**: Next.js `output: 'standalone'`이 필요한 모듈만 자동으로 추림

---

## 각 필드의 정의

<style scoped>section { font-size: 20px; }</style>

### `dependencies`

> **이 패키지가 동작하기 위해 런타임에 반드시 필요한 패키지**

- `npm install my-package` 시 함께 설치됨
- 예: `lodash`, `express`, `react` (라이브러리가 직접 import하는 것)

### `devDependencies`

> **개발, 테스트, 빌드 시에만 필요한 패키지**

- `npm install my-package` 시 설치되지 **않음**
- `npm install` (패키지 자체 개발 시) 에서만 설치됨
- 예: `typescript`, `jest`, `eslint`, `rollup`

### `peerDependencies`

> **이 패키지가 동작하려면 호스트 프로젝트가 직접 설치해야 하는 패키지**

- 뒤에서 자세히 다룸

---

## 서비스 프로젝트에서는 왜 구분이 덜 중요할까?

<style scoped>section { font-size: 18px; }</style>

```text
  패키지 A ──►┌──────────────┐
  패키지 B ──►│ 서비스(Next.js)│── 번들 ──► 사용자
  패키지 C ──►└──────────────┘
                ▲ 의존성 체인의 "끝"
```

- 서비스 프로젝트는 **누군가에 의해 `npm install`되는 대상이 아님**
- `dependencies`에 넣든 `devDependencies`에 넣든 **빌드 시 전부 설치**됨
- webpack, vite 등 번들러는 `import` 그래프를 따라갈 뿐, `package.json` 필드를 구분하지 않음
- 대부분의 현대 배포(Vercel, Docker 멀티스테이지)는 **전체 설치 → 빌드 → 결과물만 배포**

**단, Node 런타임이 남는 환경에서는 여전히 중요:**

- SSR / API 서버 / CLI / Docker에서 `npm ci --omit=dev`로 배포 시, 런타임 의존성이 devDeps에 있으면 **깨질 수 있음**
- Next.js `output: 'standalone'`도 런타임 파일만 추려내므로 올바른 분류가 필요
- 클라이언트 번들만 배포하는 SPA라면 영향이 상대적으로 작음

> **패키지** 를 만들 때는 이야기가 완전히 달라진다

---

## `peerDependencies`는 왜 생겼나? — mocha로 이해하기

<style scoped>section { font-size: 20px; }</style>

`mocha`는 Node.js 테스트 러너. 요즘의 `jest`나 `vitest` 같은 역할

```js
// test/math.test.js
describe('add 함수', () => {
  it('1 + 2 = 3', () => {
    assert.strictEqual(add(1, 2), 3)
  })
})
```

여기서 핵심: **`describe`와 `it`을 어디서도 import하지 않았다**

- mocha는 테스트 파일을 실행하기 전에 `describe`, `it`, `before`, `after` 등을 **글로벌에 주입**
- 테스트 파일은 mocha가 만들어 놓은 글로벌 환경 위에서 동작
- 이건 jest도 마찬가지 — `describe`, `it`, `expect` 전부 import 없이 사용

> import 없이 동작한다 = **어떤 mocha 인스턴스가 글로벌을 세팅했느냐**가 중요하다

---

## 왜 import 없이 글로벌 주입이었을까?

<style scoped>section { font-size: 20px; }</style>

mocha는 Ruby의 **RSpec**에서 영향을 받았음. Ruby에서 `describe`, `it`은 그냥 메서드 호출:

```ruby
# Ruby — 괄호 없이 메서드 호출 + 블록(do...end) = 자연어처럼 읽힘
describe "Calculator" do
  it "adds numbers" do
    expect(1 + 2).to eq(3)
  end
end
```

- Ruby는 최상위 스코프가 `Object` 인스턴스 → 어디서든 메서드 주입 가능 → **import 개념 자체가 불필요**
- 언어 설계 철학이 "코드가 영어 문장처럼 읽혀야 한다"(Matz) → 글로벌 DSL이 안티패턴이 아니라 **관용적 Ruby**
- mocha, Jasmine이 이 패턴을 JS로 그대로 가져옴
- 현대 JS에서는 글로벌 오염이 안티패턴이므로, `vitest`나 `node:test`는 **명시적 import** 방식으로 전환

---

## mocha 플러그인의 문제

<style scoped>section { font-size: 18px; }</style>

mocha용 플러그인 `mocha-sugar`를 만든다고 가정. mocha의 글로벌 `it`을 확장하는 패키지

```js
// mocha-sugar/index.js — mocha의 글로벌 it을 감싸서 기능을 추가
const originalIt = global.it // ← mocha가 주입한 글로벌 it을 가져옴

global.it.todo = (title) => {
  originalIt(title, function () {
    this.skip() // ← mocha의 skip() API를 사용
  })
}
```

```js
// 사용하는 쪽
require('mocha-sugar')          // 글로벌 it을 확장
describe('my test', () => {
  it.todo('나중에 구현')         // ← mocha-sugar가 추가한 기능
  it('동작하는 테스트', () => { ... })
})
```

**`mocha-sugar`는 mocha를 직접 import하지 않지만, mocha가 세팅한 글로벌 환경에 의존한다**

---

## mocha가 두 벌이면 무슨 일이 생기나?

<style scoped>section { font-size: 18px; }</style>

`mocha-sugar`가 `mocha`를 `dependencies`에 넣으면?

```text
my-project
├── mocha@10.0.0                     ← npx mocha가 실행하는 mocha (글로벌 세팅)
└── node_modules/mocha-sugar/
    └── node_modules/mocha@8.0.0     ← mocha-sugar가 가져온 다른 mocha
```

- `npx mocha`는 mocha@10이 `describe`, `it`을 글로벌에 세팅
- `mocha-sugar`는 자기 `node_modules`의 mocha@8을 참조
- **두 mocha의 글로벌 컨텍스트가 다름** → `this.skip()`이 mocha@10의 러너와 연결 안 됨
- 결과: 플러그인이 정상 동작하지 않거나, 알 수 없는 에러 발생

### peerDependencies로 해결

```json
{
  "name": "mocha-sugar",
  "peerDependencies": {"mocha": ">=6"}
}
```

→ mocha-sugar는 mocha를 직접 갖지 않음 → 호스트의 mocha@10 하나만 존재
→ 글로벌 `it`, `describe`가 **단일 mocha 인스턴스**에서 온 것이 보장됨

---

## peerDependencies의 핵심 원리

<!-- _class: invert -->

> **"나는 이 패키지가 필요하지만, 내가 가져오면 안 된다. 호스트가 가져와야 한다."**

이것이 필요한 패턴:

- **플러그인**: eslint-plugin-\*, babel-plugin-\*, mocha reporter (공식 문서의 대표 예시: chai plugin 계열)
- **프레임워크 확장**: React 컴포넌트 라이브러리, Next.js 플러그인
- **싱글턴이 보장되어야 하는 패키지**: react, react-dom

---

## npm 3 — nested에서 flat으로 (2015)

<style scoped>section { font-size: 20px; }</style>

npm 3에서 `node_modules`를 **nested → flat** 구조로 바꿈

```text
# npm 1~2 (nested) — 패키지마다 자기 의존성을 중첩으로 가짐
node_modules/
├── A/
│   └── node_modules/lodash@3/
└── B/
    └── node_modules/lodash@4/

# npm 3+ (flat) — 같은 패키지를 최상위로 끌어올림 (hoisting)
node_modules/
├── A/
├── B/
└── lodash@4/   ← 하나만 남김. A가 원하는 @3은?
```

flat 구조는 디스크를 절약하지만, **최상위에 어떤 버전을 올릴지** 판단이 복잡해짐

---

## npm 3 — peerDeps 자동 설치 제거

<style scoped>section { font-size: 22px; }</style>

flat 구조에서 peerDeps까지 자동으로 끼워넣으면 호이스팅 충돌이 더 심해짐

- nested 시절에는 각 패키지가 자기 `node_modules` 안에 독립적으로 의존성을 가졌으므로, peerDeps를 자동 설치해도 **다른 패키지에 영향이 없었음**
- flat에서는 최상위 `node_modules`에 **하나의 버전만** 올라가야 하는데, peerDeps가 요구하는 버전까지 끼워넣으면 "최상위에 어떤 버전을 올릴지" 판단이 **더 복잡해지고 충돌 확률이 급증**
- 예: A가 `react@16`을 peerDep으로, B가 `react@15`를 peerDep으로 요구 → 둘 다 최상위에 올릴 수 없음

→ **자동 설치를 제거하고 경고만 출력**하는 것으로 후퇴

```text
npm WARN mocha-junit-reporter@2.0.0 requires a peer of mocha@>=6
but none is installed. You must install peer dependencies yourself.
```

- 결과: 많은 개발자들이 peerDeps 경고를 **무시**하기 시작
- peerDependencies의 의미가 흐려지기 시작한 시점

---

## npm 7 — peerDeps 자동 설치 부활 (2020)

<style scoped>section { font-size: 22px; }</style>

npm 7에서 다시 peerDependencies를 **기본 설치 대상**으로 변경

```bash
npm install mocha-junit-reporter
# → peerDep인 mocha도 설치를 시도하지만, 충돌 시 에러가 발생할 수 있음
```

- peerDeps를 자동 해결하되, **충돌 시 설치 자체가 실패** → `--legacy-peer-deps` 플래그 등장
- 동시에 `peerDependenciesMeta`도 추가

```json
{
  "peerDependencies": {"react": "^18 || ^19"},
  "peerDependenciesMeta": {"react": {"optional": true}}
}
```

이 자동 설치 부활이 **서비스 코드의 무분별한 비대화의 시작점**이기도 하다 — 패키지 제작자는 "어차피 자동 설치되니까"라는 안이함으로 peerDeps를 늘리고, 사용자의 `node_modules`는 의도와 무관하게 불어난다

---

## 현대에 와서 의미가 변질된 이유

<style scoped>section { font-size: 20px; }</style>

### 1. 패키지 매니저마다 동작이 다름

| 패키지 매니저   | peerDeps 자동 설치 | 동작                                         |
| --------------- | ------------------ | -------------------------------------------- |
| npm 3~6         | ❌                 | 경고만                                       |
| npm 7+          | ✅                 | 자동 설치, 충돌 시 에러                      |
| yarn 1          | ❌                 | 경고만                                       |
| yarn berry (2+) | ❌                 | 자동 설치 안 함. PnP에서 누락 시 런타임 에러 |
| pnpm 7          | ❌                 | `auto-install-peers` 기본값 `false`          |
| pnpm 8+         | ✅                 | `auto-install-peers` 기본값 `true`로 변경    |

### 2. 원래 의도와 다른 사용

- 원래: "플러그인이 호스트 패키지를 참조"하기 위한 메커니즘
- 현재: "이 패키지를 같이 설치해야 한다"는 **단순한 알림** 용도로 변질
- `react`를 peerDep에 넣는 것은 원래 의미에 부합하지만, 단순히 "함께 쓰면 좋은 패키지"를 peerDep에 넣는 사례도 증가

---

## 요즘 peerDependencies 사례

<style scoped>section { font-size: 18px; }</style>

### 올바른 사용 — 싱글턴 / 플러그인 패턴

```json
// React 컴포넌트 라이브러리
{ "peerDependencies": { "react": "^18 || ^19", "react-dom": "^18 || ^19" } }
// ESLint 플러그인
{ "peerDependencies": { "eslint": "^8.0.0 || ^9.0.0" } }
// Tailwind 플러그인
{ "peerDependencies": { "tailwindcss": "^3.0.0 || ^4.0.0" } }
```

### 논쟁적 사용 — "같이 쓰면 좋은 것"

```json
// 어떤 상태관리 라이브러리의 React 바인딩
{
  "peerDependencies": {
    "react": "^18 || ^19", // ← 이건 맞음 (싱글턴)
    "immer": ">=9" // ← 이건 논쟁적 (dependencies로도 가능)
  }
}
```

- `immer`가 두 벌이어도 문제는 없지만, 번들 크기 최적화를 위해 peerDep으로 선언하는 경우도 있음

---

## 요즘 peerDependencies 사례 (2) — 사내 공통 패키지

<style scoped>section { font-size: 20px; }</style>

```json
// @my-company/design-system
{
  "peerDependencies": {
    "react": "^19.0.0",
    "@my-company/tokens": "^2.0.0",
    "lodash": "^4.17.0"
  }
}
```

- 여러 서비스가 공유하는 사내 패키지에서, **버전 불일치와 중복 설치를 방지**하고 싶을 때
- `lodash`처럼 두 벌이어도 동작은 하지만, 번들에 중복 포함되면 **크기가 낭비**되는 패키지도 대상

**peerDeps는 책임의 선언이기도 하다**

- `dependencies` = "이 버전은 **내가(패키지 팀이) 책임**진다"
- `peerDependencies` = "이 버전은 **호스트(서비스 팀)가 책임**져라"
- lodash를 peerDep으로 → **"lodash 버전 관리는 각 서비스 팀의 몫"**
- lodash를 dependencies에 → 패키지 팀이 올릴 때마다 모든 서비스에 영향

---

## 패키지 개발자로서 주의할 점 (1)

<style scoped>section { font-size: 18px; }</style>

### 런타임 의존성을 `devDependencies`에 넣지 말 것

```json
// ❌ 잘못된 예
{
  "devDependencies": {
    "lodash": "^4.17.0" // 런타임에 import 하는데 devDeps에 있음
  }
}
```

- 사용자가 `npm install`하면 `lodash`가 설치 안 됨 → **런타임 에러**. 패키지에서는 치명적

### 프레임워크를 `dependencies`에 넣지 말 것

```json
// ❌ 잘못된 예 — React 컴포넌트 라이브러리
{
  "dependencies": {
    "react": "^19.0.0" // 호스트와 다른 버전의 react가 설치될 수 있음
  }
}
```

- React가 두 벌 → hooks 에러, context 공유 불가 등 심각한 버그

---

## 패키지 개발자로서 주의할 점 (1-2)

<style scoped>section { font-size: 22px; }</style>

### `@types/*`를 `dependencies`에 넣지 말 것

```json
// ❌ 잘못된 예
{
  "dependencies": {
    "@types/node": "^20.0.0"
  }
}
```

- `npm install my-package` 시 `@types/node`가 사용자에게 딸려옴
- 사용자 프로젝트에 **의도하지 않은 Node.js 타입이 주입**되어 타입 오염 발생
- `@types/*`는 빌드 시에만 필요하므로 반드시 **devDependencies**에

---

## 패키지 개발자로서 주의할 점 (2)

<style scoped>section { font-size: 20px; }</style>

### peerDeps + devDependencies 패턴

peerDeps에 넣은 패키지는 **개발 시에는** devDependencies에도 넣어야 한다

```json
{
  "peerDependencies": {
    "react": "^18.0.0 || ^19.0.0"
  },
  "devDependencies": {
    "react": "^19.0.0"
  }
}
```

- peerDeps만 선언하면 `npm install`(패키지 개발 시)에서 react가 설치되지 않음
- 로컬 개발, 테스트, 빌드를 위해 **devDependencies에도** 넣어야 함
- 배포 시에는 devDeps가 제외되므로, 사용자에게 중복 설치되지 않음

> peerDeps = "사용자에게 요구하는 것", devDeps = "내가 개발할 때 필요한 것"
> 둘은 역할이 다르므로 **같은 패키지가 양쪽에 있어도 정상**

---

## 패키지 개발자로서 주의할 점 (3)

<style scoped>section { font-size: 18px; }</style>

### peerDependencies 버전 범위는 최대한 넓게, 단 `>=`는 쓰지 말 것

```json
// ❌ 너무 좁은 범위
{ "peerDependencies": { "react": "^19.0.0" } }
// ❌ 검증 없이 넓은 범위 — 미래 메이저 버전까지 보장하는 건 거짓말
{ "peerDependencies": { "eslint": ">=8" } }
// ✅ 실제 테스트한 범위만 명시적으로 나열
{ "peerDependencies": { "react": "^17.0.0 || ^18.0.0 || ^19.0.0" } }
```

- 좁은 범위 → 호스트 프로젝트에서 버전 충돌 발생 확률 증가
- `>=`는 **아직 출시되지 않은 메이저 버전까지 지원을 약속**하는 것 — 테스트하지 않은 버전에 대한 보장은 사용자에게 거짓 안전감을 줌
- `^`와 `||`로 **실제 CI에서 검증한 범위만** 명시적으로 나열할 것
- 새 메이저가 나오면 테스트 후 범위에 추가하는 것이 올바른 프로세스

---

## 패키지 개발자로서 주의할 점 (3-2)

<style scoped>section { font-size: 18px; }</style>

### `0.x` 버전은 `^`의 동작이 다르다

semver에서 `0.x`는 "아직 안정되지 않은 API"를 의미하며, `^`의 범위 해석이 달라진다

```text
^1.2.3  →  >=1.2.3 <2.0.0   (메이저 고정)
^0.2.3  →  >=0.2.3 <0.3.0   (마이너 고정 — 훨씬 좁음!)
^0.0.3  →  >=0.0.3 <0.0.4   (패치 고정 — 사실상 고정 버전)
```

- `0.x`에서는 마이너 업데이트도 breaking change가 허용되므로, `^`가 마이너를 고정시킴
- react-native가 오래 `0.x`에 머물렀던 대표적 사례 — `^0.70.0`은 `>=0.70.0 <0.71.0`이므로, 패치만 허용하는 매우 좁은 범위
- 필요 시 `||`로 마이너 범위를 명시적으로 나열할 것

```json
// ❌ ^0.70.0은 0.70.x만 허용 — 의도보다 좁을 수 있음
{ "peerDependencies": { "react-native": "^0.70.0" } }
// ✅ 지원하는 마이너 범위를 명시적으로 나열
{ "peerDependencies": { "react-native": "^0.70.0 || ^0.71.0 || ^0.72.0 || ^0.73.0" } }
```

---

## 패키지 개발자로서 주의할 점 (3-3)

<style scoped>section { font-size: 22px; }</style>

### 선택적 peerDependencies 활용

```json
{
  "peerDependencies": {"webpack": "^5.0.0"},
  "peerDependenciesMeta": {
    "webpack": {"optional": true}
  }
}
```

- "webpack이 있으면 활용하고, 없어도 동작한다"는 의미
- 불필요한 설치 강요를 줄일 수 있음

---

## pnpm으로 올바르게 의존성 검사하기 (1)

<style scoped>section { font-size: 20px; }</style>

### pnpm의 의존성 관리

- npm/yarn의 flat `node_modules`와 달리, pnpm은 **symlink 기반** 구조
- phantom dependency를 **크게 줄여주지만**, `hoist=true`(기본값)에서는 일부 unlisted dependency 접근이 가능할 수 있음

```text
node_modules/
├── .pnpm/                    ← 실제 패키지가 여기에 설치됨
│   ├── lodash@4.17.21/
│   └── express@4.18.2/
├── lodash -> .pnpm/lodash@4.17.21/   ← symlink
└── express -> .pnpm/express@4.18.2/  ← symlink
```

```bash
# npm에서는 되지만 pnpm에서는 에러
import chalk from 'chalk'  // dependencies에 없으면 못 찾음
```

→ pnpm을 쓰는 것 자체가 의존성을 올바르게 선언하도록 **유도**하는 효과 (`hoist=false`로 더 엄격하게 설정 가능)

---

## pnpm으로 올바르게 의존성 검사하기 (2)

<style scoped>section { font-size: 20px; }</style>

### `.npmrc` 설정

```ini
# peerDependencies 자동 설치 끄기 (기본값: true, pnpm v8+)
# 자동 설치는 의도하지 않은 의존성이 슬금슬금 늘어나는 원인
auto-install-peers=false

# peerDependencies 누락/불일치 시 에러로 처리 (기본값: false)
# 개발자가 필요한 peerDep을 명시적으로 설치하도록 강제
strict-peer-dependencies=true
```

- `auto-install-peers=false` → 어떤 패키지가 설치되는지 개발자가 직접 인지하고 결정
- `strict-peer-dependencies=true` → 누락된 peerDep이 있으면 설치 자체가 실패하므로, 빠르게 발견 가능
- 이 조합이 **"무분별한 node_modules 비대화를 막으면서 의존성 누락도 잡는"** 가장 엄격한 설정

---

## pnpm으로 올바르게 의존성 검사하기 (2-2)

<style scoped>section { font-size: 18px; }</style>

### `pnpm catalog` — 모노레포에서 버전을 한 곳에서 관리

```yaml
# pnpm-workspace.yaml
catalog:
  react: ^19.0.0
  react-dom: ^19.0.0
  lodash: ^4.17.0
```

```json
// packages/ui/package.json
{
  "peerDependencies": { "react": "catalog:" },
  "devDependencies": { "react": "catalog:" }
}
// packages/hooks/package.json
{
  "peerDependencies": { "react": "catalog:" }
}
```

- 모든 workspace 패키지가 **동일한 버전 범위**를 참조 → peerDeps 충돌 원천 차단
- 버전 올릴 때 `pnpm-workspace.yaml` **한 곳만** 수정하면 전체 반영
- 단, **모노레포 내부에서만** 유효 — 외부에서 설치 시 resolve된 실제 버전이 들어감

---

### 유용한 검사 명령어

```bash
# 특정 패키지가 왜 설치되었는지 확인
pnpm why react

# 의존성 트리 확인
pnpm ls --depth 2

# 전체 의존성 중 outdated 확인
pnpm outdated

# 보안 취약점 검사
pnpm audit
```

---

## pnpm으로 올바르게 의존성 검사하기 (3)

<style scoped>section { font-size: 18px; }</style>

### `publint` — 패키징 구조 검사 도구

```bash
# 패키지 디렉토리에서 실행
pnpm dlx publint

# 결과 예시
┌ publint results
│ ✗ pkg.main is ./dist/index.js but the file does not exist.
│ ✗ pkg.module is ESM but pkg.type is not "module".
│ ✓ pkg.exports["."].import is valid
└ 1 error, 1 warning
```

- `exports` / `main` / `module` / `types` / 조건부 export 순서 등 **패키징 호환성**을 검사
- dependencies import 일치 검사가 아니라, **패키지 진입점과 파일 존재 여부** 중심

### `arethetypeswrong` (attw) — 타입 패키징 검사

```bash
pnpm dlx @arethetypeswrong/cli --pack .
```

- **타입 패키징과 모듈 해석 문제**를 검사 (CJS/ESM 타입 불일치 등)

---

## 정리

<style scoped>section { font-size: 20px; }</style>

| 필드               | 설치 시점     | 핵심 질문                                     |
| ------------------ | ------------- | --------------------------------------------- |
| `dependencies`     | 항상          | "이 패키지 없이 내 코드가 런타임에 돌아가나?" |
| `devDependencies`  | 개발 시에만   | "빌드/테스트/린트에만 필요한가?"              |
| `peerDependencies` | 호스트가 설치 | "호스트와 같은 인스턴스를 공유해야 하나?"     |

### 서비스 프로젝트

- deps vs devDeps 구분은 **관례적** 의미만 있음. 실질적 차이 없음.

### 패키지 개발

- deps/devDeps/peerDeps 구분이 **치명적으로 중요**
- 잘못 넣으면 사용자의 앱이 깨짐
- pnpm + `publint` + `attw`로 배포 전 반드시 검증할 것

---

<!-- _class: invert -->

## 한 줄 요약

> 서비스에서는 신경 안 써도 되지만,
> 패키지에서는 잘못 넣으면 **남의 앱이 터진다**.

---

## Q&A — 예상 질문

<!-- _class: invert -->

---

## Q. 모노레포에서 workspace 패키지끼리는 deps? peerDeps?

<style scoped>section { font-size: 20px; }</style>

**상황에 따라 다르다**

```json
// packages/ui (사내 디자인 시스템)
{
  "peerDependencies": {"react": "^19.0.0"},
  "dependencies": {"@my-company/tokens": "workspace:*"}
}
```

- **같은 모노레포 안의 유틸 패키지** → `dependencies` + `workspace:*`
  - 항상 함께 배포되고, 버전을 모노레포가 통제하므로 dependencies가 자연스러움
- **호스트가 직접 설치해야 하는 프레임워크** → `peerDependencies`
  - react, vue 등은 모노레포 안이라도 peerDeps가 맞음
- 핵심 기준: "이 패키지를 모노레포 **밖에서** 설치할 때, 누가 버전을 통제해야 하는가?"

---

## Q. `--legacy-peer-deps`를 계속 쓰고 있는데 괜찮나요?

<style scoped>section { font-size: 22px; }</style>

**괜찮지 않다.** 기술 부채를 쌓고 있는 것

```bash
npm install --legacy-peer-deps   # peerDeps 충돌을 무시
```

- 충돌을 무시한다 = 런타임에 **어떤 버전이 쓰일지 보장할 수 없다**
- React 18과 19가 공존하거나, ESLint 플러그인이 섞이는 등의 문제가 잠복
- 당장은 동작하지만, 어느 날 알 수 없는 에러로 터질 수 있음

**해결**: 충돌 원인을 찾아서 해소

```bash
npm ls react              # 어떤 패키지가 어떤 버전을 요구하는지 확인
npm explain react         # 왜 이 버전이 설치되었는지 추적
```

---

## Q. `--legacy-peer-deps` 대신 쓸 수 있는 것은?

<style scoped>section { font-size: 18px; }</style>

`overrides`(npm) / `resolutions`(yarn) / `pnpm.overrides`(pnpm)로 버전을 **명시적으로 해소**:

```json
// npm (package.json) — npm 8.3+
{
  "overrides": {
    "react": "^19.0.0",
    // 특정 패키지의 의존성만 교체할 수도 있음
    "some-ui-lib": {"react": "^19.0.0"}
  }
}
```

```json
// pnpm (package.json)
{
  "pnpm": {
    "overrides": {"react": "^19.0.0"},
    // peerDep 충돌만 허용하고 싶을 때 (설치 버전은 건드리지 않음)
    "peerDependencyRules": {
      "allowedVersions": {"react": "19"}
    }
  }
}
```

---

## Q. `--legacy-peer-deps` 대신 쓸 수 있는 것은? (2)

<style scoped>section { font-size: 20px; }</style>

```json
// yarn berry (package.json)
{"resolutions": {"react": "^19.0.0"}}
```

핵심 차이:

- `--legacy-peer-deps`는 충돌을 **무시** — 어떤 버전이 설치될지 비결정적
- `overrides`는 충돌을 **해소** — "이 버전을 써라"고 명시적으로 선언
- `overrides`는 `package.json`에 남으므로 **의도가 코드로 추적** 가능

---

## Q. 라이브러리가 peerDeps를 안 올려주면?

<style scoped>section { font-size: 18px; }</style>

예: 내 서비스는 React 19인데, 어떤 라이브러리가 `"react": "^18.0.0"`만 peerDep으로 선언

```text
npm warn ERESOLVE Could not resolve dependency:
npm warn peer react@"^18.0.0" from some-ui-lib@3.2.1
```

**매니저별 우회 수단은 있지만, 근본 해결은 upstream 이슈/PR**

| 매니저     | 방법                                  | 용도                      |
| ---------- | ------------------------------------- | ------------------------- |
| npm        | `overrides`                           | 의존성 트리 버전 교체     |
| pnpm       | `peerDependencyRules.allowedVersions` | 특정 peer 버전 허용       |
| pnpm       | `packageExtensions`                   | 누락/잘못된 peerDeps 보완 |
| yarn berry | `resolutions`                         | 버전 고정                 |

- 이런 설정은 라이브러리가 peerDeps를 업데이트해줄 때까지의 **임시 조치**
- 단, 실제로 호환되는지는 **사용하는 쪽이 검증할 책임**이 있음
- 라이브러리에 이슈/PR을 올려서 peerDeps 범위를 넓혀달라고 요청하는 것이 근본적 해결

---

## Q. dependencies와 peerDependencies에 같은 패키지를 넣으면?

<style scoped>section { font-size: 22px; }</style>

```json
{
  "dependencies": {"lodash": "^4.17.0"},
  "peerDependencies": {"lodash": "^4.17.0"}
}
```

**안티패턴이다.** peer + dev 조합을 사용할 것

---

## Q. `optionalDependencies`와 `peerDependenciesMeta.optional`의 차이는?

<style scoped>section { font-size: 18px; }</style>

이름이 비슷해서 혼동하기 쉽지만, **완전히 다른 메커니즘**

### `optionalDependencies` — 설치 실패를 허용

**"실패"란 구체적으로 무엇인가?**

- 현재 OS/아키텍처에 해당하는 바이너리가 **존재하지 않음** (예: Linux ARM용 패키지를 macOS에서 설치)
- 네이티브 애드온 **컴파일 실패** (예: `node-gyp` 빌드에 필요한 C++ 컴파일러가 없음)
- npm registry에서 해당 패키지를 **다운로드할 수 없음** (네트워크, 프라이빗 레지스트리 등)

이 중 하나라도 발생하면, `dependencies`는 전체 설치가 실패하지만, `optionalDependencies`는 **해당 패키지만 건너뛰고 설치를 계속**한다.

---

## `optionalDependencies` — `@swc/core` 사례

<style scoped>section { font-size: 18px; }</style>

```json
// @swc/core의 package.json (실제)
{
  "optionalDependencies": {
    "@swc/core-darwin-arm64": "1.15.24", // macOS Apple Silicon
    "@swc/core-darwin-x64": "1.15.24", // macOS Intel
    "@swc/core-linux-x64-gnu": "1.15.24", // Linux x64 (glibc)
    "@swc/core-linux-x64-musl": "1.15.24", // Linux x64 (musl/Alpine)
    "@swc/core-linux-arm64-gnu": "1.15.24", // Linux ARM64
    "@swc/core-win32-x64-msvc": "1.15.24" // Windows x64
    // ... 12개 플랫폼별 바이너리
  }
}
```

- `npm install @swc/core` → 12개 바이너리를 **모두 시도**하되, 현재 플랫폼과 맞지 않는 건 실패해도 넘어감
- macOS ARM에서 설치하면 `@swc/core-darwin-arm64`만 성공, 나머지 11개는 조용히 스킵
- 코드에서는 런타임에 맞는 바이너리를 `require()`로 로드하고, 없으면 fallback 처리

---

## `peerDependenciesMeta.optional` — `@swc/core` 사례

<style scoped>section { font-size: 20px; }</style>

같은 `@swc/core`에서 이것도 사용:

```json
// @swc/core의 package.json (실제)
{
  "peerDependencies": {"@swc/helpers": ">=0.5.17"},
  "peerDependenciesMeta": {"@swc/helpers": {"optional": true}}
}
```

- `@swc/helpers`는 SWC가 트랜스파일할 때 주입하는 런타임 헬퍼 (`_async_to_generator` 등)
- 모든 프로젝트가 이 헬퍼를 쓰는 건 아님 → **optional**로 선언
- 호스트가 `@swc/helpers`를 설치하지 않아도 경고/에러 없음
- 설치되어 있으면 SWC가 활용하고, 없으면 헬퍼를 인라인으로 생성

---

## 언제 어떤 걸 써야 하나?

<style scoped>section { font-size: 16px; }</style>

|           | `optionalDependencies` | `peerDependenciesMeta.optional` |
| --------- | ---------------------- | ------------------------------- |
| 설치 주체 | **패키지가** 설치 시도 | **호스트가** 직접 설치          |
| 실패 시   | 조용히 넘어감          | 경고/에러 없음                  |
| 주요 용도 | 플랫폼 종속 바이너리   | 선택적 통합                     |
| 버전 통제 | 패키지 팀              | 호스트(서비스 팀)               |

### `@swc/core`로 보는 실전 예시

```json
{
  // 플랫폼별 네이티브 바이너리 — 패키지가 설치를 시도하고, 실패하면 스킵
  "optionalDependencies": {
    "@swc/core-darwin-arm64": "1.15.24",
    "@swc/core-linux-x64-gnu": "1.15.24"
  },
  // 런타임 헬퍼 — 호스트가 설치 여부를 결정
  "peerDependencies": {"@swc/helpers": ">=0.5.17"},
  "peerDependenciesMeta": {"@swc/helpers": {"optional": true}}
}
```

- `optionalDependencies`: "내가 깔아볼 테니, 안 되면 말고" (설치 주체 = 패키지)
- `peerDependenciesMeta.optional`: "필요하면 네가 깔아, 안 깔아도 됨" (설치 주체 = 호스트)

---

## Q. React 19 신규 API를 쓰려면 peerDeps 범위를 좁혀야 하지 않나요?

<style scoped>section { font-size: 18px; }</style>

**꼭 그렇지는 않다.** feature detection으로 넓은 범위를 유지하면서 신규 API를 활용할 수 있음

```js
import * as React from 'react'

// React 18+에서만 존재하는 API
const useStore =
  typeof React.useSyncExternalStore === 'function'
    ? React.useSyncExternalStore
    : require('./shims/useSyncExternalStore') // React 17 이하 폴백

// React 19 전용
function usePromise(promise) {
  if (typeof React.use === 'function') return React.use(promise)
  // React 18 이하 폴백...
}
```

```json
// peerDeps는 넓게 유지
{"peerDependencies": {"react": "^17.0.0 || ^18.0.0 || ^19.0.0"}}
```

- 사용자의 React 업그레이드를 강제하지 않음 → **점진적 개선**

---

## Q. 그래도 peerDeps 범위를 올려야 하는 경우는?

<style scoped>section { font-size: 22px; }</style>

- 신규 API가 핵심 동작에 **필수적**이고, 폴백이 불가능하거나 의미 없을 때
- breaking change로 인해 구버전 동시 지원이 **코드 복잡성 대비 가치가 없을 때**

> 버전 범위를 좁히는 건 최후의 수단. **feature detection이 가능하면 먼저 시도할 것**

---

## Q. 특정 버전에 버그가 있으면 peerDeps에서 제외해야 하나요?

<style scoped>section { font-size: 20px; }</style>

**아니요.** peerDeps는 그런 용도가 아닙니다.

```json
// ❌ 이렇게 하고 싶은 유혹 (실제로 이런 negation 문법은 semver에 존재하지 않음)
{"peerDependencies": {"react": ">=19.0.0 <19.0.1 || >19.0.1"}}
```

**이런 식으로 특정 버전을 제외하면 안 되는 이유:**

- peerDeps는 **호환 가능한 버전 범위**를 선언하는 것이지, 특정 버전의 버그를 차단하는 도구가 아님
- 그 버그가 내 패키지와 관련 없는 곳에서 발생한 것일 수 있음
- 해당 버전이 패치되면 의미 없는 제약만 남음
- 제외할 버전이 쌓이면 peerDeps가 점점 복잡해지고 관리 불가능

**대신:**

- **README / CHANGELOG**에 "X 버전에 알려진 이슈가 있으니 Y 이상으로 업데이트를 권장합니다" 안내
- 심각한 경우 **GitHub 이슈에 pinned comment**로 고정
- peerDeps는 넓게 유지하되, **문서로 소통**하는 것이 올바른 접근

---

## Q. 한 패키지가 여러 버전으로 설치되면 번들러는 어떤 버전을 선택하나요?

<style scoped>section { font-size: 18px; }</style>

번들러는 Node.js module resolution 알고리즘을 따름 — `import`하는 파일 위치에서 **가장 가까운 `node_modules`** 의 버전을 선택

```text
node_modules/
├── lodash@4.17.21/          ← 최상위에 hoisting된 버전
├── A/                       ← lodash@4를 import → 최상위 4.17.21 사용
└── B/
    └── node_modules/
        └── lodash@3.10.0/   ← B가 lodash@3을 import → 이 버전 사용
```

- flat hoisting(npm/yarn)에서는 먼저 설치된 버전이 최상위에 올라감 → 나머지는 nested
- pnpm은 symlink 구조라 각 패키지가 **자기 peerDeps에 맞는 버전만** 접근 가능하므로 더 명확
- **번들 크기 외 문제**: React가 두 벌이면 hooks 에러, context 공유 불가 등 **런타임 버그** 발생 (앞서 mocha 사례 참고)
- 싱글턴이어야 하는 패키지가 두 벌 → **디버깅이 극히 어려운** 유형의 버그

---

## Q. `bundleDependencies`는 실제로 사용한 적 있나요?

<style scoped>section { font-size: 20px; }</style>

**사용한 적 없다.** 현대 패키지 개발에서 쓸 일이 거의 없는 레거시 기능

```json
{
  "bundleDependencies": ["lodash", "express"]
}
```

- `npm pack` 시 해당 의존성을 tarball 안에 **함께 포함**시키는 기능
- 용도: 프라이빗 레지스트리 접근 불가 환경, 오프라인 설치, 특정 버전 고정이 필수인 극단적 상황
- npm, yarn, pnpm 모두 lockfile과 `overrides`가 충분히 발전하여 **이 기능 없이도 버전 통제가 가능**
- 패키지 tarball 크기만 키우므로 일반적으로 권장하지 않음

---

## Q. peerDependencies 경고를 모두 해결하는 게 현실적으로 가능한가요?

<style scoped>section { font-size: 18px; }</style>

**가능하다.** 다만 참여자 모두의 책임이 전제되어야 한다

### 원칙 1 — 패키지 개발자의 책임

- peerDeps **버전 범위를 최대한 넓게** 잡아야 함 (`^17 || ^18 || ^19`)
- 새 메이저가 나오면 CI에서 검증 후 **빠르게 범위에 추가**
- 좁은 범위는 소비자에게 불필요한 충돌을 강요하는 것

### 원칙 2 — 소비자(서비스 개발자)의 책임

- peerDeps 경고를 **무시하지 않고** 항상 의구심을 가질 것
- 범위가 좁거나 업데이트가 안 된 패키지에는 **적극적으로 이슈/PR 제기**
- "경고니까 괜찮겠지"가 아니라, "왜 이 경고가 뜨는지" 추적

### 원칙 3 — 그럼에도 해결 안 되면

- `overrides`(npm) / `resolutions`(yarn) / `pnpm.overrides`로 **명시적으로 해소**
- 이는 `--legacy-peer-deps`와 달리 의도가 코드에 남고, 추적 가능

---

## Q. 라이브러리에서 peerDeps를 정의해도 사용처 npm 버전에 따라 강제가 안 되지 않나요?

<style scoped>section { font-size: 20px; }</style>

**맞다. 그리고 그건 라이브러리의 책임 범위가 아니다.**

- npm 3~6은 경고만, npm 7+는 에러, yarn은 또 다름 — 패키지 매니저별 동작 차이표 참고
- `engines`로 npm 버전을 강제하거나 `postinstall`로 체크하는 방법이 있긴 하지만:
  - `engines`는 패키지 매니저 버전을 강제하는 것이지, peerDeps 자체를 강제하는 게 아님
  - `postinstall` 스크립트는 사용자 경험이 나쁘고, 보안상 꺼져 있는 환경도 많음
- **라이브러리가 할 수 있는 것**: peerDeps를 정확하게 선언하고, README에 명시하는 것까지
- **사용처의 패키지 매니저 동작까지 통제하는 것은 범위 밖**

---

## Q. `--legacy-peer-deps`를 어쩔 수 없이 사용한 경험이 있나요?

<style scoped>section { font-size: 22px; }</style>

**없다.**

- 충돌이 발생하면 원인을 추적해서 `overrides`로 해소하거나, 라이브러리에 이슈를 올려 해결해왔음
- `--legacy-peer-deps`는 충돌을 **무시**하는 것이므로, 어떤 버전이 설치될지 비결정적
- 당장 돌아가더라도 **시한폭탄**을 심는 것과 같음

앞서 설명한 대안을 다시 정리하면:

| 상황                          | 해결 방법                              |
| ----------------------------- | -------------------------------------- |
| 라이브러리가 peerDeps 안 올림 | upstream에 이슈/PR                     |
| 기다릴 수 없음                | `overrides`로 명시적 해소              |
| pnpm 환경                     | `peerDependencyRules.allowedVersions`  |
| 근본적으로                    | 충돌 원인을 `npm ls`/`pnpm why`로 추적 |

---

## Q. `@types/react`가 여러 메이저 버전으로 설치되면 문제가 되나요?

<style scoped>section { font-size: 18px; }</style>

**된다.** `@types/react`도 사실상 **싱글턴이 보장되어야 하는 패키지**

```text
node_modules/
├── @types/react@19/          ← 내 프로젝트
└── some-ui-lib/
    └── node_modules/
        └── @types/react@18/  ← 라이브러리가 가져온 다른 버전
```

- **타입 충돌**: `@types/react@18`과 `@types/react@19`의 인터페이스 시그니처가 다름 → duplicate identifier 에러
- **API 유무 불일치**: `@types/react@18`에는 `React.use()`가 없고, `@types/react@19`에는 있음 → 어떤 파일에서 어떤 버전이 잡히느냐에 따라 타입 체크 결과가 달라짐
- **비결정적**: hoisting 결과에 따라 **환경마다 다른 에러**가 발생할 수 있어 디버깅이 극히 어려움

런타임이 아니라 **타입 레벨에서 깨지는 문제**이므로, 앞서 설명한 "`@types/*`를 `dependencies`에 넣지 말 것" 원칙과 직결됨

---

<!-- _class: invert -->

# 감사합니다

@yceffort
