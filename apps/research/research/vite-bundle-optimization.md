---
title: Vite 번들 최적화 실전 가이드
marp: true
paginate: true
theme: default
tags:
  - bundler
  - web-performance
date: 2026-08-05
description: '내 앱의 번들은 무엇으로 결정되는가: 네 가지 질문으로 나눠서 점검하고 고치는 법'
published: true
art:
  undraw: static-assets
---

# Vite 번들 최적화 실전 가이드

내 앱의 번들은 무엇으로 결정되는가

<!-- _class: invert deck-cover -->

@yceffort

---

## 이 글이 답하려는 질문

번들이 크다는 건 알겠는데, **뭘 건드려야 줄어드는가?**

흔한 답("코드 스플리팅 하세요", "트리셰이킹 되게 하세요")은 대부분 순서가 틀렸거나, 그 기법이 실제로 무엇을 줄이는지 오해하고 있다.

이 글은 번들 크기를 **네 개의 파트**로 분해하고, 파트마다 무엇을 점검하고 어떤 규모의 이득을 기대할 수 있는지 정리한다.

수치는 두 개의 Vite 앱 **A**(라우트 23개 게시판형)와 **B**(단일 스크롤 화면)를 `vite build` before/after로 잰 값이다. **절대값이 아니라 "이 기법은 이 정도 규모"라는 감각으로 보면 된다.**

측정 환경: Vite 8 (Rolldown 기반). 이 글의 옵션 이름은 rolldown의 `codeSplitting` rename 이후 기준이라 버전에 민감하다. Q&A에 Vite 7 이하 대응표가 있다.

---

## 번들 크기를 결정하는 네 가지 질문

| Part                     | 질문                       | 주요 수단                          | 단독 SPA라면                |
| ------------------------ | -------------------------- | ---------------------------------- | --------------------------- |
| **1. 무엇이 들어오는가** | 이 모듈이 왜 번들에 있나   | `external`, dedupe, `define`, 타깃 | 부분 해당 (`external` 제외) |
| **2. 얼마나 걷히는가**   | 안 쓰는 JS가 남아 있나     | 트리셰이킹, `sideEffects`          | 해당                        |
| **3. CSS**               | 안 쓰는 스타일이 남아 있나 | subpath import, entry CSS          | 해당                        |
| **4. 언제 받는가**       | 첫 화면에 필요한 것만 받나 | 코드 스플리팅, entry 분리          | 해당                        |

---

## 본론에 들어가기 전에 못 박아둘 것

- **압축(minify·gzip)은 최적화 항목이 아니라 측정의 전제다.** Part 0에서 다룬다
- **네 파트는 서로 독립이 아니다.** 같은 바이트를 겨냥한 기법끼리는 이득이 합산되지 않는다 (Part 5)
- **Part 4는 크기를 줄이지 않는다.** 받는 시점을 옮길 뿐이다. 이걸 크기 최적화로 착각하는 게 가장 흔한 실수다

---

## 어디부터 봐야 하나

```text
0. 측정 환경이 맞는지 확인 (미니파이·gz)  ← 여기서 틀리면 나머지가 전부 무의미
   ↓
1. 안 들어와도 되는 게 있나      (Part 1)  : 코드 한 줄, 효과 큼
   ↓
2. 들어온 것 중 안 쓰는 게 있나  (Part 2)  : 설정 몇 줄, 효과 큼
   ↓
3. 안 쓰는 CSS가 통째로 오나     (Part 3)  : import 경로 변경, 효과 중간
   ↓
4. 받는 시점을 옮길 수 있나      (Part 4)  : 구조 변경, 크기 이득 없음
```

**위에서부터 내려온다.** 4번부터 시작하는 팀이 많은데, 가장 비싸고 크기 이득은 0이다.

---

# Part 0. 측정

무엇을 재고 있는지부터 확인한다

---

## 재기 전 확인 ①: 미니파이된 산출물인가

같은 코드가 **두 배 차이로 측정된다.** 여기서 틀리면 뒤의 모든 비교가 무의미하다.

```bash
head -c 200 dist/assets/index-*.js   # 변수명이 길면 미니파이 안 된 것
wc -l dist/assets/index-*.js         # 수천 줄이면 안 된 것
```

Vite 8의 `build.minify` 기본값은 `'oxc'`다(CSS는 lightningcss). **그냥 두면 mode와 무관하게 항상 켜져 있다.** 그런데도 이 함정에 빠지는 팀이 많다.

---

## 왜 꺼지는가: config가 기본값을 덮는다

```ts
export default defineConfig(({mode}) => {
  const isProd = mode === 'production'
  return {
    build: {
      minify: isProd, // ← 기본값을 mode에 묶었다
      sourcemap: isProd ? 'hidden' : true,
    },
  }
})
```

배포 환경이 하나면 드러나지 않는다. `vite build`의 mode 기본값이 `production`이라 `isProd`가 항상 참이기 때문이다.

---

## mode가 늘어나는 순간 갈라진다

```bash
vite build                      # mode=production → minify O
vite build --mode staging       # mode=staging    → minify X  ⚠️
vite build --mode qa            # mode=qa         → minify X  ⚠️
```

**환경별 API 주소를 바꾸려고 `--mode`를 도입했을 뿐인데 미니파이가 같이 꺼진다.**

---

## 그래서 어떻게 할 것인가

`minify`를 mode에 묶을 이유는 없다. 디버깅이 필요하면 `sourcemap`만 켜면 된다.

```ts
build: {
  // minify는 건드리지 않는다 (기본값 'oxc')
  sourcemap: isProd ? 'hidden' : true,
}
```

두 가지가 따라온다.

- **측정 신뢰성**: 어느 모드로 빌드하든 같은 조건으로 잰다
- **QA 정합성**: QA가 스테이징을 본다면, 프로덕션과 같은 크기의 번들을 받게 된다

참고로 `'hidden'`은 **`.map` 파일을 생성하되 JS 끝의 참조 주석만 뺀 것**이다. 파일은 나오므로, 뒤에서 볼 소스맵 귀속 분석의 재료가 된다.

---

## 재기 전 확인 ②: raw가 아니라 gz로 본다

- **raw**: 디스크에 놓인 파일 크기
- **gz**: gzip 압축 후 크기. 사용자가 실제로 받는 양에 가깝다 (서버·CDN이 압축해서 보내므로)

의존성 중복 버전을 정리한 실측이다.

```text
before  패키지 A ×4 · B ×3 · C ×2      합계 232.0 kB raw
after                                  합계  79.8 kB raw   (−152.2 kB)
gz 이득                                −2.25 kB
```

**raw로 세운 기대치는 거의 항상 과대평가다.** 이 사례는 raw Δ의 1/60만 전송량에 반영됐다.

---

## 잠깐: gzip은 어떻게 압축하나

핵심은 "**이미 나온 바이트열이 또 나오면, 뒤로 가리키는 참조로 치환**한다"는 것이다.

```text
function add(a,b){return a+b}     ← 원본 그대로 기록
...
function add(a,b){return a+b}     ← "3,000바이트 앞의 30바이트와 동일"이라는 참조 몇 바이트로 치환
```

단, 뒤돌아볼 수 있는 범위(슬라이딩 윈도우)가 **직전 32KB**로 제한된다.

- 반복이 32KB 안에서 일어나면 → 참조 몇 바이트로 줄어든다
- 두 사본 사이가 32KB를 넘으면 → 앞의 사본은 이미 윈도우 밖이라, 두 번째 사본도 처음 보는 데이터처럼 통째로 압축한다

"사본이 멀리 떨어져 있다"는 건 **한 파일 안에서 두 사본 사이의 바이트 거리가 32KB를 넘는다**는 뜻이다.

---

## 단, 두 가지로 일반화하면 안 된다

**① "중복은 gzip이 알아서 지워주니 dedupe는 무의미하다" (✗)**

방금 봤듯 gzip은 직전 32KB 안의 반복만 참조로 줄일 수 있다. 미니파이된 번들에서 중복 패키지 두 벌은 대개 그보다 멀리 배치되므로, raw Δ와 gz Δ의 비율은 사본의 배치에 따라 케이스마다 다르다. 그래서 결론은 "dedupe 하지 마라"가 아니라 "**기대치를 gz로 재고 나서 판단하라**"다.

**② 크기가 0이어도 정리해야 하는 중복이 있다 (✗)**

`react`처럼 **상태를 가진 패키지의 중복은 크기가 아니라 정합성 문제**다. 두 벌 로드되면 훅이 깨진다. 이건 gz 이득과 무관하게 정리 대상이다. Part 1의 external 절에서 다시 나온다.

---

## 재기 전 확인 ③: "왜 들어왔는지"를 볼 수 있는가

`rollup-plugin-visualizer`류의 트리맵은 기본이 raw 면적이라 위 함정을 재현한다. 그리고 더 큰 한계가 있다. **모듈이 왜 들어왔는지 알려주지 않는다.**

크기를 줄이려면 "이게 왜 있지?"에 답할 수 있어야 한다. 소스맵이 그 출발점이다.

```js
// 이 청크에 어느 패키지의 모듈이 몇 개 들어갔는지 1차 스캔
const map = JSON.parse(fs.readFileSync('dist/chunks/foo.js.map'))
const byPkg = {}
for (const src of map.sources) {
  const pkg = src.includes('node_modules')
    ? src.split('node_modules/').pop().split('/').slice(0, 2).join('/')
    : '(app)'
  byPkg[pkg] = (byPkg[pkg] ?? 0) + 1
}
```

---

## 주의: 이 스캔은 모듈 개수지, 바이트가 아니다

바이트 단위 귀속은 소스맵 `mappings` 디코딩이 필요하고, `source-map-explorer`·`sonda` 같은 도구가 해준다.

**앞 스크립트로 수상한 청크를 찾고, 도구로 바이트를 확인하는 순서다.**

---

## 측정 루프

```bash
# 1. 깨끗한 워크트리에서 베이스라인
git stash && pnpm install --frozen-lockfile
vite build --mode production
cp -r dist /tmp/baseline

# 2. 레버 하나만 적용
git stash pop
vite build --mode production

# 3. diff
node scripts/diff-dist.mjs /tmp/baseline dist
```

**한 번에 레버 하나씩.** 두 개를 같이 적용하면 어느 쪽이 효과를 냈는지 모른다. 합산 가능 여부도 따로 확인해야 한다. **두 기법이 같은 바이트를 겨냥하면 합산되지 않는다.**

---

## diff 스크립트 ①: 해시 정규화와 스캔

산출물 파일명에는 content hash가 들어가서(`index-Ab3xK9.js`) **이름 그대로는 before/after가 매칭되지 않는다.** 해시를 정규화하고 디렉토리를 스캔한다.

```js
import {gzipSync} from 'node:zlib'
import {readFileSync, readdirSync, statSync} from 'node:fs'

const norm = (f) => f.replace(/-[\w-]{8,}(?=\.\w+$)/, '') // index-Ab3xK9.js → index.js

const scan = (dir) => {
  const out = {}
  for (const f of readdirSync(dir, {recursive: true})) {
    if (!statSync(`${dir}/${f}`).isFile()) continue
    const buf = readFileSync(`${dir}/${f}`)
    out[norm(String(f))] = [buf.length / 1024, gzipSync(buf).length / 1024]
  }
  return out
}
```

---

## diff 스크립트 ②: 비교

한쪽에만 있는 파일(신규·삭제)은 0으로 친다.

```js
const [a, b] = [scan(before), scan(after)]
for (const f of new Set([...Object.keys(a), ...Object.keys(b)])) {
  const [rawA = 0, gzA = 0] = a[f] ?? []
  const [rawB = 0, gzB = 0] = b[f] ?? []
  console.log(f, (rawB - rawA).toFixed(2), (gzB - gzA).toFixed(2))
}
```

Vite 콘솔의 gz 값과 `zlib` 기본 레벨 값은 1~2KB 다르다. **절대값이 아니라 Δ로 본다.**

---

# Part 1. 무엇이 들어오는가

가장 싸고 가장 확실한 지점

---

## 번들에 안 들어와도 되는 것들

같은 코드를 두 번 넣고 있지 않은지 본다. 네 가지 패턴이 흔하다.

| 패턴                  | 증상                              | 수단                     |
| --------------------- | --------------------------------- | ------------------------ |
| 런타임이 이미 제공    | CDN·호스트 앱이 주는 걸 또 번들   | `external`               |
| 같은 패키지 여러 버전 | `react@18`과 `react@19`가 공존    | `overrides` / `dedupe`   |
| 개발 전용 코드        | mock, 디버그 패널이 프로덕션에    | 조건부 import + `define` |
| 타깃·폴리필 과잉      | 안 쓰는 legacy 번들·폴리필을 서빙 | browserslist 정책        |

**첫 번째가 압도적으로 이득이 크다.** 코드 변경량은 배열에 문자열 하나다. 다만 성립 조건이 있어서, 단독 SPA라면 두 번째부터 보면 된다.

---

## `external`: 런타임이 주는 건 빼기

```ts
build: {
  rolldownOptions: {
    external: ['react', 'react-dom', 'react/jsx-runtime'],
  },
}
```

번들에서 빠지고, 런타임이 제공하는 인스턴스를 쓴다. 이게 성립하려면 **런타임에 그 모듈이 실제로 있어야 한다.**

---

## external이 성립하는 상황

- **마이크로 프론트엔드(MFE)**: 호스트 셸 하나가 여러 리모트 앱을 조립하는 구조. 호스트가 import map으로 공통 의존성을 싱글톤 제공
- **라이브러리 빌드**: `peerDependencies`(소비자가 직접 설치해서 제공하는 의존성)는 소비자가 갖고 있음
- **CDN import map**: `<script type="importmap">`으로 bare import(`'react'`)를 URL에 직접 매핑

일반 SPA 단독 배포라면 이 절은 대부분 해당이 없다. 뒤의 dedupe·개발 전용 코드부터 보면 된다.

---

## 실측: 문자열 한 줄이 얼마나 되나

MFE 환경에서 HTTP 클라이언트 패키지 하나를 `external`에서 뺐다 넣었다 하며 측정했다.

```text
external에 있을 때   shared.js   43.08 kB gz
빼면                shared.js   76.58 kB gz     (+33.50)
```

**단일 기법 중 가성비 최고다.** 다만 공짜는 아니다.

---

## external 후보를 고르는 기준은 크기가 아니다

앱 B에서 후보를 하나씩 `external`로 옮겨가며 측정했다.

| 대상                   |       Δ gz | 조율 범위                          |
| ---------------------- | ---------: | ---------------------------------- |
| 디자인시스템 JS 전체   | **−31.81** | 런타임 + 전 소비자 버전 단일화     |
| HTTP 클라이언트        |     −33.50 | config 1줄 (런타임이 이미 제공 중) |
| 데이터 페칭 라이브러리 |     −11.83 | 런타임 import map에 신규 노출      |
| 브릿지 유틸            |      −4.89 | config 1줄                         |

**크기 순으로 착수하면 안 된다.** 디자인시스템(−31.81)은 소비자 전체가 같은 버전으로 정렬돼야 해서 몇 개 팀이 붙어야 하고, HTTP 클라이언트(−33.50)는 이미 런타임이 서빙 중이라 문자열 하나면 끝난다.

> **`Δ ÷ 조율 비용`으로 정렬한다.** config 1줄짜리부터 전부 털고 나서 조율이 필요한 걸 본다.

---

## `external`의 함정: 조용한 버전 다운그레이드

**런타임이 제공하는 버전과 로컬 lockfile 버전이 semver를 만족하는지 확인해야 한다.**

```text
런타임 제공  : @platform/http@0.19.0
로컬 lockfile: 0.100.2
```

이 조합에서 `external`로 빼면 앱은 **0.19.0을 쓰게 된다.** 빌드는 통과하고, 런타임에 없는 API를 호출하다 터진다.

---

## external 체크리스트

1. 런타임이 실제 제공하는 버전 확인 (기록 말고 실물)
2. 로컬 버전과 semver 정합성 확인
3. **`external`은 빌드 옵션이다.** dev 서버는 번들링을 하지 않으므로 `build.rolldownOptions.external`이 적용되지 않는다. dev에서 멀쩡했다는 건 아무것도 검증하지 않은 것이다
4. 그러므로 반드시 **빌드 산출물로** 배포 전 런타임 스모크

---

## 크기가 아니라 정합성이 목적인 경우

`external`을 크기 때문에만 쓰는 게 아니다.

```text
리모트 6개가 각자 WebSocket 클라이언트 번들 → 연결 6개
external + 런타임 싱글톤                    → 연결 1개
```

React가 `external` 1순위인 것도 같은 이유다. **두 벌 로드되면 훅이 깨진다.** 상태를 가진 라이브러리(라우터, 쿼리 클라이언트, 스토어)는 크기와 무관하게 싱글톤이어야 한다.

> 크기 이득은 부수 효과로 따라온다. Part 0에서 본 "gz가 0이어도 정리해야 하는 중복"이 바로 이것이다.

---

## 중복 버전은 gz로 먼저 재본다

`pnpm why`로 같은 패키지의 여러 버전이 들어오는지 본다.

```bash
pnpm why react
pnpm dedupe --check
```

정리하면 raw는 확실히 준다. 하지만 Part 0에서 봤듯 **gz 이득은 그 1/60 수준일 수 있다.**

모노레포에서 `overrides`로 강제 정렬하면 **전체 패키지의 의존성 해석이 바뀐다.** 전 패키지 typecheck·build를 다시 돌려야 하고 런타임 회귀 위험을 떠안는다.

**먼저 gz로 재고, 그 값이 비용을 정당화하는지 보고 결정한다.** 단, 상태를 가진 패키지(react, 스토어류)의 중복은 gz와 무관하게 정리한다.

---

## 개발 전용 코드: 조건부 동적 import + 상수 치환

mock 서버, 디버그 패널, 데모 데이터가 프로덕션 번들에 들어 있는 경우가 있다.

```ts
// entry
if (import.meta.env.DEV) {
  const {worker} = await import('./mocks/browser')
  await worker.start()
}
```

- `import.meta.env.DEV`는 빌드 시 `false` 리터럴로 치환된다 → 조건문째 데드 코드로 제거되고, **동적 import라 청크도 아예 안 생긴다**
- 최상단에 정적 `import {worker} from './mocks/browser'`로 올려두면, 조건문이 지워져도 모듈 평가가 남아 번들에 포함될 수 있다. **반드시 조건부 동적 import로**
- 커스텀 플래그가 필요하면 `define: {__DEBUG_PANEL__: 'false'}`로 같은 효과를 낸다

---

## 타깃·폴리필: 조용히 두 배가 되는 지점

- **`@vitejs/plugin-legacy`**: 번들을 modern/legacy 두 벌로 만들고 core-js 폴리필까지 얹는다. 지원 브라우저 정책이 정말 legacy 빌드를 요구하는지부터 확인한다. 정책에 없는 브라우저를 위해 전 사용자가 두 벌 분량의 빌드를 유지하는 경우가 있다
- **`build.target` 하향**: 문법 다운레벨링으로 코드가 팽창한다. 반대로 상향의 이득은 실측 −0.48 kB에 그쳤다 (Part 5). **어느 방향이든 지원 브라우저 정책 결정이 먼저다**
- **수동 폴리필 import**: 타깃 브라우저가 이미 지원하는 폴리필을 들고 있지 않은지 본다

---

# Part 2. 얼마나 걷히는가

트리셰이킹이 왜 안 되는가

---

## 먼저: 부수효과란 무엇인가

**모듈이 "평가되는 것만으로" 프로그램 상태를 바꾸는 것.**

```js
// 이 파일은 import 되는 순간 바깥 세계를 바꾼다
window.__APP__ = {} // 전역 오염
Array.prototype.at ??= function () {} // 폴리필
registry.register('button', Button) // 전역 레지스트리 등록
import './styles.css' // 스타일시트 주입
```

반대로 이런 파일은 부수효과가 없다.

```js
// export만 있고, 평가해도 바깥에 아무 일도 안 일어난다
export function Button(props) { ... }
```

**부수효과가 없는 모듈은 export를 아무도 안 쓰면 통째로 지워도 된다.** 있는 모듈은 지우면 안 된다. 폴리필이 사라지고 스타일이 없어진다.

---

## `sideEffects`는 그걸 알려주는 필드

번들러는 정적 분석만으로 부수효과 유무를 100% 판정할 수 없다. 그래서 패키지가 직접 선언한다.

```jsonc
// package.json
{
  "sideEffects": false, // 모든 모듈을 안전하게 지워도 됨
  "sideEffects": ["**/*.css"], // CSS만 부수효과 있음
  "sideEffects": ["./src/polyfill.js"],
}
```

webpack이 도입했고, esbuild·Rolldown은 네이티브로, Rollup은 `@rollup/plugin-node-resolve` 계층에서 존중한다. Vite를 쓴다면 버전과 무관하게 동작한다고 보면 된다.

**문제는 기본값이다. 필드가 없으면 `true`로 간주한다.** "전부 부수효과 있음, 아무것도 지우지 마."

---

## 배럴에서 왜 재앙이 되는가

배럴(barrel)은 re-export만 하는 `index.js`다.

```js
// @ds/ui/index.js
export {Button} from './button'
export {Badge} from './badge'
// ... 컴포넌트 100개
```

```ts
import {Button} from '@ds/ui' // 하나만 썼는데
```

---

## 하나만 썼는데 100개가 남는 과정

1. `Button`을 쓰려면 번들러는 배럴 `index.js`를 평가해야 한다
2. 배럴은 100개 하위 모듈을 참조한다
3. `sideEffects` 선언이 없으니 **하위 모듈 100개가 전부 "부수효과 있음"**
4. 부수효과 있는 모듈은 **export를 안 써도 평가는 해야 한다** → 드롭 불가
5. 100개가 전부 번들에 남는다

**배럴 자체에 부수효과가 있는 게 아니다.** 배럴이 끌어오는 모듈들이 "지워도 되는지 모르겠다"로 표시돼 있는 것뿐이다.

---

## 내 앱에 이 문제가 있는지 확인하기

```bash
# 1. 배럴로 import하는 무거운 패키지의 sideEffects 선언 확인
cat node_modules/@ds/ui/package.json | grep -A3 sideEffects
# → 아무것도 안 나오면 = 전부 부수효과 있음으로 취급됨

# 2. 그 패키지가 번들에서 차지하는 비중 확인 (소스맵 귀속, Part 0)
# 3. 배럴 import 지점 세기
grep -roh "from '@ds/ui'" src | wc -l
```

선언이 없고 + 비중이 크고 + 쓰는 컴포넌트가 일부라면, 아래 두 수단이 후보다.

---

## 수단 ①: `moduleSideEffects`로 로컬 오버라이드

업스트림이 선언 안 해주면 우리가 대신 선언한다.

```ts
build: {
  rolldownOptions: {
    treeshake: {
      moduleSideEffects: [
        { test: /@ds[\\/](?:ui|headless|icon)[\\/].*\.css$/, sideEffects: true },
        { test: /@ds[\\/](?:ui|headless|icon)[\\/]/,          sideEffects: false },
      ],
    },
  },
}
```

**CSS만은 부수효과로 남겨야 한다.** `import '@ds/ui/css'` 같은 bare import가 드롭되면 스타일이 통째로 사라진다.

실측: 이 오버라이드를 끄면 **+38.65 kB gz.**

---

## 오버라이드 전에 확인할 것

- **그 패키지 JS에 진짜 부수효과가 없는지 먼저 확인한다.** bare import와 JS 내 CSS import를 grep으로 세어보는 정도면 충분하다
- 함수형 `moduleSideEffects: (id) => ...`도 동작하지만 모듈마다 Rust→JS 왕복이 생긴다. rolldown 타입 정의가 룰 배열을 권장한다

---

## 수단 ②: subpath import, 되면 가장 깔끔

```ts
// before
import {Button} from '@ds/ui'

// after: 배럴을 거치지 않음
import {Button} from '@ds/ui/button'
```

패키지가 subpath `exports`를 제대로 선언했다면 이게 정공법이다. 설정 없이 import 경로만 바꾸면 된다.

**다만 안 되는 패키지가 있다.** `exports`의 subpath `types` 조건이 per-subpath 선언이 아니라 배럴 `index.d.mts`를 가리키는데 런타임 모듈은 `default`로만 export하는 경우:

| 작성                                     | 결과                                     |
| ---------------------------------------- | ---------------------------------------- |
| `import Button from '@ds/ui/button'`     | `TS2613: has no default export`          |
| `import { Button } from '@ds/ui/button'` | tsc 통과 → **런타임 `undefined` 크래시** |

두 번째가 무섭다. **타입 체크를 통과하고 런타임에 터진다.** 이럴 땐 수단 ①로 우회한다.

---

## 안 되는 패키지는 이렇게 생겼다

```jsonc
// @ds/ui/package.json
"exports": {
  "./button": {
    "types": "./dist/index.d.mts", // ← 타입은 배럴 하나로 퉁침
    "import": "./dist/button/index.mjs"
  }
}
```

```ts
// dist/index.d.mts: tsc가 보는 파일. named로 선언돼 있다
export declare const Button: ButtonComponent

// dist/button/index.mjs: 런타임이 실행하는 파일. default뿐이다
export default Button
```

`import {Button} from '@ds/ui/button'`을 쓰면 **tsc는 위 파일로 통과시키고, 런타임은 아래 파일을 실행한다.** 타입과 값이 서로 다른 파일을 보고 있는 것이다. 런타임 모듈에 named `Button`이 없으니 `undefined`가 된다.

---

## 트리셰이킹이 원천적으로 안 되는 패턴: 동적 참조

```tsx
<Icon icon="arrow-right" />
// 내부: registry[icon]
```

`registry`가 120개 아이콘을 전부 참조하므로 **하나만 써도 120개가 다 번들된다.** 문자열은 정적 분석이 불가능하다.

같은 모양의 다른 사례들:

```js
require(`./locales/${lang}.json`) // 전 언어 번들
components[type] // 전 컴포넌트 번들
await import(`./icons/${name}.svg`) // 전 아이콘 청크 생성
```

**"키로 조회하는 레지스트리"를 보면 트리셰이킹을 의심한다.**

---

## 해결: 정적 참조로 바꾸는 얇은 어댑터

```tsx
import { Common, Navigation } from '@ds/icon'

export const Icon = ({ icon: Svg, size, color }) => (
  <Svg width={size} height={size} className={`ds-color-${color}`} />
)

// 사용처: 문자열이 아니라 컴포넌트를 넘긴다
<Icon icon={Navigation.ArrowRight} size={16} />
```

실측: 아이콘 120개 → **실제 번들 13개.**

---

## 왜 이 어댑터는 걷히는가

`@ds/icon`이 `export * as Navigation`처럼 네임스페이스로 내보내고, 번들러는 네임스페이스에 대한 **정적 property access**(`Navigation.ArrowRight`)를 추적할 수 있다. 패키지 내부가 런타임 객체 리터럴 레지스트리(`const Navigation = {ArrowRight, ...}`)라면 같은 코드를 써도 안 걷힌다. 따라 하기 전에 패키지의 export 형태부터 확인한다.

> 이 수단은 수단 ①과 짝이다. 동적 참조를 없애도 `sideEffects` 선언이 없으면 안 걷히고, 오버라이드만 있고 `registry[icon]`이 남아 있어도 안 걷힌다. **그래서 개별 기여도를 분리 측정하기 어렵다.** 위의 `+38.65 kB`에도 아이콘 몫이 섞여 있다.

---

## 걷어내기 전에: 낯선 모듈 = 죽은 코드가 아니다

번들을 뜯어보면 이런 게 나온다.

```text
디자인시스템 청크 구성
  embla-carousel + wheel-gestures   71.9 KB   ← 우리 화면에 캐러셀 없는데?
```

"미사용 Carousel이 끌고 온 죽은 코드"라고 결론 내리기 쉽다. 그렇게 판단하고 −29 KB를 기대했다가, 빌드해보니 **−7 KB**였다.

---

## 누가 끌고 왔는지 추적한다

```bash
grep -rl "embla" node_modules/@ds/headless/dist/
# → dist/tab-group/TabGroup.mjs
```

```text
TabGroup          ← 앱이 실제로 쓰는 컴포넌트
  └ Carousel      ← TabGroup 내부 구현
      └ embla-carousel
```

**탭 전환을 캐러셀로 구현한 컴포넌트**를 쓰고 있었다. 정당한 의존성이라 트리셰이킹으로 걷힐 리가 없다.

> 걷어낼 계획을 세우기 전에 `grep -rl`로 누가 끌고 왔는지 확인한다.

---

## 잠깐, 지금까지의 수단은 전부 워크어라운드다

이 파트에서 쓴 기법들을 다시 보자.

| 우리가 한 것                        | 원래 누가 했어야 하나                    |
| ----------------------------------- | ---------------------------------------- |
| `moduleSideEffects` 로컬 오버라이드 | 패키지가 `sideEffects` 선언              |
| subpath 스왑 시도 → 실패            | 패키지가 `exports.types`를 per-subpath로 |
| 아이콘 어댑터 작성                  | 패키지가 컴포넌트를 named export         |
| CSS `dist/` 사적 경로 참조          | 패키지가 컴포넌트별 CSS subpath 제공     |

**전부 소비자가 패키지의 배포 실수를 뒤집어쓴 것이다.** 워크어라운드는 업스트림 버전업마다 깨질 수 있다.

내부 디자인시스템을 만드는 쪽이라면, 소비자가 이런 짓을 하게 두지 않을 수 있다.

---

## 트리셰이킹 가능한 패키지의 조건

```jsonc
{
  // ① CSS만 부수효과로 선언: 가장 중요
  "sideEffects": ["**/*.css"],
  "exports": {
    ".": {
      "types": "./dist/index.d.ts",
      "import": "./dist/index.js",
    },
    "./button": {
      // ② 배럴이 아니라 per-subpath 타입
      "types": "./dist/button/index.d.ts",
      "import": "./dist/button/index.js",
    },
    // ③ CSS도 컴포넌트 단위로 노출
    "./button/css": "./dist/button/index.css",
    "./css": "./dist/index.css",
  },
}
```

---

## 조건 ④: 런타임 export는 named로

subpath가 `default`만 내보내면 타입은 배럴에서 오고 값은 default라 어긋난다. 이게 앞에서 본 `TS2613` / 런타임 `undefined`의 정체다.

```ts
// ✗ subpath가 default만 내보냄: 배럴 타입과 어긋난다
export default Button

// ✓ named export
export {Button}
```

---

## 아이콘·아이콘류는 특히 조심

```tsx
// ✗ 문자열 키: 정적 분석 불가, 전체 레지스트리가 번들에 남음
;<Icon name="arrow-right" />

// ✓ 컴포넌트 export: 쓴 것만 남음
import {ArrowRight} from '@ds/icon'
;<ArrowRight size={16} />
```

문자열 API가 편해 보이지만, **소비자 번들에 아이콘 전체를 강제로 넣는 대가**를 치른다. 편의를 유지하려면 codegen으로 named export를 함께 내보내면 된다.

같은 원리가 로케일·차트 타입·컴포넌트 팩토리 전반에 적용된다.

---

## 왜 사내 디자인시스템에서 이게 자주 깨지나

- **소비자가 하나일 때는 안 드러난다.** 첫 앱이 컴포넌트 대부분을 쓰면 죽은 코드가 없다
- **번들러 라이브러리 모드 기본값**에는 `sideEffects` 선언이 없다. 명시적으로 넣어야 한다
- **타입 생성 도구가 배럴 `.d.ts` 하나만 뱉는 경우**가 많다. 런타임은 subpath로 나뉘는데 타입은 안 나뉜다
- 소비자가 늘어난 뒤에는 **breaking change 없이 고치기 어렵다**

> 릴리스 체크에 `sideEffects` 선언 여부와 subpath 타입 정합성을 넣어두면 대부분 예방된다.

---

# Part 3. CSS

JS와 다른 파이프라인, 다른 함정

---

## CSS에는 트리셰이킹이 없다

```scss
@use '@ds/ui/css'; // dist/index.css (492 KB raw)
```

번들러는 어떤 셀렉터가 실제로 쓰이는지 알 수 없다. 클래스명이 런타임에 조합될 수도 있기 때문이다. **그래서 CSS 배럴은 통째로 남는다.**

실제로 쓰는 컴포넌트가 15~24개인데 안 쓰는 `app-bar`(75KB), `textfield`, `chip`, `dialog`, `snackbar` … **~208 KB가 죽은 CSS**였다.

Part 2의 배럴 문제와 같은 모양이지만, JS와 달리 `sideEffects`로 풀 수 없어서 파트를 따로 뒀다.

---

## 문제: 공개된 CSS 통로가 배럴 하나뿐

`exports`는 패키지가 "이 경로만 공개한다"고 선언하는 **화이트리스트**다. 선언되지 않은 경로는 import가 차단된다.

```jsonc
// @ds/ui/package.json
"exports": {
  "./css": "./dist/index.css", // ← 공개된 CSS 경로는 배럴 하나뿐
  // "./button/css" 같은 컴포넌트 단위 경로는 선언돼 있지 않다
}
```

컴포넌트별 CSS 파일은 `dist/` 안에 실제로 존재한다. 그런데 공식 통로가 없으니, **배럴 492KB를 통째로 쓰는 것 말고는 방법이 없어 보이는 상황**이다.

---

## 우회로: sass `@use`는 `exports`를 안 거친다

JS의 import는 번들러가 `exports` 화이트리스트로 검사한다. 그런데 **sass 컴파일러는 자체 resolver로 node_modules 안의 실제 파일을 직접 찾는다.** 화이트리스트 검사가 없다.

```scss
// 안 됨: exports에 선언된 공개 경로가 아니다
@use '@ds/ui/button/css';

// 됨: 디스크에 실제로 있는 파일 경로를 그대로 적으면 찾아간다
@use '@ds/ui/dist/button/index.css';
@use '@ds/ui/dist/badge/index.css';
```

실측:

```text
CSS   763.74 → 460.51 kB raw   |   49.36 → 30.37 kB gz   (−18.99)
JS    바이트 완전 동일
```

**같은 패키지인데 JS에서 막힌 게 CSS에서는 통한다.**

---

## 대신 구조적 리스크를 안는다

방금 쓴 `dist/<component>/index.css`는 패키지가 공개를 약속한 경로가 아니라 **내부 폴더 구조를 직접 파고든 것**이다. 공개 API가 아니라 구현 디테일에 의존하는 셈이다.

그래서 이런 일이 생길 수 있다.

- 업스트림이 dist 폴더 구조를 바꾸면 (파일명 하나만 바뀌어도) 빌드가 깨진다
- 새 컴포넌트를 쓰기 시작할 때 `@use` 한 줄을 잊으면, 그 컴포넌트만 스타일이 빠진 채 배포된다
- 그래서 버전업 때마다 이 경로들이 살아있는지 확인해야 한다

**−19 KB gz와 이 리스크를 저울질하는 건 팀의 판단이다.** 모르고 넣는 것과 알고 넣는 건 다르다.

---

## 동적 마운트 앱이라면: CSS를 entry에서 import

```text
entry(index.ts) → 동적 import → App.tsx → import './app.css'
```

이 구조에서는 CSS가 **entry가 아니라 동적 청크에 딸린다.** 마운트/언마운트를 직접 제어하는 앱(MFE 리모트, 위젯)에서는 스타일이 안 걷히거나 첫 페인트에 스타일이 없는 증상이 나온다.

```ts
// entry에서 직접 import
import './styles/global.scss'
export {mount, unmount} from './app'
```

일반 SPA는 Vite가 `index.html`에 `<link>`를 넣어주므로 해당 없다.

---

## 참고: entry가 여러 개면 CSS도 갈린다

```text
index.js  / index.css    ← 메인 화면
detail.js / detail.css   ← 상세 화면
```

`detail.css`는 메인 화면 첫 페인트에 포함되지 않는다. **라우트 lazy 없이 entry 분리만으로 얻은 결과다.**

Part 4로 이어지는 지점이다.

---

# Part 4. 언제 받는가

크기는 안 줄어든다

---

## 먼저 못 박기: 스플리팅은 크기 기법이 아니다

수동 청크 그룹을 **꺼보고** 재봤다.

```text
켬:  index 33.04 + ds-ui 31.87 + shared 43.08 + runtime 0.41 = 108.40 gz
끔:  index 57.60 +                shared 50.58              = 108.18 gz
```

**전송량은 사실상 동일하다 (−0.22 KB).** 당연하다. 같은 모듈을 어느 파일에 담느냐의 문제이지 모듈이 없어지는 게 아니다.

그럼 왜 하는가. 두 가지다.

- **첫 화면에 필요 없는 걸 나중에 받기** (초기 로드 감소)
- **배포 간 캐시 분리** (앱 코드만 바뀌면 vendor는 재다운로드 안 함)

**KB 절감으로 계산하면 안 된다.**

---

## 그런데 잘못 나누면 파일이 폭발한다

같은 `vite.config` 베이스를 공유하는 두 앱을 비교했다.

|                 |      앱 A |    앱 B |
| --------------- | --------: | ------: |
| JS 파일         | **164개** | **5개** |
| CSS 파일        |      44개 |     3개 |
| `React.lazy`    |      23개 |     0개 |
| 동적 `import()` |      27개 |     5개 |

A는 라우트를 전부 `React.lazy`로 나눴고, B는 라우트 2개를 빌드 타임 entry로 나눴다.

---

## 왜 164개가 되는가

Rolldown(과 Rollup)의 기본 청킹은 **"동일한 동적 진입점 집합에서 참조되는 모듈"을 한 청크로 묶는다.**

```text
라우트 A만 쓰는 모듈       → 청크 1
라우트 B만 쓰는 모듈       → 청크 2
라우트 A·B가 공유하는 모듈 → 청크 3
라우트 A·C가 공유하는 모듈 → 청크 4
...
```

동적 진입점이 27개면 그 **공유 조합**마다 청크가 생긴다. 앱 A에서 2곳 이상이 공유하는 청크가 **134개** 나왔다.

**`lazy`를 하나 추가할 때마다 조합의 밑이 하나 늘어난다.**

---

## 파편의 정체: vendor가 아니다

앱 A의 청크 중 소스맵으로 귀속 가능한 161개를 분류했다.

```text
node_modules 전용 : 50개 / 2,792 KB
앱 소스 전용      : 93개 /   599 KB   ← 평균 6.4 KB
혼합             : 18개 / 3,277 KB
```

흔히 "vendor를 안 묶어서 그렇다"고 진단하지만, 이 앱에서는 이미 잘 묶여 있었다.

```text
디자인시스템 ui : 2개 청크에만 분포
아이콘         : 1개 청크
```

**개수의 주범은 6KB짜리 앱 소스 조합 청크 93개다.**

---

## 파편의 진짜 비용

```text
초기(entry 정적)     13개 파일 / 1,480 KB
─────────────────────────────────────────
목록 화면 진입      +79개 파일 / 3,164 KB
게시글 상세         +66개 파일 / 3,075 KB
```

HTTP/2 멀티플렉싱이 있어도 공짜가 아니다. 정확히 하자.

- Vite는 동적 청크의 **정적 의존 청크 목록을 빌드 타임에 알고**, 프리로드 헬퍼로 병렬 fetch한다. "받아봐야 다음 파일을 안다"는 식의 워터폴은 생각보다 얕다
- 그래도 남는 비용: **요청 시작 자체가 클릭 이후**라는 것, **동적 import가 중첩된 체인**은 병렬화가 안 된다는 것, 그리고 요청 수십 개 분량의 스케줄링·캐시 조회 오버헤드

모바일 네트워크에서는 이 남은 왕복이 그대로 체감이 된다.

---

## "페이지마다 lazy"가 왜 안티패턴인가

`React.lazy` 자체는 좋은 도구다. 문제는 **라우트 개수만큼 기계적으로 거는 것**이다.

**① 요청이 사용자 클릭 이후에 시작된다**

```text
정적 import : 앱 로드 시점에 이미 받아둠     → 클릭 즉시 렌더
lazy        : 클릭 → 청크 요청(정적 의존은 병렬) → 렌더
```

라우트 전환마다 최소 한 번의 네트워크 왕복이 낀다. 첫 로드를 줄인 만큼 **전환이 느려진다.**

**② 조합 폭발**

앞에서 본 그대로다. lazy를 하나 추가할 때마다 공유 조합의 밑이 하나 늘어난다.

---

## ③ 여러 화면을 도는 사용자는 손해를 본다

라우트 A와 B가 공통 모듈을 쓴다고 하자.

```text
안 나눴을 때:  A+B 한 청크        → 요청 1개
나눴을 때  :  A청크 + B청크 + 공유청크 → 요청 3개
```

**총 바이트는 같은데 요청만 늘었다.** 게시판 앱처럼 사용자가 목록↔상세↔작성을 오가는 구조라면 거의 모든 청크를 결국 다 받는다.

**④ 첫 화면을 lazy로 하면 순손해**

랜딩 라우트를 lazy로 감싸면 `entry → route chunk` 2-hop이 무조건 발생한다. 첫 화면은 정적으로 두는 게 맞다.

**⑤ Suspense fallback 깜빡임**

전환마다 스켈레톤이 스쳐 지나간다. 100ms짜리 로딩은 UX를 개선하지 않는다.

---

## 그럼 lazy는 언제 쓰나

**"이 코드를 안 받고도 대부분의 사용자가 목적을 달성하는가?"** 에 예라고 답할 수 있을 때.

| 대상               | lazy | 이유                           |
| ------------------ | ---- | ------------------------------ |
| 리치 텍스트 에디터 | ✅   | 글 쓰는 사용자만 필요, 수백 KB |
| 차트 라이브러리    | ✅   | 특정 탭에서만                  |
| PDF 뷰어           | ✅   | 첨부 열 때만                   |
| 결제 모듈          | ✅   | 구매 플로우에서만              |
| 관리자 전용 화면   | ✅   | 대부분의 사용자가 안 감        |

---

## 반대로 lazy가 손해인 곳

| 대상               | lazy | 이유                    |
| ------------------ | ---- | ----------------------- |
| 목록 / 상세 / 작성 | ❌   | 앱의 주 동선            |
| 첫 진입 화면       | ❌   | 무조건 2-hop 손해       |
| 상시 노출 섹션     | ❌   | 청크만 늘고 전송량 동일 |

**"무겁고, 일부만 쓰고, 지연돼도 되는 것."** 셋 다 맞을 때만.

---

## 순수 SPA라면: 라우트가 아니라 묶음 단위로

entry를 나눌 수 없는 구조에서는 lazy를 쓰되 **경계를 굵게** 잡는다.

```ts
// ✗ 라우트마다: 진입점 23개
const BoardList = lazy(() => import('./pages/board/list'))
const BoardDetail = lazy(() => import('./pages/board/detail'))
const BoardWrite = lazy(() => import('./pages/board/write'))
// ... 20개 더

// ✓ 기능 묶음마다: 진입점 3개
const BoardRoutes = lazy(() => import('./features/board/routes'))
const ProfileRoutes = lazy(() => import('./features/profile/routes'))
const AdminRoutes = lazy(() => import('./features/admin/routes'))
```

게시판에 들어온 사용자는 목록·상세·작성을 다 볼 가능성이 높다. **함께 쓰이는 것은 함께 받는 게 낫다.**

조합의 밑이 23에서 3으로 줄고, 전환 시 네트워크 왕복도 사라진다.

---

## 남은 전환 지연은 prefetch로 상쇄한다

경계를 굵게 잡았어도 lazy 경계를 넘는 첫 전환에는 왕복 한 번이 남는다. 이건 숨길 수 있다.

```tsx
const preloadBoard = () => import('./features/board/routes')

<Link to="/board" onMouseEnter={preloadBoard} onFocus={preloadBoard}>
  게시판
</Link>
```

- `import()`는 같은 청크를 두 번 받지 않는다. **hover 시점에 미리 불러두면 클릭 시점엔 이미 캐시에 있다**
- 링크가 뷰포트에 들어올 때(IntersectionObserver) 미리 부르는 것도 같은 원리다
- 단, 첫 화면에서 전부 preload하면 lazy를 한 의미가 없다. **다음에 갈 확률이 높은 경계만** 골라서 건다

---

## entry 분리란 무엇인가

**라우트 분할을 런타임(`import()`)이 아니라 빌드 타임에 하는 것.**

```ts
build: {
  rolldownOptions: {
    input: {
      main:  'src/entries/main.ts',
      admin: 'src/entries/admin.ts',
    },
  },
}
```

각 entry는 **독립된 번들 그래프의 루트**다. `admin.ts`에서 도달할 수 없는 코드는 `admin.js`에 없다. 반대도 마찬가지다.

---

## entry 분리의 산출물

```text
dist/
  main.js   / main.css      ← 일반 사용자용
  admin.js  / admin.css     ← 관리자용
  chunks/shared-*.js        ← 둘 다 쓰는 것만
```

`lazy`와 달리 **런타임 요청 왕복이 없다.** 해당 진입점을 로드하는 순간 필요한 게 정적으로 다 딸려 온다.

---

## entry 분리가 가능한 구조인가

| 구조                       | 가능? | 어떻게 나뉘나                               |
| -------------------------- | ----- | ------------------------------------------- |
| **MPA**                    | ✅    | 페이지마다 HTML → 각 HTML이 다른 entry 로드 |
| **마이크로 프론트엔드**    | ✅    | 호스트가 경로별로 다른 리모트를 마운트      |
| **위젯 / 임베드**          | ✅    | 삽입 지점마다 다른 번들                     |
| **어드민·유저 분리 SPA**   | ✅    | 도메인·경로로 갈리므로 HTML을 나눌 수 있음  |
| **단일 SPA (라우터 하나)** | ❌    | 진입 HTML이 하나라 나눌 방법이 없음         |

**마지막 줄이 중요하다.** `index.html` 하나로 서비스되는 일반 SPA는 entry를 나눠도 브라우저가 결국 하나만 로드한다. 그 경우엔 앞 슬라이드의 **묶음 단위 lazy + prefetch**가 답이다.

---

## 수단 ①: 임계값으로 파편 억제

```ts
build: {
  rolldownOptions: {
    output: {
      codeSplitting: {
        minSize: 20 * 1024,      // 이보다 작으면 청크로 안 뺌
        minShareCount: 2,        // N개 이상 진입점이 참조할 때만 분리
        maxSize: 200 * 1024,     // 이보다 크면 다시 쪼갬
      },
    },
  },
}
```

6KB짜리 조합 청크가 100개라면 이게 가장 직접적이다.

`minSize`에 걸려 탈락한 모듈은 사라지는 게 아니라 **자동 청킹으로 폴백**한다. 자기를 쓰는 청크로 흡수된다.

---

## 수단 ②: `groups`로 캐시 경계 만들기

```ts
codeSplitting: {
  groups: [
    {
      name: 'ds',
      test: /node_modules[\\/](?:\.pnpm[\\/])?@ds(?:\+|[\\/])(?:ui|headless|icon)/,
      priority: 10,
    },
  ],
}
```

**pnpm을 쓴다면 `.pnpm` 디렉토리와 `+` 구분자를 정규식에 넣어야 한다.** `@ds/ui`는 디스크에서 `node_modules/.pnpm/@ds+ui@2.4.0/node_modules/@ds/ui/...`로 존재한다. 빠뜨리면 그룹이 아무것도 못 잡는다.

Windows 대응으로 경로 구분자는 `[\\/]`를 쓰라고 rolldown 문서가 권고한다.

---

## 설정으로는 한계가 있다

`minSize`·`groups`는 **이미 생긴 파편을 사후에 묶는** 수단이다. 조합이 27개 진입점에서 나온다는 사실 자체는 그대로다.

```text
설정 수단 (minSize / groups)   →  파편을 묶어 개수를 줄임
구조 수단 (묶음 lazy / entry)   →  조합의 밑을 줄임
```

**둘 다 필요하지만 순서가 있다.** 구조를 먼저 정하고, 남는 파편을 설정으로 정리한다. 반대로 하면 정규식만 늘어난다.

---

## Part 4 판단 기준

| 상황                                          | 선택                                    |
| --------------------------------------------- | --------------------------------------- |
| 라우트 2~3개, 단일 화면                       | **아무것도 안 함.** entry 하나          |
| 진입 HTML이 갈리는 구조 (MPA·MFE·어드민 분리) | **entry 분리**                          |
| 단일 SPA, 라우트가 많음                       | **기능 묶음 단위 lazy** (라우트 단위 ✗) |
| 무겁고 일부만 쓰는 라이브러리                 | 그 지점만 `lazy`                        |
| lazy 전환이 느리다는 불만                     | hover·뷰포트 **prefetch**               |
| 전 화면 공통 vendor                           | `groups`로 캐시 분리                    |
| 6KB짜리 파편이 100개                          | `minSize` / `minShareCount`             |

**"모든 라우트를 lazy로"는 기본값이 아니다.** 에디터·차트·PDF처럼 무겁고 일부만 쓰는 것에 먼저 쓴다.

---

# Part 5. 기대치 관리

측정해서 기각한 것들

---

## 이미 되어 있는 걸 또 하지 않기

| 시도                     | 왜 무의미한가                                  |
| ------------------------ | ---------------------------------------------- |
| `output.comments: false` | production은 미니파이가 이미 주석을 전부 제거  |
| cssnano 등 CSS 압축 추가 | Vite 8은 lightningcss로 이미 압축 (1줄로 나옴) |
| gzip 플러그인            | 압축은 보통 CDN·서버가 한다                    |

> production CSS의 raw가 개발 빌드보다 **큰** 경우가 있다. lightningcss가 legacy 브라우저 타깃으로 다운레벨링하기 때문이지 미압축이 아니다. **raw만 보고 판단하면 또 틀린다.**

---

## 이득이 비용을 못 넘는 것들

| 시도                           |     이득 | 비용                         |
| ------------------------------ | -------: | ---------------------------- |
| `classnames` → `clsx`          |  −0.3 KB | 43개 파일 수정               |
| 금융 계산 라이브러리 경량 대체 |  −1.5 KB | **금액 계산 정확성 리스크**  |
| 중복 버전 dedupe               | −2.25 KB | 모노레포 전체 의존성 재해석  |
| `build.target` 상향            | −0.48 KB | 지원 브라우저 정책 결정 선행 |

**번들 크기는 정확성 앞에서 후순위다.** 금융 계산을 1.5KB 때문에 바꾸는 건 나쁜 거래다.

---

## 효과가 없는 것들

- **상시 노출 컴포넌트 `lazy`**: 첫 화면에 무조건 보이는 걸 lazy로 만들면 청크만 늘고 전송량은 그대로다. 오히려 요청 시작만 늦어진다
- **이미 트리셰이킹된 패키지에 추가 조치**: `sideEffects: false`가 선언된 패키지는 그냥 두면 된다

---

## 레버는 서로 독립이 아니다

도입부에서 예고한 그 얘기다. 디자인시스템을 통째로 `external`로 빼는 계획의 기대치가 처음엔 **−89 KB gz**였다. 나중에 실제로 재보니 **−31.81 KB**였다.

계획을 세운 시점과 실행 시점 사이에 **트리셰이킹(Part 2)을 먼저 적용했기 때문이다.**

```text
Part 2 적용 전 : 디자인시스템 청크 89.99 kB gz   ← external로 빼면 −89
Part 2 적용 후 : 디자인시스템 청크 31.87 kB gz   ← external로 빼면 −31.81
```

**같은 바이트를 겨냥하는 기법끼리는 먼저 적용한 쪽이 이득을 가져간다.** 죽은 코드를 이미 걷어냈으면, 그걸 외부화해도 걷어낼 게 없다.

---

## 그래서 순서가 중요하다

두 가지가 따라온다.

**① 개별 Δ를 단순 합산하면 과대평가된다**

각각 재서 `−38 + −33 + −31 + −19`를 더하면 실제 총합보다 크다. 로드맵을 세울 때는 **적용 순서대로 누적 측정**해야 한다.

**② 싼 것부터 하는 게 유리한 또 다른 이유**

비싼 레버(조율이 필요한 것)의 기대치가 깎이더라도, 그건 **이미 싼 레버로 이득을 확보했다는 뜻**이다. 반대로 비싼 것부터 하면 여러 팀을 동원해 −89를 얻고, 그 다음 싼 레버는 −7밖에 안 나온다.

> 총량은 비슷해도 **비용 구조가 완전히 다르다.**

---

## 기각 목록을 남기는 이유

측정해서 기각한 항목을 기록해두면:

- 6개월 뒤 같은 제안이 올라왔을 때 재측정 없이 답한다
- 신규 입사자가 "이건 왜 안 했지?"에 스스로 답을 찾는다
- **추정만으로 착수하는 문화를 막는다**

기각 사유에는 **숫자와 측정 조건**을 남긴다. "작아서 안 함"이 아니라 "production 빌드 기준 −0.3 KB gz, 43파일 수정 필요".

---

# Part 6. 검증

"줄었다"를 어떻게 증명하나

---

## CSS 제거는 셀렉터 diff로 교차검증

CSS를 걷어낼 때 가장 무서운 건 **쓰는 셀렉터를 지우는 것**이다. 빌드는 통과하고 화면만 깨진다.

```text
클래스 셀렉터 : 589 → 465   (제거 124, 추가 0)
CSS 변수 선언 : 13 → 9

검사                                        결과
─────────────────────────────────────────────────
제거 클래스가 빌드된 JS 문자열에 등장        0건
제거 클래스의 접두사가 JS에 등장(동적 조합)   0건
제거 클래스를 소스(SCSS/TSX)가 참조          0건
예상치 못하게 추가된 클래스                  0건
```

접두사 검사가 중요하다. `` `ds-color-${color}` `` 같은 **런타임 조합**은 완전 문자열 검색으로 안 잡힌다.

---

## 이름이 애매한 건 정의 파일까지 추적

```text
ds-container-color-*, ds-placement-*  → floating-layout (미사용)
ds-type-solid/line, ds-check*         → checkbox (미사용)
ds-size-compact, ds-text-size-*       → textfield (미사용)
```

"이 클래스가 우리 게 아닌 것 같은데"에서 멈추지 않고 **어느 컴포넌트 소유인지**까지 확인한다.

정적 분석은 시각 회귀 테스트를 대체하지 못한다. 다만 시각 회귀를 돌릴 수 없을 때 **할 수 있는 만큼은 한다.**

---

## 안 잰 건 안 쟀다고 쓴다

측정 기록에 이 섹션을 반드시 넣는다.

> **확인이 남은 불확실성**
>
> 1. CSS 변경의 시각 회귀 미실시: 셀렉터 diff 정적 분석만
> 2. 두 기법의 개별 기여도 미분리: 커플링돼 있어 합산치만 측정
> 3. 런타임 스모크 미실시: 빌드 + `tsc --noEmit`까지

**"−19 KB 줄었다"와 "−19 KB 줄어드는 것으로 측정됐고 시각 검증은 안 했다"는 다른 문장이다.**

---

## 정리: 착수 순서

| 순서 | Part        | 확인                               | 기대 규모       |
| ---- | ----------- | ---------------------------------- | --------------- |
| 0    | 측정        | 미니파이 켜져 있나, gz로 보고 있나 | -               |
| 1    | 들어오는 것 | 런타임이 주는 걸 또 번들하나       | **수십 KB**     |
| 2    | 걷히는 것   | 배럴 + `sideEffects` 미선언        | **수십 KB**     |
| 3    | CSS         | CSS 배럴을 통째로 로드하나         | **십수 KB**     |
| 4    | 받는 시점   | 첫 화면에 필요 없는 게 있나        | **크기 이득 0** |

**Part 1~3은 앱 단독 설정 변경이고, Part 4는 구조 변경이다.** 싸고 확실한 것부터.

---

## 누적으로 보면 이렇게 간다

앱 B를 조율 비용 순으로 밟았을 때의 실측 누적이다. 첫 진입 전송량 기준(JS + CSS, gz).

```text
베이스라인                                    165.04
  + CSS 배럴 정리(−18.99) · 중복버전 dedupe(−2.25)
                                              143.80   (−12.9%)  ← 앱 단독
  + 브릿지 유틸 external(−4.89)
                                              138.91             ← config 1줄
  + 데이터 페칭 라이브러리 external(−11.83)
                                              127.08             ← 런타임 협조
  + 디자인시스템 JS external(−31.81)
                                               95.27   (−42%)    ← 대규모 조율
```

**앞의 두 단계(−26.13)는 다른 팀에 요청할 게 없다.** 여기까지가 협상 없이 갈 수 있는 지점이고, 그 아래부터는 조율 비용이 급격히 올라간다.

---

## 이 표에서 읽어야 할 것

- **첫 두 줄이 전체 이득의 37%**: 앱 단독 변경만으로
- 마지막 한 줄이 나머지의 대부분인데, **런타임과 전 소비자의 버전 단일화**가 선행돼야 한다
- Part 4(코드 스플리팅)는 이 표에 **한 줄도 없다**: 크기를 안 줄이니까

> 조직에 "번들 42% 줄입니다"를 약속하기 전에, 그 42% 중 얼마가 내 결정권 안에 있는지부터 본다.

---

# Q&A

---

## Q. 청크는 적을수록 좋은가?

아니다. 기준이 세 개다.

- **전송량**: 청크 개수와 무관
- **캐시 효율**: 자주 바뀌는 앱 코드와 안 바뀌는 vendor를 분리하면 이득
- **요청 수 / 왕복**: 너무 잘게 쪼개면 손해

"5개가 164개보다 좋다"가 아니라 **164개가 될 만한 이유가 있었는지**를 물어야 한다. 라우트를 전부 lazy로 만든 결과라면, 의도한 분할이 맞는지 한번 돌아볼 필요가 있다.

---

## Q. Vite 7 이하(Rollup)를 쓰고 있다면?

개념은 그대로 옮겨진다. 이름만 다르다.

| Rollup (Vite 5~7)                 | Rolldown (Vite 8)             |
| --------------------------------- | ----------------------------- |
| `build.rollupOptions`             | `build.rolldownOptions`       |
| `output.manualChunks`             | `output.codeSplitting`        |
| `output.experimentalMinChunkSize` | `codeSplitting.minSize`       |
| (없음)                            | `codeSplitting.minShareCount` |
| `treeshake.moduleSideEffects`     | 동일 (룰 배열 형태 추가)      |

`manualChunks`와 `advancedChunks`는 rolldown 타입 정의에 `@deprecated`로 명시돼 있다. `codeSplitting`은 `advancedChunks`를 rename한 최근 옵션이므로, **rolldown 버전이 오래됐다면 이름부터 확인한다.** 새로 쓴다면 `codeSplitting`이다.

---

## Q. gz가 아니라 brotli로 봐야 하지 않나?

CDN이 brotli를 서빙한다면 그쪽이 실제 전송량에 가깝다. 다만:

- **Δ의 방향과 상대 크기는 gz와 거의 같다**: 판단이 뒤집히는 경우는 드물다
- gz는 `zlib` 내장이라 스크립트가 간단하다
- Vite 콘솔 출력이 gz 기준이라 눈으로 대조하기 쉽다

성능 예산을 절대값으로 계약해야 하는 상황이면 brotli로 재고, **레버를 비교하는 단계에서는 gz로 충분하다.**

---

## Q. 어디까지 줄여야 하나

**목표는 KB가 아니라 사용자 지표여야 한다.** LCP, TBT, 첫 상호작용까지의 시간.

번들 최적화는 그 지표를 개선하는 여러 수단 중 하나다. 크기 레버를 다 훑고 나면 대개 병목은 다른 쪽(API 워터폴, 캐시 설정 누락)에 있다.

**165 KB를 145 KB로 줄이는 것보다, 순차 요청 하나를 병렬로 바꾸는 게 체감이 클 수 있다.** 크기만 보다 보면 이걸 놓친다.

---

# 감사합니다

<!-- _class: invert -->

@yceffort
