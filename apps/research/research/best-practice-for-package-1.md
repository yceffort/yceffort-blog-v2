---
title: 패키지 가이드 1탄) 패키지 제작을 위한 package.json 가이드
marp: true
paginate: true
theme: default
tags:
  - npm
  - nodejs
date: 2025-03-22
description: 'package.json 만 잘만들어도 반은 먹고 간다.'
published: true
art:
  undraw: setup
---

# 패키지 가이드 1탄) 패키지 제작을 위한 package.json 가이드

<!-- _class: invert deck-cover -->

@yceffort

---

## 패키지 제작이 어렵게 느껴지는 이유

- 평소에 개발하는 웹 서비스는 이미 프레임워크 형태로 제작되고 있기 때문에 실행 방법이나 진입점을 고려할 필요가 없음
- 그러나 패키지는 별도 프레임워크가 없어 이에 대한 도움을 받기 어려움
- 패키지가 모듈을 찾아가는 방식을 이해하기 위해서는 node module resolution algorithm 을 잘 이해해야 함
  - esm: <https://nodejs.org/api/esm.html#resolution-and-loading-algorithm>
  - cjs: <https://nodejs.org/api/modules.html#all-together>

> 저 내용을 100% 이해하고 개발할 수는 없으니까. 🤔

---

## <!-- fit --> `package.json` 만 이해해도 패키지 개발 반은 먹고 간다

- 서비스 개발시에 쓰는 `package.json` 사실상 `dependencies`, `scripts` 만 사용하고 있다고 봐도 무방
  - `version`, `main`, `repository` 없어도 그만
- 패키지 개발시에는 이야기가 다름
  - `package.json`의 몇몇 필드는 Node.js 가 해당 패키지에서 자바스크립트 모듈을 찾아가는 데 필수적인 정보를 제공
  - 이를 이해하면 패키지 개발이 훨씬 수월해짐

---

## 패키지 개발은 순서를 바꿔서 접근해보자

1. 내가 만들고 싶은 패키지가 무엇인지 생각하기
2. 해당 패키지에서 내보내야할 함수, 변수가 무엇인지 생각하기
3. 👉 2번을 만족할 수 있도록 `package.json` 작성하기 👈
4. 필요한 코드 작성하기
5. 3번에 구조에 맞게 4번 코드를 빌드할 수 있도록 번들러 및 트랜스파일러 설정하기

---

## 1. `name`과 `version`

### `name`

- 패키지의 이름으로, 용도에 맞게 작성
- `@pie` `@financial-auth`와 같이 스코프와 함께 사용 가능

### `version`

- 패키지의 버전으로, [유의적 버저닝](https://semver.org/) 을 따라 작성
- 호스트 패키지에서 해당 패키지가 어떤식으로 변경되었는지 알 수 있는 가장 중요한 정보
- 유의적 버저닝을 만족할 수 있도록 정확하게 버전을 작성하는 것이 중요

---

## 2. `main`과 `module`, `types`

<style scoped>section { font-size: 18px; }</style>

### `main`

- `main` 은 과거 node@12 미만 `CommonJS` 모듈만 지원하던 런타임에서 읽던 필드로, `require('something')` 호출시 가장 읽는 파일
- 뒤이어 설명할 `exports` 필드를 이해 못하는 번들러, 환경 등에서는 `main` 필드를 참조하여 모듈을 찾으므로 필수
- 따라서 여기는
  - `type: commonjs`거나 없다면 `js`파일로 CommonJS 로 작성하거나
  - `cjs` 파일로 작성해야 함

### `module`

- `main`과 비슷하게 `exports`를 이해하지 못하는 번들러나 환경에서 사용하는 필드
- `exports` 필드는 이해 못하지만 `ESModule` 를 지원하는 경우, `main` 대신 `module` 필드를 참조하여 모듈을 찾음
- 따라서 여기는
  - `type: commonjs`거나 없다면 `mjs`파일로 `ESModule` 임을 명시하거나
  - `type: module` 을 사용하여 `.js` 작성해야 함

### `types`

- 위 두 필드와 마찬가지로 `exports`를 이해하지 못하는 번들러나 환경에서 사용하는 필드
- 구형 타입스크립트 (4.8 ?), 구형 번들러 등에서 `main` 또는 `module` 에 매칭되는 타입스크립트 타입을 제공하기 위하여 사용

---

## 3. `files`

- `files`란 실제 npm 패키지에 배포 될 파일들을 지정하는 필드
- `files`에 없어도 몇몇 파일 `package.json` `LICENSE` 등은 기본으로 업로드 됨
- 타입스크립트와 같이 원본 파일이 아닌, `.js`, `.d.ts` 또는 소스맵과 같이 실제 호스트 패키지 입장에서 필요한 파일만 업로드 할 수 있도록 잘 설정해야함
- `./dist`와 같은 경로 이해 못하므로 잘 작성해야함
- `npm publish --dry-run`을 사용하면 실제 배포하지 않고도 배포 대상 파일을 확인할 수 있으므로 한번씩 해보시는 것을 추천

---

## 4. `type`

- `type`은 `.js` 파일을 어떻게 읽을지 Node.js 에 알려주는 힌트
- `commonjs`, `module` 두 가지를 지원하며, 기본값은 `commonjs` 임
- 이 타입과 상관없이 모듈을 강제하고 싶다면 `.mjs`와 `.cjs` 를 사용할 수 있음
- `.js` 에 대한 기본 모듈 동작을 조금이라도 빨리 알려주기 위해 둘 중 하나라도 기재하는 것이 좋음.
- 아니면 `/esm` 과 같이 폴더 레벨로 나누었다면, 해당 폴더에 `type` 만 있는 `package.json` 을 작성하는 것도 방법이 될 수 있음
  - Node.js 는 어떤 모듈인지 알기 위해 가장 가까운 `package.json` 을 읽기 때문
  - 그러나 가독성이 매우 떨어지므로 별로 추천하지는 않음

---

## 5. `sideEffects`

- 이 필드는 Node.js 가 아닌 [번들러](https://webpack.js.org/guides/tree-shaking/#mark-the-file-as-side-effect-free)를 위한 힌트 필드로, 해당 파일이나 패키지가 부수효과가 없는 순수 함수 여부를 알려줄 수 있는 필드
- 해당 필드가 없다면, 번들러는 모든 모듈이 순수하지 않다는 가정하에 다 때려넣어서 번들링을 하게 됨
- `false`를 넣어 모든 함수가 순수 함수임을 알려줄 수 있고, 혹은 배열로 특정 파일을 넣어준다면 그 파일을 제외하고 모두 순수하다는 것을 알려줄 수 있음

---

## 6-1. `exports`

> 진짜 우리가 제대로 알아야 하는 가장 중요한 필드

<style scoped>section { font-size: 18px; }</style>

- 해당 패키지가 제공하고 있는 API 를 선언하는 필드로, 호스트 패키지로 하여금 이 패키지에서 사용할 수 있는 모듈을 알려주는 역할

```json
{
  "main": "./index.js",
  "exports": "./index.js"
}
```

```jsonc
{
  "name": "my-package",
  "exports": {
    ".": "./lib/index.js",
    "./lib": "./lib/index.js",
    "./lib/index": "./lib/index.js",
    "./lib/index.js": "./lib/index.js",
    "./package.json": "./package.json",
  },
}
```

---

## 6-2. `exports`

<style scoped>section { font-size: 18px; }</style>

<!-- prettier-ignore -->
```jsonc

{
  "exports": {
    ".": { //  패키지의 엔트리 포인트. 위에서 부터 탐색하므로 서순이 매우 중요함
      // 롤업, 웹팩에서 사용하는 비공식 필드. import, require 보다 앞에 와야 하며 ESModule을 가리킨다.
      // 이 필드는 확장자가 js 여도 가능하므로, 반드시 require 보다 앞에  있어야 함
      "module": "./dist/index.mjs", 
      "import": { // 해당 경로로 패키지를 'import' 할 때 사용되는 파일
        "types": "./dist/index.d.mts",
        "default": "./dist/index.mjs",
      },
      "require": { // 해당 경로로 패키지를 'require' 할 때 사용되는 파일
        "types": "./dist/index.d.cts",
        "default": "./dist/index.cjs",
      },
      "default": "./dist/index.mjs", // 어느 조건에도 걸리지 못하는 경우 사용되는 폴백. 
      // 번외로 롤업에서만 쓰는 production, development 도 있다.
    },
    "./package.json": "./package.json", // package.json 도 명시적으로 내보내 주는 것이 요즘 트렌드
  }  
}
```

---

## 6-3. `.d.mts`, `.d.cts` 과연 의미가 있는것일까?

<style scoped>section { font-size: 22px; }</style>

- 파일명에서 유추할 수 있듯, `ESModule`과 `CommonJS` 에 대한 타입스크립트 타입을 의미
- 그러나 실제 대부분 사례에서는 두 파일 내용이 거의 동일 [참고.d.cts](https://app.unpkg.com/date-fns@4.1.0/files/add.d.cts) [참고.d.ts](https://app.unpkg.com/date-fns@4.1.0/files/add.d.ts)
- 그렇다면 두 파일은 실제로 의미가 있는 것일까? 아니면 그냥 두 파일을 만들어 놓은 것일까?
  - 사실 차이가 있음. 타입스크립트 역시 Node.js 와 동일한 알고리즘을 사용하기 위해 확장자를 통해 파일의 모듈 시스템을 구별함 [참고](https://www.typescriptlang.org/docs/handbook/modules/reference.html#module-format-detection)
  - 이러한 내용은 `module`의 값을 `node16` `node18` `nodenext` 등 으로 선언 할 때 사용되며, 실제 타입스크립트가 모듈 시스템을 이해하는데 있어 중요함 (어떤식으로 모듈을 읽어야 하는지 결정해야 하므로)
- 별도 types 필드에 값이 없어도 타입스크립트 컴파일러가 타입을 어떻게든 찾아갈수는 있음.그러나 굉장히 비효율적이라 선언해주는게 나음
  - `/mod.js`: `/mod.ts` > `/mod.tsx` > `/mod.d.ts` > `/mod.js` > `/mode.jsx`
  - .js 파일을 만나면 위와 같은 순서로 타입을 찾아가게 되므로, 반드시 **`types`필드가 `exports` 순서 상단에 있어야 함.**

---

## 7. Wrap up

- `name`, `version`, `files`, `sideEffects` 는 패키지 작성을 위한 필수 필드
- subpath 지원, ESModule 지원, 모던 번들러 지원을 위해서는 `exports` 필드를 잘 작성해주는 것이 매우 중요
- 타입 파일 역시 타입스크립트가 Node.js 와 동일한 모듈 알고리즘을 따라가므로 그에 맞춰서 작성하는 것이 필요
- 위 내용들을 잘 작성할 수 있도록 도와주는 도구
  - 자동완성 `schemastore`: <https://github.com/SchemaStore/schemastore/blob/master/src/schemas/json/package.json>
  - 실제 lint를 도와주는 `publint`: <https://publint.dev/>
