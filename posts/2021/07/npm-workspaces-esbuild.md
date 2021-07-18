---
title: 'npm workspace와 esbuild로 monorepo 구축해보기'
tags:
  - javascript
  - npm
published: true
date: 2021-07-06 19:34:21
description: '계속 찍먹만 해보는 중'
---

매번 느끼는 거지만 자바스크립트 생태계는 진짜 쉴새 없이 변한다. 하루에도 수십 수백가지의 패키지가 만들어지고, 또 잘나가는 프로젝트는 오늘도 버전업과 기능 추가에 여념이 없다.

그런 와중에 내 눈에 들어온 것이 npm workspace와 esbuild다. npm workspace는 과연 lerna의 아성을 넘을 만큼 잘만들어졌을까? esbuild는 또 걔네들이 말하는 것처럼 엄청 빠를까?

## 예제 레파지토리

https://github.com/yceffort/workspaces-esbuild-example

## npm workspace

npm v7 이 정식으로 나오면서 모노레포를 지원하게 되었다. 원래 monorepo는 lerna와 yarn이 꽉잡고 있던 영역이었는데, 이번에 npm이 등장하게 되면서 npm cli로도 workspace를 활용하면 모노레포를 구축할 수 있게 되었다.

https://docs.npmjs.com/cli/v7/using-npm/workspaces

이 기능을 활용하면, 로컬 파일시스템에서 연결된 패키지를 훨씬 더 효율적으로 관리할 수 있게 해준다. 기존에 원래 있던 명령어인 `npm install`을 활용하면 자동으로 패키지를 link 해주고, 서로 다른 패키지 레벨에서 `npm link`할 필요 없이 알아서 현재 폴더의 `node_modules`를 가지고 연결해준다.

npm workspace를 어떻게 구축하는지 먼저 살펴보자.

```json
{
  "name": "@yceffort/monorepo",
  "version": "0.0.1",
  "description": "",
  "main": "index.js",
  "scripts": {
    "build:all": "npm run build --workspaces",
    "deploy:all": "npm run deploy --workspaces",
    "lint": "eslint '**/*.{js,ts,tsx}'",
    "lint:fix": "npm run lint -- --fix",
    "prettier": "prettier '**/*.{json,yaml,md}' --check",
    "prettier:fix": "prettier '**/*.{json,yaml,md}' --write"
  },
  "author": "yceffort",
  "license": "ISC",
  "devDependencies": {
    "esbuild": "^0.12.12",
    "esbuild-node-externals": "^1.3.0",
    "eslint-config-yceffort": "0.0.5",
    "typescript": "^4.3.4"
  },
  "workspaces": ["./packages/*"]
}
```

먼저 프로젝트 루트 디렉토리에서 `workspaces`를 정의해줘야 한다. 위 예제에서는, `packages` 디렉토리 이하에 있는 프로젝트를 각각 모노레포의 모듈로 가져가기 위해 설정해두었다.

그리고 모노레포로 가져갈 레파지토리에 package.json을 설정해 둬야 한다. 이는 일반적인 패키지 설정과 별 차이가 없다.

여기서 `npm install`을 해보자.

> 한가지 중요한 것은 (당연한 이야기지만) npm v7 이어야 한다는 것이다. 이 실험을 위해서 무작정 npm v7을 설치하는 것은 권장하지 않는다. v7의 또한가지 변경점은 package-lock.json의 관리방식이 변경되었다는 것이다. (lock-version이 2로 올라갔다) 그래서 npm v7 을 버전업 한 후에 다른 프로젝트에서 npm i 를 날리면 package-lock.json에 무지막지한 diff 가 생성될 것이다. 이를 방지하기 위해 `npx npm@7` 를 사용하도록 하자. 뭐, 다른 프로젝트도 lockversion v2를 가져가도 괜찮다면 상관없다.

![npm-workspace](./images/npm-workspace-1.png)

여러개의 `package.json`이 있지만, `node_modules`는 루트에만 생성된 것을 볼 수 있다. 하위 패키지들의 참조는 모두 이 루트의 `node_modules`로 이어져 있다. (앞서 장황하게 설명한 그것)

이러한 패키지 설정을 매번 수동으로 할 필요는 없다.

```bash
npm init -w ./packages/some-package
```

이렇게 하면 알아서 하위 패키지를 만들어둔다. 단, 루트에 있는 `package.json`에 `workspaces` 값이 새롭게 추가된 패키지도 가르키고 있는지 확인해야 한다. 난 그게 귀찮아서 `*`로 처리했다.

만약 워크 스페이스에 의존성을 설치하고 싶다면,

```bash
npm install react -w some-package
```

와 같은 방식으로 하면 된다. 물론, `package.json`에 직접 방문해서 루트에서 `npm install`을 설치해도 된다.

## esbuild

자바스크립트는 느리다. 애초에 이렇게까지 쓰기 위해서 설계된 언어가 아니기 때문이다. 애초에 웹페이지에 있는 폼이나, 간단한 연산 정도만 처리할 용도로만 만들어졌기 때문이다. (싱글 스레드) 따라서 번들러가 느린 것도 어느정도 어쩔 수 없는 문제(?) 로 다들 받아드리고 있었다. 그런데, 아예 번들러를 저수준의 다른언어인 GO로 만들어버려서 이 속도문제를 해결한 것이 바로 esbuild다.

https://esbuild.github.io/

이 esbuild로 모노레포를 한번 만들어보려고 한다.

https://esbuild.github.io/getting-started/#your-first-bundle

```javascript
// esbuild
const esbuild = require('esbuild')
// 빌드시에 자동으로 node_modules를 제외 해준다.
// https://github.com/pradel/esbuild-node-externals
const { nodeExternalsPlugin } = require('esbuild-node-externals')

esbuild
  .build({
    entryPoints: ['./src/index.ts'],
    outfile: 'dist/index.js',
    bundle: true,
    minify: true,
    platform: 'browser',
    format: 'esm',
    sourcemap: true,
    target: 'es6',
    plugins: [nodeExternalsPlugin()],
  })
  .catch(() => process.exit(1))
```

처음 설정을 하고 느꼈던 첫인상은, webpack이나 rollup 처럼 json 설정이 불가능하다는 것이다. `.babelrc`와 같은 [cosmicconfig](https://github.com/davidtheclark/cosmiconfig)로 설정파일을 만드는 것이 불가능하다.

본격적으로 설정파일을 하나씩 파헤쳐보자.

- [entryPoints](https://esbuild.github.io/api/#entry-points): 번들링 알고리즘이 들어가게 되는 애플리케이션의 entry 포인트다. 보시다시피 ts가 자동지원 되기 때문에 (다 지원되는 건 아니다.) 타입스크립트 파일을 넣어도 무방하다.
- [outfile](https://esbuild.github.io/api/#outfile): 번들의 결과물이다. `entryPoints`와는 다르게, 딱 하나의 파일만 (문자열만) 가능한 것을 볼 수 있다. 단하나의 번들된, 그리고 minified된 파일이 나오게 된다.
- [bundle](https://esbuild.github.io/api/#bundle): 번들링 여부
- [minify](https://esbuild.github.io/api/#minify): minification (자바스크립트 파일 축소) 여부
- [platform](https://esbuild.github.io/api/#platform): 번들링된 파일이 어느 환경에서 실행될지를 결정하게 된다.
- [format](https://esbuild.github.io/api/#format): 생성된 파일의 형태를 나타낸다. `iife`, `cjs` `esm`이 가능하다.
- [sourcemap](https://esbuild.github.io/api/#sourcemap): 디버깅을 용이하게 해주는 소스맵 제공 여부
- [target](https://esbuild.github.io/api/#target): 어떤 플랫폼의 버전에서 사용할 수 있을지 명시한다. 가능한 옵션은 https://esbuild.github.io/content-types/#javascript 여기에 있다.

위 설정대로 esbuild를 실행해보자.

**sum.ts**

```typescript
export default function sum(...args: number[]) {
  return args.reduce((prev, acc) => acc + prev, 0)
}
```

**formatNumber.ts**

```typescript
export default function formatNumberWithComma(value: string | number): string {
  if (typeof value === 'string' && isNaN(+value)) {
    return value
  }

  let formattedNumber = `${value}`

  const reg = /(^[+-]?\d+)(\d{3})/

  while (reg.test(formattedNumber)) {
    formattedNumber = formattedNumber.replace(reg, '$1,$2')
  }

  return formattedNumber
}
```

**index.ts**

```typescript
export { default as sum } from './sum'
export { default as formatNumberWithComma } from './formatNumber'
```

**결과**

(minified된 파일을 보기 쉽게 하기 위해 unminifed함.)

```javascript
function m(...r) {
  return r.reduce((t, e) => e + t, 0)
}

function o(r) {
  if (typeof r == 'string' && isNaN(+r)) return r
  let t = `${r}`,
    e = /(^[+-]?\d+)(\d{3})/
  for (; e.test(t); ) t = t.replace(e, '$1,$2')
  return t
}
export { o as formatNumberWithComma, m as sum }
//# sourceMappingURL=index.js.map
```

## 삽질하면서 깨달은 것들, 그리고 감상

- target을 ES5로 할수 없다. 이는 공식 문서 https://esbuild.github.io/content-types/ 에도 나와있고, 찾아보니 제작자도 지원할 생각이 없는 것 같다. https://github.com/evanw/esbuild/issues/182#issuecomment-646297130 따라서 별도로 transpile 후에, 다시 esbuild를 해줘야 한다.
- 타입스크립트를 지원하지만, d.ts를 emit 해주지 않는다. 이를 위해서는 별도로 `tsc`를 실행해서 타입을 만들어야 한다. 그래서 빌드시 별도로 `tsc`를 실행했다.

```json
{
  "extends": "../../tsconfig.base.json",
  "compilerOptions": {
    "baseUrl": ".",
    "rootDir": "src",
    "outDir": "./dist",
    "declarationDir": "./dist",
    // 이 아래 두개가 중요
    "emitDeclarationOnly": true,
    "declaration": true,
    "types": ["jest", "node"]
  },
  "include": ["./src/**/*.ts"],
  "exclude": ["node_modules", "dist"]
}
```

- css module 번들링이 아직 완벽하지는 않은 것 같다. https://github.com/evanw/esbuild/issues/20 현재 제작자가 최선을 다해서(?) 작업중이라고 한다.
- 그 밖에도 아직 작업중인 것들이 많다. 0.x 버전인데에는 이유가 있었다.
- `npx npm@7`을 매번 해주는 것이 너무 귀찮았다 (...) 가끔 이 사실을 까먹고 workspace를 쓰면 당연히 안된다. default로 7을 쓰고 싶지만, 다른 프로젝트 때문에 쓰지 못해서 아쉬웠다. 물론, `nvm`을 써서 node@16을 쓰고, 여기의 기본 npm 버전에 의존하는 방법 (7.18.1)도 있었지만, 생각만큼 잘되지는 않았다.
- 방금 예제는 정말 간단한 패키지라서 속도를 체감할 수는 없었지만, 큰 패키지를 대상으로 실험해본 결과 정말로 크게 속도차이가 나긴 했다.
- `webpack` 환경에서도 사용할 수 있도록 [esbuild-loader](https://github.com/privatenumber/esbuild-loader)가 존재한다. 앞서 살펴본것처럼, minify, uglify도 esbuild가 해주기 때문에 terser를 대체할 수 있다.
  - 벤치마킹 : https://github.com/privatenumber/minification-benchmarks
- 제법 많은 플러그인들이 존재했다. https://github.com/esbuild/community-plugins
- 개발자가 혼자서 고군 분투 중이었다. 정말 멋있었다. (존경)
- esbuild로 SSR도 가능한 것처럼 보인다. https://github.com/egoist/maho 깊게 살펴보진 않았지만
  - 차라리 vite를 사용해보는 걸 추천하고 싶다. https://vitejs.dev/guide/ssr.html 물론 여기도 experimental이다.

웹팩은 이미 메이저버전이 5까지 나와있을 정도로 성숙한 프로젝트고, 또 많은 사람들이 널리 사용하고 있는 프로젝트다. (롤업과 parcel도 잊으면 안된다) 하지만 기존의 당연시 생각되는 것들을 깨는 새로운 것의 등장은 언제나 보는 사람으로 하여금 설레게 하는 것 같다. esbuild도 그런 프로젝트 중 하나로, 정식 버전 업데이트 까지 잘 만들어졌으면 좋겠다.

아, workspace도 npm 생태계에 모노레포를 잘 녹여낸 것 같아서 좋았다. lerna보다는 쓰기 편한 것 같은 느낌?하지만, npm@7에서 workspace말고 그외에는 글쎄... 🤔

https://blog.logrocket.com/whats-new-in-npm-v7/
