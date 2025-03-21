---
title: '자바스크립트 의존성 지옥'
tags:
  - javascript
  - npm
published: true
date: 2020-11-20 23:14:25
description: 'package-lock.json은 정말 복잡 😈'
---

모든 자바스크립트 프로젝트들은 시작할 때만 하더라도 많은 NPM 패키지를 의존성으로 갖지 않으려고 노력한다. 이런 노력에도 불구하고, 결국 몇몇 패키지를 사용하기 시작한다. `package.json`에 한줄 한줄이 추가될 수록, PR에서 보이는 `package-lock.json`의 추가/삭제 라인 수는 끔찍해진다.

물론 이렇한 과정이 팀리더나 동료들의 반대에 부딪히지는 않는다. 자바스크립트 생태계가 살아있고 계속해서 번창한다는 것은 굉장한 행운이다. 매번 바퀴를 새롭게 발명하거나, 오픈소스 커뮤니티가 해결한 문제를 또 해결하려고 시도해서는 안된다.

블로그를 만들기 위해 gatsby를 쓴다고 가정해보자. 이를 설치하고 dependency에 추가해보자. 이제 1800개의 추가 dependency를 추가했다. 이는 정말 괜찮은 걸까? 자바스크립트의 dependency 트리는 얼마나더 복잡해질 수 있을까? 어떻게 의존성 지옥이 만들어지는 걸까?

## 자바스크립트 패키지

NPM (Node Package Manager)는 세계에서 가장 큰 자바스크립트 패키지 레지스트리르르 보유하고 있다. 이는 RubyGems, PyPi, Maven을 합친 것보다 크다.

![Module Count](./images/module-counts.png)

출처: http://www.modulecounts.com/

정말 많다. 이러한 npm 패키지를 사용하기 위해서는, 프로젝트에 `package.json`을 추가해야 한다.

## package.json

`package.json`은 무엇인가?

- 프로젝트가 의존하고 있는 패키지의 목록
- 시멘틱 버전에 따라서 프로젝트가 의존하고 있는 패키지의 특정버전을 구체적으로 나열
- 빌드를 언제든 다시 만들 수 있게 하여 다른 개발자들이 공유를 쉽게 함

패키지가 다른 패키지에 의존한다고 상상한다면, 왜 `gatsby`가 1.9만개의 추가 종속성을 갖게 되는지 알 수 있을 것이다.

## package.json의 종속성 타입

종속성이 어떻게 누적되는지 이해하기 위해서는, 프로젝트가 가질 수 있는 다양한 종속성 타입을 이해해야 한다.

- `dependencies`: 프로젝트의 코드를 호출하는데 있어 필수적으로 의존하고 있는 종속성
- `devDependencies`: 개발단계에서 필요한 종속성. `prettier`와 같은 코드를 이쁘게 하는 라이브러리 등
- `peerDependencies`: `package.json`에 `peerDependencies`를 설정해둔다면, 패키지를 설치하는 다른 사람들에게 여기에 지정된 버전에 대한 종속성이 필요하다고 말하는 것이다.
- `optionalDependencies`: 옵션 성격의 종속성으로, 이 종속성을 설치 하는데 실패한다 하더라도 설치 과정에 문제가 되지는 않는다.
- `bundleDependencies`: 패키지를 번들링 하는데 같이 들어가게 되는 의존성. NPM에 있지 않은 제3의 라이브러리나, 일부 프로젝트 모듈로 포함하려는 경우 유용하다.

## package-lock.json의 목적

`package-lock.json`은 자동으로 `package.json`이나 `node_modules` 디렉토리가 변할 때 마다 자동으로 생성된다. 이는 설치로 만들어진 정확히 똑같은 의존성 트리를 보관하고 있으며, 후속 설치에도 동일한 트리를 생성할 수 있도록 한다. 이는 나와 다른 사용자가 다른 의존성 트리를 만드는 것을 막는다.

`package.json`에 `react`를 설치한다고 가정해보자. `package-lock.json`에는 이렇게 나와있을 것이다.

```json
{
  "react": {
    "version": "17.0.1",
    "resolved": "https://registry.npmjs.org/react/-/react-17.0.1.tgz",
    "integrity": "sha512-lG9c9UuMHdcAexXtigOZLX8exLWkW0Ku29qPRU8uhF2R9BN96dLCt0psvzPLlHc5OWkgymP3qwTRgbnw5BKx3w==",
    "requires": {
      "loose-envify": "^1.1.0",
      "object-assign": "^4.1.1"
    }
  }
}
```

`package-lock.json`은 프로젝트의 거대한 종속성 목록을 가지고 있다. 여기에는 버전, module의 위치 (URI), 정합성을 위한 해싱값과 패키지가 요구하는 모듈들이 나와있다.

## Gatsby.js의 의존성 살펴보지.

Gatsby는 왜 1800개의 의존성을 갖게 되는 것일까? 답은 의존성의 의존성이다.

```bash
$ npm install --save gatsby

...

+ gatsby@2.27.0
added 1889 packages from 1011 contributors and audited 1889 packages in 51.894s
```

`package.json` 에는 의존성이 딱 하나만 존재하지만,

```json
{
  "name": "test",
  "version": "1.0.0",
  "description": "",
  "main": "index.js",
  "scripts": {
    "test": "echo \"Error: no test specified\" && exit 1"
  },
  "author": "",
  "license": "ISC",
  "dependencies": {
    "gatsby": "^2.27.0"
  }
}
```

`package-lock.json`에는 이제 만 오천줄이 넘는 종속성이 명시되어있다. 이 문제의 원인은 [gatsby의 package.json](https://github.com/gatsbyjs/gatsby/blob/347be6f6fbaa2f9b2e252e6f329ce7fe96f6f2b2/packages/gatsby/package.json#L12-L162)에 있다.

```bash
test@1.0.0 /Users/yceffort/private/test
└─┬ gatsby@2.27.0
  ├─┬ @babel/core@7.12.3
  │ ├─┬ @babel/helper-module-transforms@7.12.1
  │ │ └── lodash@4.17.20  deduped
  │ └── lodash@4.17.20  deduped
  ├─┬ @babel/traverse@7.12.5
  │ └── lodash@4.17.20  deduped
  ├─┬ @babel/types@7.12.6
  │ └── lodash@4.17.20  deduped
  ├─┬ @typescript-eslint/parser@2.34.0
  │ └─┬ @typescript-eslint/typescript-estree@2.34.0
  │   └── lodash@4.17.20  deduped
  ├─┬ babel-plugin-lodash@3.3.4
  │ └── lodash@4.17.20  deduped
  ├─┬ babel-preset-gatsby@0.7.0
  │ └─┬ @babel/preset-env@7.12.1
  │   ├─┬ @babel/plugin-transform-classes@7.12.1
  │   │ └─┬ @babel/helper-define-map@7.10.5
  │   │   └── lodash@4.17.20  deduped
  │   └─┬ @babel/plugin-transform-sticky-regex@7.12.1
  │     └─┬ @babel/helper-regex@7.10.5
  │       └── lodash@4.17.20  deduped
  ├─┬ css-loader@1.0.1
  │ └── lodash@4.17.20  deduped
  ├─┬ devcert@1.1.3
  │ └── lodash@4.17.20  deduped
  ├─┬ eslint@6.8.0
  │ ├─┬ inquirer@7.3.3
  │ │ └── lodash@4.17.20  deduped
  │ ├── lodash@4.17.20  deduped
  │ └─┬ table@5.4.6
  │   └── lodash@4.17.20  deduped
  ├─┬ eslint-plugin-flowtype@3.13.0
  │ └── lodash@4.17.20  deduped
  ├─┬ gatsby-cli@2.14.0
  │ ├─┬ gatsby-recipes@0.4.0
  │ │ ├─┬ contentful-management@5.28.0
  │ │ │ ├─┬ contentful-sdk-core@6.4.6
  │ │ │ │ └── lodash@4.17.20  deduped
  │ │ │ └── lodash@4.17.20  deduped
  │ │ ├── lodash@4.17.20  deduped
  │ │ └─┬ remark-mdxjs@2.0.0-next.8
  │ │   └─┬ @babel/core@7.10.5
  │ │     └── lodash@4.17.20  deduped
  │ ├── lodash@4.17.20  deduped
  │ └─┬ pretty-error@2.1.2
  │   ├── lodash@4.17.20  deduped
  │   └─┬ renderkid@2.0.4
  │     └── lodash@4.17.20  deduped
  ├─┬ gatsby-plugin-page-creator@2.5.0
  │ ├─┬ gatsby-page-utils@0.4.0
  │ │ └── lodash@4.17.20  deduped
  │ └── lodash@4.17.20  deduped
  ├─┬ gatsby-telemetry@1.5.0
  │ └── lodash@4.17.20  deduped
  ├── lodash@4.17.20
  ├─┬ optimize-css-assets-webpack-plugin@5.0.4
  │ └─┬ last-call-webpack-plugin@3.0.0
  │   └── lodash@4.17.20  deduped
  ├─┬ react-dev-utils@4.2.3
  │ └─┬ inquirer@3.3.0
  │   └── lodash@4.17.20  deduped
  ├─┬ webpack-dev-server@3.11.0
  │ ├─┬ http-proxy-middleware@0.19.1
  │ │ └── lodash@4.17.20  deduped
  │ └─┬ portfinder@1.0.28
  │   └─┬ async@2.6.3
  │     └── lodash@4.17.20  deduped
  └─┬ webpack-merge@4.2.2
    └── lodash@4.17.20  deduped
```

gatsby의 lodash 의존을 살펴보면, 모두 같은 버전의 lodash를 사용하고 있기 때문에, `node_modules` 에는 하나의 `lodash` 만 설치해도 된다는 것을 알 수 있다. 그렇지만 만약 다른 버전에 각각 의존하고 있다면 해당 버전을 모두 설치해야 되므로 사이즈가 커지게 된다.

```bash
» du -sh node_modules
348M  node_modules
```

300메가 정도면 괜찮은 편이다. 만약 `node_modules`에서 무엇이 비중을 많이 차지 하는지 살펴보고 싶다면 아래 명령어를 실행하면 된다.

```bash
» du -sh ./node_modules/* | sort -nr | grep '\dM.*'
 30M  ./node_modules/@graphql-tools
 20M  ./node_modules/date-fns
 17M  ./node_modules/rxjs
 14M  ./node_modules/gatsby
 14M  ./node_modules/@babel
8.7M  ./node_modules/prettier
8.4M  ./node_modules/babel-runtime
8.3M  ./node_modules/gatsby-recipes
6.9M  ./node_modules/core-js
6.8M  ./node_modules/core-js-pure
5.5M  ./node_modules/eslint
5.1M  ./node_modules/moment
5.1M  ./node_modules/@types
4.9M  ./node_modules/webpack
4.8M  ./node_modules/lodash
...
```

(저놈의 graphql...)

`node_modules`의 사이즈를 줄이고, 종속성을 평평하게 만드는 명령어는 `npm dedup`이다. 중복된 종속성을 정리하는데 도움을 준다.

```bash
» npm dedup
audited 1889 packages in 3.36s

134 packages are looking for funding
  run `npm fund` for details

found 0 vulnerabilities
```

[Deduplication](https://docs.npmjs.com/cli/dedupe)은 종속성 사이의 공통 패키지를 찾고, 이러한 패키지가 재사용될 수 있도록 하여 종속성 트리 구조를 단순화 시키는 작업이다.

## 의존성 한눈에 보기

https://npm.anvaka.com/#/view/2d/eslint-config-yceffort

![npm-anvaka](./images/npm-anvaka.png)

http://npm.broofa.com/?q=eslint-config-yceffort

![npm-broofa](./images/npm-broofa.png)

https://packagephobia.com/result?p=eslint-config-yceffort@0.0.5

![npm-phobia](./images/npm-phobia.png)

## npm install, ci

`npm install`이 이따금씩 `package-lock.json`을 업데이트 하는 이유는, `package.json`에 정확하게 지정된 버전이 아닌 시멘틱 버전으로 작성되어 있기 때문이다. 예를 들어 `^1.1.0`으로 설치된 패키지가 있고, 시간이 흘러 `1.1.9`버전이 나온다면 `package-lock.json`은 기존 버전에서 `^1.1.9`로 설치하려 할것이다.

https://github.com/npm/npm/issues/18103

이를 막기 위한 명령어가 `npm ci`다. `package.json`이 아닌 `package-lock.json`에 명시된 버전 그 자체로 `package-lock.json`의 변경이 없이 설치를 수행한다. 많은 프로젝트에서 놓치는 것 중 하나가, 빌드나 배포단계에서 `npm ci`대신 `npm install`을 쓰는 것이다. 이는 개발단계에서는 몰랐던 얘기치 않은 에러를 낳을 수 있다.

https://blog.npmjs.org/post/621733939456933888/npm-v7-series-why-keep-package-lockjson

항상 감사하십시오, javascript developers.
