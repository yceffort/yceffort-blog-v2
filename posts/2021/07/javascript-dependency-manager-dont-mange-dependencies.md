---
title: '자바스크립트 의존성 관리자(npm, yarn, pnpm)보다 더 의존성 관리 잘하는 방법'
tags:
  - javascript
  - npm
  - yarn
published: true
date: 2021-07-28 22:17:24
description: '일단 제목으로 어그로를 끈다.'
---

## Table of Contents

## 시작하며

npm, yarn, pnpm 등은 자바스크립트 생태계가 성장하는데 지대한 공헌을 했다. 자바스크립트 영역에서 새로운 솔루션을 끊임없이 쉽게 찾고 이용할 수 있도록 도와주고 있다. 그러나 사용 편의성이라던가, 패키지의 모듈화가 심해진다는 단점 또한 존재한다. 사실, 앞서 소개한 자바스크립트 패키지 관리자들은 실제로 의존성을 제대로 관리하고 있지 못한다.

이 글에서는 결국 의존성 관리자(이하 패키지 관리자라고 칭하겠다)의 근본적인 문제를 해결하고자 하는 것은 아니다. 대신, 하드드라이브의 블랙홀을 막고, 종속성을 제대로 관리할 수 있는 가이드를 제공하고자 한다.

`node_modules`은 패키지가 설치될 수록 커지고 느려지게 된다. 또한 일부 라이브러리에서 패키지 의존성이 심해지면, 이를 사용하는 전세계 모든 사용자들의 작업이 느려지는 결과를 초래한다. 이를 막기 위해 아래 방법을 적용하면 종속성 크기를 점차 줄여나가고, 안정적으로 유지할 수 있다.

## 의존성 제어를 되찾기 위한 방법

이 가이드에서는 주로 `yarn1`에 초점을 맞추고 있지만, 대부분의 많은 권장사항 등이 다른 패키지 관리자에도 적용된다. 한가지 유의할 점은 일부 패키지 관리자의 경우 의존성을 직접 설치하지 않기 위해, 심링크, 하드링크 등의 까다롭고 불투명한 방법을 사용한다는 것이다.

### 1. 의존성 분석

먼저, `node_modules`에 도대체 무슨 패키지들이 설치되어 있는지 이해하는 작업이 필요하다. 여기에서는 다양한 방법을 활용하여 분석해보려고 한다.

- [Disk Inventory X](http://www.derlien.com/)와 `du -sh ./node_modules/* | sort -nr | grep '\dM.*'`를 사용하는 것이다. 이 방법을 사용하면, 현재 `node_modules`의 각 패키지의 크기를 확인해볼 수 있다.

다음은 내 블로그를 위 방법으로 살펴본 결과다.

![disk-inventory-x](./images/disk-inventory-x.png)

```bash
 58M    ./node_modules/typescript
 38M    ./node_modules/@babel
 34M    ./node_modules/tailwindcss
 30M    ./node_modules/next
 22M    ./node_modules/date-fns
 19M    ./node_modules/prettier
 17M    ./node_modules/rxjs
 15M    ./node_modules/@firebase
 11M    ./node_modules/@types
8.1M    ./node_modules/esbuild
7.8M    ./node_modules/@typescript-eslint
7.3M    ./node_modules/protobufjs
7.2M    ./node_modules/core-js-pure
6.4M    ./node_modules/webpack
6.4M    ./node_modules/@google-cloud
6.3M    ./node_modules/google-gax
5.2M    ./node_modules/eslint
4.9M    ./node_modules/lodash
4.6M    ./node_modules/katex
4.5M    ./node_modules/rollup
4.3M    ./node_modules/jsdom
4.1M    ./node_modules/es-abstract
3.6M    ./node_modules/es5-ext
3.2M    ./node_modules/prismjs
3.2M    ./node_modules/caniuse-lite
2.9M    ./node_modules/react-dom
2.6M    ./node_modules/table
2.3M    ./node_modules/@grpc
2.2M    ./node_modules/terser
2.1M    ./node_modules/terser-webpack-plugin
2.0M    ./node_modules/firebase-admin
1.9M    ./node_modules/axe-core
1.8M    ./node_modules/regenerate-unicode-properties
1.8M    ./node_modules/node-forge
1.8M    ./node_modules/@tailwindcss
1.5M    ./node_modules/language-subtag-registry
1.4M    ./node_modules/refractor
1.4M    ./node_modules/eslint-plugin-import
1.3M    ./node_modules/espree
1.3M    ./node_modules/eslint-plugin-react
1.3M    ./node_modules/eslint-plugin-jsx-a11y
1.3M    ./node_modules/acorn-node
1.2M    ./node_modules/node-libs-browser
1.2M    ./node_modules/acorn-globals
1.2M    ./node_modules/@hapi
1.1M    ./node_modules/rollup-plugin-terser
1.1M    ./node_modules/postcss
1.1M    ./node_modules/jest-worker
1.1M    ./node_modules/csstype
1.1M    ./node_modules/ajv
```

- `yarn why <packagename>`: 이 명령어를 사용하면, 해당 패키지가 왜 의존성 트리에 포함되어 있는지 알 수 있다. `npm ls <packagename>` 도 동일하다. 이를 사용하면 버전별로 어떤 패키지가 어떤 의존성으로 설치되어 있는지 알려준다.
- [Packagephobia](https://packagephobia.com/): 패키지가 대략 디스크에서 얼마나 차지하고 있는지 알 수 있다.

![packagephobia-eslint-config-yceffort](./images/packagephobia-eslint-config-yceffort.png)

> 이 패키지가 50메가가 넘다니,, 말세다

https://packagephobia.com/result?p=eslint-config-yceffort

- [Bundlephobia](https://bundlephobia.com/): 앱을 번들링 했을 때 패키지가 얼마나 커지는 지 확인할 수 있다.

![bundlephobia-react](./images/bundlephobia-react.png)

https://bundlephobia.com/package/react@17.0.2

### 2. 미사용 의존성 제거

어떻게 보면 정말 당연한 내용이지만, 많은 프로젝트에서 이러한 죽은 패키지들을 볼 수 있다. 제거하는 것보다는 설치하는게 쉽기 때문이다. 여러 의존성이 존재하고 있을 수도 있고, 사용하지 않는 종속성을 계속 업그레이드 하고 있을 수도 있다. 따라서 패키지에 나열 되어 있는 모든 기본 종속성을 살펴보는 것이 중요하다. 추천해주고 싶은 방법은, `package.json`을 보고, 모든 패키지를 하나씩 살펴본다음 사용하지 않는 것을 제거하는 것이다. 이는 보통 수동으로 하는 것이 좋다. 예를 들어, moment.js를 날리고 이제 date-fns를 사용한다고 가정하자. `moment`가 설치되어 있는지 확인하기 위해서는, 텍스트 에디터의 검색 기능을 사용하거나, [rg](https://github.com/BurntSushi/ripgrep) 패키지를 사용하여 아래 명령어를 날려보자.

```bash
rg '(require\(|from\s+)(?:"|\')moment'
```

그 다음엔 `moment`를 검색해서 현재 사용되고 있는 부분이 있는지 한번더 확인해본다. babel, eslint 또는 jest와 같은 도구의 일부 설정 파일이 해당 모듈을 사용하고 있을 수도 있다.

혹은 [eslint의 룰](https://eslint.org/docs/rules/no-restricted-imports)를 사용해서 패키지 import 시에 경고문을 날려버릴 수도 있다.

```javascript
"no-restricted-imports": [
    "error",
    {
        name: "moment",
        message:
            "moment has been deprecated. use date-fns instead.",
    },
],
```

아무튼, 이제 사용되지 않는 것이 확인되면 `package.json`에서 확실하게 제거하자. 위와 같은 방법을 `dependencies`, `devDependencies`에 반복해서 작업해주자. 또는 [depcheck](https://github.com/depcheck/depcheck)와 같은 도구를 사용해 볼 수도 있다.

### 3. 의존성을 최신화

혹시 '업그레이드 절벽'을 경험해본적이 있는가? 의존성 버전이 너무 뒤쳐져 업그레이드나 마이그레이션에 상당한 노력이 필요하고, 전체 개발 작업속도가 뒤쳐질 수가 있다. (나는 최근에는 husky를 쓰면서 느꼈다) 업그레이드는 조직 전체에 걸쳐 모든 사람의 지속적인 책임으로 하는 것이, 한사람에게 모두 맡겨버리는 것보다는 낫다. 모두가 같이 조금씩 움직여야, 변화를 깨는 것을 덜 주저하게 될 것이다. 한사람이 변화를 주도하는 것은 (리더가 아니라면 더더욱) 너무 힘들다.

모든 의존성을 최신으로 유지하면, 레거시 패키지에서 벗어나고 이후 작업을 더 수월하게 진행할 수 있다. `yarn outdated`나 `yarn upgrade-interactive`를 사용하여 의존성을 확인하면서 최신버전으로 업그레이드할 수 있다. 물론, 이 작업에는 변경사항을 확인하고, 버그 문제를 해결하는 작업이 필요하다. 이를 위해서는 신뢰도를 높일 수 있는 자동화된 테스트가 많을 수록 좋다. 변경사항을 제대로 파악하지 못하고 최신버전을 사용하기 위해 패키지 버전을 올리는 것 만큼 끔찍한 일은 없다.

### 4. 중복되는 패키지는 제거

일부 자바스크립트 패키지 관리자에서 사용되는 알고리즘이 의존성 그래프를 지속적으로 최적화 하지는 않는다. 우리는 `lock`파일이 동일 패키지의 여러버전을 `semver`가 목표로 하는 버전관리에 맞게 패키지를 설치할 책임이 있다고 가정한다. [yarn-deduplicate](https://github.com/atlassian/yarn-deduplicate) 를 사용하면 lock file을 한번더 최적화 할 수 있다. 기본적으로 패키지를 설치하고, 업데이트하거나, 제거할 때마다 `npx yarn-deduplicate yarn.lock`를 하는 것을 추천한다. 또는 CI 과정에 `yarn-deduplicate yarn.lock --list --fail`를 추가하여 지속적으로 이를 확인해볼 수 있다.

이 문제와 관련하여 가장많은 문제를 일으키는 곳은 babel, jest와 같은 모노레포로 이루어진 라이브러리다. 최악의 경우, 여러개의 babel parser, jest package 등이 존재할 수 있다. `yarn-deduplicate`를 사용하면 이러한 문제를 어느 정도 해결할 수는 있지만, 모든 패키지를 업데이트 할 수 있는 확실한 방법은 없다. 이를 해결하기 위해 시도해본 방법은

- 모노레포의 모든 `package.json`를 확인하여 최신버전으로 설치하는 방법
- `yarn upgrade` `yarn upgrade-interactive`
- [yarn resolution](https://classic.yarnpkg.com/en/docs/selective-version-resolutions/)을 활용하여 시멘틱 버전 제한을 덮어쓰고, 모든 패키지를 최신으로 관리

등이 있었다. 하지만 이들 중 어떤 것도 잘 해결하지 못했고 이따금 상황을 악화시키기도 했다. 프로젝트에 단일 babel 버전의 패키지를 설치할 수 있는 확실한 방법은, 수동으로 `yarn.lock`에서 모두 제거한 다음, 다시 처음부터 `@babel/`을 설치하여 의존성 그래프를 다시 그리게 하는 방법이다.

또다른 방법으로는, 패키지의 semver 범위를 직접 탐색하여 해당 패키지에 pull requests를 보내서 의존성을 업그레이드하거나버전을 변경하여 semver버전을 조정하는 것이다.

### 5. 단일 패키지를 명확한 목적에 따라 정리할 것

대규모 프로젝트에서는 동일한 용도의 여러 패키지가 사용될 수 있으며, 동일한 패키지의 여러 major 버전이 설치되어 있을 수도 있다. 규모가 큰 팀에서는, 누군가가 큰 의존성을 들고와서 고작 딱 한번 사용하고는, 번들 크기를 크게 부풀릴 수도 있다. 또는 비슷한 목적의 비슷한 크기의 작은 패키지를 여기저기서 사용하고 있을 수도 있다. 이를 해결하기 위해서는 엄격한 스타일 가이드, 문서화, 코드 리뷰등을 통해 이 문제를 방지할 수 있다. 그러나 최상으로 환경을 준비한다 한들 직접 의존성을 통해 포함된 유사한 기능의 패키지가 존재할 수 있다. 예를 들어, 두개의 패키지에 명령줄 옵션을 해석하는 동일한 기능이지만 다른 패키지가 설치되어 있을 수 있다. 따라서 어떤 패키지가 `node_modules`에 있는지를 분석하고, 각 목적에 따라 하나의 패키지로 정리하는 것이 좋다.

### 6. 필요에 따라 패키지를 포크하여 커스텀

어떤 경우에는 패키지가 너무 유지보수가 안되고 있거나, 너무 급격하게 발전하고 있을 수 있다. 내가 사용하고 있는 오픈소스의 패키지 릴리즈를 기다리기 위해 내 제품을 연기하는 것은 결코 바람직하지 못하다. 이를 해결하기 위한 좋은 방법은 적극적으로 패키지를 포크하여 사용하는 것이다. 포크가 오래 지속될 필요는 없다. 예를 들어, 어떤 작업에 대한 수정을 앞당기기 위해 포크를 하고, 이를 나중에 제거할 수 있다. 이렇게 하면 유지 보수 부담이 생길 수 는 있지만, 프로젝트에서 실행중인 코드의 제어권을 넘겨받을 수 있다.

[Yarn resolution](https://classic.yarnpkg.com/en/docs/selective-version-resolutions/)을 사용하여 기존 패키지를 포크 버전으로 교체할 수 있다.

```javascript
"resolutions": {
  "bloated-package": "npm:@yceffort/not-bloated-package",
  "unmaintained-package": "npm:@yceffort/well-maintained-package"
}
```

포크된 버전은 오래 살려두지 말고, 직접 PR을 날려주는 것이 좋다.

### 7. 의존성의 숫자와 크기를 계속 추적

의존성 크기와 숫자를 한번에 줄이면 좋겠지만, 그것보다는 지속적으로 관리하는 것이 좋다. 개인적으로는, CI 단계에서 `package.json`이나 `yarn.lock`의 변화가 있을 때마다 `node_modules`를 `du -sh node_modules`로 자동으로 확인하는 것이 좋다. PR단계에서 CI를 수행하고, 크기가 커졌다면, 한번쯤 눈길이 갈 것이다.

자동화를 통해 계속해서 확인할 수 있지만, 중요한 것은 동일한 코드베이스로 작업하는 다른 모든 사람들에게 대화하고 책임감을 공유하는 것이 최상의 결과를 가져오는데 도움이 된다. 의존성이 커지면 모든 사람의 작업속도가 느려지거나, 이미 사용하고 있는 유사한 패키지가 설치 될 수 있다는 것을 알리자. 대부분의 경우에는 많은 사람들이 감사하게 생각할 것이다.

예를 들어, 누군가가 `node_modules`의 크기를 두배로 늘리는 패키지를 추가헀다고 가정해보자. 간단하게 왜 이것이 옳지 않은지, 이상적이지 않은지를 설명하고 문제를 해결할 다른 두세가지 방법을 제시하기만 하면 또다시 PR을 만들 필요가 없이 해결할 수 있다. 누군가 100줄의 코드를 추가하면, 우리는 코드리뷰를 통해 꼼꼼하게 확인한다. 그러나 패키지에 한줄을 추가하는 것은 최악의 경우 엄청난 크기의 코드를 프로젝트로 가져오고 번들링을 부풀리는데, PR에서는 이게 어떤 영향을 미치는지 알 수 없기 때문에 순식간에 적용되어 버릴 수 있다. 써드파티 종속성에 대한 문제를 버전 컨트롤에서 지속적으로 확인하면 이런 문제를 사전에 방지할 수 있다.

### 다른 방법

yarn에는 [autoclean](https://classic.yarnpkg.com/en/docs/cli/autoclean/) 명령어가 있는데, 이를 통해 제외목록과 일치하는 파일을 자동으로 제거할 수 있다. 이를 통해 프로젝트와 관계없는 예제, 테스트, 마크다운 파일 등을 제거할 수 있다. `yarn autoclean --init`을 실행하고, `.yarnclean`파일을 확인하여 결과를 살펴볼 수 있다. 그러나 이 명령어는 설치하는 동안이 아니라 이미 의존성이 설치된 이후에 실행된다. 이는 yarn의 호출이 몇초 정도 느려질 수 있다는 것을 의미한다. 좋은 기능이지만, 버전 컨트롤에서 `node_modules`를 확인하는 프로젝트에만 사용하는 것이 좋다.
