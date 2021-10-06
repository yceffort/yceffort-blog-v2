---
title: 'package.json에 쌓여있는 개발 부채'
tags:
  - javascript
published: true
date: 2021-10-06 22:01:12
description: '설치와 업데이트 시에는 신중에 신중을.'
---

## Table of Contents

## Introduction

npm은 자바스크립트 개발자에게 있어 한 줄기 빛 같은 도구다. 자바스크립트 프로젝트를 시작한다고 하면, 열에 아홉은 `npm init` 명령어와 함께 시작한다. 그렇게 생성된 `package.json`에 필요한 npm 패키지를 하나 둘 씩 설치해 나가다 보면 어느새 프로젝트가 완성되어 있다. `don't reinvent the wheel again` 이라는 개발의 오랜 격언 처럼, 개발자가 필요로 하는 자바스크립트 패키지들 대부분은 npm에 존재하고 그리고 손쉽게 설치한다. 그러나 설치는 쉽게 하지만, 쉽게 설치되는 만큼 그 안에 개발 부채가 쌓이고 있다는 사실은 다들 간과하고 있는 것 같다. 점점 커져가고 있는 `package.json`에서는 무슨 일이 일어날 수 있을까? 그리고 이를 방지하기 위해서는 어떻게 해야할까?

## 1. 설치하고자 하는 패키지의 dependencies를 파악하자

아래 package.json을 살펴보자.

```json
{
  "name": "sample",
  "dependencies": {
    "react-scripts": "^3.4.2"
  },
  "devDependencies": {
    "eslint": "^7.32.0"
  }
}
```

`create-react-app`으로 react 프로젝트를 시작한다면 `react-scripts`가 설치되어 있을 것이다. 그리고 코딩 컨벤션을 위해 `eslint`를 설치할 수도 있다. 그러나 위 dependencies는 한가지 문제가 있다.

```
➜  playground npm list eslint
playground@ /Users/yceffort/private/playground
├── eslint@7.32.0
└─┬ react-scripts@3.4.4
  ├─┬ @typescript-eslint/eslint-plugin@2.34.0
  │ ├─┬ @typescript-eslint/experimental-utils@2.34.0
  │ │ └── eslint@6.8.0 deduped
  │ └── eslint@6.8.0 deduped
  ├─┬ @typescript-eslint/parser@2.34.0
  │ └── eslint@6.8.0 deduped
  ├─┬ babel-eslint@10.1.0
  │ └── eslint@7.32.0 deduped
  ├─┬ eslint-config-react-app@5.2.1
  │ └── eslint@6.8.0 deduped
  ├─┬ eslint-loader@3.0.3
  │ └── eslint@6.8.0 deduped
  ├─┬ eslint-plugin-flowtype@4.6.0
  │ └── eslint@6.8.0 deduped
  ├─┬ eslint-plugin-import@2.20.1
  │ └── eslint@6.8.0 deduped
  ├─┬ eslint-plugin-jsx-a11y@6.2.3
  │ └── eslint@6.8.0 deduped
  ├─┬ eslint-plugin-react-hooks@1.7.0
  │ └── eslint@6.8.0 deduped
  ├─┬ eslint-plugin-react@7.19.0
  │ └── eslint@6.8.0 deduped
  └── eslint@6.8.0
```

`react-scripts@3.4.4`에서는 `eslint@6`를 사용 중인데, 이를 무시하고 `eslint@7`을 설치했을 경우 위와 같이 의도치 않는 패키지 트리를 생성하게 된다. `node_modules`에는 `eslint@7`이 설치될 것이고, [eslint의 7에는 breaking change](https://eslint.org/docs/user-guide/migrating-to-7.0.0)가 있기 때문에 잠재적으로 문제가 될 수 있다.

`react-scripts`를 `4.x`로 업데이트 하기로 결정했다고 가정해보자. 단순히 버전업만 하면 될까? 그렇지 않다. 버전업 하기전에는 패키지의 CHANGE LOG와 package.json에서의 dependencies의 변경에 주목해야 한다.

https://github.com/facebook/create-react-app/blob/main/CHANGELOG.md#migrating-from-34x-to-400

jest 버전이 24.x에서 26으로 업그레이드 된 것을 볼 수 있다. major 버전 업이기 때문에, jest를 사용하고 있는 테스트 코드에도 문제가 생길 수 있다.

- https://jestjs.io/blog/2020/01/21/jest-25
- https://jestjs.io/blog/2020/05/05/jest-26

그러나 놀랍게도 현재 시간 기준으로 jest의 최신 버전은 27 이다. https://jestjs.io/blog/2021/05/25/jest-27 최신버전을 포기하고 `react-scripts`의 버전에 의존할 것인가? 혹은 무시하고 최신버전을 설치할 것인가?

개발자가 설치하고자 하는 package의 dependencies는 항상 눈여겨 봐야 한다. 설치할 때 뿐만 아니라, major 버전업을 감행할 때 또한 주의를 기울여야 한다. 그리고 dependencies가 복잡한 패키지는 더더욱 설치할 때 주의를 기울여야 한다. 프로젝트에서 꼭 필요로 하는 package 인가? 전체 package의 버전이 여기에 좌우되도 괜찮은가?

## 2. peerDependencies 도 자세히 확인해보자

`peerDependencies`의 중요성을 알기전에, `peerDependencies`의 정의와 `dependencies`의 차이점을 알아야 한다.

> `peerDependencies`: In some cases, you want to express the compatibility of your package with a host tool or library, while not necessarily doing a require of this host. This is usually referred to as a plugin. Notably, your module may be exposing a specific interface, expected and specified by the host documentation.

> https://docs.npmjs.com/cli/v7/configuring-npm/package-json

`peerDependencies`란 실제로 패키지에서 `require`나 `import` 하지는 않지만, 특정 라이브러리나 툴에 호환성을 필요로 할 경우에 명시하는 dependencies다. npm3 부터 6까지는 `peerDependencies`가 자동으로 설치되지 않았고, 설령 버전이 맞지 않더라도 경고 문구만 뜰 뿐이었다. 그러나 npm@7 부터는 기본으로 설치되고, 이 버전이 맞지 않으면 에러도 발생한다.

> In npm versions 3 through 6, peerDependencies were not automatically installed, and would raise a warning if an invalid version of the peer dependency was found in the tree. As of npm v7, peerDependencies are installed by default.

```json
{
  "name": "playground",
  "dependencies": {
    "react": "16.8.6"
  },
  "devDependencies": {
    "@testing-library/react-hooks": "^7.0.2"
  }
}
```

위 `package.json`을 살펴보자. react 버전은 16.8.6으로 고정되어 있고, `@testing-library/react-hooks`를 설치하려고 시도하고 있다. 그러나 [`@testing-library/react-hooks`는 `peerDependencies`로 `react@>=16.9`를 요구하기 때문](https://github.com/testing-library/react-hooks-testing-library/blob/565c9f80ff969c3b9f20d8b2efdc033996d9ec27/package.json#L78)에 아래와 같이 npm@7 환경에서는 설치가 되지 않는다.

```
➜  playground npm install
npm ERR! code ERESOLVE
npm ERR! ERESOLVE could not resolve
npm ERR!
npm ERR! While resolving: @testing-library/react-hooks@7.0.2
npm ERR! Found: react@16.8.6
npm ERR! node_modules/react
npm ERR!   react@"16.8.6" from the root project
npm ERR!
npm ERR! Could not resolve dependency:
npm ERR! peer react@">=16.9.0" from @testing-library/react-hooks@7.0.2
npm ERR! node_modules/@testing-library/react-hooks
npm ERR!   dev @testing-library/react-hooks@"^7.0.2" from the root project
npm ERR!
npm ERR! Conflicting peer dependency: react@17.0.2
npm ERR! node_modules/react
npm ERR!   peer react@">=16.9.0" from @testing-library/react-hooks@7.0.2
npm ERR!   node_modules/@testing-library/react-hooks
npm ERR!     dev @testing-library/react-hooks@"^7.0.2" from the root project
npm ERR!
npm ERR! Fix the upstream dependency conflict, or retry
npm ERR! this command with --force, or --legacy-peer-deps
npm ERR! to accept an incorrect (and potentially broken) dependency resolution.
npm ERR!
npm ERR! See /Users/yceffort/.npm/eresolve-report.txt for a full report.

npm ERR! A complete log of this run can be found in:
npm ERR!     /Users/yceffort/.npm/_logs/2021-10-06T14_57_04_722Z-debug.log
```

반대로 npm@6 환경에서는 그냥 경고문구만 뜨는 것을 확인할 수 있다.

```
➜  playground npx npm@6 install
npm WARN @testing-library/react-hooks@7.0.2 requires a peer of react@>=16.9.0 but none is installed. You must install peer dependencies yourself.
npm WARN react-error-boundary@3.1.3 requires a peer of react@>=16.13.1 but none is installed. You must install peer dependencies yourself.
npm WARN playground@ No description
npm WARN playground@ No repository field.
npm WARN playground@ No license field.
```

npm@6 환경에서 `peerDependencies`가 단순히 경고 문구만 내뱉고, 설치는 잘된다 하더라도 이를 간과해서는 안된다. 패키지 마다 `peerDependencies`를 버전에 맞게 선언하는데는 이유가 있을 것이고, 이를 지키지 않아서 발생할 수 있는 잠재적인 문제는 모두 오롯이 개발자가 안게 된다.

## 3. 살아있는 패키지를 설치해라

프로젝트에서 꼭 필요로 하는 패키지를 찾았다고 가정해보자. 우리가 필요로 하는 기능도 있고, 사용하기에도 어렵지 않다. 그리고 사용해보니 별 문제도 없었다. 그렇다면 설치해도 될까? 그렇지 않다. 장기적으로 서비스를 안정적으로 관리하기 위해서는 살아있는 패키지, 즉 활발하게 업데이트나 피드백이 오가는 패키지를 설치해야 한다.

우리 프로젝트에서는 [react-swipe-views](https://github.com/sanfilippopablo/react-swipeable-routes)라고 하는 패키지를 설치하여 사용했다. 사용 당시 까지만 하더라도 크게 이슈는 없었지만, 문제는 몇가지 개선사항이 필요해지면서 시작되었다. 수정이 필요한 코드도 찾았고, PR도 열어두었다. 이제 메인테이너의 머지와 버전업만 기다리고 있었는데... 문제는 해당 패키지가 더이상 관리가 되고 있지 않다는 것이었다. 패키지가 살아있기 위해서는 해당 패키지가 꾸준히 관리되고 활발하게 버전업되어야 하지만 그렇지 않으면 죽은 패키지가 되어 버린다. 물론 해당 패키지를 fork하여 개선하는 방법도 있지만 메인테이너의 도움 없이 패키지 코드를 읽고 원하는 형태로 동작하게 만들기 위해서는 상당한 노력이 필요하다.

이는 우리가 오픈소스를 사용하면서 발생하는 일종의 '빚'이라고 생각한다. 관리되고 유지보수 되면 좋지만, 그렇지 않더라도 비난할 수는 없다. 그것이 오픈소스 생태계 이기 때문에.

> 메인테이너를 찾고 있지만, 생각만큼 잘되고 있는 것 같지는 않다.
> https://github.com/oliviertassinari/react-swipeable-views/issues/558

이러한 일들을 막기 위해서는, 다수의 사용자가 사용하고 있는 패키지, star가 많은 패키지를 선택하는 것이 첫번째 방법이다.

![react-swipeable-views](./images/react-swipeable-views.png)

> `react-swipeable-views` 도 4k star에 used 30.6k 이건만,,

그리고 이 패키지를 운영하고 있는 주체가 누구인지도 확인해보고, 메인스트림 브랜치 기준으로 마지막 commit, PR, issue closed 등을 확인해보는 것도 필요하다. 최근까지 패키지가 '살아있다'는 증거를 찾았는가? 그렇다면 설치해도 좋다. 그렇지 않다면 장기적인 관점으로 봤을 때 설치를 재고해봐야 한다.

## 4. 핵심 라이브러리는 직접 구현하자

npm에서 제공되는 다양한 오픈소스 패키지를 활용하여 프로젝트를 꾸미는 것도 좋지만, 프로젝트의 핵심이 되는 코어 기능들은 외부 패키지에 의존하는 것 보다는 자체적으로 만들어서 제공하는 것이 장기적으로 안정적으로 서비스를 운영하는데 도움이된다. 우리가 사용하고 있는 npm 패키지들은 모두 오픈소스임을 명시해야 한다. 오픈 소스이기에 (라이센스만 준수한다면) 무료로 쉽게 이용할 수 있지만, 반대로 언제든지 오픈소스 프로젝트가 중단될 가능성도 존재한다.

> Babel is used by millions, so why are we running out of money?

> ... So, our ask is to help fund our work, via Open Collective and GitHub Sponsors. Though individual contributions do matter (and we deeply appreciate them), we are really looking for more companies to step up and become corporate sponsors, alongside our current sponsors like AMP, Airbnb, Salesforce, GitPod, and others. If it would be better for your company to sustain us in other ways, we are also open to hearing any ideas. Reach out to us directly or by email at team@babeljs.io.

> 프론트엔드 개발자라면 당연히 한번씩은 다 써봤을 babel도 현재 자금난에 시달리고 있다. (그 와중에 눈에 띄는 월 11,000달러 급여...)

> https://babeljs.io/blog/2021/05/10/funding-update

반대로 멀쩡히 잘 쓰고 있던 패키지가 어느 순간 라이센스 정책의 변화로 인해 사용이 어려워 질 수도 있다.

> If you’re a startup, you should not use React (reflecting on the BSD + patents license)

> https://medium.com/@raulk/if-youre-a-startup-you-should-not-use-react-reflecting-on-the-bsd-patents-license-b049d4a67dd2

> 3~4년 전쯤, 리액트가 갑자기 라이센스 정책을 바꾸려고 시도했던 적이 있다. 물론 이는 오픈소스 커뮤니티의 반발로 인해 무산되었다.

물론 프로젝트 전반에 있는 모든 라이브러리, 패키지들을 다 걷어내고 0에서 구현하자는 것은 아니다. 하지만 프로젝트의 핵심 기능, npm 에 존재하는 패키지만으로는 커버가 어려운 기능 등은 직접 구현해서 가지고 있는 것이 좋다. 물론, 내부적으로 패키지화해서 관리한다면 더 좋다. 이러한 이유 때문에 많은 회사들이 내부적으로 관리하는 패키지를 도입하고 있는 추세다. [private package](https://docs.npmjs.com/creating-and-publishing-private-packages)로 관리해도 좋고, 또 오픈소스 커뮤니티의 건강한 성장을 위해 보안상의 위협만 없다면 직접 세상에 알리는 것도 도움이 될 것이다. 그리고 이러한 경험이 개발자를 한단계 성장시키는데 큰 도움이 되리라 믿어 의심치 않는다.
