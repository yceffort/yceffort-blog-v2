---
title: 'package.json에 쌓여있는 개발 부채'
tags:
  - javascript  
published: true
date: 2021-10-06 22:01:12
description: '설치할 땐 즐겁지만, 어느 순간 부메랑으로...'
---

## Introduction

npm은 자바스크립트 개발자에게 있어 한 줄기 빛 같은 도구다. 자바스크립트 프로젝트를 시작한다고 하면, 열에 아홉은 `npm init` 명령어와 함께 시작한다. 그렇게 생성된 `package.json`에 필요한 npm 패키지를 하나 둘 씩 설치해 나가다 보면 어느새 프로젝트가 완성되어 있다. `don't reinvent the wheel again` 이라는 개발의 오랜 격언 처럼, 개발자가 필요로 하는 자바스크립트 패키지들 대부분은 npm에 존재하고 그리고 손쉽게 설치한다. 그러나 설치는 쉽게 하지만, 쉽게 설치되는 만큼 그 안에 개발 부채가 쌓이고 있다는 사실은 다들 간과하고 있는 것 같다. 점점 커져가고 있는 `package.json`에서는 무슨 일이 일어날 수 있을까? 그리고 이를 방지하기 위해서는 어떻게 해야할까?

## 1. 패키지의 dependencies를 잘 파악하자

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

## 2. 살아있는 패키지를 설치해라

프로젝트에서 꼭 필요로 하는 패키지를 찾았다고 가정해보자. 우리가 필요로 하는 기능도 있고, 사용하기에도 어렵지 않다. 그리고 사용해보니 별 문제도 없었다. 그렇다면 설치해도 될까? 그렇지 않다. 장기적으로 서비스를 안정적으로 관리하기 위해서는 살아있는 패키지, 즉 활발하게 업데이트나 피드백이 오가는 패키지를 설치해야 한다.

우리 프로젝트에서는 [react-swipe-views](https://github.com/sanfilippopablo/react-swipeable-routes)라고 하는 패키지를 설치하여 사용했다. 사용 당시 까지만 하더라도 크게 이슈는 없었지만, 문제는 몇가지 개선사항이 필요해지면서 시작되었다. 수정이 필요한 코드도 찾았고, PR도 열어두었다. 이제 메인테이너의 머지와 버전업만 기다리고 있었는데... 문제는 해당 패키지가 더이상 관리가 되고 있지 않다는 것이었다. 패키지가 살아있기 위해서는 해당 패키지가 꾸준히 관리되고 활발하게 버전업되어야 하지만 그렇지 않으면 죽은 패키지가 되어 버린다. 물론 해당 패키지를 fork하여 개선하는 방법도 있지만 메인테이너의 도움 없이 패키지 코드를 읽고 원하는 형태로 동작하게 만들기 위해서는 상당한 노력이 필요하다. 

이는 우리가 오픈소스를 사용하면서 발생하는 일종의 '빚'이라고 생각한다. 관리되고 유지보수 되면 좋지만, 그렇지 않더라도 비난할 수는 없다. 그것이 오픈소스 생태계 이기 때문에. 

> 메인테이너를 찾고 있지만, 생각만큼 잘되고 있는 것 같지는 않다.
> https://github.com/oliviertassinari/react-swipeable-views/issues/558

이러한 일들을 막기 위해서는, 다수의 사용자가 사용하고 있는 패키지, star가 많은 패키지를 선택하는 것이 첫번째 방법이다.

![react-swipeable-views](./images/react-swipeable-views.png)

> `react-swipeable-views` 도 4k star에 used 30.6k 이건만,,

그리고 이 패키지를 운영하고 있는 주체가 누구인지도 확인해보고, 메인스트림 브랜치 기준으로 마지막 commit, PR, issue closed 등을 확인해보는 것도 필요하다. 최근까지 패키지가 '살아있다'는 증거를 찾았는가? 그렇다면 설치해도 좋다. 그렇지 않다면 장기적인 관점으로 봤을 때 설치를 재고해봐야 한다.

