---
title: 'npm, yarn, pnpm 비교해보기'
tags:
  - javascript
  - npm
  - yarn
  - pnpm
published: true
date: 2022-05-20 22:26:01
description: '그리고 승자는 🤔'
---

## Table of Contents

## Introduction

npm 에서 시작한 node package management의 역사는, 이제 3가지 옵션이 주어져 있다. yarn 1.0 (이제 yarn classic 이라고 부르겠다) 과 yarn 2.0 (yarn berry) 두 가지 버전도 사뭇 다른 점이 많다는 것을 감안한다면, 이제 크게 4가지 선택지가 존재 한다고 볼 수 있다.

그리고 위 3가지 패키지 관리자들은 아래와 같은 기본적인 기능 (node 모듈을 설치하고, 관리하는 등)을 제공하고 있다.

- metadata 작성 및 관리
- 모든 dependencies 일괄 설치 또는 업데이트
- dependencies 추가, 업데이트, 삭제
- 스크립트 실행
- 패키지 퍼블리쉬
- 보안 검사

따라서 설치 속도나 디스크 사용량, 또는 기존 워크 플로우 등과 어떻게 매칭 시킬지와 같은 기능 외적인 요구 사항에 따라 패키지 관리자를 선택하는 시대가 도래했다고 볼 수 있다.

겉으로는 기능적으로 비슷해보이고 무엇을 선택하든 별 차이는 없어보이지만, 패키지 관리자들의 내부 동작은 매우 다르다. npm 과 yarn의 경우 flat 한 node_modules 폴더에 dependencies 를 설치했다. 그러나 이러한 전략은 비판에서 자유롭지 못하다. (어떤 문제인지는 뒤에서 설명하도록 한다.)

그래서 등장한 pnpm은 이러한 dependencies를 중첩된 node_modules 폴더에 효율적으로 저장하기 시작했고, yarn berry는 plug and play (pnp) 모드를 도입하여 이러한 문제를 해결하기 시작했다.

이 세가지 패키지 관리자는 각각 어떤 특징과 역사를 가지고 있으며, 무엇을 선택해야할까?

- [npm](https://www.npmjs.com/)
- [yarn classic](https://classic.yarnpkg.com/lang/en/)
- [yarn berry](https://github.com/yarnpkg/berry)
- [pnpm](https://pnpm.io/ko/)

> pnpm 홈페이지에 있는 yarn과 npm을 쓰레기통에 쳐박은 이미지가 매우 인상적이다. 🤔 vue가 react를 쓰레기통에 쳐박는 이미지를 달아놨다면...

## 자바스크립트 패키지 관리자의 역사

모두가 잘 알고 있는 것 처럼 최초의 패키지 매니저는 2010년 1월에 나온 npm 이다. npm 은 패키지 매니저가 어떤 동작을 해야하는 지에 대한 핵심적인 개념을 잡았다고 볼 수 있다.

10여년이 넘는 시간 동안 npm 이 존재했는데, yarn, pnpm 등이 등장하게 된 것일까?

- `node_modules` 효율화를 위한 다른 구조 (nested vs flat, node_modules, vs pnp mode)
- 보안에 영향을 미치는 호이스팅 지원
- 성능에 영향을 미칠 수 있는 `lock`파일 형식
- 디스크 효율성에 영향을 미치는 패키지를 디스크에 저장하는 방식
- 대규모 모노레포의 유지 보수성 과 속도에 영향을 미치는 workspace라 알려진 멀티 패키지 관리 및 지원
- 새로운 도구와 명령어 관리에 대한 관리
  - 이와 관련된 다양하고 확장가능한 플러그인과 커뮤니티 툴
- 다양한 기능 구현 가능성과 유연함

npm 이 최초로 등장하 이래로 이러한 니즈가 어떻게 나타났는지, yarn classic은 그 이후 등장해서 어떻게 해결햏ㅅ는지, pnpm이 이러한 개념을 어떻게 확장했는지, yarn berry가 전통적인 개념과 프로테스에 의해 설정된 틀을 깨기 위해 어떠한 노력을 했는지 간략한 역사를 파악해보자.

### 선구자 npm

본격적으로 시작하기에 앞서 재밌는 사실을 이야기 해보자면, `npm`은 `node package manager`의 약자가 아니다. npm의 전신은 사실 `pm`이라 불리는 bash 유틸리티인데, 이는 `pkgmakeinst`의 약자다. 그리고 이의 node 버전이 `npm`인 것이다.

> [https://github.com/npm/cli#is-npm-an-acronym-for-node-package-manager](https://github.com/npm/cli#is-npm-an-acronym-for-node-package-manager)

npm 이전에는 프로젝트의 dependencies를 수동으로 다운로드하고 관리하였기 때문에 엄청난 혁명을 가져왔다고 볼 수 있다. 이와 더불어 메타데이터를 가지고 있는 `package.json`와 같은 개념, dependencies를 `node_modules`라 불리는 폴더에 설치한다는 개념, 커스텀 스크립트, public & private 패키지 레지스트리와 같은 개념들 모두 npm에 의해 도입되었다.

### 많은 혁명을 가져온 yarn classic

[2016년의 블로그 글](https://engineering.fb.com/2016/10/11/web/yarn-a-new-package-manager-for-javascript/)에서, 페이스북은 구글과 몇몇 다른 개발자들과 함께 npm이 가지고 있던 일관성, 보안, 성능 문제 등을 해결하기 위한 새로운 패키지 매니저를 만들기 위한 시도를 진행 중이라고 발표 했다. 그리고 이듬해 `Yet Another Resource Negotiator`의 약자인 yarn을 발표했다.

yarn은 대부분의 개념과 프로세스에 npm을 기반으로 설계했지만, 이외에 패키지 관리자 환경에 큰 영향을 미쳤다. npm과 대조적으로, yarn은 초기버전의 npm의 주요 문제점 중 하나였던 설치 프로세스의 속도를 높이기 위해 작업을 병렬화 하였다.

yarn은 dx(개발자 경험), 보안 및 성능에 대한 기준을 높였으며, 다음과 같은 개념을 패키지 매니저에 도입하였다.

- native 모노레포 지원
- cache-aware 설치
- 오프라인 캐싱
- lock files

yarn classic은 2020년 부터 유지보수 모드로 전환되었다. 그리고 1.x 버전은 모두 레거시로 간주하고 yarn classic으로 이름이 바뀌었다. 현재는 yarn berry에서 개발과 개선이 이루어지고 있다.

### pnpm 빠르고 휴올적인 디스크 관리

pnpm은 2017년에 만들어졌으며, npm 의 drop-in replacement(설정을 바꿀 필요 없이 바로 사용가능하며, 속도와 안정성 등 다양한 기능 향상이 이루어지는 대체품) 으로, npm만 있다면 바로 사용할 수 있다.
