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
