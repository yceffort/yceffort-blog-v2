---
title: Webpack Module Federation에 대해 알아보자
tags:
  - javascript, webpack
published: false
date: 2020-09-15 18:52:29
description: ''
category: javascript, webpack
template: post
---

[이 글](https://webpack.js.org/concepts/module-federation/)을 위주로 번역한 글이며, 추가적으로 micro frontend에 대한 개념도 넣어두었습니다.

```toc
tight: false,
from-heading: 2
to-heading: 3
```

## Motivation

여러개로 쪼개진 빌드들을 모으면 하나의 단일 어플리케이션으로 구성할 수 있다. 이러한 여러 빌드들이 서로 의존성을 갖고 있지 않는다면, 이들을 개별적으로 만들어서 빌드하고 배포하는 것이 가능할 것이다.

이는 종종 [마이크로 프론트엔드](https://micro-frontends.org/)라고도 알려져 있지만, 여기에서 말하는 것은 단순히 마이크로 프론트엔드에 그치지 않는다.

## Low-level concepts

우리는 로컬과 리모트 모듈을 구분할 수 있다. 로컬 모듈은 일반적인 모듈로, 현재 빌드의 일부분을 의미한다. 반면 리모트 모듈은 현재 빌드의 일부분은 아니지만, 런타임 단계에서 로딩되는 일종의 컨테이너라고 볼 수 있다.

리모트 모듈을 불러오는 것은 비동기 동작으로 이해할 수 있다. 리모트 모듈을 사용할때, 이러한 비동기 작업은 리모트 모듈과 진입점(entrypoint) 사이에 있는 다음 청크를 불러오는 작업에 배치된다. 청크를 불러오는 작업 없이는 리모트 모듈을 사용할 수 없다.

청크를 불러오는 작업은 보통 `import()`를 호출하는 것이 일반적이지만, 과거에는 `require.ensure`나 `require([...])` 와 같은 동작도 존재했었다.

컨테이너 entry를 통해서 컨테이너가 생성되며, 이는 특정 모듈에 대한 비동기 엑세스를 노출한다. 이러한 노출에 접근하는 방법은 두 단계로 구분할 수 있다.

1. 모듈 로딩 (비동기)
2. 모듈 실행 (동기)

1단계는 청크를 로딩하는 순간에 완료될 것이다. 2단계는 서로 다른 (로컬 및 리모트) 모듈을 끼워 넣는 단계에서 완료될 것이다. 이러한 방식을 활용하면, 모듈을 리모트에서 로컬로 변환한거나, 혹은 다른 방식으로 변환하더라도 실행 순서에는 영향 받지 않게 될 것이다.

이는 컨테이너를 내재화 하는 것을 가능케 한다. 컨테이너들은 다른 컨테이너에 있는 모듈을 활용할 수 있다. 컨테이너간 순환참조 또한 가능해진다.

### Overriding

컨테이너는 선택된 로컬 모듈들을 `overridable`한 것으로 지정할 수 있다.

## High-level concepts

## Building Blocks

### Overridables Plugin

### Container Plugin

### ContainerReferencePlugin

### ModuleFederationPlugin

## Concept goals

### Use cases

### Separate build per page

### Components library as container

## Dynamic Remote Containers

## TroubleShooting
