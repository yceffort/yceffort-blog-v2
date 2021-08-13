---
title: '비동기 리소스 (async resources)와 비동기 훅 (async hooks) 이해하기'
tags:
  - javascript
  - nodejs
published: true
date: 2021-08-14 00:20:56
description: '비동기로 불타는 금요일'
---

## Table of Contents

## Introduction

실제 우리가 사용하는 nodejs 애플리케이션은 비동기 작업, 그리고 이로 인해 만들어지고 없어지기를 반복하는 비동기 리소스 등으로 인해 매우 복잡하게 운영되고 있을 수도 있다. 때문에 코드에서 이러한 비동기 리소스의 라이프 사이클을 확인하는 기능은 애플리케이션에 대하나 통찰력, 그리고 실제 실행 가능한 성능 및 잠재적인 최적화 정보를 제공할 수 있기 때문에 매우 유용할 수 있다.

이러한 고급기능을 사용하기 위해 [AsyncListener](https://github.com/nodejs/node-v0.x-archive/pull/6011)라 든가 [async_wrap](https://github.com/nodejs/node-v0.x-archive/commit/709fc160e5) 와 같은 것들을 통해 많은 시도가 있었다. 그리고 감사하게도, 이제는 비동기 리소스의 라이프 사이클을 추적할 수 있는, 매우 성숙하나 기능인 `async_hooks`을 갖게 되었다. 지금 2021-08-14 00:26:43 을 기준으로도 여전히 실험 단계 이긴 하지만 (nodejs 16.6.2 기준), 꽤 많이 다듬어졌고, 형태를 갖추고 있다.

`async_hooks`는 다양한 기능을 가지고 있지만, 그 중에 가장 흥미로운 것은 파일 읽기, http 요청, http server 생성, 데이터 베이스에 쿼리 수행과 같은 응용프로그램에서 자주 수행하는 작업에서 어떤 일들이 벌어지는지 쉽게 이해할 수 있다는 것이다.

같이 보면 좋은 글 들

- https://nodejs.org/api/async_hooks.html
- https://itnext.io/a-pragmatic-overview-of-async-hooks-api-in-node-js-e514b31460e9

## 비동기 리소스의 생명주기