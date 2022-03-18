---
title: 'Rust로 web assembly로 게임 만들어보기 (1)'
tags:
  - web
  - javascript  
  - rust
published: true
date: 2022-03-18 23:56:56
description: '아이고야'
---

## Table of Contents

## Introduction

이 튜토리얼은 https://rustwasm.github.io/docs/book/game-of-life/introduction.html 에서 제공하는 Rust WebAssembly로 만드는 Game of Life 을 기반으로 작성되었습니다. 직접 튜토리얼을 따라하면서 단순히 번역 이외에도 최신 라이브러리 버전 기준으로 재작성하였으며, 설명이 부족하거나 생략된 부분에 대해서도 별도로 주석을 달았습니다. 

기본적으로 러스트에 대한 완벽한 이해를 기반으로 하지 않고 작성되었기 때문에, 일부 러스트 문법에 대한 설명이 적혀있을 수도 있습니다.

## Game of Life?

[라이프 게임, 또는 생명 게임](https://ko.wikipedia.org/wiki/%EB%9D%BC%EC%9D%B4%ED%94%84_%EA%B2%8C%EC%9E%84)은 처음에 입력된 초기값을 기준으로 알아서 시작되는 게임이다. 

이 게임은 무한한 개수의 사각형 (이하 세포)로 이루어진 격자위에서 실행된다. 각 세포 주위에는 8개의 이웃 세포가 있으며, 각 세포는 살아있거나 죽어있는 상태를 가진다. 그리고 이 세포의 다음 상태는 다음과 같이 결정된다.

- 죽은 세포 이웃 중 딱 세개 가 살아 있으면 살아난다.
- 살아 있는 세포 중 이웃이 두 개나 세개가 살아있으면 그 세포는 살아있고, 그외에는 죽는다.

## 1. Setup

시작에 앞서 설치해야 하는 기본 언어와 라이브러리는 다음과 같다.

- `rust`
- `rustup`
- `cargo`
- [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/)
- [cargo-generate](https://github.com/ashleygwilliams/cargo-generate): 이 라이브러리는 이미 존재하는 깃 저장소를 기본 템플릿으로 사용하여 빠르게 러스트 프로젝트를 만드는데 도움을 준다.
- `npm`


## 2. Hello, World

### 프로젝트 클론

`wasm-pack` 을 기반으로 한 프로젝트를 빠르게 만들기 위해, 기본적으로 제공하는 프로젝트 템플릿이 존재하는데, 이를 기본으로 프로젝트를 만든다.

```shell
cargo generate --git https://github.com/rustwasm/wasm-pack-template
```

그리고 게임 이름을 입력한다. `wasm-game-of-life`


### 내부 살펴보기

