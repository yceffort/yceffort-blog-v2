---
title: '[Rust] 자바스크립트에서 러스트로 - RustUp'
tags:
  - javascript
published: true
date: 2022-02-26 13:40:28
description: ''
---

## tools

rust에서 사용하는 대표적인 툴을 nodejs 입장에서 비교해 보았다.

- [nvm](https://github.com/nvm-sh/nvm) ➝ [rustup](https://rustup.rs/)
- `npm` ➝ [cargo](https://rustup.rs/) (rust package manager)
- `eslint`  ➝ [clippy](https://github.com/rust-lang/rust-clippy)
- `prettier` ➝ [rustfmt](https://github.com/rust-lang/rustfmt)

## rustup 설치 및 사용

가장먼저 할일은 [rustup](https://rustup.rs/)을 설치하는 것이다. 설치하는 방법은 간단하다.

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

기본으로 설치하면 알아서 잘 설치되는 것을 볼 수 있다. 몇가지 명령어를 사용해보자.

- `rustup show`: 현재 시스템에 설치된 러스트 버전을 알 수 있다.
- `rustup completions`: cli에서 tab 등으로 자동완서응ㄹ 할 수 있도록 도와주는 도구. `rustup completions zsh`를 입력하면 `zsh`에서 자동완성을 할 수 있도록 도와준다.
- `rustup update`: 가장 최신버전으로 업데이트 한다.
- `rustup install [version]`: 특정 버전, stable, nightly 버전 등으로 설치할 수 있다.

## npm에서 cargo로 전환하기

cargo는 앞서 언급했던 것처럼 npm과 비슷하게 rust세계에서 사용하는 패키지 매니저다. cargo는 [crates.io](https://crates.io/)에서 의존성을 다운로드 하고 설치한다. npmjs.com 와 동작방식이 유사한데, 개발자들이 가입해서 여기에 모듈을 업로드할 수도 있다. 쉽게 공부하기 위해서, `npm`과 `cargo`를 매핑하는 방식으로 이해해보자.

## npm vs cargo

### 프로젝트 세팅 파일

node.js에 `package.json`이 있다면 rust에는 `Cargo.toml`이 있다. 확장자에서 알 수 있는 것 처럼, `json` 형식이 아닌 `toml` 형식으로 되어 있다. 그다지 어려운 설정 파일이 아니므로, 파일 형태에 대한 설명을 생략한다. 여기에는 어떤 의존성을 다운로드할지, 테스트는 어떻게 할지, 빌드는 어떻게 할지 등을 나타낼 수 있다.

> https://doc.rust-lang.org/cargo/reference/manifest.html

### 프로젝트 시작하기

`npm init`과 유사하게 `cargo init`과 `cargo new`가 있다. `cargo init`은 현재 디렉토리에서, `cargo new`는 새로운 디렉토리에서 시작한다.

### 의존성 설치

`npm install [dep]`가 있다면, rust에는 `cargo add [dep]`이 있다. 이 명령어를 사용하기 위해서는 [cargo-edit](https://github.com/killercup/cargo-edit)을 설치해야 한다.

> $ cargo install cargo-edit

`cargo-edit`은 `add` `rm` `upgrade` `set-version`등을 지원한다.

> https://github.com/killercup/cargo-edit

### 글로벌하게 tool 설치

앞서 눈치챘을 수도 있지만, `npm install -g`는 `cargo install`과 같다. 