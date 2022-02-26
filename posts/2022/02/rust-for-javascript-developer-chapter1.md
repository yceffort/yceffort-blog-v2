---
title: '[Rust] 자바스크립트에서 러스트로 - RustUp'
tags:
  - rust
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

### 테스트

`npm test`는 `cargo test`와 같다. `cargo test`를 거치면 유닛테스트, 통합 테스트, 문서화 테스트를 자동으로 실행하게 된다. 

### 모듈 publish

`npm publish`는 `cargo publish`와 같다. 앞서 언급했던 것 처럼, [crates.io](https://crates.io/) 계정과 인증이 필요하다.

### 그밖에 작업 실행하기

그밖에 cargo에서 대응되는 작업은 다음과 같다.

- `npm run start`: `cargo run`
- `npm run benchmarks`: `cargo bench`
- `npm run build`: `cargo build`
- `npm run clean`: `cargo clean` 이 작업을 실행하면 `target` 폴더를 청소한다.
- `npm run docs`: `cargo doc`

그외의 경우에는 rust 개발자가 개별적으로 대응해야 한다.

## 그밖에 다른 도구들

### `cargo-edit`

`cargo-edit` 는 앞서 언급했던 것 처럼 `cargo add` `cargo rm`과 같은 명령어를 가능하게 해준다.

### `cargo-workspaces`

cargo-workspaces는 워크스페이스를 만들고 관리할 수 있도록 도와주는 도구다. 이는 node의 lerna에 영감을 받아 만들어졌다. 여기에는 패키지 자동 publish, local 의존성을 publish 버전으로 대체하는 등 다양한 도구를 제공한다.

## VSCode에서 설치하면 도움이되는 도구들

- https://marketplace.visualstudio.com/items?itemName=rust-lang.rust
- https://marketplace.visualstudio.com/items?itemName=matklad.rust-analyzer
- https://marketplace.visualstudio.com/items?itemName=vadimcn.vscode-lldb (debug)
- https://marketplace.visualstudio.com/items?itemName=bungcip.better-toml
- https://marketplace.visualstudio.com/items?itemName=serayuzgur.crates
- https://marketplace.visualstudio.com/items?itemName=belfz.search-crates-io

## Hello World

자, 이제 hello world를 작성해보자.

```bash
cargo new my-app
```

기본값으로, `cargo new`는 바이너리 애플리케이션 템플릿을 사용한다. 코드를 실행 한뒤에는, 아래와 같은 디렉토리 구조를 볼 수 있다.

```
my-app/
├── .git
├── .gitignore
├── Cargo.toml
└── src
  └── main.rs
```

`cargo run`을 실행해보자.

```bash
» cargo run
  Compiling my-app v0.1.0 (./my-app)
  Finished dev [unoptimized + debuginfo] target(s) in 0.89s
  Running `target/debug/my-app`
Hello, world!
```

`cargo run`은 `cargo build`를 실행하여 애플리케이션을 빌드하고, 그리고 실행한다. 빌드된 바이너리는 `./target/debug/my-app`에서 확인할 수 있다. 실행 없이 빌드만 하고 싶다면, `cargo build`를 실행하면 된다. 기본적으로, 빌드는 `dev` 프로파일에서 실행되기 떄문에 파일의 크기, 성능과 같은 디버그에 유용한 정보를 얻을 수 있다. 실제 프로덕션에 필요한 프로그램을 얻기 위해서는 `cargo build --release`를 실행하면 되고, 해당 결과는 `./target/release/my-app`에 위치한다.

`src/main.rs`를 살펴보자.

```rust
fn main() {
  println!("Hello, World!")
}
```

음 별다르게 특이한건 없다. 🤔

- `main()`은 단독 실행 되는 애플리케이션을 만들 때 필요한 함수다. cli app의 시작지점이 된다.
- `println!()`는 받은 인수를 STDOUT해주고 있다. 
- `"Hello, world!"`는 string이다. 


### 자바스크립트와 다른 것 1

먼저 앞선 string을 변수에 넣어서 실행해보자. rust도 마찬가지로 변수를 선언할때 `let`을 쓴다. 자바스크립트 세계엔 `let` `const`가 있고, 대부분 `const`를 쓰지만, rust는 대부분 `let`을 쓴다.

