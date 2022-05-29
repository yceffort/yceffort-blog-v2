---
title: '[Rust] 자바스크립트에서 러스트로 (1) - rustup, hello world, 그리고 소유권과 빌림'
tags:
  - rust
published: true
date: 2022-02-26 13:40:28
description: 'Rust 공부해보기 (1)'
---

## Table of Contents

## tools

rust에서 사용하는 대표적인 툴을 nodejs 입장에서 비교해 보았다.

- [nvm](https://github.com/nvm-sh/nvm) ➝ [rustup](https://rustup.rs/)
- `npm` ➝ [cargo](https://rustup.rs/) (rust package manager)
- `eslint` ➝ [clippy](https://github.com/rust-lang/rust-clippy)
- `prettier` ➝ [rustfmt](https://github.com/rust-lang/rustfmt)

## rustup 설치 및 사용

가장먼저 할일은 [rustup](https://rustup.rs/)을 설치하는 것이다. 설치하는 방법은 간단하다.

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

기본으로 설치하면 알아서 잘 설치되는 것을 볼 수 있다. 몇가지 명령어를 사용해보자.

- `rustup show`: 현재 시스템에 설치된 러스트 버전을 알 수 있다.
- `rustup completions`: cli에서 tab 등으로 자동완성을 할 수 있도록 도와주는 도구. `rustup completions zsh`를 입력하면 `zsh`에서 자동완성을 할 수 있도록 도와준다.
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

`cargo run`은 `cargo build`를 실행하여 애플리케이션을 빌드하고, 그리고 실행한다. 빌드된 바이너리는 `./target/debug/my-app`에서 확인할 수 있다. 실행 없이 빌드만 하고 싶다면, `cargo build`를 실행하면 된다. 기본적으로, 빌드는 `dev` 프로파일에서 실행되기 때문에 파일의 크기, 성능과 같은 디버그에 유용한 정보를 얻을 수 있다. 실제 프로덕션에 필요한 프로그램을 얻기 위해서는 `cargo build --release`를 실행하면 되고, 해당 결과는 `./target/release/my-app`에 위치한다.

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

`let`을 사용하여 변수를 할당해서 사용해보자.

```rust
fn main() {
  let message = "Hello, World!";
  println!(message)
}
```

```shell
@yceffort ➜ /workspaces/rust-playground/chapter1/hello_cargo (main ✗) $ cargo run
   Compiling hello_cargo v0.1.0 (/workspaces/rust-playground/chapter1/hello_cargo)
error: format argument must be a string literal
 --> src/main.rs:3:14
  |
3 |     println!(message)
  |              ^^^^^^^
  |
help: you might be missing a string literal to format with
  |
3 |     println!("{}", message)
  |              +++++

error: could not compile `hello_cargo` due to previous error
```

자바스크립트 개발자의 시선에서는 동작해야할 코드였던 것 같은데, 동작하지 않았다. 대부분의 언어에서는 잘 동작할 코드일 것 같은데, 러스트는 그렇지 않다. 에러 메시를 일단 잘 살펴보자.

> format argument must be a string literal

`println!()`은 첫번째 인수를 string literal을 요구하고, 변수를 활용하여 formatting하는 것을 지원한다. 따라서 우리는 코드를 아래와 같이 고쳐야 한다.

```rust
fn main() {
  let message = "Hello, World!";
  println!("{}", message)
}
```

### 자바스크립트와 다른 것 2

이번엔 함수를 사용한다고 가정해보자.

```rust
fn main() {
    greet("world")
}

fn greet(target: String) {
    println!("hello, {}", target)
}
```

이 코드 역시 에러가 난다.

```shell
@yceffort ➜ /workspaces/rust-playground/chapter1/hello_cargo (main ✗) $ cargo run
   Compiling hello_cargo v0.1.0 (/workspaces/rust-playground/chapter1/hello_cargo)
error[E0308]: mismatched types
 --> src/main.rs:2:11
  |
2 |     greet("world")
  |           ^^^^^^^- help: try using a conversion method: `.to_string()`
  |           |
  |           expected struct `String`, found `&str`

For more information about this error, try `rustc --explain E0308`.
error: could not compile `hello_cargo` due to previous error
```

`String`을 `target`으로 예상했지만, 그것이 아닌 `&str`을 전달받았다는 에러다. 이러한 일이 왜 일어나는지 알기 위해서는, rust에서 String이 무엇인지 알아봐야 하고, 그것보다 이전에 우리는 러스트의 '소유권' 과 '빌림' 의 개념에 대해서 알아야 한다. rust의 가장 핵심이 되는 개념이다.

## 소유권과 빌림

소유권은 러스트를 이해하는데 있어 첫번째 난관이다. 이해하기가 어렵다기보다는, 러스트의 규칙은 다른 언어에서는 잘 통용되는 논리와 구조를 다시금 생각하게 만드는 구조이기 때문이다.

러스트는 가비지 컬렉터 없이 안전한 메모리 해제를 약속한 덕분에 많은 인기와 지지를 얻을 수 있었다. 자바스크립트나 GO는 메모리를 관리하기 위해 가비지 컬렉션을 사용한다. 객체에 대한 모든 참조를 추적하고, 이 참조 카운트가 0으로 감소했을 때만 메모리를 해제한다. 이 가비지 컬렉터는 자완과 성능을 희생하여 개발자를 좀더 편하게 만들어 준다. 물론 이정도로도 충분할 수 있다. 그러나, 이것으로 부족할때, 이 가비지 컬렉터 문제를 해결하고 최적화 하는 것은 굉장히 어려운 일이다. 러스트에서는 가비지 컬렉터의 오버헤드 없이 메모리 안정성을 달성할 수 있다. 모든 자원을 특별히 노력을 기울이지 않아도 돌려 받을 수 있다.

메모리의 안정성은 단순히 프로그램이 예기치 않은 크래쉬를 방지하는 것 그 이상의 것을 의미한다. 모든 종류의 보안 취약점을 차단한다는 것을 의미한다. SQL 인젝션을 들어보았는가? SQL 인젝션은 미처 관리되고 있지 않은 사용자 입력을 활용하여 의도치 않은 SQL 문을 만들어내고, 데이터를 빼돌리는 데이터베이스 클라이언트 쪽 취약성이다. 이 공격은 그다지 어려운 것이 아니라서 3관리가 가능하고 100% 예방 또한 가능하다. 그러나 오늘 날 웹 애플리케이션에서 가장 흔한 취약점으로 남아 있다. 메모리 측면에서 안전하지 않은 코드는 어디서나 나타날 수 있는 SQL 인젝션 취약성을 찾기 어려워진다는 것과 비슷하다. 메모리 안정성 측면의 버그는 심각한 취약점의 대부분을 차지한다. 그러므로, 성능에 영향을 미치지 않고 이러한 위협요소를 모두 제거할 수 있다는 것은 매력적인 개념이라고 볼 수 있다.

### 변수 할당과 mutability

앞서 이야기 한 것처럼 자바스크립트에는 `let` `const`가 있으며, `const`는 다시 재할당 할 수 없는 변수를 선언할 때 쓴다. 러스트에도 `let` `const`가 있지만, 일단 `let`만 쓴다.

자바스크립트에서 `const`가 쓰고 싶다면, rust에서는 `let`을 쓰면 된다. `let`을 쓰고 싶다면, `let mut`을 쓰면 된다. `mut`은 변수 중에서도 재할당 가능한 변수를 선언할 때 사용한다.

```javascript
let one = 1
console.log(one) // 1
one = 3
console.log(one) // 3
```

러스트에서는

```rust
fn main() {
  let mut one = 1;
  println!("{}", one);
  one = 3;
  println!("{}", one)
}
```

이렇게 작성하면 된다.

한가지 큰 다른점은, 오로지 같은 타입일때만 가능하다는 것이다. 즉 아래와 같은 코드는 불가능하다.

```rust
fn main() {
    let mut one = 1;
    println!("{}", one);
    one = "3";
    println!("{}", one)
}
```

```
@yceffort ➜ /workspaces/rust-playground/chapter1/hello_cargo (main ✗) $ cargo run
   Compiling hello_cargo v0.1.0 (/workspaces/rust-playground/chapter1/hello_cargo)
error[E0308]: mismatched types
 --> src/main.rs:4:11
  |
2 |     let mut one = 1;
  |                   - expected due to this value
3 |     println!("{}", one);
4 |     one = "3";
  |           ^^^ expected integer, found `&str`

For more information about this error, try `rustc --explain E0308`.
error: could not compile `hello_cargo` due to previous error
```

다른 타입을 변수에 할당하고 싶다면 `let`을 선언하여 같은 이름에 할당하는 방법을 쓰면 된다.

```rust
fn main() {
    let one = 1;
    println!("{}", one);
    let one = "3";
    println!("{}", one)
}
```

### 러스트에서 빌림을 확인하는 법

러스트에는 데이터를 전달하는 방법, 즉 데이터를 "빌리는 방법" 과 "소유권" 에대한 기본적인 규칙을 적용함으로써 메모리 안전성을 보장한다.

### 규칙1. 소유권

값을 전달하면, 호출하는 코드는 더이상 해당 데이터에 접근할 수 없다. 간단히 말해 소유권을 포기한 것이다. 아래 코드를 확인해보자.

```rust
use std::{collections::HashMap, fs::read_to_string};

fn main() {
    let source = read_to_string("./README.md").unwrap();
    let mut files = HashMap::new();
    files.insert("README", source);
    files.insert("README2", source);
}
```

```shell
@yceffort ➜ /workspaces/rust-playground/chapter1/hello_cargo (main ✗) $ cargo run
   Compiling hello_cargo v0.1.0 (/workspaces/rust-playground/chapter1/hello_cargo)
error[E0382]: use of moved value: `source`
 --> src/main.rs:7:29
  |
4 |     let source = read_to_string("./README.md").unwrap();
  |         ------ move occurs because `source` has type `String`, which does not implement the `Copy` trait
5 |     let mut files = HashMap::new();
6 |     files.insert("README", source);
  |                            ------ value moved here
7 |     files.insert("README2", source);
  |                             ^^^^^^ value used here after move
```

앞으로 rust를 공부하면서 가장 많이 마주하게될 에러 메시지, `use of moved value: source.`다. 처음 `source`를 HashMap에 넘겼을때, 이때는 우리는 소유권을 포기한 것이다. 따라서 두번째 줄에서는 동일하게 호출할 수 없었던 것이다. 위 코드가 실행되기 위해서는, 다음과 같이 고쳐야한다.

```rust
use std::{collections::HashMap, fs::read_to_string};

fn main() {
    let source = read_to_string("./README.md").unwrap();
    let mut files = HashMap::new();
    files.insert("README", source.clone());
    files.insert("README2", source);
}
```

### 규칙2. 빌림

데이터를 빌릴때, 즉 데이터의 참조를 가져가고 싶다면, `&` 키워드를 사용해서 참조를 가져올 수 있다. 이를 사용하면 앞서 했던 것 처럼 굳이 번거롭게 데이터를 계속 복사하지 않아도 참조를 안전하게 가져올 수 있다.

```rust
use std::{collections::HashMap, fs::read_to_string};

fn main() {
    let source = read_to_string("./README.md").unwrap();
    let mut files = HashMap::new();
    files.insert("README", source.clone());
    files.insert("README2", source);

    // rust 참조 가져오기
    let files_ref = &files;
    let files_ref2 = &files;

    print_borrowed_map(files_ref);
    print_borrowed_map(files_ref2)
}


fn print_borrowed_map(map: &HashMap<&str, String>) {
    println!("{:?}", map)
}
```

만약 map에 mutable reference가 필요하다면, `let files_ref = &mut files;`를 사용하면 된다.

```rust
use std::{collections::HashMap, fs::read_to_string};

fn main() {
    let source = read_to_string("./README.md").unwrap();
    let mut files = HashMap::new();
    files.insert("README", source.clone());
    files.insert("README2", source);

    let files_ref = &mut files;
    let files_ref2 = &mut files;

    print_borrowed_map(files_ref);
    print_borrowed_map(files_ref2);

    needs_mutable_ref(files_ref);
    needs_mutable_ref(files_ref2);
}

fn needs_mutable_ref(map: &mut HashMap<&str, String>) {}

fn print_borrowed_map(map: &HashMap<&str, String>) {
    println!("{:?}", map)
}
```

그러나 빌드 하면 에러가 나게된다.

```bash
@yceffort ➜ /workspaces/rust-playground/chapter1/hello_cargo (main ✗) $ cargo build
   Compiling hello_cargo v0.1.0 (/workspaces/rust-playground/chapter1/hello_cargo)
warning: unused variable: `map`
  --> src/main.rs:19:22
   |
19 | fn needs_mutable_ref(map: &mut HashMap<&str, String>) {}
   |                      ^^^ help: if this is intentional, prefix it with an underscore: `_map`
   |
   = note: `#[warn(unused_variables)]` on by default

error[E0499]: cannot borrow `files` as mutable more than once at a time
  --> src/main.rs:10:22
   |
9  |     let files_ref = &mut files;
   |                     ---------- first mutable borrow occurs here
10 |     let files_ref2 = &mut files;
   |                      ^^^^^^^^^^ second mutable borrow occurs here
11 |
12 |     print_borrowed_map(files_ref);
   |                        --------- first borrow later used here

For more information about this error, try `rustc --explain E0499`.
warning: `hello_cargo` (bin "hello_cargo") generated 1 warning
error: could not compile `hello_cargo` due to previous error; 1 warning emitted
```

보면 볼수록 rust 컴파일러의 메시지가 참 친절하다고 느낀다. 만약 다른 참조를 사용하기 전에, 하나의 참조가 끝날 수 있도록 순서를 조정한다면, 이 에러는 더이상 나타나지 않을 것이다.

```rust
use std::{collections::HashMap, fs::read_to_string};

fn main() {
    let source = read_to_string("./README.md").unwrap();
    let mut files = HashMap::new();
    files.insert("README", source.clone());
    files.insert("README2", source);

    let files_ref = &mut files;
    needs_mutable_ref(files_ref);
    let files_ref2 = &mut files;
    needs_mutable_ref(files_ref2);
}

fn needs_mutable_ref(map: &mut HashMap<&str, String>) {}
```

러스트를 시작할때, 코드의 순서를 조정하는 것만으로도 에러를 해결할 수 있는 경우가 많다.
