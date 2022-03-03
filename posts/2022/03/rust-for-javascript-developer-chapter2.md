---
title: '[Rust] 자바스크립트에서 러스트로 (2) - String'
tags:
  - rust
published: false
date: 2022-03-03 16:20:26
description: 'Rust 공부해보기 (2)'
---

## Table of Contents

## Introduction

먼저 자바스크립트의 string에 대해서 다시 한번 생각해보자.

```javascript
'Hi' === 'Hi' // true
'Hi' === new String('Hi') // false
typeof 'Hi' // string
typeof new String('Hi') // object
typeof String('Hi') // string
'Hi' === String('Hi') // true
String('Hi') === new String('Hi') //false
```

## Rust에서 string 살펴보기

### &str

러스트 컴파일러는 모든 리터럴 문자열을 버킷 어딘가에 넣고, 이 값을 포인터로 대체한다. 이렇게 하는 이유는, Rust가 중복되는 문자열을 최적화 할 수 있으므로, 단일 문자열 슬라이스에 대한 포인터와, 단일 문자열에 대한 포인터를 사용할 수 있다.

무슨말인지 잘 이해가 안된다면, 아래 코드를 한번 사용해보자.

```rust

fn main() {
    print("TESTING:123456789012345678901234567890");
    // ... 위 코드를 여러번 반복
    print("TESTING:123456789012345678901234567890");
  }

  fn print(msg: &str) {
    println!("{}", msg);
  }
```

```shell
@yceffort ➜ /workspaces/rust-playground/chapter3 (main ✗) $ cargo build --release
   Compiling chapter3 v0.1.0 (/workspaces/rust-playground/chapter3)
    Finished release [optimized] target(s) in 0.35s
@yceffort ➜ /workspaces/rust-playground/chapter3 (main ✗) $ strings target/release/chapter3 | grep TESTING
TESTING:123456789012345678901234567890
```

아무리 같은 글자를 여러번 반복해도 실제 실행파일 크기에는 크게 차이가 없는 것을 알 수 있다. 즉, 러스트가 내부에서 string에 대해서 포인터를 활용하여 최적화를 하고 있다는 것을 의미한다.

### String

String은 자바스크립트에서 알던 String과 거의 비슷하다.

### &str을 String으로 만드는 법

```rust
let borrowed_string = "string literal!"; // &str
let real_string = "string".to_owned() // std::string::String
```

애초에 string literal은 소유권을 넘겨주고 시작한다. 만약 String이 필요하다면, 위와 같은 방식으로 변환해야 한다. (본질적으로 이는 복사와 같다.)

### `.to_string()` `.into()` `String::from()`

위 옵션 모두 `to_owned()`와 동일하게 변경해주는 역할을 한다. 그러나 이들 동작에는 약간씩 차이가 있다.

#### `.to_string()`

```rust
fn main() {
  let real_string: String = "string literal".to_string();
  needs_a_string(real_string)
}

fn needs_a_string(arg: String) {}
```
