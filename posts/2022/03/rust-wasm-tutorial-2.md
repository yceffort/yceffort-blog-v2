---
title: 'Rust로 web assembly 만들어보기 (2) - Rust로 간단한 Web Assembly 만들기'
tags:
  - web
  - javascript
  - html
  - rust
published: true
date: 2022-03-10 17:45:43
description: '오 이거 신기하네'
---

## Table of Contents

## 개발 환경

1. [Install Rust](https://www.rust-lang.org/install.html)로 먼저 Rust를 설치한다.
2. 그리고 wasm을 만들기 위해, [wasm-pack](https://github.com/rustwasm/wasm-pack)을 설치한다.

```shell
cargo install wasm-pack
```

## 패키지 만들기

```shell
cargo new --lib hello-wasm
```

이제 아래와 같은 파일이 생성되었을 것이다.

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
```

일반적으로 단위테스트는 src 디렉토리의 각 파일에 테스트 할 코드와 함께 작성한다. 여기서 사용되는 규칙은 각 파일에 `mod tests`라는 모듈을 `#[cfg(test)]`와 함께 선언하고, 그안에 테스트할 코드를 작성하면 된다. `#[cfg(test)]` 로 선언된 모듈은 `cargo test`를 할 때만 실행되고, build시에는 컴파일 되지 않는다. 따라서 빌드 시 시간과 공간을 절약할 수 있다.

> cfg는 configuration 이라는 뜻이다.

`[#test]`는 이 함수가 테스트 함수임을 가리키는 역할을 한다.

## Rust 작성하기

먼저 `Cargo.toml`에 `wasm_bindgen`을 의존성 목록에 추가해주자.

```toml
[package]
name = "hello-wasm"
version = "0.1.0"
authors = ["yceffort <yceffort@gmail.com>"]
description = "A sample project with wasm-pack"
license = "MIT/Apache-2.0"
repository = "https://github.com/yceffort/rust-playground/tree/main/wasm/tutorial/hello-wasm"

[lib]
crate-type = ["cdylib"]

[dependencies]
wasm-bindgen = "0.2"
```

```rust
// import * from wasm_bindgen/prelude와 같다.
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern {
    pub fn alert(s: &str);
}

#[wasm_bindgen]
pub fn greet(name: &str) {
    alert(&format!("Hello, {}!", name));
}
```

[wasm-bindgen](https://github.com/rustwasm/wasm-bindgen)은 자바스크립트와 러스트 사이에 일종의 다리 역할을 한다고 보면 된다. 자바스크립트에서 rust api를 호출하거나, 반대로 rust가 js에서 발생한 예외처리를 하는 등의 처리를 할 수 있도록 해준다.

`#[XXX]`는 일종의 wrapper를 생성하는 속성 값인데, 이것이 무슨일을 하는지는 이후에 알아보자.

`extern` 키워드는, 이 것이 rust 외부에 정의된 함수라는 것을 알린다. 외부에 `alert`라는 함수가 있으며, 이는 문자열 타입의 `s` 를 받는 다는 것을 의미한다. 눈치 챘을 수도 있지만, 이는 `window.alert`를 의미한다.

즉, 자바스크립트에 무언가 함수를 호출 하고 싶다면 `extern` 키워드와 함께 추가하면 된다.

```rust
#[wasm_bindgen]
pub fn greet(name: &str) {
    alert(&format!("Hello, {}!", name));
}
```

이번에는 `extern` 키워드 대신 다른 것이 나왔다. 이번에는 `fn` 구문을 wrapping 하고 있다. 이는 rust 함수를 자바스크립트에 의해 호출될 수 있도록 처리한다는 것을 의미한다. 즉 `extern`과는 반대가 되는 기능이다.

함수를 보면 알겠지만, `greet()`는 문자열 타입 `name`을 받고 `hello {name}`이라는 문자열을 만들고 이를 alert에 넘겨주고 있다.

이제 이 코드를 빌드해보자

## 빌드하기

```shell
wasm-pack build --scope yceffort
```

마지막 scope는 npm 계정의 아이디를 넣어주면된다.

이 빌드는 다음과 같은 과정을 수행한다.

1. Rust 코드를 WebAssembly로 컴파일
2. WebAssembly위에서 `wasm-bindgen`을 실행하여, WebAssembly가 npm이 이해할 수 있는 모듈로 감싸는 자바스크립트 파일을 생성
3. `pkg` 폴더를 만들고, 자바스크립트 파일과 WebAssembly 코드를 그 안으로 옮긴다.
4. `Cargo.toml`과 동등한 `package.json`을 생성
5. `README.md`가 있다면 패키지로 복사

빌드가 완료되었다면, `pkg` 폴더가 생성되어 있는 것을 볼 수 있다.

`hello_wasm.js`

```javascript
import * as wasm from './hello_wasm_bg.wasm'
export * from './hello_wasm_bg.js'
```

`package.json`

```json
{
  "name": "@yceffort/hello-wasm",
  "collaborators": ["yceffort <yceffort@gmail.com>"],
  "description": "A sample project with wasm-pack",
  "version": "0.1.0",
  "license": "MIT/Apache-2.0",
  "repository": {
    "type": "git",
    "url": "https://github.com/yceffort/rust-playground.git"
  },
  "files": [
    "hello_wasm_bg.wasm",
    "hello_wasm.js",
    "hello_wasm_bg.js",
    "hello_wasm.d.ts"
  ],
  "module": "hello_wasm.js",
  "types": "hello_wasm.d.ts",
  "sideEffects": false
}
```

## 빌드한 패키지 사용해보기

이 npm package를 사용할 수 있도록 한번 설정해보자.

```json
{
  "name": "hello-wasm-npm",
  "version": "1.0.0",
  "description": "",
  "main": "index.js",
  "scripts": {
    "serve": "webpack-dev-server"
  },
  "dependencies": {
    "@yceffort/hello-wasm": "../hello-wasm/pkg"
  },
  "devDependencies": {
    "webpack": "^4.25.1",
    "webpack-cli": "^3.1.2",
    "webpack-dev-server": "^3.1.10"
  },
  "author": "",
  "license": "ISC"
}
```

```javascript
const path = require('path')
module.exports = {
  entry: './index.js',
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: 'index.js',
  },
  mode: 'development',
}
```

```html
<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>hello-wasm example</title>
  </head>
  <body>
    <script src="./index.js"></script>
  </body>
</html>
```

```javascript
const js = import('./node_modules/@yceffort/hello-wasm/hello_wasm.js')
js.then((js) => {
  js.greet("yceffort's first WebAssembly")
})
```

![first-wasm](./images/first-wasm.png)

## `wasm-bindgen`의 대략적인 원리

`wasm-bindgen`의 가장 중요한 개념은, wasm module이 ES Module의 한 가지 종류로 인식하고 연동한다는 것이다. `pkg`에 있는 `d.ts`를 보면 (타입스크립트 시그니쳐까지..!) 다음과 같이 선언되어 있다.

```typescript
/* tslint:disable */
/* eslint-disable */
/**
 * @param {string} name
 */
export function greet(name: string): void
```

WebAssembly는 이러한 처리가 불가능하므로, 이 것을 수행해주는 것이 `wasm-bindgen`이다. 이 중 자바스크립트 파일은 러스트를 호출할때 사용되는 인터페이스 역할을 하고, `*_bg.wasm` 파일이 실제로 방금 컴파일한 것과 구현체를 가지고 있다.

`hello_wasm_bg.js` 파일은 다음과 같이 구현되어 있다.

```javascript
import * as wasm from './hello_wasm_bg.wasm'

// ...

function getStringFromWasm0(ptr, len) {
  return cachedTextDecoder.decode(getUint8Memory0().subarray(ptr, ptr + len))
}

/**
 * @param {string} name
 */
export function greet(name) {
  var ptr0 = passStringToWasm0(
    name,
    wasm.__wbindgen_malloc,
    wasm.__wbindgen_realloc,
  )
  var len0 = WASM_VECTOR_LEN
  wasm.greet(ptr0, len0)
}

export function __wbg_alert_a5a2f68cc09adc6e(arg0, arg1) {
  alert(getStringFromWasm0(arg0, arg1))
}
```

`wasm.greet(ptr0, len0);`를 보면, 이 함수는 문자열이 아닌 포인터와 length를 인수로 받고 있는 것을 알 수 있다.

조금 더 깊이 들어가서, WebAssembly의 `greet`함수가 러스트 컴파일러에 의해 컴파일되는 시점을 보면 이런식으로 코드가 작성되어 있다.

```rust
pub fn greet(name: &str) {
    alert(&format!("Hello, {}!", name));
}

#[export_name = "greet"]
pub extern fn __wasm_bindgen_generated_greet(arg0_ptr: *mut u8, arg0_len: usize) {
    let arg0 = unsafe { ::std::slice::from_raw_parts(arg0_ptr as *const u8, arg0_len) }
    let arg0 = unsafe { ::std::str::from_utf8_unchecked(arg0) };
    greet(arg0);
}
```

원래 작성한 코드와 함께, 이상한 이름의 함수와 `#[export_name = "greet"]`가 붙어 있다. 이는 JS가 던진 pointer와 length를 받는 부분이다. 이 두개 인자를 받아서, `greet` 함수에 전달한다.

정리하자면, `#[wasm_bindgen]`는 두개의 wrapper를 생성한다.

- JS 타입을 받아서 wasm으로 변환 (자바스크립트)
- wasm 타입을 rust 타입으로 변환 (러스트)

즉, 앞서 언급했던 것 처럼, `wasm-bindgen`는 자바스크립트 - WASM - 러스트 사이에 다리 역할을 하고 있으며, 이를 위해 많은 일들이 뒷단에서 일어나고 있음을 알 수 있다.
