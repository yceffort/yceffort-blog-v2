---
title: 'Rust로 web assembly로 game of life 만들어보기 (2)'
tags:
  - web
  - javascript
  - rust
published: true
date: 2022-04-08 11:45:16
description: '사이드 프로젝트도 열심히 하고 싶은데 바쁘기도 바쁘고 체력도 안되는 것 같고 아무튼 핑계입니다.'
---

## Table of Contents

## 디자인

본격적인 구현에 앞서, 어떤식으로 개발하면 좋을지 고민해보자.

### Infinite Universe

game of life (이하 라이프 게임)은 무한대의 우주에서 펼쳐지지만, 아쉽게도 우리의 컴퓨팅 파워는 무한대가 아니다. 이러한 한계를 극복하기 위하나 방법으로는, 세가지 정도가 있을 것이다.

1. 계속해서 어떤일이 일어나고 있는지 추적하기 위해 영역을 지속적으로 확장하는 것. 그러나 이러한 확장은 제한적이고, 구현속도는 점점 느려지고 메모리도 부족하게 될 것
2. 고정된 크기의 우주를 만들되, 모서리에 있는 셀이 가운데에 있는 셀보다 더 적은 수의 이웃을 갖게 하는 방법. 그러나 이 패턴은 글라이더와 같은 무한 패턴을 구현하지 못한다.
3. 일정한 크기의 주기적으로 구현되는 우주를 만드는 방법. 이 우주 가장자리에 우주의 반대편으로 둘러싼 이웃을 존재하게 된다. (쉽게 말해 좌우를 잇는다고 보면 된다.)

3번째 방법으로 구현한다고 생각해보자.

### 자바스크립트와 러스트의 인터페이스

자바스크립트의 가비지 컬렉팅 힙은 Object, array, DOM 노드 등이 할당되며, 이는 로스트의 값이 존재하는 웹 어셈블리의 선형 메모리 공간과는 구별되는 영역이다. 웹 어셈블리는 자바스크립트의 가비지 컬렉팅 힙에 직접 접근할 수가 없다. 하지만 자바스크립트는 웹 어셈블리의 이러한 선형 메모리 공간에 접근하여 읽고 쓸 수는 잇지만, 스칼라 값 (u8, i32, f64 등)의 Array Buffer만 가능하다. 웹 어셈블리 함수는 스칼라 값을 가져오고 반환한다. 이것이 모든 웹 어셈블리와 자바스크립트 통신을 구성하는 요소로 볼 수 있다.

`wasm_bindgen`은 이 바운더리를 가로지르는 복잡한 구조물을 다루는 방법에 대한 공통적인 방법을 정의한다고 볼 수 있다. 이것은 러스트 구조를 박스화하고, 포인터를 자바스크립트 클래스로 래핑하거나, 러스트에서 자바스크립트 객체의 테이플로 인덱싱 하는 것 등등을 포함한다. `wasm_bindgen`은 이런면에서 매우 편리하지만, 데이터 표현과 어떤 값 구조가 이 바운더리를 가로질러 전달되는지를 개발자가 고려하도록 만들어 두엇다. 단순히 `wasm_bindgen`은 선택한 인터페이스 설계를 구현을 위한 도구라고 생각하면 된다.

웹 어셈블리와 자바스크립트 사이의 인터페이스를 설계할 때, 다음의 내용을 최적화 하고자 한다.

1. 웹 어셈블리 선형 메모리로 복사하는 것을 최소화 한다. 불필요한 복사본은 불필요한 오버헤드를 만든다.
2. 직렬화 및 역직렬화를 최소화 한다. 1번과 마찬가지로, 직렬화와 역직렬화도 오버헤드를 초래하고 종종 복사도 강제하는 등의 부작용이 있다.

일반적으로, 좋은 자바스크립트와 웹어셈블리간의 인터페이스 설계는, 대용량의, 그리고 수명을 오래 가져가야 하는 데이터를 러스트의 선형메모리에 구현하고, 이를 자바스크립트에 제한적인 핸들러로 노출시키는 것이다. 자바스크립트에서 이 제한적인 핸들러를 사영하여 웹 어셈블리를 호출하면 러스트에서는 데이터를 변환하고, 무거운 계산을 수행하고, 데이터를 쿼리하고, 궁극적으로 복사 가능한 아주 작은 데이터를 반환하는 것이다. 웹 어셈블리의 작은 계산 결과만 반환함으로써, 자바스크립트의 가비지 콜렉팅 힙과 웹 어셈블리의 메모리 사이에 직렬화를 피하는 것이 좋다.

### 라이프 게임에서 러스트와 자바스크립트 인터페이스

먼저 피해야 할 것들 부터 알아보자. 우리는 매틱 마다 웹 어셈블리의 메모리로 온 우주의 정보를 보내서 복사할 필요가 없다. 우주에 있는 모든 세포들에 객체를 할당해서느 안되고, 각 세포를 읽고 쓰기 위해 경계를 넘나들며 (자바스크립트 웹어셈블리) 호출을 할 필요는 없다.

어떻게 해야할까? 우리는 우주를 웹 어셈블리 선형 메모리에 나타내고, 각 셀에 대한 바이트를 갖는 평평한 배열로 나타낼 수 있다. 0은 죽은 상태, 1은 살아있는 상태다.

아래는 메모리에서 4x4 우주를 어떻게 나타내는지 보여준다.

![4x4 universe](https://rustwasm.github.io/docs/book/images/game-of-life/universe.png)

우주에서 행과 열을 주어줬다면, 이 상태를 알기 위해 우리는 아래 공식을 사용할 수 있다.

```rust
index(row, column, universe) = row * width(universe) + column
```

이를 자바스크립트에 표현하기 위해 사용할 수 있는 방법은 무엇이 있을까? 먼저 `Universe`에서 [std::fmt::Display](https://doc.rust-lang.org/1.25.0/std/fmt/trait.Display.html)를 구현하여 텍스트 문자로 렌더링 할 수 있는 Rust String을 생성할 수 있다. 그런 다음 이 러스트 문자령르 웹 어셈블리의 선형 메모리에서 자바스크립트의 문자열로 보낸다음, HTML의 `textContent`로 설정하여 표시하면 된다. 이러한 구현을 한단계 진화시켜서 `<canvas>`에 그리는 방법도 있을 것이다.

또다른 방법으로는, 러스트가 모든 우주를 자바스크립트에 노출시키는 대신, 각 틱이 발생한 후에 상태가 변경된 모든 셀의 목록을 반환하는 방법도 있다. 이렇게 하면 자바스크립트는 렌더링할 때 모든 전체 우주를 반복할 필요가 없고, 렌더링이 필요한 부분 집합만 구할 수 있다. 단점은 이 방법이 조금더 구현이 어렵다는 것이다.

## 러스트 구현

`greet`과 `alert`를 제거하고, 아래 세포를 정의한 코드로 대체하자.

```rust
#[wasm_bindgen]
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cell {
    Dead = 0,
    Alive = 1,
}
```

여기서 주목해야 할 것은 `#[repr(u8)]`다. 이 정의는, 하나의 셀이 싱글 바이트로 표현된다는 것을 의미한다. 또한 `Dead`가 0, `Alive`가 1로 설정해놓음으로써, 이웃에 얼마나 많은 셀들이 살아있는지 쉽게 구할 수 있다.

> `repr`은 struct의 alighment를 설정하는 방법이다. 즉 `Cell`을 `u8` 구조체로 설정한 것이다.
> u8은 숫자를 표현할 수 있는 최소 단위다

> `derive`일부 특성에 대한 기본 구현을 할 수 있도록 도와주는 도구다.

다음으로는 우주를 정의하자. 우주는 너비와 높이를 가진다.

```rust
#[wasm_bindgen]
pub struct Universe {
    width: u32,
    height: u32,
    cells: Vec<Cell>,
}
```

> `Vec`은 벡터로 불리는 배열이다.

주어진 행과 열에 존재하는 셀에 접근하기 위하여, 앞서 설명한 것과 같이 인덱스로 접근할 수 있는 함수도 만들 것이다.

```rust
impl Universe {
    fn get_index(&self, row: u32, column: u32) -> usize {
        (row * self.width + column) as usize
    }

    // ...
}
```

인접해 있는 세포가 얼마나 살아있는지 판단할 수 있는 함수를 만들어야 한다.

```rust

impl Universe {
    // ...
    fn live_neighbor_count(&self, row: u32, column: u32) -> u8 {
        let mut count = 0;
        for delta_row in [self.height - 1, 0, 1].iter().cloned() {
            for delta_col in [self.width - 1, 0, 1].iter().cloned() {
                if delta_row == 0 && delta_col == 0 {
                    continue;
                }

                let neighbor_row = (row + delta_row) % self.height;
                let neighbor_col = (column + delta_col) % self.width;
                let idx = self.get_index(neighbor_row, neighbor_col);
                count += self.cells[idx] as u8;
            }
        }
        count
    }
}
```

다음으로는 자바스크립트가 틱이 발생했을 때 제어할 수 있는 메소드를 추가해보자.

```rust

#[wasm_bindgen]
impl Universe {
    // public method. 이를 자바스크립트에서 쓸 수 있게 할 것이다.
    pub fn tick(&mut self) {
      // 현재 세포들을 모두 꺼내와서 복사해둔다.
        let mut next = self.cells.clone();

      // 현재 모든 셀을 순환한다.
        for row in 0..self.height {
            for col in 0..self.width {
                let idx = self.get_index(row, col);
                // 현재 세포
                let cell = self.cells[idx];
                // 주변 세포가 몇개나 살아 있는지 계산한다
                let live_neighbors = self.live_neighbor_count(row, col);

                // 다음 셀은 다음과 같이 결정된다.
                let next_cell = match (cell, live_neighbors) {

                    // 규칙1. 살아있는 세포 근처에 두명 미만의 세포가 살아있다면, 죽는다.
                    (Cell::Alive, x) if x < 2 => Cell::Dead,
                    // 규칙 2: 살아있는 세포 규칙에 2~3의 살아있는 세포가 있다면, 산다.
                    (Cell::Alive, 2) | (Cell::Alive, 3) => Cell::Alive,                    .
                    // 규칙3: 살아있는 이웃세포가 3 보다 많다면 죽는다
                    (Cell::Alive, x) if x > 3 => Cell::Dead,
                    // 규칙4: 살아있는 이웃이 정확히 3개있는 죽은세포는 살아난다.
                    (Cell::Dead, 3) => Cell::Alive,
                    // 그외의 다른 셀은 그대로...
                    (otherwise, _) => otherwise,
                };
                next[idx] = next_cell;
            }
        }
        self.cells = next;
    }
    // ...
}
```

지금까지 우주의 상태는 셀의 벡터로 표현되고 있다. 이제 이것을 사람이 볼 수 있는 텍스트 형태로 만들어보자. 살아있는 셀은 ◼로, 죽어있는 셀은 ◻를 나타내게 할 것이다.

러스트 표준라이브러리에서 `Display` trait을 사용한다면, 사용자가 볼 수 있는 방식으로 구조를 포맷팅하는 메소드를 추가할 수 있다. 그리고 자동으로 `to_string` 메소드를 제공한다.

```rust
use std::fmt;

impl fmt::Display for Universe {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for line in self.cells.as_slice().chunks(self.width as usize) {
            for &cell in line {
                let symbol = if cell == Cell::Dead { '◻' } else { '◼' };
                write!(f, "{}", symbol)?;
            }
            write!(f, "\n")?;
        }

        Ok(())
    }
}
```

마지막으로, constructor와 `to_string`을 도와주는 `render`를 만들자.

```rust
#[wasm_bindgen]
impl Universe {
    // ...

    pub fn new() -> Universe {
        let width = 64;
        let height = 64;

        let cells = (0..width * height)
            .map(|i| {
                if i % 2 == 0 || i % 7 == 0 {
                    Cell::Alive
                } else {
                    Cell::Dead
                }
            })
            .collect();

        Universe {
            width,
            height,
            cells,
        }
    }

    pub fn render(&self) -> String {
        self.to_string()
    }
}
```

이정도면, 절반정도 구현한 셈이다. 이제 다시 `wasm-pack build`로 빌드해보자.

## 자바스크립트에서 렌더링하기

`wasm-game-of-life/www/index.html`를 다음과 같이 작성해보자.

```html
<body>
  <pre id="game-of-life-canvas"></pre>
  <script src="./bootstrap.js"></script>
</body>
```

그리고 `<pre>`를 중앙으로 배치하기 위해 css를 활용하자.

그리고 `wasm-game-of-life/www/index.js`의 최상단에, 우리가 만든 `Universe`를 import 하자.

```javascript
import { Universe } from 'wasm-game-of-life'
```

그리고, `<pre>` 엘리먼트에 우리가 만든 `Universe`를 새로 만든다.

```javascript
const pre = document.getElementById('game-of-life-canvas')
const universe = Universe.new()
```

매틱마다 매끄러운 렌더링을 구현하기 위해 [requestAnimationFrame](https://developer.mozilla.org/en-US/docs/Web/API/window/requestAnimationFrame)을 사용한다. 매 iteration 마다, 현재 우주 상태를 `<pre>`에 그리고, `Universe::tick`을 호출한다.

```javascript
const renderLoop = () => {
  pre.textContent = universe.render()
  universe.tick()

  requestAnimationFrame(renderLoop)
}
```

그리고 이제 매 틱마다 실행될 수 있도록, 최초 한번 실행한다.

```javascript
requestAnimationFrame(renderLoop)
```

`npm run start`로 실행하여, http://localhost:8080 에서 무슨일이 일어나는지 확인해보자.

![game-of-life](./images/game-of-life.gif)

> 잘 모르겠지만,, 뭔가 일어나고 있음,,,

## 메모리에서 캔버스로 바로 렌더링 해보기

앞서 언급했던 것처럼, 러스트에서 문자열을 생성하고 할당한 다음, `wasm-bindgen`으로 유효한 자바스크립트 문자열로 반환하는 작업은 불필요하게 셀의 복사본을 두번 만드는 것이다. 자바스크립트는 이미 전체 너비와 높이를 알고 있고, 셀을 구성하고 있는 웹 어셈블리의 선형 메모리를 직접 읽을 수 있으므로, 렌더링 방법을 수정해보자.

이에 추가로 유니코드 텍스트를 그리는 대신, Canvas api를 사용해보자.

먼저 `<pre>`를 `<canvas>`로 변경해보자.

```html
<body>
  <canvas id="game-of-life-canvas"></canvas>
  <script src="./bootstrap.js"></script>
</body>
```

러스트에서 필요한 정보를 읽기 위해, 우주의 너비, 높이, 셀 배열에 대한 포인터 정보를 알 수 있는 함수를 추가해보자.

```rust
#[wasm_bindgen]
impl Universe {
    // ...

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn cells(&self) -> *const Cell {
        // 슬라이스 버퍼에 있는 포인터 정보를 리턴한다.
        self.cells.as_ptr()
    }
}
```

그리고 자바스크립트에서, 셀을 표현하는데 필요한 상수를 정의 해두자.

그리고, 자바스크립트 코드에서 `<canvas>`를 그리도록 변경해보자.

```javascript
// Construct the universe, and get its width and height.
const universe = Universe.new()
const width = universe.width()
const height = universe.height()

// 세포 사이에 1px border
const canvas = document.getElementById('game-of-life-canvas')
canvas.height = (CELL_SIZE + 1) * height + 1
canvas.width = (CELL_SIZE + 1) * width + 1

const ctx = canvas.getContext('2d')

const renderLoop = () => {
  universe.tick()

  drawGrid()
  drawCells()

  requestAnimationFrame(renderLoop)
}
```

그리드를 그리기 위해서, 같은 간격의 수평선과 수직선 세트를 그린다. 이 선들을 교차하여 그리드를 그리게 될 것이다.

```javascript
const drawGrid = () => {
  ctx.beginPath()
  ctx.strokeStyle = GRID_COLOR

  // Vertical lines.
  for (let i = 0; i <= width; i++) {
    ctx.moveTo(i * (CELL_SIZE + 1) + 1, 0)
    ctx.lineTo(i * (CELL_SIZE + 1) + 1, (CELL_SIZE + 1) * height + 1)
  }

  // Horizontal lines.
  for (let j = 0; j <= height; j++) {
    ctx.moveTo(0, j * (CELL_SIZE + 1) + 1)
    ctx.lineTo((CELL_SIZE + 1) * width + 1, j * (CELL_SIZE + 1) + 1)
  }

  ctx.stroke()
}
```

웹 어셈블리의 선형메모리에 바로 접근하기 위해, raw wasm module인 `wasm_game_of_life_bg`를 활용할 것이다. 세포를 그리기 위해 현재 우주의 세포에 대한 포인터를 얻고, 세포 버퍼가있는 `Unit8Array`를 구성하고, 각 세포를 순회하면서 세포가 죽었는지 살았는지에 따라 각각 사각형을 그린다. 포인터와 오버레이로 작업하여 매 틱에서 경계를 넘어 셀을 복사하는 것을 피한다.

```javascript
// Import the WebAssembly memory at the top of the file.
import { memory } from 'wasm-game-of-life/wasm_game_of_life_bg'

// ...

const getIndex = (row, column) => {
  return row * width + column
}

const drawCells = () => {
  const cellsPtr = universe.cells()
  const cells = new Uint8Array(memory.buffer, cellsPtr, width * height)

  ctx.beginPath()

  for (let row = 0; row < height; row++) {
    for (let col = 0; col < width; col++) {
      const idx = getIndex(row, col)

      ctx.fillStyle = cells[idx] === Cell.Dead ? DEAD_COLOR : ALIVE_COLOR

      ctx.fillRect(
        col * (CELL_SIZE + 1) + 1,
        row * (CELL_SIZE + 1) + 1,
        CELL_SIZE,
        CELL_SIZE,
      )
    }
  }

  ctx.stroke()
}
```

최초 렌더링 프로세스를 시작 하기 위해서, `renderLoop`에 있는 프로세스를 가져와 실행한다.

```javascript
drawGrid()
drawCells()
requestAnimationFrame(renderLoop)
```

다시한번 빌드하고, 실행해보자.

```bash
@yceffort ➜ /workspaces/rust-playground/wasm-game-of-life (main ✗) $ wasm-pack build
[INFO]: Checking for the Wasm target...
[INFO]: Compiling to Wasm...
   Compiling wasm-game-of-life v0.1.0 (/workspaces/rust-playground/wasm-game-of-life)
    Finished release [optimized] target(s) in 0.39s
[INFO]: Installing wasm-bindgen...
[INFO]: Optimizing wasm binaries with `wasm-opt`...
[INFO]: Optional fields missing from Cargo.toml: 'description', 'repository', and 'license'. These are not necessary, but recommended
[INFO]: :-) Done in 0.93s
[INFO]: :-) Your wasm pkg is ready to publish at /workspaces/rust-playground/wasm-game-of-life/pkg.
```

```bash
@yceffort ➜ /workspaces/rust-playground/wasm-game-of-life (main ✗) $ cd www/
@yceffort ➜ /workspaces/rust-playground/wasm-game-of-life/www (master ✗) $ npm run start

> create-wasm-app@0.1.0 start /workspaces/rust-playground/wasm-game-of-life/www
> webpack-dev-server

ℹ ｢wds｣: Project is running at http://localhost:8080/
ℹ ｢wds｣: webpack output is served from /
ℹ ｢wds｣: Content not from webpack is served from /workspaces/rust-playground/wasm-game-of-life/www
ℹ ｢wdm｣: Hash: 9a501699d68560154eeb
Version: webpack 4.43.0
Time: 524ms
Built at: 04/08/2022 6:13:05 AM
                           Asset       Size  Chunks                         Chunk Names
                  0.bootstrap.js   10.5 KiB       0  [emitted]
8ca3edcd4459872d299d.module.wasm   20.6 KiB       0  [emitted] [immutable]
                    bootstrap.js    369 KiB    main  [emitted]              main
                      index.html  494 bytes          [emitted]
Entrypoint main = bootstrap.js
[0] multi (webpack)-dev-server/client?http://localhost:8080 ./bootstrap.js 40 bytes {main} [built]
[../pkg/wasm_game_of_life.js] 95 bytes {0} [built]
[../pkg/wasm_game_of_life_bg.wasm] 20.6 KiB {0} [built]
[./bootstrap.js] 279 bytes {main} [built]
[./index.js] 1.79 KiB {0} [built]
[./node_modules/ansi-html/index.js] 4.16 KiB {main} [built]
[./node_modules/strip-ansi/index.js] 161 bytes {main} [built]
[./node_modules/webpack-dev-server/client/index.js?http://localhost:8080] (webpack)-dev-server/client?http://localhost:8080 4.29 KiB {main} [built]
[./node_modules/webpack-dev-server/client/overlay.js] (webpack)-dev-server/client/overlay.js 3.51 KiB {main} [built]
[./node_modules/webpack-dev-server/client/socket.js] (webpack)-dev-server/client/socket.js 1.53 KiB {main} [built]
[./node_modules/webpack-dev-server/client/utils/createSocketUrl.js] (webpack)-dev-server/client/utils/createSocketUrl.js 2.91 KiB {main} [built]
[./node_modules/webpack-dev-server/client/utils/log.js] (webpack)-dev-server/client/utils/log.js 964 bytes {main} [built]
[./node_modules/webpack-dev-server/client/utils/reloadApp.js] (webpack)-dev-server/client/utils/reloadApp.js 1.59 KiB {main} [built]
[./node_modules/webpack-dev-server/client/utils/sendMessage.js] (webpack)-dev-server/client/utils/sendMessage.js 402 bytes {main} [built]
[./node_modules/webpack/hot sync ^\.\/log$] (webpack)/hot sync nonrecursive ^\.\/log$ 170 bytes {main} [built]
    + 23 hidden modules
ℹ ｢wdm｣: Compiled successfully.
```

![game-of-life-canvas](./images/game-of-life-canvas.gif)

> 지금까지 작성한 코드는 [github](https://github.com/yceffort/rust-playground/tree/main/wasm-game-of-life)에서 확인하실 수 있습니다.
