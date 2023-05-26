---
title: 'commonjs 와 esmodule 은 어떤 차이가 있는가?'
tags:
  - nodejs
published: true
date: 2023-05-26 13:52:26
description: '아 require 랑 import 는 안다구요'
---

## Table of Contents

## 서론

nodejs가 15.3.0 부터 esmodule을 정식 지원하기 시작한 이래로, 많은 자바스크립트 개발자들이 모듈을 불러오는 과정이 `require`와 `import` 로 차이가 있다는 것을 알 뿐, 그 외의 동작에도 차이가 있다는 것을 잘 모르는 것 같다. (일단 나부터 모른다면 ㄱㅊ) 구체적으로 이 둘은 어떤 차이가 있고, 궁극적으로 npm 라이브러리가 이 두 모듈을 동시에 지원하기 위해 어떠한 노력을 기울여야 하는지 종합적으로 살펴보자.

> 과거 [CommonJS와 ES Modules은 왜 함께 할 수 없는가?
> ](/2020/08/commonjs-esmodules) 라는 글을 작성한 적이 있는데 이 보다 더 심오하게 들어간 내용을 작성해보았다.

## Commonjs

### 정의

commonjs 모듈은 원래 nodejs에서 자바스크립트 패키지를 불러올 때 사용하는 근본있는 방식이다. 앞서 이야기 한 것 처럼 현재는 ECMAScript module(이하 esmodule)을 지원하지만, 태초에는 commonjs 방식만 존재했다. (amd나 뭐이것저것 있었는데 일단 nodejs 환경에서는 commonjs가 유일했다.)

먼저 모듈이라는 말의 정의를 먼저 짚고 넘어가야 한다. nodejs에서 모듈은 각각의 분리된 파일을 모듈이라 칭한다. 예를 들어 다음과 같은 코드가 있다고 가정해보자.

```javascript
// foo.js
const math = require('./math.js')
console.log(math.sum(1, 2))
```

위 코드에서 첫번째 줄에는 `./math.js`라는 별도의 파일, 즉 같은 디렉토리에 있는 별도의 모듈을 참조하고 있는 것을 볼 수 있다. 그리고 `./bar.js`는 다음과 같은 내용을 담고 있다고 가정해보자.

```javascript
const { PI } = Math

exports.sum = (a, b) => a + b

exports.circumference = (r) => 2 * PI * r
```

`math.js`는 `sum`과 `circumference` 함수 두개를 export 하는 것을 볼 수 있다. 이처럼 nodejs는 `exports`라고 하는 특별한 객체를 통해 모듈의 루트에 추가할 수 있게 된다.

여기에서 주목할 것은 최상단의 `Math` 객체에서 구조분해할당을 한 `PI`다. nodejs는 모듈을 module wrapper라고 하는 함수로 래핑하기 때문에 `PI`와 같은 로컬 변수는 위 두 함수와 다르게 비공개가 된다. 이에 대한 자세한 내용은 뒤에서 다룬다.

또 하나 알아두어야 할 것은 `module.exports`라고 하는 속성이다. 이 속성에는 함수나 객체와 같은 새로운 값을 선언할 수 있다. 다음 예시를 살펴보자.

```javascript
// foo.js
const Square = require('./square.js')
const mySquare = new Squre(2)
```

```javascript
// square.js
module.exports = class Square {
  constructor(width) {
    this.width = width
  }

  area() {
    return this.width ** 2
  }
}
```

여기에서는 `exports`가 `module.exports`을 사용하였다. 그 결과 `foo`에서 `require`해온 `Square` 는 `square.js`에서 선언한 `Square`클래스가 할당되어 있는 것을 볼 수 있다.

그렇다면 `module.exports`랑 `exports`는 어떤 차이가 있는 것일까? 먼저 앞선 `math`의 예제 처럼 `exports.sum`을 하거나 `module.exports.sum`을 하는 것 은 동일하다.

```javascript
const { PI } = Math

module.exports.area = (r) => PI * r ** 2
module.exports.circumference = (r) => 2 * PI * r

module.exports === exports // true
```

```javascript
const { PI } = Math

exports.area = (r) => PI * r ** 2
exports.circumference = (r) => 2 * PI * r

module.exports === exports // true
```

그러나 큰 차이를 보이는 건 바로 `module.exports = // ... something`을 하는 경우다.

```javascript
module.exports = class Square {
  constructor(width) {
    this.width = width
  }

  area() {
    return this.width ** 2
  }
}

console.log('exports >>>', exports) // [class Square]
console.log('module.exports >>>', module.exports) // {}
console.log('compare', exports === module.exports) // false

// index.js
const Square = require('./Math.js') // {}
```

```javascript
module.exports = class Square {
  constructor(width) {
    this.width = width
  }

  area() {
    return this.width ** 2
  }
}

console.log('exports >>>', exports) // {}
console.log('module.exports >>>', module.exports) // [class Square]
console.log('compare', exports === module.exports) // false

// index.js
const Square = require('./Math.js') // Square
```

이러한 차이가 발생하는 이유는 무엇일까? **그 이유는 바로 `exports`자체가 `module.exports`를 가리키고 있기 때문이다.** 이는 nodejs의 문서에도 나와있다.

> A reference to the module.exports that is shorter to type.
>
> https://nodejs.org/api/modules.html#exports

`exports`는 `module.exports`의 일종의 숏컷으로 볼 수 있다. `module.exports`는 모듈이 평가되기 전에 미리 할당되는 값이다. 그렇다면 아래의 코드에서 `export`되는 것은 무엇일까?

```js
module.exports.hello = true
exports = { hello: false }
```

정답은 `{hello: true}`다. (이유는 생략한다.)

### nodejs 는 언제 commonjs를 사용할까?

앞서 이야기 한 것 처럼 nodejs에서 사용되는 모듈 시스템은 `Commonjs`와 `esmodule` 두가지가 있다. 그렇다면 nodejs는 이 두 모듈 시스템 중 어떤 모듈 시스템을 사용할지 어떻게 결정할까? nodejs가 `Commonjs` 모듈 시스템을 사용하는 경우는 다음과 같다.

- 파일 확장자가 `.cjs`로 되어 있는 경우
- 파일 확장자가 `.js`로 되어 있으며
  - 가장 가까운 부모의 `package.json`의 파일의 `type`필드에 값이 `commonjs`인 경우
  - 가장 가까운 부모의 `package.json`파일에 `type` 필드가 명시되어 있지 않은 경우
    - 이것이 바로 그 commonjs 라이브러리로 대표되는 `lodash`의 사례다. [lodash의 경우 package.json에 `type`이 할당되어 있지 않다.](https://github.com/lodash/lodash/blob/master/package.json)
    - 라이브러리 제작자라면, 어쩄거나 이 `type` 필드에 값을 `commonjs`든 뭐든 넣어주는 것이 좋다. 이는 빌드 도구나 번들러들이 모듈을 빠르게 결정해서 작업하는데 도움을 준다.
- 파일 확장자가 `.mjs` `.cjs` `.json` `.node` `.js` 가 아닌 경우. 이 경우 가장 가까운 부모의 `package.json`이 `type: "module"`로 되어 있다고 하더라도, 모듈 내부에 `require()`를 쓰고 있다면 commonjs로 인식한다.
- 모듈이 `require()`로 호출 되는 경우 내부 파일에 상관없이 무조건 `commonjs`로 인식한다.

### module wrapper

TBD

### 특징

#### 동기로 실행된다

commonjs의 특징은 모듈을 동기로 불러온다는 것이다. 이 말인 즉슨 모듈을 하나씩 순서대로 불러오고 처리한다는 뜻이다. 다음 예제를 살펴보자.

```javascript
// module1.js
console.log('module1 로드 시작')

setTimeout(() => {
  console.log('module1 실행')
}, 2000)

console.log('module1')
```

```javascript
// index.js
console.log('시작')
const module1 = require('./module1')
console.log('index!')
const module2 = require('./module2')
console.log('종료')
```

> 깜짝 면접 퀴즈: 다음 실행결과는?

```bash
시작
module1 로드 시작
module1
index!
module2 로드 시작
module
종료
```

`require`는 동기로 불러온다는 점을 반드시 기억해야 한다. `require`를 선언하면 디스크 또는 네트워크로 해당 모듈을 읽어서 즉시 스크립트를 실행한다. 따라서 `require`를 실행하게 되면 그 자체만으로 I/O나 부수효과를 발생시키고, 그 이후에 `module.exports`에 있는 값을 반환한다.

따라서 성능이 좋은 nodejs 프로그램을 만드려면 `require`를 최소화 하는 것이 좋다. 이에 대해서는 이후에 다룬다.

#### 캐싱

모듈은 한번 로딩되고 난 뒤에는 캐싱된다. 즉, 같은 `reuiqre()`를 호출하게 되면, 한번 이 값을 resolve한 뒤에는 동일한 값을 반환한다. 다음 예제를 살펴보자.

```js
// data.js
console.log('call data')

module.exports = 'hello'
```

```js
const data1 = require('./data.js')
const data2 = require('./data.js')
const data3 = require('./data.js')

console.log(data1, data2, data3)
```

```js
// call data
// hello hello hello
```

최초에는 미처 `require(./data.js)`가 캐싱되지 않아 전체 모듈을 evaluation 하여 값을 가져왔다. 이렇게 한번 캐싱된 이후에는 앞서 캐싱원리에 따라 동일하나 값을 `resolve`하면 되므로 더이상 `console.log`가 실행되지 않는 것을 확인할 수 있다.

이러한 캐싱 정보는 `require.cache`에 존재한다. 필요에 따라서 이 캐시정보를 삭제할 수도 있다.

```js
const data1 = require('./data.js')

delete require.cache[require.resolve('./data.js')]

const data2 = require('./data.js')
const data3 = require('./data.js')

console.log(data1, data2, data3)

// call data
// call data (캐시가 지워져 한번더 호출되었다.)
// hello hello hello
```

이 모듈 캐싱에 대해 알아둬야 할점은, 캐싱의 기준은 파일명이 된다는 것이다. `node_modules`와 같이 모듈은 호출하는 모듈의 위치에 따라 다른 파일명으로 해석될수도 있으므로, 다른파일로 해석될 여지가 존재하는 경우 항상 동일한 객체를 반환한다는 보장을 할수는 없다.

또한 OS나 파일시스템에 따라 대소문자를 구분하지 아흔 경우, 서로 다른 파일 이름이 동일한 파일을 가리킬 수 는 있지만, 모듈은 여전히 다른 것으로 취급하여 파일을 여러번 다시 로드할 수도 있다. 즉, OS나 파일시스템에 따라 `./foo`나 `./FOO`는 같은 파일로 취급될 수도 있지만, `require('./foo')` `require('./FOO')`는 서로 다른 두 객체를 반환한다. 즉, nodejs의 파일명 기반 모듈 캐싱은 대소문자에 따라 결과가 달라진다.

#### 트리쉐이킹이 되지 않는다?

#### `module.exports`로만 `export`가 가능하다
