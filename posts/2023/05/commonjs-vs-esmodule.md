---
title: '1부) commonjs vs esmodule'
tags:
  - nodejs
published: true
date: 2023-05-26 13:52:26
description: '아 require 랑 import 는 안다구요'
---

> 이 글은 `commonjs`와 `esmodule` 의 동작 원리와 차이점을 알기 위해 작성된 글이다. 총 3부작으로 작성할 예정이고, 작성될 때 마다 본문에 링크를 추가해 두겠다.

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

그렇다면 `module.exports`랑 `exports`을 사용하는 것에는 어떤 차이가 있는 것일까? 먼저 앞선 `math`의 예제 처럼 `exports.sum`을 하거나 `module.exports.sum`을 하는 것 은 동일하다.

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

정답은 `{hello: true}`다.

즉, `module.exports`와 `exports`는 아래와 같은 관계를 가지고 있다고 보면 된다.

```js
module.exports = exports = class Square {
  // something...
}
```

결론적으로 `exports`가 아무리 일부 케이스에서 정상적으로 동작한다 하더라도 `module.exports`를 쓰는 것이 옳다.

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

여기서 중요한 것은 항상 기본값은 `commonjs`를 사용하는 것이다. `package.json`또는 파일명에 별다른 조치를 취해주지 않으면 항상 `commonjs`를 사용한다. 이러한 이유는

1. `commonjs`와 `esmodule`간에 호환이 되지 않음
2. 이미 많은 패키지가 `commonjs`를 기반으로 제작됨

이기 때문이다. 호환이 되지 않는 이유는 뒤이어서 다룬다.

### module wrapper

앞서 `module wrapper`라는 함수 덕분에, 모듈에서 `export`되지 않은 값들이 로컬 변수로 남아 숨겨질 수 있다고 언급했다. 이 `module wrapper` 함수는 다음과 같이 생겼다.

```js
;(function (exports, require, module, __filename, __dirname) {
  // 내부 모듈 코드는 실제로 여기에 들어감
})
```

이렇게 함으로써 얻을 수 있는 이점은 다음과 같다.

- 모듈 최상단에 있는 `var` `const` `let` 등으로 선언된 변수가 글로벌 객체 (`global`)에 등록되는 것을 막는다.
- 모듈에서 글로벌 객체 있는 `exports` `require` `module` `__filename` `__dirname`을 사용할 수 있게 해준다.
  - 그렇다. `esmodule`에서 `__filename`, `__dirname` 등을 사용하지 못하는 이유는 `module wrapper`가 없기 때문이다.

### 순환 참조에서는 어떻게 동작할까?

백문이 불여일견이다. 코드를 보면서 살펴보자.

```js
// a.js
console.log('a starting')
exports.done = false
const b = require('./b.js')
console.log('in a, b.done = %j', b.done)
exports.done = true
console.log('a done')

// b.js
console.log('b starting')
exports.done = false
const a = require('./a.js')
console.log('in b, a.done = %j', a.done)
exports.done = true
console.log('b done')

// index.js
console.log('main starting')
const a = require('./a.js')
const b = require('./b.js')
console.log('in main, a.done = %j, b.done = %j', a.done, b.done)
```

이 코드에서 예상되는 서순은 다음 과 같다.

1. `index.js`가 실행됨
2. `a.js`를 불러옴
3. `a.js`가 `b.js`를 불러옴
4. `b.js`가 `a.js`를 불러옴
5. 무한루프?????????

실제 실행 결과를 살펴보자.

```text
main starting
a starting
b starting
in b, a.done = false
b done
in a, b.done = true
a done
in main, a.done = true, b.done = true
```

실제 실행 시에는 무한루프에 빠지지 않고 잘 끝난 것을 볼 수 있다. 그 이유는 앞서 이야기 한 캐싱 덕분이다. 캐싱 작업으로 인해, 한번 불러온 모듈은 다시 불러오지 않게 된다. 여기에서는 `b.js`가 `a.js`를 불러오는 순간, `a.js`의 `exports.done`이 `false`인 상태의 객체가 리턴된다. 그 이유는 최초에 `index.js`에서 `require(./a.js)`가 아직 끝나지 않았기 때문이다. 이렇게 nodejs가 무한 순환 참조를 방지하면, `main.js`가 `./a.js`와 `./b.js`를 모두 불러온 순간 각각 모듈의 `done`이 `false`가 된다.

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

자바스크립트 개발자라면 `commonjs`가 트리쉐이킹이 되지 않는 다는 이야기를 많이 들어보았을 것이다. 결론부터 말하자면 어느정도는 사실이다. 사실 `commonjs` 는 nodejs 환경에서만 사용될 목적으로 만들어졌었다. 즉 그당시만 하더라도 브라우저에서는 복잡한 모듈 시스템을 만들 필요가 없었고, (복잡한 자바스크립트 자체가 필요하지 않았으므로) 서버, 즉 많은 서로다른 모듈을 불러와야 했던 nodejs에서만 필요했기 때문이다. 그리고 서버는 애초에 모듈 크기가 커지는게 크게 상관이 없기도 하다. (브라우저 처럼 사용자가 다운로드 하거나 그럴 필요가 있는 것은 아니므로) 그러한 사실을 방증하듯, 애초에 `commonjs`의 이름은 `serverjs`였다.

> 2009년 `commonjs` 의 창시자 Kevin Dangoor 가 쓴 글에 그 흔적을 볼 수 있다.
>
> https://www.blueskyonmars.com/2009/01/29/what-server-side-javascript-needs/

아무튼 다시 본론으로 돌아와서, `commonjs`와 트리쉐이킹의 관계를 살펴보자. 앞서 `commonjs`환경에서는 모든 각각의 파일단위의 모듈을 `module wrapper`라고 하는 함수로 감싸서 실행한다고 하였다. 이러한 `commonjs`의 방식이 문제가 된 것은 브라우저에서 `commonjs` 모듈 방식을 사용하기 시작하면서 부터다. 서버는 어느 정도 컴퓨팅 속도나 성능이 보장되어있었지만, 브라우저의 경우 이러한 사용자의 성능을 담보할 수 없다. 각 모듈이 `module wrapper`로 인해 생성된 개별 함수 클로저에 의해 래핑되서 실행된다는 점은, 브라우저에서 자바스크립트 성능을 매우 안좋게 만들었다. 프레임워크 기반의 자바스크립트 환경을 생각해보자. 각종 모듈이 얽혀서 불러오는 과정에서 매번 클로저가 생성되서 참조된다는 것은 분명히 성능상 문제가 있었다. 그래서 그당시 인기있는 번들러였던 [Closure Compiler](https://developers.google.com/closure/compiler?hl=ko)나 [rollupjs](https://rollupjs.org/)는 모든 모듈을 하나의 클로즈로 호이스팅하거나 연결해서 `require`로 인한 성능 저하 현상을 방지하였다.

이러한 작업은 지금까지도 가장 널리 쓰이고 있는 번들러인 웹팩에서도 마찬가지다. 웹팩은 [ModuleConcatenationPlugin](https://webpack.kr/plugins/module-concatenation-plugin/) 라는 프로덕션 모드에서만 동작하는 플러그인을 용하여 여러 모듈을 하나로 연결하여 클로져 생성을 최소화 하는 작업을 한다.

그렇다면 이게 왜 문제가 되는 것일까? 답은 `module.exports`의 객체 방식 `exports` 때문이다. 아래 코드를 살펴보자.

```javascript
// test.js
module.exports = {
  [globalThis.hello]: 'world',
}
```

```javascript
// index.js
const hello = 'hello'

globalThis[hello] = hello

const test = require('./test.js')

console.log(test[hello])
```

이 정신나가 보이는 코드는 동작할까? 놀랍게도 `world`라는 값이 정상적으로 출력된다.

> https://replit.com/@yceffort/YellowishNeatDictionary#index.js

**`module.exports`의 객체라는 특성 때문에, 빌드 타임에서는 모듈에서 어떠한 값이 불러와서 사용해질 수 있을지 가늠할 수 없다.** 따라서 번들러들은 `commonjs`로 되어 있는 모듈의 성능을 위해 하나의 거대한 클로저로 합쳐버린 대신, 무엇이 실행될 지를 결정하는 작업을 포기해버린다. 그에 반해, `esmodule`은 `export`라는 명확한 키워드를 사용하고 있으므로 사용 여부를 결정할 수 있기 때문에 트리쉐이킹이 가능하다.

그렇다면 아까 '어느 정도는 사실' 이다 라는 말은 무엇일까? 위 코드 처럼 동적으로 `exports`을 하지 않는 등 몇가지 규칙을 지키다면, [webpack-common-shake](https://github.com/indutny/webpack-common-shake) 모듈을 사용하는 등의 방법으로 트리쉐이킹을 수행할 수 있다. (`rollup`에서는 별도 설정없이 기본으로 된다)

#### `module.exports`로만 `export`가 가능하다

이는 앞서 `module.exports`에서 알아보았던 내용과 동일하다. `module.exports`가 `export`할 수 있는 유일한 방법이기 때문에, 모듈에서 여러 값을 `export`하려면 `module.exports` 자체를 객체로 사용하는 수 밖에 없다. 이는 `export const ...`으로 `export` 키워드로 모듈 어디서든 내보내기를 사용할 수 있는 `esmodule`과 대비되는 지점이다.

### commonjs의 시대는 끝났는가?

답은 그렇다고 볼 수 있다. 최근 많은 라이브러리들이 순수한 esmodule로 구현하고 있는 추세다.

- `query-string`: https://github.com/sindresorhus/query-string/releases/tag/v8.0.0
- `d3.js`: https://github.com/d3/d3/releases/tag/v7.0.0

등등 유명한 라이브러리들이 `commonjs` 지원을 중단하고 `esmodule`로 넘어가고 있는 추세다. 그 이유는 여러가지 있다.

- `webpack@4`와 같은 `commonjs` 만 지원하는 번들러가 점차 사라지고 있음
- 라이브러리 관리자들이 유지보수하기 굉장히 빡셈
  - 라이브러리를 두종류로 번들링 해야하는 데 따른 시간 증가 및 관리 포인트 증가
- 트리쉐이킹을 지원하지 못함

`commonjs`를 표준에서 제외해야 하는가, `deprecated` 해야 하는가, `nodejs`에서 지원을 중단해야 하는가 여부는 매우 논쟁적인 부분이지만, 대부분의 자바스크립트 개발자들은 `esmodule`을 더 선호한다는 것에는 동의할 것이다.

### 마치며

지금까지 `commonjs`의 특징에 대해서 살펴보았다. 비록 이제 저물어가는 모듈 방식이지만, 여전히 많은 코드가 `commonjs`에 의존하고 있기 때문에 `commonjs` 동작 방식을 이해하는 것은 중요하다. 그리고 `commonjs` 방식을 이해한다면, `esmodule`의 필요성에 대해 이해하게 되는 좋은 계기가 될 것이다.
