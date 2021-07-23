---
title: 'Export에 숨겨져 있는 심오함'
tags:
  - javascript
  - nodejs
published: true
date: 2021-07-22 21:12:37
description: '자바스크립트는 멋져 짜릿해 늘 새로워'
---

자, 흔히 쓰는 import 가 있다.

`module.js`

```javascript
export let data = 5
```

`index.js`

```javascript
import { data } from './module'
```

그런데 만약에 이렇게 import를 해보면 어떨까?

```javascript
const module = await import('./module.js')
const { data: value } = await import('./module.js')
```

첫번째 import 에서 `module.data`를 하는 것은 맨 처음에 import 했던 결과와 완전히 동일 할 것이다. 두번째는, `data`를 `value`라는 새로운 identifier로 할당하고 있다. 그리고 이 동작은 앞선 두 케이스와 묘하게 다르다.

만약에 export 하는 쪽에서 값의 변경이 있다고 가정해보자.

```javascript
export let data = 5

setTimeout(() => {
  data = 10
}, 500)
```

```javascript
import { data } from './module.js'
const module = await import('./module.js')
const { data: value } = await import('./module.js')

setTimeout(() => {
  console.log(data) // 10
  console.log(module.data) // 10
  console.log(value) // 5
}, 1000)
```

또다른 변수로 아예 할당을 해버렸던 3번째 케이스를 제외하고 나머지 모든 값들은 변했다는 것을 알 수 있다. 그렇다. `import`는 일종의 참조 처럼 동작을 한다는 것을 알 수 있다. 사실 이러한 3번째 케이스의 동작은 아래처럼 생각하면 당연하다고 느껴 질 수 있다.

```javascript
const obj = { foo: 'bar' }
const { foo } = obj
obj.foo = 'baz'
console.log(foo) // 'bar'
```

내 개인적으로 봤을 때는 위 케이스, 즉 3번째 케이스가 제일 자연스러워 보인다. 🤔 여전히 자바스크립트는 신비로운 언어다. 근데 잠깐, `import { data }`도 어떻게 보면 분해할당이 아닌가? 근데 이 것은 놀랍게도 분해 할당처럼 동작하지 않는 다는 것을 알 수 있다.

자 정리해보자.

```javascript
// 특정 값을 참조하는 것 처럼 동작하여, 값이 바뀌면 서순에 따라서 그 바뀐 값을 들고 올 수도 있다.
import { data } from './module.js'
import { data as value } from './module.js'
import * as all from './module.js'
const module = await import('./module.js')
// 현재 값을 새로운 변수에 그대로 할당해서, 참조측에서 값이 바뀌든 말든 최초의 값을 계속 간직한다.
let { data } = await import('./module.js')
```

자 그럼, `export default`의 경우는 어떤가?

> 요즘 핫하게 클릭되는 https://yceffort.kr/2020/11/avoid-default-export 이글도 살펴보세여 😘

```javascript
export { data }
export default data

setTimeout(() => {
  data = 10
}, 500)
```

```javascript
import { data, default as data2 } from './module.js'
import data3 from './module.js'

setTimeout(() => {
  console.log(data) // 10
  console.log(data2) // 5
  console.log(data3) // 5
}, 1000)
```

그렇다, default는 모두 값이 변하든 말든 상관없이 초기의 값을 간직하고 있다.

`export default`는 , 혹시 이렇게 써본 적이 있는지는 모르겠지만, `default`로 바로 그냥 값을 내보내 버릴 수 있다.

```javascript
export default 'direct'
```

그러나 named exports, 이름으로 export를 하는 경우에는 불가능하다.

```javascript
// 이런 코드는 존재할 수 없다.ㄴㄴ
export {'direct' as direct}
```

`export default 'direct'`가 동작하게 하기 위해서, default export는 named export와는 다르게 동작한다. `export default`는 일종의 표현식처럼 동작하여 값을 바로 내보내거나, 연산을 통한 결과 값이 나가는 것이 가능하다. (`export default 'direct'` `export default 1+2`) 근데 여기서 또한 `export default data`도 가능하다. 두가지 모두를 가능하게 하기 위하여, `default`뒤에 오는 변수를 모두 값으로 처리를 하는 것이다. 따라서 export 하는 쪽에서 새로운 값으로 변하게 했다 하더라도, `export default`의 동작의 특성상 변한 값이 내보내지는게 아니라, 그 순간의 값이 나가게 된다.

정리하자면,

```javascript
// 특정 값을 참조하는 것 처럼 동작하여, 값이 바뀌면 서순에 따라서 그 바뀐 값을 들고 올 수도 있다.
import {data} from './module.js'
import {data as value} from './module.js'
import * as all from './module.js'
const module = await import('./module.js')
// 현재 값을 새로운 변수에 그대로 할당해서, 참조측에서 값이 바뀌든 말든 최초의 값을 계속 간직한다.
let  { data } = await import('./module.js')

// 참조를 export
export {data}
export {data as data2}
// 현재 값 그 자체를 export
export default data
export default 'direct'
```

자 여기에 하나만 더 끼얹어보자. `export {}`는 값을 바로 내보낼 수는 없고 참조만 내보낼 수 있다.

```javascript
let data = 5

export {data, data as default}
setTimeout(() => {
  data = 10
}, 500)}
```

```javascript
import { data, default as data2 } from './module.js'
import data3 from './module.js'

setTimeout(() => {
  console.log(data) // 10
  console.log(data2) // 10
  console.log(data3) // 10
}, 1000)
```

뭐야 이건 또, 값이 다 바꼈다. `export default data`와는 다르게, `export {data as default}`는 값이 아닌 참조를 내보낸 것을 알 수 있다. `as default`는 named export 와 같은 문법이므로, 참조를 내보낸 것을 알 수 있다.

그래서 또또 정리하자면,

```javascript
// 특정 값을 참조하는 것 처럼 동작하여, 값이 바뀌면 서순에 따라서 그 바뀐 값을 들고 올 수도 있다.
import {data} from './module.js'
import {data as value} from './module.js'
import * as all from './module.js'
const module = await import('./module.js')
// 현재 값을 새로운 변수에 그대로 할당해서, 참조측에서 값이 바뀌든 말든 최초의 값을 계속 간직한다.
let  { data } = await import('./module.js')

// 참조를 export
export {data}
export {data as data2}
export {data as default}
// 현재 값 그 자체를 export
export default data
export default 'direct'
```

함수는 어떨까?

```javascript
export default function getData() {}

setTimeout(() => {
  getData = '사실 변수 였습니다. 짜잔'
}, 500)
```

```javascript
import getData from './module.js'

setTimeout(() => {
  console.log(getData) // 사실 변수 였습니다. 짜잔
}, 1000)
```

.......?

```javascript
function getData() {}

export default getData

setTimeout(() => {
  getData = '사실 변수 였습니다. 짜잔'
}, 500)
```

```javascript
import getData from './module.js'

setTimeout(() => {
  console.log(getData) // [Function: getData]
}, 1000)
```

![...](https://t1.daumcdn.net/news/202105/25/maxim/20210525050708362gnmn.jpg)

`export default function`와 `export default class`는 조금 특별하다.

```javascript
function someFunction() {}
class SomeClass {}

console.log(typeof someFunction) // "function"
console.log(typeof SomeClass) // "function"
```

```javascript
;(function someFunction() {})
;(class SomeClass {})

console.log(typeof someFunction) // "undefined"
console.log(typeof SomeClass) // "undefined"
```

`function`과 `class` 문은 스코프/블록내에서는 identifier, 식별자를 만드는 반면, `function` `class` 표현식은 그렇지 않다.

따라서,

```javascript
export default function someFunction() {}
console.log(typeof someFunction) // "function"
```

만약, `export default function`이 값으로 내보내졌다면, 즉 기존의 `export default`와 동일하게 동작하여 표현식으로 동작했다면, `function`이 아닌 `undefined`로 찍혔을 것이다.

그래서 또또또또 요약을 하자면,

```javascript
// 특정 값을 참조하는 것 처럼 동작하여, 값이 바뀌면 서순에 따라서 그 바뀐 값을 들고 올 수도 있다.
import {data} from './module.js'
import {data as value} from './module.js'
import * as all from './module.js'
const module = await import('./module.js')
// 현재 값을 새로운 변수에 그대로 할당해서, 참조측에서 값이 바뀌든 말든 최초의 값을 계속 간직한다.
let  { data } = await import('./module.js')

// 참조를 export
export {data}
export {data as data2}
export {data as default}
export default function getData() {}
// 현재 값 그 자체를 export
export default data
export default 'direct'
```

여기서 한가지 명심해야할 것은, `export default 'direct'`는 값 그자체를 내보내는 반면, `export default function`은 참조를 내보낸다는 것이다.

> `export default = data` 와 같은게 차라리 더 나았을 지도 모른다..

호이스팅의 경우를 잠깐 생각해보자.

```javascript
work()

function work() {
  console.log("job's done")
}
```

이는 잘 알겠지만 동작한다. 함수 정의를 파일 위로 끌어올린다.

```javascript
// 둘다 안됨
assignedFunction()
new SomeClass()

const assignedFunction = function () {
  console.log('nope')
}
class SomeClass {}
```

`let` `const` `class` 식별자를 초기화 전에 쓰려고 하면, 에러가 발생한다.

```javascript
var foo = 'bar'

function test() {
  console.log(foo) // undefined
  var foo = 'hello'
}

test()
```

왜 undefined가 찍히는가? `var foo`는 함수 내에도 존재하고 있고, 함수 레벨에서 호이스팅이 있었고, `hello`로 할당되기 전에 호출되었기 때문에 값이 없는 것이다.

자바스크립트 내부에서는 아래와 같이 순환참조가 허용된다. 물론, 권장하지는 않는다.

```javascript
import { hi } from './module.js'

hi()

export function hello() {
  console.log('hello')
}
```

```javascript
import { hello } from './index.js'

hello()

export function hi() {
  console.log('hi')
}
```

"hello", 그 다음에 "hi" 가 나온다.이는 호이스팅 때문에 가능한 것이다. 호이스팅은 함수 정의를 호출 보다 위로 끌어올리기 때문이다.

그러나... 아래의 경우에는 안된다.

```javascript
import { hi } from './module.js'

hi()

export const hello = () => console.log('hello')
```

```javascript
import { hello } from './index.js'

hello()

export const hi = () => console.log('hi')
```

```
hello()
^

ReferenceError: Cannot access 'hello' before initialization
```

호이스팅이 일어나지 않아 `module.js`를 먼저 실행했고, `module.js`에서는 아직 있지도 않은 (호이스팅 되지도 않은) `hello`를 실행해서 에러가 발생하는 것이다.

하지만 아래 처럼 `export default`를 써보자.

```javascript
import foo from './module.js'

foo()

function hello() {
  console.log('hello')
}

export default hello
```

```javascript
import hello from './index.js'

hello()

function hi() {
  console.log('hi')
}

export default hi
```

이것도, 실패한다.

```
hello();
^

ReferenceError: Cannot access 'hello' before initialization
```

`module.js`에 있는 `hello`는 아직 초기화 되지않은 값이므로, 이를 호출하려다가 에러가 발생하게 된다.

그렇다, `export {hello as default}`로 바꿨다면 에러가 발생하지 않았을 것이다. 왜냐면 함수를 참조로 넘겨줬고, 그리고 그 순간 호이스팅이되었기 때문이다. `export default function hello()`도 마찬가지로 에러가 나지 않았을 것이다. 앞서 말했듯, `export default function`은 특별하게 처리한 케이스이기 때문이다.

## 결론!

```javascript
// 특정 값을 참조하는 것 처럼 동작하여, 값이 바뀌면 서순에 따라서 그 바뀐 값을 들고 올 수도 있다.
import {data} from './module.js'
import {data as value} from './module.js'
import * as all from './module.js'
const module = await import('./module.js')
// 현재 값을 새로운 변수에 그대로 할당해서, 참조측에서 값이 바뀌든 말든 최초의 값을 계속 간직한다.
let  { data } = await import('./module.js')

// 참조를 export
export {data}
export {data as data2}
export {data as default}
export default function getData() {}
// 현재 값 그 자체를 export
export default data
export default 'direct'
```

그리고, 위를 잘 참조하여 호이스팅이 발생할지 예측해보자.

> - https://jakearchibald.com/2021/export-default-thing-vs-thing-as-default/
> - https://developer.mozilla.org/ko/docs/orphaned/Web/JavaScript/Reference/Statements/export
> - https://nodejs.org/api/esm.html
