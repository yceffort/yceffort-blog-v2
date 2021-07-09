---
title: 'ESModule을 동적으로 import 하기'
tags:
  - javascript
published: true
date: 2021-06-19 20:38:43
description: '무지성 import 멈춰!'
---

ECMAScript (ES2015, ES) 모듈이란 자바스크립트에서 각 코드를 하나의 청크로 구성할 수 있게 해주는 방법을 제공한다. 먼저, 자바스크립트의 값을 외부로 노출을 시킨다.

```javascript
export const sum = (a, b) => a + b
```

그리고 이 값(함수)을 필요로 하는 곳에서 아래와 같이 사용한다.

```javascript
import { sum } from './test'

sum(1, 2)
```

대부분의 경우에는 위의 예제 처럼 상단에 필요한 모듈들을 static하게 import하는 것이 일반적이지만, 때때로 이를 필요에 따라 조건부로 불러올 수도 있다. [dynamic import](https://caniuse.com/?search=dynamic%20import)라고 불리며, 이는 ES2020(ES11)에 포함된 기능이다.

```javascript
const sum = await import('./test')
```

`await import`는 프로미스를 리턴하고 이 때부터 동적으로 모듈을 불러오기 시작한다. 그리고 모듈을 성공적으로 불러오면, promise는 모듈을 resolve하거나 실패할 경우 reject 하게 된다.

그리고 `import`내의 구문은 모듈의 위치를 가리키는 string이면 되기 때문에, 함수 외부에서 인수로 받거나 계산된 string을 받는 것들도 가능하다.

```javascript
export const sum = (a, b) => a + b
```

```javascript
async function calc(a, b) {
  const { sum } = await import('./test')
  sum(a, b)
}

calc(1, 2) // 3
```

이제 default 의 경우를 살펴보자.

```javascript
const sum = (a, b) = > a + b
export default sum
```

이 경우에는 `default` 키워드를 사용하면 된다.

```javascript
async function calc() {
  const { default: sum } = await import('./test')
  sum(1, 2)
}
```

한가지 유의 할 것은,

```javascript
const sum = await import('./test')
```

이렇게는 안된다는 것이다.

default와 그 외의 것들이 섞여있는 경우라면 아래와 같이 처리하면 된다.

```javascript
const sum = (a, b) => a + b
export const sum1 = (a, b) => a + b
export const sum2 = (a, b) => a + b

export default sum
```

```javascript
const { default: sum, sum1, sum2 } = await import('./test')
```
