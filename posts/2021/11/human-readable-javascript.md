---
title: '읽기 좋은 자바스크립트 코드 작성하기'
tags:
  - javascript
published: true
date: 2021-11-10 23:11:23
description: '나는야 자바스크립트 키보드 워리어'
---

다른 사람들이 읽기에 좋은 코드를 작성하는 것은 중요하다. 같은 일을 하는 코드라면, 성능에 크게 영향을 미치지 않는 한에서 읽기 쉬운 코드를 작성하는 것이 좋다. 읽기 쉬운 이란 무엇인가? 복잡한 것을 간단하게 보일 수 있는 코드다. 그러나 '간단함' 이라는 것은 무엇인가? 이는 보는 사람의 수준에 기초한다. 읽기 쉬운 코드를 작성하고자 할 때 우리는 무엇을 목표로 해야할까? 이에 대한 해답은 상황에 따라 다를 것이다.

자바스크립트 개발자의 개발 경험을 개선하기 위해, TC39는 꾸준히 새로운 기능을 ECMAScript에 추가해 왔다. ES2019에서 추가된 기능 중 하나는, [Array.prototype.flat()](https://developer.mozilla.org/ko/docs/Web/JavaScript/Reference/Global_Objects/Array/flat) 이다. 이는 배열을 평평하게 만드는 역할을 담당하고 있다. 이 메소드가 존재하기 이전에는, 우리는 이 작업을 하기 위해 아래와 같이 코드를 작성해 왔다.

```javascript
let arr = [1, 2, [3, 4]]
let flatted1 = [].concat.apply([], arr)
let flatted2 = arr.flat()
```

`flat`을 사용한게 더 읽기 좋은가? 당연히 그 대답은 예스 일 것이다. 첫번째 코드는 알고리즘 테스트나 면접에서 볼 법한 코드로 한번에 코드의 목적을 이해하기 어렵다. 이 코드 중 어떤 코드가 더 읽기 쉬운지에 대한 대답은 뉴비나 전문가나 같을 것이다.

그러나 모든 개발자가 `flat()`의 존재를 아는 것은 아니다. 그러나 그 메소드의 존재를 알지 못하더라도, 메소드 자체가 기술 동사로 의미를 전달하기 때문에 보기만 해도 한눈에 알 수 있을 것이다. 이는 `concat.apply`보다 훨씬 더 직관적이다.

아마도 이러한 경우는 '어떤 코드가 더 읽기 좋은가' 의 질문에서 명확하게 대답할 수 있는 희귀한 케이스 일 것이다.

자바스크립트의 놀라운 점 중 하나는 바로 다재다능하다는 것이다. 이게 좋은 점이든 단점이든 간에, 암튼 다재다능하기 때문에 널리 사용되고 있을 것이다.

그러나 이러한 다재다능함과 함께 선택의 순간도 찾아온다. 여러가지 방법으로 동일한 코드를 작성할 수 있다. 우리는 과연 어떠한 방법이 옳은지 어떻게 결정할까?

자바스크립트 함수형 프로그래밍의 사례로 `map`을 살펴보자. `map`은 배열을 순회하면서 똑같은 길이의 배열을 만들 수 있다.

<!-- prettier-ignore-start -->

```javascript
const arr = [1, 2, 3]
let double = arr.map(item => item * 2)
// double is [2, 4, 6]
```
<!-- prettier-ignore-end -->

이제 다음 예제를 살펴보자.

```javascript
const arr = [1, 2, 3]
let double = arr.map((item) => item * 2)
```

두 예제의 차이점은 매개변수에 괄호가 있고 없고 일 뿐이다. 둘 이상의 매개변수가 있는 함수는 괄호를 항상 사용해야 하지만, 하나의 경우엔 그렇지 않다. 여기에 괄호를 넣어도 결과에는 차이가 없다. 단순히 prettier가 괄호가 없는 함수를 참지 못하는 것인가?

다른 예제를 살펴보자.

```javascript
let double = arr.map((item) => {
  return item * 2
})
```

이번에는 괄호와 return을 화살표 함수에 추가했다. 이제 좀 전통적인 함수 처럼 보이기 시작했다. 보통 함수 내부에 논리가 들어가야 한다면 이런식으로 작성한다.

```javascript
let double = arr.map(function (item) {
  return item * 2
})
```

자 이번엔 화살표 함수를 제거하고, 함수 키워드를 사용했다. 처음 코드를 작성했을 때 보다 복잡해졌는데, 이게 과연 나쁜 것일까?

```javascript
const timesTwo = (item) => item * 2
let double = arr.map(timesTwo)
```

이번엔 함수를 넘겨줬다. 함수 이름만 전달한다면 위와 같은 경우에는 문제가 발생하지 않는다. 하지만 이 코드가 혼란을 야기할 수 있지 않을까? `timesTwo`가 객체가 아닌 함수라는 것을 어떻게 확신할 수 있을까? `map`이라는 메소드가 이에 대한 힌트를 줄 수 있지만, 자세한 것을 알기엔 부족하다. `timesTwo`가 다른 곳에서 초기화 되거나 선언된다면 어떻게 되는가? 찾기 쉬워질까? 코드가 무엇을 하고 있고 어떤 영향을 미치지는지 확인하는 것은 매우 중요하다.

보는 것처럼, 명확한 대답은 없다. 그러나 코드 베이스에 적합한 선택을 한다는 것은 특정 동작을 수행하는 코드를 작성할 수 있는 모든 옵션과 그 한계를 명확히 이해한다는 것을 의미한다. 일관성을 유지하기 위해서는 괄호, 중괄호, return 키워드 등이 중요하다.

코드를 작성할 때는 항상 스스로에게 질문을 해야 한다. 가장 먼저 해야할 질문은 성능이다. 하지만 기능적으로, 그리고 성능적으로 동일한 코드를 볼 떄는 인간이 코드를 읽는 방식으로 판단해야 한다.

```javascript
const { node } = exampleObject
```

위 코드는 변수를 초기화하고 할당하는 것을 모두 한줄에서 한다. 물론, 그렇게 할 필요는 없다.

```javascript
let node
;({ node } = exampleObject)
```

이 코드도 같은일을 한다. 하지만 이 코드를 자세히 보면 어색한 점이 많다. 세미 콜론을 사용하지 않는 코드에 어색한 코드를 강제로 줄 시작에 붙인다. 명령을 괄호안에 넣고 또 중괄호로 묶었다. 중괄호가 무엇을 하는지는 전혀 알수 없다. 결론적으로, 읽기가 쉽지 않다.

```javascript
let node
node = exampleObject.node
```

반면에 이 코드는 무엇을 하는지 명확하고, 누가 봐도 이해할 수 있다. 꼭 구조 분해 할당이 있다고 해서, 꼭 그것을 사용해야하는 것은 아니다.

물론, 그렇다고 해서 위코드로 써야 한다는 것이 아니다. `let`이 있다면 코드를 읽는 사람은 항상 이 변수가 언제 어디서 재할당되는지 긴장감을 가지고 살펴봐야 한다. 따라서, 이 경우에는 구조 분해할당을 하는 것이 좋다.

이번엔 전개 연산자와 `concat`에 대해 알아보자.

전개 연산자는 ECMAScript에서 새로나온 기능으로, 코드에 널리 사용되고 있다. 다양한 작업을 할 수 있는데, 그중 하나가 두 배열을 합치는 것이다.

```javascript
const arr1 = [1, 2, 3]
const arr2 = [9, 11, 13]
const nums = [...arr1, ...arr2]
```

물론 자바스크립트 코드에 익숙한 사람이 본다면 쉽게 이해할 수 있지만, 그렇지 않다면 직관적으로 무엇을 하는지 이해하기 어렵다. 모든 사람이 자바스크립트 코드에 익숙하다면 상관없지만, 그렇지 않은 사람들에게 혼란을 빚을 수 있다. 이 경우에는, `concat`이 훨씬 직관적일 것이다.

```javascript
const arr1 = [1, 2, 3]
const arr2 = [9, 11, 13]
const nums = arr1.concat(arr2)
```

자바스크립트 코드를 짜고 있는 인적 요인이 코드 선택에 이런식으로 영향을 미칠 수 있다. 또 자바스크립트 코드가 최신 코드 베이스가 아니라면 보다 엄격하게 표준을 유지해야한다.

> 이와 별개로, 전개연산자는 성능에 그다지 좋지 못하다.

```javascript
const array1 = []
const array2 = []
const mergeCount = 50
let spreadTime = 0
let concatTime = 0

for (let i = 0; i < 10000000; ++i) {
  array1.push(i)
  array2.push(i)
}

// The spread syntax performance test.
for (let i = 0; i < mergeCount; ++i) {
  const startTime = performance.now()
  const array3 = [...array1, ...array2]

  spreadTime += performance.now() - startTime
}

// The concat performance test.
for (let i = 0; i < mergeCount; ++i) {
  const startTime = performance.now()
  const array3 = array1.concat(array2)

  concatTime += performance.now() - startTime
}

console.log(spreadTime / mergeCount)
console.log(concatTime / mergeCount)

/*
 * Performance results.
 * Browser           Spread syntax      concat method
 * --------------------------------------------------
 * Chrome 75         626.43ms           235.13ms
 * Firefox 68        928.40ms           821.30ms
 * Safari 12         165.44ms           152.04ms
 * Edge 18           1784.72ms          703.41ms
 * Opera 62          590.10ms           213.45ms
 * --------------------------------------------------
 */
```

> 따라서, 이 경우에는 `concat`을 쓰는 것이 좋다.

전문가는 스펙의 모든 부분을 사용하는 사람이 아니라, 현명하게 문법을 배치하고, 합리적인 결정을 내릴 수 있을 만큼 스펙을 이해하고 있는 사람이다. 우리 스스로가 전문가가 되기 위해서는 어떻게 해야할까? 코드를 작성해야 하는 것은 스스로에게 많은 질문을 하는 것을 의미하기도 한다. 이는 개발자가 다른 개발자를 고객으로 바라보고 고려하는 것을 의미하기도 한다. 작성할 수 있는 최상의 코드는 복잡한 기능을 수행하면서도, 다른 사람이 코드를 보고 쉽게 이해할 수 있도록 만드는 코드다. 그리고 이는 쉽지 않다. 그리고 대부분의 경우 명확한 답이 없다. 하지만 우리는 매번 코드를 쓸 때 이점을 뒤돌아 봐야 한다.
