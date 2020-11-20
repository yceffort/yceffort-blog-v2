---
title: Nodejs 성능 최적화를 위한 방법
tags:
  - javascript
  - nodejs
  - V8
published: true
date: 2020-11-19 23:22:28
description: '성능은 좋을 수록 좋다 그것이 성능이니까'
---

## Table of Contents

## 자바스크립트의 메모리 관리

Nodejs의 메모리 누수를 이해하기 위해서는, nodejs의 메모리 관리에 대해서 이해하고 있어야 한다. nodejs는 자바스크립트의 V8엔진을 사용하고 있다. 이에 대해 이해하기 위해서는, 아래 포스트를 먼저 참고할 필요가 있다.

- https://dev.to/deepu105/visualizing-memory-management-in-v8-engine-javascript-nodejs-deno-webassembly-105p
- https://yceffort.kr/2020/11/v8-memory-management

메모리는 크게 스택과 힙메모리로 구별할 수 있다.

- Stack: 메소드, 함수 프레임, 원시값, 객체의 포인터등 정적인 데이터가 저장되는 곳
- Heap: 객체 또는 다이나믹 데이터 등이 저장되는 곳. 메모리 블록중 가장 큰 영역이며, GC가 작업을 하는 곳

> V8은 가비지 콜렉션을 이용해서 힙 메모리를 관리한다. 간단히 얘기해, 스택에서 더이상 참조하지 않는 객체의 메모리를 해제하여 다른 객체가 메모리를 할당하여 쓸 수 있도록 한다. V8의 가비지 컬렉터는 더이상 사용하지 않는 메모리를 해제하여 공간을 확보하는 책임이 있다. V8 가비지 컬렉터는 객체를 생성시점으로 묶어서 각각 다른 스테이지별로 별도로 관리한다. V8 가비지 컬렉터는 2개의 다른 스테이지와 세개의 다른 알고리즘을 사용한다.

![V8 Garbage Collector](https://d33wubrfki0l68.cloudfront.net/e3979bee7b7b51e6124594ea36dfde4eb7015da5/5c860/images/blog/2020-05/mark-sweep-compact.gif)

## 메모리 누수란 무엇인가

간단히 말해 메모리 누수랑, 애플리케이션에서 더이상 사용하지 않는 메모리가 힙에서 계속 남아 있고, 그래서 이를 가비지 컬렉터가 OS로 메모리로 반환하지 못하는 상황을 의미한다. 이는 메모리에서 쓸모없는 블록으로 존재하게 된다. 이러한 블록이 계속해서 생기게 되면 애플리케이션에서는 더이상 사용할 메모리가 존재하지 않게 되고, 나아가 OS 또한 할당할 메모리가 남아나지 않아서 애플리케이션이 느려지고 크래쉬되거나, 혹은 OS 단에서 문제가 발생할 수 있다.

## 자바스크립트에서는 무엇이 메모리 누수를 발생시키는가?

V8의 가비지 콜렉터와 같은 자동 메모리 관리는 메모리 누수를 피하는데 초점이 맞춰져있다. 예를 들어 순환 참조는 가비지 콜렉터의 고려대상이 아니지만, 힙의 원치 않는 참조로 인해 문제가 발생할 수 있다. 일반적인 메모리 누수의 상황은 아래와 같다.

- 전역변수: 자바스크립트의 전역 변수는 루트 노드를 참조하기 때문에 (`window`, `global`) 애플리케이션의 생명주기 동안 절대로 가비지 콜렉팅이 되지 않아 계속해서 메모리를 점유하고 있게 된다. 따라서 글로벌 변수를 참조 하고 있는 객체 또한 가비지 콜렉팅의 대상이 되지 않는다는 것을 의미한다. 루트로부터 커다란 객체 참조 그래프를 가지고 있다는 것은 결국 메모리 누수로 이어지게 된다.
- 동시참조: 하나의 동일한 객체가 다양한 객체에서 참조될 때, 이 중 하나의 참조가 잘못된다면 전체 객체에서 메모리 누수가 발생할 수 있다.
- 클로져: 자바스크립트의 클로져는 코드를 둘러싼 콘텍스트를 기억한다는 점에서 멋진 기능이다. 클로져가 힙의 큰객체의 클로져를 참조하고 있다면, 클로져가 사용될 때 까지 그 객체는 메모리에 남아 있게 된다. 이는 메모리 누수의 원인으로 이어질 수 있다.
- 타이머 & 이벤트: `setTimeout` `setInterval` `Observer` 이벤트 리스너 등의 콜백이 적절한 조치 없이 무거운 객체의 참조를 가지고 있을 경우 메모리 누수가 발생할 수 있다.

## 메모리 누수를 피하는 방법

### 전역 변수의 사용을 줄인다.

전역 변수는 절대로 가비지 컬렉팅 되지 않으므로, 전역변수를 남용하지 않는 것이 제일 좋다.

#### 실수로 전역변수를 선언하는 것을 주의하자.

만약 undeclare한 변수를 선언하게 되면, 자동으로 자바스크립트는 이를 호이스팅해서 전역 변수로 만들어 버린다. 이는 곧 메모리 누수로 이어지게 된다.

```javascript
function hello() {
  // 전역변수로 호이스팅 된다.
  foo = 'Message'
}

function hello() {
  // 여기서 this는 global 이기 때문에 마찬가지로 호이스팅되어 전역변수가 된다.
  this.foo = 'Message'
}
```

이러한 원치 안흔ㄴ 사고를 방지 하기 위해서는, 자바스크립트 파일 상단에 `'use strict';`를 선언해 두면된다. 엄격한 모드에서는, 위의 코드는 에러를 발생시킨다. 만약 ES 모듈이나 타입스크립트 또는 바벨과 같은 프랜스파일러를 사용한다면, 굳이 하지 않아도 된다. 최근 버전의 Nodejs에서는, `--use_strict` 옵션으로 nodejs 환경 전역에 이 모드를 활성화 시킬 수 있다.

```javascript
'use strict'

// This will not be hoisted as global variable
function hello() {
  foo = 'Message' // will throw runtime error
}

// This will not become global variable as global functions
// have their own `this` in strict mode
function hello() {
  this.foo = 'Message'
}
```

화살표 함수를 사용하면, 마찬가지로 전역변수를 생성할수도 있다는 사실을 조심해야 한다. 이러한 경우에는 엄격모드로는 해결할 수가 없고, eslint의 `no-invalid-this`로 해결하면 된다.

```javascript
// 전역변수로 할당된다.
const hello = () => {
    this.foo = 'Message";
}
```

마지막으로, `bind`와 `call`을 사용하는 함수에 전역 `this`를 바인딩하지 않도록 주의한다.

#### 글로벌 스코프 사용을 줄인다.

글로벌 스코프의 사용은 가능한 줄이는 것이 좋다.

1. 가능한, 글로벌 스코프는 사용하지 않는 것이 좋다. 대신, 함수의 지역 스코프를 사용하여 카비지 콜렉터가 원할 때 메모리를 수집할 수 있게 해주자. 만약 특별한 제한 때문에 전역 스코프를 사용해야 한다면, 더이상 사용하지 않게 되는 시점에 `null`을 넣어주면 된다.
2. 전역변수는 오직 상수, 캐시 또는 재사용할 싱글턴 패턴에만 사용해야 한다.함수와 클래스간에 데이터를 공유하기 위해서는, 파라미터와 객체의 속성값으로 전달해주는 것이 좋다.
3. 큰 객체를 전역 변수에 저장하지 말자. 만약 꼭 저장해야 한다면, 더이상 사용하지 않을 때 null 처리를 해줘야 한다. 캐시 객체의 경우, 이 객체가 점점 커지는 것을 방지해야 한다.

### 스택 메모리를 잘 활용하자.

스택 접근은 힙 접근 보다 성능적으로도 우월하고, 메모리의 효율성도 높기 때문에 가능한 스택 변수를 많이 활용해야 한다. 이는 또한 실수로 일어나는 메모리 누수도 방지해준다. 물론, 실무상으로 오로직 스태틱 데이터만 쓸 수 있는 일은 없다. 실제 에플리케이션은, 다양한 객체와 다이나믹 데이터를 사용해야 한다. 하지만 몇가지 트릭을 사용하여 스택을 조금 더 효율적으로 쓸 수 있다.

1. 스택 변수로부터 힙객체 참조하는 것을 가능한 피해야 한다. 또한, 사용하지 않는 변수를 그냥 둬서는 안된다.
2. 객체나 배열 내부의 값을 넘길 때는 전체 객체를 통째로 넘기는 대신에 이를 분해해서 필요한 것만 넘기는 것이 좋다. 이는 클로져 내부에서 불필요한 객체 참조를 피할 수 있다. 객체내부의 값은 대부분 원시값이므로, 이는 스택을 사용하는데 도움이 될 것이다.

```javascript
function outer() {
    const obj = {
        foo: 1,
        bar: "hello",
    };

    const closure = () {
      // 구조분해 할당을 써서 필요한 foo만 꺼내왔다.
        const { foo } = obj;
        myFunc(foo);
    }
}

function myFunc(foo) {}
```

### 힙 메모리를 효율적으로 활용하자

실제 애플리케이션에서 힙메모리의 사용을 피할수는 없지만, 아래 팁들을 이용하면 좀더 효율적으로 사용할 수 있다.

1. 참조를 넘기는 대신에 가능하면 객체를 복사하는게 좋다. 참조를 넘기는 것은 객체가 크거나, 복사하는 작업이 비쌀때만 활용한다.
2. 객체의 변이를 가능한 피해야 한다. 그 대신 전개 연산자를 사용하거나 `Object.assign`으로 복사하는 것이 좋다.
3. 하나의 객체에 여러가지 참조를 만드는 것을 피해야 한다. 대신에 객체를 복사하는 것이 좋다.
4. 수명이 짧은 변수를 활용하자.
5. 큰 객체 트리를 만드는 것을 피해야 한다. 만약 이러한 것이 불가능하다면, 지역변수 내에서 보관하는 것이 좋다.

### 클로저, 타이버, 이벤트 핸들러를 적절히 활용하자.

앞서 언급했던 것처럼 클로져, 타이버, 그리고 이벤트 핸들러는 메모리 누수가 일어날 수 있는 영역이다. 아래 코드를 살펴보자.`longStr`은 절대 가비지 콜렉팅이 되지 않고, 또한 점점 커지기 때문에 메모리 누수의 원인이 된다.

참고: https://blog.meteor.com/an-interesting-kind-of-javascript-memory-leak-8b47d2e7f156?gi=275d4bdd446b

```javascript
var theThing = null
var replaceThing = function () {
  var originalThing = theThing
  var unused = function () {
    if (originalThing) console.log('hi')
  }
  theThing = {
    longStr: new Array(1000000).join('*'),
    someMethod: function () {
      console.log(someMessage)
    },
  }
}
setInterval(replaceThing, 1000)
```

위 코드는 여러 클로져를 만들고, 이 클로져들은 각각 객체 참조를 가지고 있게 된다. 이 경우 메모리 누수를 해결하기 위해서는 `replaceThing`함수 끝에서 `originalThing`을 `null`로 선언해주어야 한다. 이러한 경우도 객체의 복사본을 만들거나, 앞서 언급한 `null`을 하는 전략으로 메모리 누수를 피할 수 있다.

이벤트 리스너와 observer도 마찬가지다. 작업이 끝나면 이들을 클리어해주어야 한다. 이들이 영원히 참조하고 있게 해서는 안된다. 특히 부모 스코프의 객체를 참조하고 있다면 더욱 위험하다.

## 결론

자바스크립트 엔진의 진화와 언어의 성장으로 인하여, 자바스크립트의 메모리 누수는 우리가 생각하는 것 만큼 잦은 이슈는 아니다. 그러나 주의를 기울이지않으면, 성능 문제를 야기하거나 애플리케이션과 OS의 크래쉬를 야기할 수 있다. 메모리 누수가 일어나지 않기 위해 첫번째로 우리가 할일은 V8이 어떻게 메모리를 관리하는 지다. 그 다음에는 무엇이 메모리 누수를 일으키는지 알아야 한다. 이에 대해 이해하고 있고, 그리고 만약 메모리 누수 문제가 발생한다면, 우리는 무엇을 살펴보아야 하는지 알 수 있게 된다. 만약 Nodejs에서 메모리 누수 문제가 발생한다면, 아래 두 개의 링크를 확인해보자.

- https://github.com/lloyd/node-memwatch
- https://nodejs.org/en/docs/guides/debugging-getting-started/

출처

- https://blog.appsignal.com/2020/05/06/avoiding-memory-leaks-in-nodejs-best-practices-for-performance.html

참고

- https://www.ibm.com/developerworks/web/library/wa-memleak/wa-memleak-pdf.pdf
- https://developer.mozilla.org/en-US/docs/Web/JavaScript/Memory_Management
- https://docs.microsoft.com/en-us/previous-versions/msdn10/ff728624(v=msdn.10)
- https://auth0.com/blog/four-types-of-leaks-in-your-javascript-code-and-how-to-get-rid-of-them/
- https://blog.meteor.com/an-interesting-kind-of-javascript-memory-leak-8b47d2e7f156
