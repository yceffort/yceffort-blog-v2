---
title: 자바스크립트의 프록시
date: 2021-03-12 18:34:41
tags:
  - javascript
published: true
description: 'IE를 주깁시다'
---

ES6에서 새롭게 나온 Proxy는 요즘 많은 프레임워크에서 (요즘이라기엔 많이 지났지만,) 주목하고 있는 기능인 것 같다.

[mobx에서도 proxy를 쓰고 있고](https://mobx.js.org/configuration.html) [2010 JSconf 에서도 관련 영상이 존재하고](https://www.youtube.com/watch?v=sClk6aB_CPk&ab_channel=JSConf) (무려 11년전,,) [vue.js](https://v3.vuejs.org/guide/reactivity.html#what-is-reactivity)에서도 reactivity 지원을 위해 Proxy를 사용하고 있다.

## Proxy

[Proxy 객체는 기본적인 동작(속성 접근, 할당, 순회, 열거, 함수 호출 등)의 새로운 행동을 정의할 때 사용합니다.](https://developer.mozilla.org/ko/docs/Web/JavaScript/Reference/Global_Objects/Proxy) 라고 되어 있다. 즉, 특정 객체의 읽기 쓰기 등 객체에 가해지는 작업을 중간에 가로채서 새로운 작업을 할 수 있는 것을 말한다. 프록시를 이해 하기 위해서는 아래의 용어에 대해 이해하고 있어야 한다.

- `target`: 기본 동작을 가로챌, 즉 감싸게 될 객체로 함수를 포함해서 모든 객체가 가능하다.
- `handler`: 동작을 가로채는 메서드인 `trap`을 가지고 있는 객체로, 여기에서 프록시를 설정한다.

먼저 트랩이 존재하지 않는 (= 동작을 가로채는 메서드가 없는) 예제를 살펴보자.

```javascript
const target = {}
const proxy = new Proxy(target, {}) // 핸들러가 없다

proxy.test = 5 // 프록시에 값을 썼는데

console.log(target.test) // 5 타겟에도 프로퍼티가 추가됐다.
console.log(proxy.test) // 5 프록시를 통해서도 읽을 수 있다.

console.log(proxy) // Proxy {test: 5}
console.log(target) // {test: 5}
```

마치 프록시가 타겟을 감싸는 래퍼처럼 작동한다.

이제 본격적으로 트랩을 추가하기 전에, 프록시가 가로챌 수 있는 작업에는 무엇이 있는지 알아보자.

- `get`
- `set`
- `has`
- `deleteProperty`
- `apply`
- `constructor`
- `getPrototypeOf`
- `setPrototypeOf`
- `isExtensible`
- `preventExtensions`
- `getOwnPropertyDescriptor`
- `ownKeys`

이 중에서, 가장 기본적인 예제인 `get` 트랩을 만들어보자.

```javascript
const arr = new Proxy([0, 1, 2, 3], {
  get(target, prop) {
    if (prop in target) {
      return target[prop]
    } else {
      console.log(`${prop}은 존재하지 않습니다`)
      return 0
    }
  },
})

console.log(arr[0]) // 1
console.log(arr[1]) // 2
console.log(arr[100]) // 0
// 100은 존재하지 않습니다 proxyConsoleLog.js:12
```

프록시에서 정의한 trap이 get을 가로채서 작동하고 있음을 알 수 있다.

이번엔 `set`트랩을 만들어보자. 다른언어의 그것 처럼, 숫자만 추가할 수 있는 배열을 만들어보자.

```javascript
const arr = new Proxy([], {
  set(target, prop, value) {
    if (typeof value === 'number') {
      target[prop] = value
      return true
    } else {
      return false
    }
  },
})

arr.push(1)
arr.push(2)
arr.push(3)
arr.push('졸려') // VM503:1 Uncaught TypeError: 'set' on proxy: trap returned falsish for property '3'

console.log([...arr]) // 1, 2, 3
```

성공적으로 push를 막은 것을 볼 수 있다.

## Reflect

[`Reflect`는 중간에서 가로챌 수 있는 작업에 대한 메소드를 제공한다.](https://developer.mozilla.org/ko/docs/Web/JavaScript/Reference/Global_Objects/Reflect) `Reflect`는 생성자 함수가 아니므로, 인스턴스를 만들거나 `new`로 호출할 없다.

요약해서 말하자면, `Proxy` 생성을 단순화 한 빌트인 객체라고 보면 된다.

```javascript
const user = new Proxy(
  {name: 'John'},
  {
    get(target, prop, receiver) {
      return target[prop]
    },
    set(target, prop, val, receiver) {
      target[prop] = value
      return true
    },
  },
)
```

```javascript
const user = new Proxy(
  {name: 'John'},
  {
    get(target, prop, receiver) {
      return Reflect.get(target, prop, receiver) // target[prop] 를 대체 해주었다.
    },
    set(target, prop, val, receiver) {
      return Reflect.set(target, prop, val, receiver) // target[prop] = value 를 대체해주고, true/false도 리턴해준다.
    },
  },
)

console.log(user.name) // John
user.name = 'Pete' // pete
```

## `Observable` 만들기

mobx의 그것인 observable을 만들어보자.

```javascript
// 모든 객체에 공통적으로 observe를 달아둘 심볼
// 심볼로 선언하여 모든 객체에서 동일한 방법으로 접근할 수 있도록 한다.
const handlers = Symbol.for('handlers')

// 객체를 넘겨 받아서 observable 하게 만든다.
function observable(target) {
  // observe가 들어가는 곳
  target[handlers] = []

  // observe에 함수가 들어오면, handlers에 넣어 둔다.
  target.observe = function (handler) {
    this[handlers].push(handler)
  }

  // 프록시를 리턴한다.
  return new Proxy(target, {
    set(target, property, value, receiver) {
      // Reflect.set으로 값을 설정한다.
      const result = Reflect.set(...arguments)
      if (result) {
        // 각 handler에 현재 set arguments를 넘겨준다.
        target[handlers].forEach((handler) =>
          handler({target, property, value, receiver}),
        )
      }
      return result
    },
  })
}

const user = observable({})

user.observe(({property, value}) => {
  console.log(`'${property}' => '${value}'`)
})

user.name = '삼전주가떡상기원' // 'name' => '삼전주가떡상기원'
```

## Polyfill

바벨의 문서에는 다음과 같이 적혀있다.

https://babeljs.io/docs/en/learn/#ecmascript-2015-features-proxies

> Unsupported feature
>
> Due to the limitations of ES5, Proxies cannot be transpiled or polyfilled. See support in various JavaScript engines.

[따라서 babel repl에서 시도해보아도, 별 소용이 없다..](https://babeljs.io/repl#?browsers=defaults%2C%20ie%2011%2C%20not%20ie_mob%2011&build=&builtIns=false&spec=false&loose=false&code_lz=MYewdgzgLgBAhgJwTAvDMBTA7jACgkADwE8AKAbQF0AaGAbwCgYYIMpSpEBzN2gBwJ9aANzgAbAK4YAlPWbMm8gJYAzGB2J8MINaMkZUKNAHIwEgLYAjDAmOzG8-ZwQ8o5ASD6VUMPVMWOCGwSCGAwUAj-jgC-MBhirPQB8kFQIWEq4qzJ0Yq50dIMQA&debug=false&forceAllTransforms=false&shippedProposals=false&circleciRepo=&evaluate=false&fileSize=false&timeTravel=false&sourceType=module&lineWrap=true&presets=env&prettier=false&targets=&version=7.13.10&externalPlugins=)

- https://kangax.github.io/compat-table/es6/#test-Proxy
- https://github.com/GoogleChrome/proxy-polyfill
- https://caniuse.com/proxy

아쉽게도, 완벽하게 polyfill이 지원이 되지 않는다. 구글 크롬팀에서 만든 폴리필도 몇가지 밖에 동작하지 않는다.

> Currently, the following traps are supported-
>
> - get
> - set
> - apply
> - construct

따라서, ie 브라우저에서는 사용할 수 없다.
