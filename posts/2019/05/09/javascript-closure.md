---
title: Javascript - Closure
date: 2019-05-09 08:11:56
published: true
tags:
  - javascript
mathjax: true
description: '## 클로저 ### 자바스크립트는 어떻게 변수의 유효 범위를 정하는가?  ```javascript function
  hello() {   var name = "yceffort"   // 내부함수이며, 클로저다.   function showName()
  {     // 부모함수가 선언한 변수를 사용한다.     alert(`hello, ${name}`)   }...'
category: javascript
slug: /2019/05/09/javascript-closure/
template: post
---
## 클로저

### 자바스크립트는 어떻게 변수의 유효 범위를 정하는가?

```javascript
function hello() {
  var name = "yceffort"
  // 내부함수이며, 클로저다.
  function showName() {
    // 부모함수가 선언한 변수를 사용한다.
    alert(`hello, ${name}`)
  }
  showName()
}
hello()
```

여기에서 `hello()`는 지역변수 `name`과 함수 `showName()`을 생성했다. `showName()`은 내부함수이므로, `hello()`에서만 사용이 가능하다. `showName()`은 별도의 지역변수가 없지만 내부함수는 외부함수에 접근할 권한을 가지고 있으므로, `name`이 정상적으로 출력될 것이다. 만약 name이라는 다른 변수가 내부 함수에 있다면, 그 변수를 우선적으로 사용할 것이다.

`Lexical`은 변수가 사용가능한 범위를 결정하기 위해 소스코드 내에서 변수가 선언된 위치를 사용한다는 것을 말한다. 따라서 내부 함수들은 그들의 외부 유효 범위 내에서 선언된 변수들에 접근할 권한을 가진다.

### 클로저란 무엇인가

```javascript
function hello() {
  var name = "yceffort"
  function showName() {
    alert(`hello, ${name}`)
  }
  return showName
}

let sayHello = hello()
sayHello()
```

이 전과 완전히 똑같은 결과를 보일 것이다. 차이점은, `hello()`가 내부 함수 `showName`를 리턴했다는 것, 그리고 그렇게 리턴한 정보를 `sayHello`변수에 저장했다는 것이다. 얼핏보면 잘 이해가 되지 않는 모습이다. `hello()`는 `showName()`만을 리턴했는데, 계속해서 `name`변수에 접근하고 있기 때문이다.

그 이유는, 자바스크립트가 함수를 리턴할때, 리턴하는 함수가 클로저를 생성하기 때문이다. 클로저는 함수와 함수가 선언된 어휘적 환경의 조합이다. (함수가 선언된 환경을 기억한다.) 여기에서 환경은, 클로저가 생성된 시점에 유효범위내에 있는 모든 지역변수로 구성된다. (내부 함수가 외부 함수의 변수에 접근할 수 있었기 때문에 그 변수들을 기억하는 것)

```javascript
function add(x) {
  var y = 1
  return function(z) {
    y = 100
    return x + y + z
  }
}

// 클로저 선언
let add5 = add(5)
let add10 = add(10)

console.log(add5(2))
console.log(add10(2))
```

`add`함수는, `x`를 인자로 받아서 새로운 내부 함수를 반환한다. 이 내부 함수는 `z`를 받아서 `x+y+z`를 반환한다. `add5`와 `add10`은 모두 클로저다. 이 두 함수의 결과는 어떻게 될까?

첫번째 선언 `let add5 = add(5)`에서 일단 x가 5로 할당이 되었다. 그리고 두번째 `add5(2)`에서는 z가 2로 할당이 되었다. 그리고 y가 두군데 할당이 되어있으므로, 내부를 우선시하여 y는 100이다. 따라서 $$x+y+z=5+100+2=107$$ 이 된다. 마찬가지로, `add10`은 $$x+y+z=10+100+2=112$$가 된다.

본질적으로, 이 두개는 같은 함수의 본문을 정의하지만, 서로 다른 환경을 저장한다. 이는 클로저가 리턴된 후에도 외부 함수의 변수에 접근이 가능하다는 것을 보여주며, 단순히 값 형태로 전달되는 것이 아니라는 것을 의미한다.

### 어디다 쓸까

클로저는 어휘적인 환경과 데이터를 조작하는 함수를 연관시켜 주기 때문에 유용하다. 이는 객체가 어떤 데이터 (속성)과 그 메소드를 연관시킨 다는 점에서 객체지향 프로그래밍과 같은 맥락에 있다. 따라서, 단 하나의 메소드 만을 가지고 있는 객체를 일반적으로 사용하는 모든 곳에 클로저를 사용할 수 있다.

이는 프론트엔드 자바스크립트 이벤트에서 흔히 볼 수 있다. 사전에 몇가지 동작을 정의한 후에, 사용자가 이벤트를 트리거 하면 이 동작들을 연결하는데 이는 이벤트에 응답하여 실행되는 단일 함수다.

```javascript
function makeFontSize(size) {
  return function() {
    document.body.style.fontSize = size + "px"
  }
}

let size12 = makeFontSize(12)
let size14 = makeFontSize(14)
let size16 = makeFontSize(16)

document.getElementById("size-12").onclick = size12
document.getElementById("size-14").onclick = size14
document.getElementById("size-16").onclick = size16
```

프라이빗 메소드를 흉내내는 것도 가능하다. 프라이빗 메소드는 코드에 제한적인 접근만 허용할 수 있고, 전역 네임스페이스를 관리하는 방법을 제공하여 불필요한 메소드가 공용 인터페이스를 혼란스럽게 만들지 않도록 할 수 있다.

```javascript
let counter = (function() {
  let privateCounter = 0
  function change(val) {
    privateCounter += val
  }

  return {
    increment: function() {
      change(1)
    },
    decrement: function() {
      change(-1)
    },
    value: function() {
      return privateCounter
    },
  }
})()

counter.increment()
counter.value()
counter.increment()
counter.increment()
counter.decrement()
counter.value()
```

`change()` `privateCounter`는 모두 익명함수 내부에서 생성되었기 때문에 접근할 수 없다. 이 익명함수에서 접근할 수 있는건 익명래퍼에서 반환된 세개의 퍼블릭함수 `increment()` `decrement()` `value()` 뿐이다. 위 처럼 즉시실행익명함수가 아니라 별도의 함수로 만들어서 따로 쓴다면, 객체지향 프로그래밍의 은닉과 캡슐화 같은 이점들을 얻을 수 있다.

### 루프에서의 클로저

<iframe width="640px" height="360px" width="100%" height="600" src="//jsfiddle.net/yceffort/n23uLwak/embedded/js,html,result/dark/" allowFullScreen="allowFullScreen" frameBorder="0"></iframe>

이 함수는 생각처럼 작동하지 않는다. 그 이유는 `onfcus`에 연결된 함수가 클로저이기 때문이다. 이 클로저는 `setupHelp()` 함수범위에서 캡쳐된 환경으로 구성된다. 루프에서 세개의 세개의 클로저가 만들어졌지만, 각 클로저는 값이 변하는 변수 `item.help`가 있는 단일 환경을 공유한다. 따라서 계속해서 마지막 변수를 가르키게 되는 것이다.

첫번째 해결방안은 `showHelp()`를 감싸는 클로저를 만드는 것이다.

<iframe width="640px" height="360px" width="100%" height="600" src="//jsfiddle.net/yceffort/tn5o29hv/embedded/js,html,result/dark/" allowFullScreen="allowFullScreen" frameBorder="0"></iframe>

`showHelp()`는 여전히 단일 환경에서 작동하지만, `makeHelpCallback()`는 매번 새로운 클로저를 만들어서 새로운 환경을 형성한다.

아니면 즉시실행익명함수를 만들어서 for 구문내의 환경이 별로 즉시로 실행되게 하는 방법도 있을 수 있다.

<iframe width="640px" height="360px" width="100%" height="600" src="//jsfiddle.net/yceffort/toc4mkbw/embedded/js,html,result/dark/" allowFullScreen="allowFullScreen" frameBorder="0"></iframe>

반드시 for 구문 내의 로직을 즉시실행함수로 감싸서 별도의 환경으로 구성되게 해야 한다.

아니면 let을 사용하여 `item`변수의 범위자체를 for문 내로 제한할 수도 있다.

<iframe width="640px" height="360px" width="100%" height="600" src="//jsfiddle.net/yceffort/7v0Lrswb/embedded/js,html,result/dark/" allowFullScreen="allowFullScreen" frameBorder="0"></iframe>

### 성능

클로저가 필요하지 않은 작업에 다른 함수내에서 함수를 불필요하게 계속 선언하고 작성하는 것은 성능에 악영향을 미친다. 예를 들어, 새로운 객체나 클래스를 생성할때 메소드를 객체 생성자에 정의하는 것 보다는 객체의 프로토타입에 연결해야 한다.

#### 안좋은 예

```javascript
function MyObject(name, message) {
  this.name = name.toString()
  this.message = message.toString()
  this.getName = function() {
    return this.name
  }

  this.getMessage = function() {
    return this.message
  }
}
```

이렇게 하기보다는 prototype에 정의하는 것이 훨씬 낫다.

```javascript
function MyObject(name, message) {
  this.name = name.toString()
  this.message = message.toString()
}
MyObject.prototype.getName = function() {
  return this.name
}
MyObject.prototype.getMessage = function() {
  return this.message
}
```

```javascript
function MyObject(name, message) {
  this.name = name.toString()
  this.message = message.toString()
}
;(function() {
  this.getName = function() {
    return this.name
  }
  this.getMessage = function() {
    return this.message
  }
}.call(MyObject.prototype))
```

이렇게 쓴다면 좀더 섹시해 보일 것이다.
