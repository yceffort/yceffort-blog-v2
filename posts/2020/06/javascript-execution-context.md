---
title: Javascript Execution Context
tags:
  - javascript
published: true
date: 2020-06-26 04:25:30
description: '들어가기에 앞서 더 좋고 제가 많이 참고한 글이
  [여기](https://poiemaweb.com/js-execution-context)에 있습니다. 이글을 보시는게 낫습니다. ```toc
  tight: true, from-heading: 1 to-heading: 4 ```  # 자바스크립트 실행컨텍스트  이번 포스팅으로
  자바스크립트 실행 컨텍스트에 대해 온...'
category: javascript
slug: /2020/06/javascript-execution-context/
template: post
---

들어가기에 앞서 더 좋고 제가 많이 참고한 글이 [여기](https://poiemaweb.com/js-execution-context)에 있습니다. 이글을 보시는게 낫습니다.

## Table of Contents

# 자바스크립트 실행컨텍스트

이번 포스팅으로 자바스크립트 실행 컨텍스트에 대해 온전히 이해하길 바라며 🤔

## 실행 컨텍스트의 정의

실행 컨텍스트에 대한 정의는 아래처럼 나타나 있다.

> Execution context (abbreviated form — EC) is the abstract concept used by ECMA-262 specification for typification and differentiation of an executable code.

> Execution Context (이하 EC2)는 ECMA-262에서 명세되어 있는 추상적인 개념으로, 실행가능한 코드를 형상화 하고 구분하는데 사용된다.

여기 저기 블로그를 들쑤시고 다닌 결과, 대체로 **실행 가능한 코드를 실행하는데 있어 필요한 환경** 정도로 의미를 부여하는 것 같다. 실행 가능한 코드는 크게 세 종류가 있다. 그러나 보통 두 종류만 이야기 한다.

Global Code (aka 전역 코드): 프로그램 레벨에서 실행되는 코드로, `.js` 파일 또는 로컬 인라인 코드 (`<script></script>`) 등을 의미한다. 전역 코드는 어떠한 함수의 바디에 포함되지 않는다. 쉽게 얘기해서 전역 레벨의 코드를 의미한다. 여기에서 ECStack은 아래와 같이 생성된다.

```
ECStack = [globalContext];
```

Function Code (aka 함수 코드): 함수 코드에 진입하게 되었을 때, ECStack에 새로운 엘리먼트가 푸쉬된다. 여기에서 중요한 것은, 함수 내부의 함수 코드는 포함되지 않는 다는 것이다. 무슨 말인고 하니, 아래 코드를 살펴보자.

```javascript
;(function foo(flag) {
  if (flag) {
    return
  }
  foo(true)
})(false)
```

이에 ECStack은 이렇게 수정된다.

```
// 처음에 foo함수를 실행했다가
ECStack = [
  <foo> functionContext
  globalContext
];

// 재귀적으로 다시 foo를 실행한다.
ECStack = [
  <foo> functionContext – recursively
  <foo> functionContext
  globalContext
];
```

모든 함수의 return 문은 현재 실행 컨텍스트를 끝내며, (에러를 던져도 `throw Error` 끝나긴 한다.) 이에 따라 `ECStack`에서 `pop()`된다. 이런 과정을 거치다보면, 결국 `ECStack`은 프로그램 종료시점에 `globalStack`만 남게 된다.

`Eval` Code: 자바스크립트 시간에 쓰지말라고 신신당부하는 그 코드다. 굳이 쓸일이 없으니 자세한 설명은 생략한다.

아무튼, 실행 가능한 코드는 이렇게 3종류가 있다.

자바스크립트 엔진은, 코드 실행을 위해 여러가지 정보를 알고 있어야 한다. 이러한 정보에는 다음과 같은 것들이 있다.

- 변수: 전역, 지역, 매개변수, 객체의 속성 등
- 함수 선언
- 변수의 유효범위 (scope)
- this

아래 코드 예제를 살펴보자.

```javascript
var x = 1

function foo() {
  var arg = arguments
  var y = 2

  function bar() {
    var z = 3
    console.log(x + y + z)
  }
  bar()
}

foo()
```

위 코드를 실행하면, 실행컨텍스트 스택이 아래와 같이 생성되고 소멸된다. 현재 실행중인 컨텍스트에서 이 컨텍스트와 관련없는 코드가 실행되면, 새로운 컨텍스트를 만든다. 이 컨텍스를 스택에 쌓고, 제어권은 이 추가된 컨텍스트에 이동된다.

```
1. [global EC]
2. [global EC, foo() EC]
3. [global EC, foo() EC, bar() EC]
4. [global EC, foo() EC]
5. [global EC]
```

- 모두가 아는 것처럼, 실행 컨텍스트는 스택 구조다. (LIFO)
- 전역 컨텍스트는 제어권이 진입하면 생성되고, 실행컨텍스트는 위처럼 생성되고 빠지길 반복하며, 전역 컨텍스트는 애플리케이션 종료 시점까지 유지된다.
- 함수를 호출하면 해당 함수의 컨텍스트를 만들고, 실행컨텍스트 스택에 쌓는다
- 함수실행이 끝나면 이를 `pop()`하고 이 직전 실행컨텍스트에 제어권을 넘긴다.

## 실행 컨텍스트의 구성요소

실행컨텍스트는 물리적으로 객체의 형태를 가지며, 3가지 프로퍼티를 가지고 있다.

- Variable Object (변수객체)
- Scope Chain (스코프 체인)
- this

### Variable Object (변수객체)

실행컨텍스트가 생성되면, 실행에 필요한 여러정보를 담을 객체를 생성하는데 이를 변수객체라고 한다. 변수객체는 아래 세가지 정보를 담는다.

- 변수
- Parameter & Arguments
- 함수 선언 (오로지 선언만)

> 파라미터는 함수에 넘기게될 값들의 alias고, argument는 parameter에 넘기는 값을 의미한다.

변수 객체는, 실행 컨텍스트의 프로퍼티이기 때문에 다른 객체를 가르키는 값을 갖는다. 그리고 전역 컨텍스트와 함수 컨텍스트의 경우에는 각각 가르키는 객체가 다르다. 예를 들어 함수 컨텍스트에는 매개변수가 있다.

#### 전역 컨텍스트의 변수 객체

최상위에 있으며, 모든 전역 변수 및 전역 함수등을 포함하는 전역 객체 (Global Object)를 가르킨다. 전역객체는 전역에 선언된 모든 전역 변수와 전역함수를 프로퍼티로 선언한다.

위 예제에서 전역으로 선언된 것은 함수 객체인 foo와 1의 값을 가진 x가 될 것이다.

#### 함수 컨텍스트의 변수 객체

함수 컨텍스트의 변수 객체는 Activation Object(활성 객체)를 가리키며, 인수들의 정보를 배열로 담고 있는 argument object가 추가된다.

`foo()`를 예로 들어보자. 변수 y, `bar()` 그리고 외부에서 파라미터를 통해 전달 받은 `arguments`가 추가된다.

### 스코프 체인 (Scope chain)

여기저기서 어렵게 설명되어 있어서 헷갈렸는데 - 스코프체인은 전역 또는 함수가 참조할 수 있는 변수, 함수선언등의 정보를 담고 있는 전역객체나 활성객체의 리스트를 말한다.

말이 어려우니, 쉽게 이야기 해보자.

`foo()`함수는 앞서 `arguments`, `bar()`, `y`를 가르키고 있는 활성객체를 변수객체로 가지고 있다. 스코프 체인에 이 활성객체가 들어가 있다.

그리고 그 다음으로 상위 컨텍스트의 활성 객체를 가르키고 있다. `foo()`의 상위 객체는 전역 컨텍스트의 변수객체인 전영객체다.

결론적으로 `foo()`의 스코프체인은 각각 `foo()`의 `AO`, 그리고 `global`의 `GO`를 가지고 있게 된다.

전역 객체는 어떨까? 전역 객체는 가장 최상위 이므로, 오로지 `GO`만을 스코프체인에 보관하게 된다.

**결론적으로, 스코프 체인은 변수 객체를 검색하는 메커니즘이다.**

엔진은 스코프 체인을 활용해서 렉시컬 스코프를 파악한다. 함수가 `foo()`처럼 중첩되어 있을때, 하위함수 안에서 상위함수, 심지어 전역 스코프 까지 참조할 수 있는건 이것은 스코프 체인이 있기 때문에 가능한 것이다. 함수 실행중에 변수를 만나면 그변수를 현재 스코프인 `AO`에서 검색해보고, 검색에 실패하면 스코프체인의 순서대로 한단계씩 위로 검색을 이어가게 된다.

그러나 이렇게 순차적으로 검색했는데 실패한다면, 정의되지 않는 변수에 접근하는 것으로 인식하고 잘 아는 에러인 Reference에러가 나게 된다.

위의 코드에 debugger를 추가해서 크롬 콘솔에서 실행해보면 위에서 설명한 내용을 볼 수가 있다.

```javascript
debugger

var x = 1

function foo() {
  console.log('foo arguments', arguments)
  var y = 2

  function bar() {
    var z = 3
    console.log(x + y + z)
  }
  bar()
}

foo()
```

![scope-chain](images/scope-chain.png)

### this

this 프로퍼티에는 this 값이 할당된다. this 값은 함수가 어떻게 호출되느냐에 따라 결정된다. this에 대한 자세한 이야기는 나중에 다뤄보도록 하자.

## 실행 컨텍스트가 실행되는 과정

아래 코드를 기준으로 살펴보자.

```javascript
debugger

var x = 1

function foo(hello) {
  var h = arguments[0]
  var y = 2

  function bar() {
    var z = 3
    console.log(x + y + z)
  }
  bar()
}

foo('hello')
```

### 1. 전역 객체 생성

![EC1](images/EC1.png)

위 스샷에서 볼 수 있는 것처럼, 전역 객체 (global object, 여기서는 window)가 생성된 것을 볼 수 있다. 그리고 이 객체에 있는 프로퍼티는 어디에서든 접근할 수 있다. 그리고 이 window에는 온갖 빌트인 객체, DOM, BOM 등이 설정되어 있는 것을 볼 수 있다.

### 2. 전역 코드로 컨트롤 진입

전역 코드로 컨트롤이 진입하면 이제 전역 실행 컨텍스트가 생성되고, 이 전역 실행 컨텍스트는 실행 컨텍스트 스택에 쌓인다.

```
ECStack = [globalContext -> {VO, SC, this}];
```

> 이렇게 생긴 코드는 없으니 그냥 느낌만 보면 될것 같다

### 2-1. 스코프 체인 생성 및 초기화

실행 컨텍스트가 생성되고 가장 먼저 하는 일은, 스코프 체인을 생성하고 초기화 하는 것이다. 여기에서는 전역 실행 컨텍스트이므로, 스코프 체인은 길이 1의 리스트로 Global Object를 가르키게 된다.

### 2-2. 변수 객체화 (Variable Instantiation) 실행

#### 순서

스코프 체인이 생성되고 초기화 되면 변수 객체화가 실행되는데 이는 Value Object에 값을 추가하는 것을 말한다. 전역 코드의 경우에는 Variable Object는 1번에서 생성하고 2번에서 가르키는 Global Object다. 그리고 아래와 같은 순서로 값을 세팅한다.

1. 함수코드인 경우 parameter가 value object의 프로퍼티로, argument가 값으로 설정된다.
2. 함수 선언을 대상으로 함수명이 value object의 프로퍼티로, 생성된 함수 객체가 값으로 설정된다. = **함수의 호이스팅**
3. 변수 선언을 대상으로 변수명이 value object의 프로퍼티로, undefined가 값으로 설정된다. = **변수의 호이스팅**

![EC2](images/EC2.png)

위 스샷을 보자.

- 1번에 따라서 hello 파라미터의 값이 'hello' argument로 설정되어 있다.
- 2번에 따라서 bar가 `f bar()`를 가르키고 있다.
- 3번에 따라서 y, h가 undefined로 세팅되어 있다.

#### 함수 선언의 처리

그 다음 눈여겨 봐야 할 것은 함수 선언 처리다. 생성된 함수 객체는 `[[Scopes]]` 프로퍼티를 갖게 된다. 이는 함수만이 소유하는 프로퍼티로, **함수 객체가 실행되는 환경**을 가리킨다.

![EC3](images/EC3.png)

여기에서 foo함수는 `[[Scopes]]`로 Global을 가르키고 있다. 근데 잠깐, 이거 어디서 본것 같은데, 싶었는데 여기서 0번의 `Global`은 스코프 체인의 0번에 있는 Global과 같다. 우리는 여기서 `[[Scopes]]`가 함수 객체가 실행되는 환경을 가지고 있다는 것을 알 수 있다.

내부 함수의 `[[Scopes]]`는 결론적으로

- 현재 자신의 실행환경
- 자신을 포함하는 외부함수의 실행환경
- 전역 객체

를 가르키게 된다. 여기서 자신의 실행환경과 외부 실행함수의 실행컨텍스트가 소멸해도, `[[Scopes]]`가 가리키는 외부 함수의 실행환경은 소멸되지 않고 참조할 수 있는데, **이를 클로져 라고 한다.**

함수 선언식은 (일반적인 `function hello() {...}`) 함수명을 프로퍼티로 함수 객체를 할당한다. 그리고 VO에 함수명을 프로퍼티로 추가하고 즉시 할당한다.

그러나 함수 표현식은 `const hello = function () {...}` 일반적인 변수의 방식을 따른다. 따라서 선언식은 함수를 선언하기 이전에 함수를 호출할 수 있다.

이것을 함수의 호이스팅이라고 한다.

#### 변수 선언의 처리

변수 선언을 세분화 하면 아래와 같다.

- 선언단계: 변수객체에 변수를 등록한다. 변수객체는 이제 스코프가 참조할 수 있게 된다
- 초기화 단계: 변수객체에 등록된 변수를 메모리에 할당한다. 변수는 undefined로 선언된다.
- 할당단계: undefined로 초기화 된 변수에 실제 값을 할당된다.

`var` 키워드는 선언과 초기화가 한번에 이루어진다. 즉 변수등록과 초기화가 한번에 이루어진다. 따라서 변수 선언 이전에 접근하여도 Variable Object에 변수가 이미 존재하고 있기 때문에 에러가 발생하지 않고 undefined가 리턴된다.

이를 변수의 호이스팅이라고 한다.

## 전역 코드의 실행

코드 실행을 위한 준비가 끝났으므로, 이제 코드가 실행된다.

```javascript
var x = 1

function foo(hello) {
  var h = arguments[0]
  var y = 2

  function bar() {
    var z = 3
    console.log(x + y + z)
  }
  bar()
}

foo('hello')
```

위 예제에서는 전역변수 x에 숫자 할당과, 함수 `foo()`의 호출이 실행된다.

### 변수에 값 할당

전역변수 x에 숫자를 할당하기 위해서, 실행 컨텍승트는 스코프체인이 참고하고 있는 Variable Object를 검색하기 시작한다. `x`를 발견하면 값 `xxx`를 할당한다.

### 함수의 실행

그리고 함수 `foo()`를 실행하게 된다. 함수를 실행하기 시작하면, 새로운 함수 실행 컨텍스트가 생성된다. 이와 동시에 컨트롤이 함수 `foo`로 이동하면서, 전역 코드와 마찬가지로

1. 스코프 체인의 생성 및 초기화
2. Variable Instantiation 실행
3. this value 결정

이 순차적으로 실행된다. 앞서 말한 것처럼 한가지 차이점은 - 전역코드가 아닌 함수코드라는 점이다.

#### 스코프 체인 생성 및 초기화

우선 Activation Object에 대한 레퍼런스를 스코프 체인에 추가하는 것으로 시작된다. 가장 먼저 arguments 프로퍼티를 초기화 한후, 그다음에 Variable Instantiation이 실행된다.

그 후, 스코프체인이 참조하고 있는 객체가 스코프 체인에 추가로 추가된다. 이 경우에는 스코프체인에 앞서 추가한 Activation Object와 두번째로 Global Object를 순차적으로 참조하게 된다.

#### Variable Instantiation 실행

앞서 만든 Activation Object를 Variable Object로서 실행된다.

먼저 `foo`함수 안에 있는 `bar`를 바인딩한다. 그리고 이 때 `bar`의 `[[Scopes]]`의 값은 GO와 AO를 참조하는 리스트가 된다.

변수 y를 Variable Object에 설정한다. 이 때 프로퍼티의 값은 y, 값은 undefined 다.

#### this 결정

this는 함수 호출 패턴에 의해 결정된다. 내부 함수는, this 는 전역 객체다.

### foo 함수의 실행

#### 값 할당

y에 2를 할당하기 위해, 스코프 체인을 탐색하면서 검색한다. 변수명 y에 해당하는 프로퍼티가 발견되면 2를 할당한다.

#### bar 함수의 실행

bar함수를 실행하기 시작하면, 새로운 실행 컨텍스트가 생성된다.여기서도 역시 마찬가지로 스코프 체인 생성 및 초기화, Variable Instantiation 실행, this value 결정이 순차적으로 실행된다.

![1](images/EC/EC.001.jpeg)
![2](images/EC/EC.002.jpeg)
![3](images/EC/EC.003.jpeg)
![4](images/EC/EC.004.jpeg)
![5](images/EC/EC.005.jpeg)
![6](images/EC/EC.006.jpeg)
![7](images/EC/EC.007.jpeg)
![8](images/EC/EC.008.jpeg)
![9](images/EC/EC.009.jpeg)
![10](images/EC/EC.010.jpeg)
![11](images/EC/EC.011.jpeg)
![12](images/EC/EC.012.jpeg)
![13](images/EC/EC.013.jpeg)
![14](images/EC/EC.014.jpeg)
![15](images/EC/EC.015.jpeg)
![16](images/EC/EC.016.jpeg)
![17](images/EC/EC.017.jpeg)
![18](images/EC/EC.018.jpeg)
![19](images/EC/EC.019.jpeg)
![20](images/EC/EC.020.jpeg)
![21](images/EC/EC.021.jpeg)
![22](images/EC/EC.022.jpeg)
![23](images/EC/EC.023.jpeg)
![24](images/EC/EC.024.jpeg)
![25](images/EC/EC.025.jpeg)
![26](images/EC/EC.026.jpeg)
![27](images/EC/EC.027.jpeg)
![28](images/EC/EC.028.jpeg)
![29](images/EC/EC.029.jpeg)
![30](images/EC/EC.030.jpeg)
![31](images/EC/EC.031.jpeg)
![32](images/EC/EC.032.jpeg)
