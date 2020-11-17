---
title: var let const, 그리고 호이스팅
tags:
  - javascript
published: true
date: 2020-05-20 07:33:36
description: "## var let const, 그리고 호이스팅 ### var  우리가 모두 아는 `var` 키워드는 아래와 같은
  특징을 가지고 있다.  1. 함수레벨 스코프를 가지고 있다.        대부분의 프로그래밍 언어들이 블록 레벨 스코프를 사용하고 있지만,
  `var`로 선언된 키워드는 함수레벨 스코프를 갖는다.     ```javascript     var ..."
category: javascript
slug: /2020/05/var-let-const-hoisting/
template: post
---
## var let const, 그리고 호이스팅

### var

우리가 모두 아는 `var` 키워드는 아래와 같은 특징을 가지고 있다.

1. 함수레벨 스코프를 가지고 있다.
   
   대부분의 프로그래밍 언어들이 블록 레벨 스코프를 사용하고 있지만, `var`로 선언된 키워드는 함수레벨 스코프를 갖는다. 
   ```javascript
    var name = 'hello'

    function t() {
        var name = 'hi'
        console.log(name) // hi
    }

    t()
    console.log(name) // hello
   ```

2. `var` 키워드는 생략이 가능하다.
   
   생략이 가능하기 때문에, 함수가 선언한 환경의 `this`에 영향을 받는다. 일반적인 웹 환경에서는 `window`일 것이다.

3. 중복 선언이 가능하다.
   
    ```javascript
    var name = 'hello'
    var name = 'hi'
    console.log(name) // hi
    ```

4. 호이스팅 당한다.

    호이스팅이란 

    > 스코프 안에 있는 선언들을 모두 스코프의 최상위로 끌어올리는 것 
    
    이다. 자바스크립트 인터프리터가 코드를 해석할 때 함수의 선언, 할당, 실행을 모두 나눠서 처리하기 때문이다. 예를 들어보자.

    ```javascript
    console.log(name) // undefined
    var name = 'hello'
    ```

    이 코드는 참조 에러가 나지 않고, `undefined`를 리턴한다. 그 이유는 자바스크립트가 호이스팅을 하면서 아래와 같은 방식으로 코드를 해석하기 때문이다.

    ```javascript
    var name // undefined
    console.log(name)
    name = 'hello'
    ```

### 2. let

`let`은 es6에서 가장 잘 알려진 기능 중 하나다. `var`랑 비슷한 것 같지만, 사실 몇가지 다른 점이 있다. 

1. 블록 수준 스코프를 가지고 있다.

    아래의 예를 살펴보자.

    ```javascript
    {{{var name = 'hello'}}}
    console.log(name) // hello

    {{{let name2 = 'hi'}}}
    console.log(name2) // ReferenceError: name2 is not defined
    ```

    `let`과 `const`는 모두 블록 수준의 스코프를 사용 하고 있다. 따라서 블록 내부에서 선언한 변수는 모두 지역변수로 취급된다. 때문에 블록 내부에서 선언한 변수들을 참고할 수가 없다.

2. 키워드 생략이 불가능하다.

    키워드를 생략하면 `var`처럼 작동하게 된다.

3. 중복선언이 불가능하다.

    그렇다.

4. 호이스팅 당한다.

    가끔 인터넷을 뒤지다보면, `let`과 `const`는 호이스팅 당하지 않는다는 이야기가 있는데, 이는 잘못된 사실이다. 두 키워드 모두 호이스팅 당하기는 마찬가지이다.

    이것을 이해하기 위해서는, `Temporary Dead Zone`에 대해 알아야 한다.


### Temporary Dead Zone

아래 코드를 살펴보자.

```javascript
name = 'hello' // ReferenceError: Cannot access 'name' before initialization
let name = 'hi'
```

자. 만약 `name`이 호이스팅 되지 않았다면, 두 번째 `let`선언에서 already declared에러가 났었어야 했다. 근데 현실은 초기화 하기전에 엑세스 할 수 없다는 에러를 내뱉었다.


아래 코드는 어떨까?

```javascript
function sayHello() {
  return name
}
let name = 'hi'
console.log(sayHello()) // hi
```

별탈없이 hi를 내뱉은 것을 볼 수 있다. 그 말인 즉, name이 분명 호이스팅 되었다는 뜻이다.

다른 이야기로 넘어가서, 자바스크립트에서는 총 3단계에 걸쳐서 변수를 생성한다.

1. 선언(Declaration): 스코프와 변수 객체가 생성되고, 스코프가 변수 객체를 참조한다.
2. 초기화(Initialization): 변수 객체 값을 위한 공간을 메모리에 할당한다. 이 때 할당되는 값은 `undefined`다.
3. 할당(Assignment): 변수 객체에 값을 할당한다.

`var`는 선언과 동시에 초기화가 이루어진다. 즉, 선언과 동시에 undefined가 할당된다.

그러나 `let`과 `const`는 다르다. 선언만 될뿐, 초기화가 이루어지지 않는다. 바로 이단계에서 TDZ에 들어가게 되는 것이다. 즉, 선언은 되어있지만, 초기화가 되지 않아 이를 위한 자리가 메모리에 준비되어 있지 않은 상태라는 것이다.


### const

`const`에서 구별되는 특징만 몇가지 살펴보자.

1. `const`는 초기화와 동시에 선언이 이루어져야 한다.

    ```javascript
    let hello
    hello = 'hello'

    const hi //SyntaxError: Missing initializer in const declaration
    hi = 'hi'
    ```

2. `const` 자체가 값을 불변으로 만드는 것이 아니다.

    아래 코드는 당연히 안된다.

    ```javascript
    const hello = 'hello'
    hello = 'hi' // TypeError: Assignment to constant variable.
    ```

    그렇다고 이것도 안되는 것은 아니다.
    
    ```javascript
    const hello = ['hi']
    hello.push('hello')
    ```

    ```javascript
    const hello = 'hello'
    var hi = hello
    hi = 'hi'
    console.log(hello) // hello
    console.log(hi) // hi
    ```

    객체 자체를 동결시키기 위해서는 [Object.freeze](https://developer.mozilla.org/ko/docs/Web/JavaScript/Reference/Global_Objects/Object/freeze)를 사용하면 된다.

3. 상수를 선언할 때 사용한다.


### 결론

- 호이스팅은 변수의 선언을 최상단에 끌어올린다는 뜻이다.
- `var` `let` `const`는 모두 호이스팅 된다.
- `let`  선언과 동시에 `TDZ`에 들어가서 초기화가 필요한 별도의 상태로 관리된다.
- `const` 는 선언과 동시에 초기화, 할당까지 이루어진다.
- `var`는 왠만하면 쓰지말자
- `let`대신 `const`를 사용하자. `let`으로 선언되어 있다면, 어디선가 이 변수가 바뀔지도 모른다는 생각을 가지고 있어야 하므로, 코드를 읽기가 어려워진다. 반면 `const`는 초기화, 선언, 할당까지 되어 있으니 변경되지 않을 것이다라는 확신으로 코드를 볼 수 있다. (물론 객체의 속성은 바뀔 수 있음.)