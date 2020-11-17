---
title: 자바스크립트의 private
tags:
  - typescript
  - javascript
published: true
date: 2020-05-08 11:33:36
description: "이 글은 [은닉을 향한 자바스크립트의 여정](https://meetup.toast.com/posts/228)을 요약한
  글입니다. ## History  자바스크립트에서는 객체에 private 한 속성을 만들 수가 없었다. 그래서 보통 자바스크립트 개발자는
  private한 것이다 라는 약속으로 `_` prefix를 붙여서 사용하고는 했었다.  ```javas..."
category: typescript
slug: /2020/04/javascript-private/
template: post
---
이 글은 [은닉을 향한 자바스크립트의 여정](https://meetup.toast.com/posts/228)을 요약한 글입니다.

## History

자바스크립트에서는 객체에 private 한 속성을 만들 수가 없었다. 그래서 보통 자바스크립트 개발자는 private한 것이다 라는 약속으로 `_` prefix를 붙여서 사용하고는 했었다.

```javascript
function Hello() {
  this.publicProp = "public"
  this._privateProp = "private"
}
```

물론 자바스크립트 개발자들은 `_`의 존재로 해당 속성을 건들지 말아야겠다는 것을 암묵적으로 공유했지만, 어디까지나 암묵적인 것일 뿐, 실제로는 밖에서 얼마든지 접근 할 수 있다.

좀 더 이 문제를 자바스크립트스럽게 해결하기 위해서는, 클로저를 활용하면 된다.

```javascript
function Hello() {
  this.publicProp = "public"
  const privateProp = "private"

  _doWithPrivateProp = () => {
    // do something
  }
}
```

비록 `this`와 `const`가 짬뽕이 되면서, 가독성이 떨어지긴 하지만, 효과적으로 데이터를 격리 시켰다. 위와 같은 방법을 사용해서 메소드도 숨길 수 있다.

```javascript
function Hello() {
  const publicProp = "public"
  const privateProp = "private"

  _doWithPrivateProp = () => {
    // ...
  }

  const publicMethod = () => {
    _doWithPrivateProp()
    // ...
  }

  return {
    publicProp,
    publicMethod,
  }
}
```

`Symbol`을 사용해 볼 수도 있다.

```javascript
const privateMethodName = Symbol()
const privatePropName = Symbol()

class Hello {
  [privatePropName] = "private"
  publicProp = "public";

  [privateMethodName]() {
    // ...
  }

  publicMethod() {
    this[privateMethodName](this[privatePropName])
  }
}
```

`Symbol`은 생성될 때 마다 고유의 값을 가지므로, 외부에서는 이를 export하지 않는 이상 접근할 수 없다.

## # 의 등장

해당 제안 내용은 [여기](https://github.com/tc39/proposal-class-fields/)에서 자세히 확인할 수 있다.

```javascript
class Hello {
  #message = 'hello'
}

const hello = new Hell()
hello.#message
```

```
Uncaught SyntaxError: Private field '#message' must be declared in an enclosing class
```

private 하기 때문에 접근 할 수 없다는 메시지가 뜬다.

여기에서 `#`은 prefix이기 때문에 꼭 접근시에 `#`을 써야 한다.

```javascript
class Hello {
  #message = "hello"

  getMessage() {
    return this.message // 안됨
  }
}
```

상속을 받는다 하더라도 접근이 되지 않는다.

```javascript
class Hello {
  #message = "hello"

  getMessage() {
    return this.#message
  }
}

class Hi extends Hello {
  getHiMessage() {
    return this.#message
  }
}

const hi = new Hi()
hi.getHelloMessage() // Uncaught SyntaxError: Private field '#message' must be declared in an enclosing clas
```

추가로 모든 private 필드는 클래스 별로 독립된 고유한 스코프를 갖는다.

```javascript
class Hello {
  #message = "hello"

  getMessage() {
    return this.#message
  }
}

class Hi extends Hello {
  #message = 'hi'

  getHiMessage() {
    return this.#message
  }
}

const hi = new Hi()
const hello = new Hello()
console.log(hello.getMessage()) // hello
console.log(hi.getHiMessage()) // hi
```

## 타입스크립트에서는?

https://www.typescriptlang.org/docs/handbook/classes.html#ecmascript-private-fields
https://www.typescriptlang.org/docs/handbook/classes.html#understanding-typescripts-private

위에서 언급한 `#` 문법과 더불어 (3.8부터) `private` 키워드 도 지원한다.

https://devblogs.microsoft.com/typescript/announcing-typescript-3-8-beta/#ecmascript-private-fields

여기에 좋은 내용이 정리되어 있다.

- Private 필드는 `#`으로 시작된다. 
- 모든 private 필드는 속한 클래스에서 고유한 스코프를 가지고 있다.
- `#`은 타입스크립트의 `public` `private`과 함게 사용할 수 없다.
- Private 필드는 클래스 밖에서 접근하거나 알아챌 수 없다. (JS도 마찬가지)
