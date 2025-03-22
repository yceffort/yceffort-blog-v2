---
title: javascript class
date: 2019-07-10 01:01:24
published: true
tags:
  - javascript
description:
  '# Class 클래스는 기본적으로 이렇게 생겼다.  ```javascript class Member
  {   getName() {     return "이름";   } }  let obj = new Member();
  console.log(obj.getName()); ```  ## 특징  ### 1. strict 모드에서 실행  딱히 `''use
  strict''...'
category: javascript
slug: /2019/07/09/javascript-class/
template: post
---

# Class

클래스는 기본적으로 이렇게 생겼다.

```javascript
class Member {
  getName() {
    return '이름'
  }
}

let obj = new Member()
console.log(obj.getName())
```

## 특징

### 1. strict 모드에서 실행

딱히 `'use strict';`를 선언하지 않아도, 클래스의 코드는 기본적으로 `strict`모드에서 실행된다. 그렇게 되면, 당연히 `strict`모드의 여러가지 특징을 자동으로 따르게 된다.

### 2. 클래스 내 메서드 작성

```javascript
class Member {
  setName(name) {
    this.name = name
  }

  getName(name) {
    this.name = name
  }
}
```

보이는 것처럼, `function`키워드와 `:`가 없이 메서드 이름만 사용한다. 그리고 메서드 사이에 `;`가 불필요하다. 다만 function 을 선언하면 글로벌 오브젝트에 설정되는 것과 다르게, class는 그렇지 않다. 그리고 class의 object property는 for()문 등으로 열거할 수 없다.

### 3. 프로퍼티에 연결

```javascript
class Member {
  setName(name) {
    this.name = name
  }
}
```

위 코드와

```javascript
Member.prototype.setName = function (namn) {
  this.name = name
}
```

위 코드는 같다.

## Constructor

`constructor`는 클래스 인스턴스를 생성하고, 생성한 인스턴스를 초기화하는 역할을 한다. `new Member()`를 실행하면, `Member.prototype.constructor`가 먼저 호출된다. 클래스에 이를 작성하지 않으면, `prototype`의 디폴트 `constructor`가 호출된다. 그리고 이 `constructor`가 없으면 인스턴스를 생성할 수 없다. 기존 es5문법에서는 자바스크립트 엔진이 디폴트 `constructor`를 호출해서 이를 활용할 수 없었지만, es6에서 부터는 개발자가 이를 정의할 수 있게 되었다.

```javascript
class Member {
  constructor(name) {
    this.name = name
  }

  getName() {
    return this.name
  }
}

let newMember = new Member('라이오넬 멧시')
console.log(newMember.getName())
```

만약 `constructor`에서 이상한 값을 반환하면 어떻게 될까?

```javascript
constructor() {
    return 1;
}
```

`constructor`에서 `number`나 `string`을 반환하면, 이를 무시하고 생성한 인스턴스를 반환하게 된다.

그러나 `object`를 반환하면 어떻게 될까?

```javascript
class Member {
  constructor(name) {
    return {name: '메켓트'}
  }

  getName() {
    return this.name
  }
}

let newMember = new Member('라이오넬 멧시')
console.log(newMember.name)
console.log(newMember.getName)
```

```
메켓트
undefined
```

name이 메켓트인 object를 반환하면서, newMember 클래스에는 name밖에 남지 않게 되었다. 그리고 getName은 존재하지 않아서 undefined가 출력된다.

## getter, setter

```javascript
class Member {
  set setName(name) {
    this.name = name
  }

  get getName(name) {
    return this.name
  }
}
```
