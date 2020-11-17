---
title: Javascript - Destructuring Assignment
date: 2019-05-21 07:18:46
published: true
tags:
  - javascript
description: "## 구조 분해 할당 구조 분해 할당은 배열이나 객체의 속성을 말그대로 분해하여, 분해 한 값을 개별변수에 담을 수
  있게 도와주는 표현식이다.  ```javascript let a, b, rest; [a, b] = [10, 20];
  console.log(a); // 10 console.log(b); // 20  // rest 패턴을 이용하여 나머지를 모두..."
category: javascript
slug: /2019/05/21/javascript-destrucuring-assignment/
template: post
---
## 구조 분해 할당

구조 분해 할당은 배열이나 객체의 속성을 말그대로 분해하여, 분해 한 값을 개별변수에 담을 수 있게 도와주는 표현식이다.

```javascript
let a, b, rest;
[a, b] = [10, 20];
console.log(a); // 10
console.log(b); // 20

// rest 패턴을 이용하여 나머지를 모두 할당 받을 수 있다.
[a, b, ...rest] = [10, 20, 30, 40, 50];
console.log(a); // 10
console.log(b); // 20
console.log(rest); // [30, 40, 50]

// 객체 리터럴 식에서 구조 분해 할당을 하기 위해서는, 객체 리터럴 형식과 마찬가지로 { }로 표기하면 된다.

({ a, b } = { a: 10, b: 20 });
console.log(a); // 10
console.log(b); // 20

({a, b, ...rest} = {a: 10, b: 20, c: 30, d: 40});
console.log(a); // 10
console.log(b); // 20
console.log(rest); // {c: 30, d: 40}

function f() {
  return [1, 2];
}

var a, b;
[a, b] = f();
console.log(a); // 1
console.log(b); // 2

// 값을 아래 처럼 무시할 수도 있다.
function f() {
  return [1, 2, 3];
}

var [a, , b] = f();
console.log(a); // 1
console.log(b); // 3
```

### 객체 구조 분해

```javascript
var o = {p: 42, q: true};
var {p, q} = o;

console.log(p); // 42
console.log(q); // true

var o = {p: 42, q: true};
var {p: foo, q: bar} = o;

// p와 q는 무시된다.
console.log(foo); // 42
console.log(bar); // true

var metadata = {
    title: "Scratchpad",
    translations: [
       {
        locale: "de",
        localization_tags: [ ],
        last_edit: "2014-04-14T08:43:37",
        url: "/de/docs/Tools/Scratchpad",
        title: "JavaScript-Umgebung"
       }
    ],
    url: "/en-US/docs/Tools/Scratchpad"
};

var { title: englishTitle, translations: [{ title: localeTitle }] } = metadata;

console.log(englishTitle); // "Scratchpad"
console.log(localeTitle);  // "JavaScript-Umgebung"

var people = [
  {
    name: "Mike Smith",
    family: {
      mother: "Jane Smith",
      father: "Harry Smith",
      sister: "Samantha Smith"
    },
    age: 35
  },
  {
    name: "Tom Jones",
    family: {
      mother: "Norah Jones",
      father: "Richard Jones",
      brother: "Howard Jones"
    },
    age: 25
  }
];

for (var {name: n, family: { father: f } } of people) {
  console.log("Name: " + n + ", Father: " + f);
}


function userId({id}) {
  return id;
}

function whois({displayName: displayName, fullName: {firstName: name}}){
  console.log(displayName + " is " + name);
}

// "Name: Mike Smith, Father: Harry Smith"
// "Name: Tom Jones, Father: Richard Jones"

var user = {
  id: 42,
  displayName: "jdoe",
  fullName: {
      firstName: "John",
      lastName: "Doe"
  }
};

console.log("userId: " + userId(user)); // "userId: 42"
whois(user); // "jdoe is John"

let key = "z";
let { [key]: foo } = { z: "bar" };

console.log(foo); // "bar"
```

```javascript
{ innerHeight } // {innerHeight: 441}
{( innerHeight )} // 441
```

여기에서 //는 전역객체 `window`일 것이다.

## 객체 리터럴 표기법과 JSON의 차이

- JSON은 "key": value 구문만 허용한다. `key`값은 큰 따옴표로 묶여 있어야한다. 그리고 값은 단축(명)일 수는 없다.
- JSON에서 값은 문자열, 숫자, 배열, `true`, `false`, `null`또는 다른 JSON객체만 가능하다.
- 함수는 JSON값에서 할당될 수 없다.
- `Date` 객체는 JSON.parse()를 거치고 나면 문자열이 된다.