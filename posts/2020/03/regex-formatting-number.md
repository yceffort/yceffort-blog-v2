---
title: Javascript Regex 숫자를 comma와 함께 Formatting 하기
tags:
  - javascript
published: true
date: 2020-03-17 04:14:07
description:
  'regex를 활용해서 숫자에 , 를 찍어서 formatting을 해보자. ## 1. 첫번째
  시도  ```javascript function formatNumber(number) {   return
  number.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",") } ```  인터넷에 가장 많이 떠돌아
  다니는 해결책으로, ...'
category: javascript
slug: /2020/03/regex-formatting-number/
template: post
---

regex를 활용해서 숫자에 , 를 찍어서 formatting을 해보자.

## 1. 첫번째 시도

```javascript
function formatNumber(number) {
  return number.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',')
}
```

인터넷에 가장 많이 떠돌아 다니는 해결책으로, 아쉽게도 소수점에 대한 대응이 되지 않는다.

```javascript
'1111.1111111'.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',')
// 1,111.1,111,111
```

## 2. 두번째 시도

```javascript
function formatNumber(number) {
  return number.toString().replace(/\B(?<!\.\d*)(?=(\d{3})+(?!\d))/g, ',')
}
```

```javascript
'1111.1111111'.toString().replace(/\B(?<!\.\d*)(?=(\d{3})+(?!\d))/g, ',')
// 1,111.1111111
```

이게 성공하는 줄 알고, test 도 넘어가길래 실제로 써보았더니 앱에서 오류가 나기 시작했다. ㅠ.ㅠ

```javascript
'1111.1111111'.toString().replace(/\B(?<!\.\d*)(?=(\d{3})+(?!\d))/g, ',')
// SyntaxError: Invalid regular expression: invalid group specifier name
```

이와 관련된 posting은 여기 [여기](https://stackoverflow.com/questions/51568821/works-in-chrome-but-breaks-in-safari-invalid-regular-expression-invalid-group)에서 찾아볼 수 있었다.

`x(?<=y)` `x(?<!y)`는 각각 lookbehind 문법으로, 아쉽게도 [사파리와 익스플로러에서는 지원하지 않는다.](https://caniuse.com/#feat=js-regexp-lookbehind). (감사합니다.)

따라서 아쉽게도, 순수 regex로 모든 브라우저 환경을 지원하면서 대체 하기는 무리인듯 하다.

## 3. (지금까지의) 정답

```javascript
function formatNumber(x) {
  var parts = x.toString().split('.')
  parts[0] = parts[0].replace(/\B(?=(\d{3})+(?!\d))/g, ',')
  return parts.join('.')
}
```

언젠가 더 좋은 방법을 찾기를 바라며 (...)
