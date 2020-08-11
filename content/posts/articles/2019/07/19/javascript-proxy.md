---
title: Javascript Proxy
date: 2019-07-19 07:12:42
tags:
  - javascript
published: false
description: "## Proxy 아래의 예제를 살펴보자.  ```javascript var obj = new
  Proxy(   {},   {     get: function(target, key, receiver)
  {       console.log(`getting ${key}!`);       return Reflect.get(target, key,
  receiver); ..."
category: javascript
slug: /2019/07/19/javascript-proxy/
template: post
---
## Proxy

아래의 예제를 살펴보자.

```javascript
var obj = new Proxy(
  {},
  {
    get: function(target, key, receiver) {
      console.log(`getting ${key}!`);
      return Reflect.get(target, key, receiver);
    },
    set: function(target, key, value, receiver) {
      console.log(`setting ${key}!`);
      return Reflect.set(target, key, value, receiver);
    }
  }
);
```

```
> obj.count = 1;
    setting count!
> ++obj.count;
    getting count!
    setting count!
    2
```

`obj`에 `.`으로 접근하려는 시도가 중간에 가로채기 당했다.
