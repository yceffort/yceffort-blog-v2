---
title: React 공부하기 8 - 함수형 컴포넌트
date: 2019-05-21 12:17:09
published: true
tags:
  - react
  - javascript
description:
  "### 함수형 컴포넌트 ```javascript import React from 'react';  function
  Hello(props) {     return (         <div>hello {props.name}</div>     ) }
  ```  함수형 컴포넌트는 컴포넌트에서 라이프사이클, state 등의 기능을 제거한 상태이므로 메모리 사용량이..."
category: react
slug: /2019/05/20/react-study-8-functional-component/
template: post
---

### 함수형 컴포넌트

```javascript
import React from 'react'

function Hello(props) {
  return <div>hello {props.name}</div>
}
```

함수형 컴포넌트는 컴포넌트에서 라이프사이클, state 등의 기능을 제거한 상태이므로 메모리 사용량이 다른 컴포넌트에 비해 적다. 따라서 성능을 최적화 하기 위해서는 위와 같이 함수형 컴포넌트를 많이 쓰는 것이 좋다.
