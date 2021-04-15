---
title: React 공부하기 2 - JSX
date: 2019-05-08 08:58:18
published: true
tags:
  - react
  - javascript
description:
  "## Create-react-app 라이브러리로 시작 ``` yarn global add create-react-app
  create-react-app hello-react  cd hello-react yarn start ```  ### app.js의
  구조  ```javascript import React from 'react'; import logo fr..."
category: react
slug: /2019/05/08/react-study-2-jsx/
template: post
---

## Create-react-app 라이브러리로 시작

```
yarn global add create-react-app
create-react-app hello-react

cd hello-react
yarn start
```

### app.js의 구조

```javascript
import React from 'react'
import logo from './logo.svg'
import './App.css'

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <p>
          Edit <code>src/App.js</code> and save to reload.
        </p>
        <a
          className="App-link"
          href="https://reactjs.org"
          target="_blank"
          rel="noopener noreferrer"
        >
          Learn React
        </a>
      </header>
    </div>
  )
}

export default App
```

코드 첫줄은 node.js의 기능, 즉 react를 불러와서 모듈화 시키는 것이다. 이를 번들링하는 작업은 webpack에서 처리해 준다. css-loader로 `./App.css`를, file-loader로 파일을 (`log.svg`), babel-loader로 es6를 es5로 변환해서 처리해줄 것이다. 원래는 이러한 작업을 위해서 세팅을 해줘야 하지만, `react-scripts`에서 이를 다 처리해주고 있다.

한가지 특이한 것은, `class`가아니고 `className`으로 되어있다는 것이다. 그렇다. jsx니까, javascript를 쓸테고, `class`는 자바스크립트의 예약어이니까 `className`을 대신 써야 하는 것이다.

그 다음으로 봐야할 것은 return() 구문에 있는 내용이다. 얼핏보면 html과 비슷하게 생긴 이 코드는, `JSX`라고 한다. 이 코드는 나중에 babel-loader로 번들링 되면서 자바스크립트로 변환된다.

#### before

```javascript
var a = (
  <div>
    <h1>
      {' '}
      hello <b>react</b>{' '}
    </h1>
  </div>
)
```

#### after

```javascript
var a = React.createElement(
    "div",
    null,
    React.createElement(
        "h1",
        null,
        "hello "
        React.createElement(
            "b",
            null,
            "react"
        )
    )
)
```

#### JSX 문법

1.컴포넌트에 여러개의 요소가 있다면 꼭 하나의 부모요소로 감싸야 한다. Virtual Dom에서 컴포넌트 변화를 감지내 낼 때, 효율적으로 비교할 수 있도록 컴포넌트 내부는 DOM 트리 구조 하나여야 한다는 규칙이 존재하기 때문이다. 만약 `<div>`를 쓰고 싶지 않다면, 리액트 v16에서 지원하는 `<Fragment>`를 사용하면 된다.

2.자바스크립트도 내부에서 표현할 수 있다.

```javascript
render () {
    const text = '집에 가고 싶다';
    return (
        <div>
            <h1>{text}</h1>
        </div>
    )
}
```

3.내부 표현식에서 if문을 사용하려면, 조건부 연산자를 사용하면 된다. 같은 맥락으로, `||` `&&` `condition ? 'true' : 'false'` 등도 가능하다.

```javascript
render () {
    const condition = true;
    <div>
        {
            condition ? '참': '거짓'
        }
    </div>

}
```

4.인라인 스타일링도 가능하다. 키는 camel_case를 사용하면 된다. `background-color`는 `backgroundColor`로, `-mos`는 `Mos`로 쓰면된다. 단 `-ms`는 그냥 소문자 `ms`로 한다.

```javascript
render() {
    const style = {
        backgroundColor: 'gray',
        border: '1px solid black',
        height: Math.round(Math.random() * 100) + 50,
        width: Math.round(Math.random() * 100) + 50,
    }
```

5.class를 쓰지말고 className을 써야 한다.

6.주석은 꼭 {}안에다가 써야 한다.

```javascript
{/* jsx에서 주석은 이렇게 써야 한다.*/}

{ /*
    여러 줄일 때는 이렇게 써야 한다.
*/ }

return {
    <div>
        <div // 하지만 이렇게 주석을 쓰는 건 괜찮다.>

        </div>
        // 이렇게 쓰면 바로 이 줄 자체가 렌더링된다.
    </div>
}
```
