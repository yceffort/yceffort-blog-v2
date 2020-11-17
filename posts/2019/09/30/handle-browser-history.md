---
title: 브라우저 히스토리 조작
date: 2019-09-30 06:28:48
published: true
tags:
  - browser
  - javascript
description: '## 브라우저 히스토리 브라우저의 히스토리는 `window.history`안에 있다.  `History {length:
  3, scrollRestoration: "auto", state: null}`  `length`만 가져올 수 있을 뿐, 실제 내부에 리스트는
  가져올 수가 없는데 이는 보안상의 문제 때문이다.  `window.history.back()` ...'
category: browser
slug: /2019/09/30/handle-browser-history/
template: post
---
## 브라우저 히스토리

브라우저의 히스토리는 `window.history`안에 있다.

`History {length: 3, scrollRestoration: "auto", state: null}`

`length`만 가져올 수 있을 뿐, 실제 내부에 리스트는 가져올 수가 없는데 이는 보안상의 문제 때문이다.

`window.history.back()` `window.history.forward()`는 각각 브라우저의 앞으로가기 뒤로 가기와 동일한 역할을 한다.

## 특정 위치로 가기

`window.history.go(n)` 현재 페이지의 index는 0 이라고 볼 수 있다. -1 은 바로 전 페이지, 1 은 다음 페이지라고 볼 수 있다.

## 히스토리 추가 및 변경

### pushState

`window.pushState(state, title, url)`

아래와 같이 한번 사용해보자.

```javascript
history.pushState({ hello: "world" }, "title", "hello");
```

현재 있는 페이지 주소창에서 `hello`가 추가되었음을 알 수 있다. 그러나 브라우저는 이를 불러오지도 않고, 해당 주소의 존재여부도 파악하지 않는다. 그저 주소만 바뀐 것이다.

아래 프로세스를 살펴보자.

1. `www.google.com` 접속 -> history: 1
2. `history.pushState({ hello: "world" }, "title", "hello");` 입력 -> 주소창: `google.com/hello` / history: 2
3. `www.naver.com` 접속 -> history: 3
4. 뒤로가기 버튼 클릭
5. `https://www.google.com/hello` 가 404를 띄움 -> history.state에 hello: world 가 있음.
6. 뒤로가기 버튼 클릭
7. `www.google.com` 으로 돌아가지만, 여전히 404

따라서 `pushState`는 history에 새로운 history만을 추가할 뿐, 실질적으로 페이지 이동은 일으키지 않는 다는 것을 볼 수 있다.

#### state

javascript object로, pushState로 새로운 히스토리를 만드는 것과 관련이 있다. 사용자가 새로운 상태로 이동할 때마다, `popState`이벤트가 발생해서, `state`의 사본을 가져온다. 파이어폭스의 경우 640k정도의 데이터를 저장할 수 있으며, 이는 브라우저를 재시작해도 사용할 수 있다. 즉, 해당 history state에서 필요한 값을 넣어두는 용도로 사용하면 좋다.

#### title

현재 파이어폭스나 크롬에서 쓰지 않는 변수로 보인다. state의 명칭을 기록해 두는 용도로 사용하면 될 것 같다.

#### URL

새로운 history의 url을 지정한다. 이 전 예제에서도 봤던 것처럼, 브라우저는 해당 URL을 호출하지 않는다.

어째 돌아가는 모양, 주소는 바뀌지만 url을 로딩하지 않는 다는 것이 `window.location = '#foo'` 와 비슷해 보이는 측면이 있다. 이렇게 쓸모없어보이는 `pushState`는 아래와 같은 장점이 있다.

- `pushState`로 생성한 URL은 현재 URL을 기준으로 한다. 반대로 window.location는 해쉬값을 지정할 경우에만 같은 document에 머물러 있다. (아무튼 URL로딩을 안함)
- URL 변경이 필요 없다면, URL값을 안넣어서 변경을 안해주어도 된다. 반대로 해쉬값 지정의 경우에는 현재 해쉬값과 다른 경우에만 새로운 히스토리를 생성한다.
- `state` 오브젝트로 데이터를 저장할 수 있다. 반면 해쉬는 해쉬값을 활용해야 한다.


### replaceState

`replaceState`는 `pushState`와 동작이 거의 비슷하다. 다만 히스토리를 추가하는 것이 아닌, 덮어 쓴다는 것에서 차이가 있다.
