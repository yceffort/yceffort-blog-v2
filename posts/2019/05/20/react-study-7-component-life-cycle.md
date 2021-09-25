---
title: React 공부하기 7 - 컴포넌트 라이프 사이클
date: 2019-05-21 12:17:09
published: true
tags:
  - react
  - javascript
description:
  '## React Component Life Cycle 라이프 사이클은 총 10가지다. `Will`접두사는 어떤 작업을
  작동하기전에 실행하는 메소드가, `Did`는 어떤 작업을 한 후에 실해오디는 메서드다. 이 메서드들은 컴포넌트 클래스에서 덮어써서 선언하여
  사용할 수 있다.  라이프사이클은 총 3가지 카테고리로 나눌 수 있는데,  `mount`, `unm...'
category: react
slug: /2019/05/20/react-study-7-component-life-cycle/
template: post
---

## React Component Life Cycle

라이프 사이클은 총 10가지다. `Will`접두사는 어떤 작업을 작동하기전에 실행하는 메소드가, `Did`는 어떤 작업을 한 후에 실해오디는 메서드다. 이 메서드들은 컴포넌트 클래스에서 덮어써서 선언하여 사용할 수 있다.

라이프사이클은 총 3가지 카테고리로 나눌 수 있는데, `mount`, `unmount`, `update`다.

### Mount

DOM이 생성되고, 웹 브라우저 상에 나타나는 것을 mount라고 한다. 이 때 호출되는 메서드는 다음과 같다.

1. 컴포넌트 만들기
2. constructor: 컴포넌트를 새로 만들 때 마다 호출되는 클래스 생성자 메서드
3. `getDerivedStateFormProps`: `props`에 있는 값을 `state`와 동기화 시키는 메서드다.
4. `render`: UI를 렌더링하는 메서드
5. `componentDidMount`: 컴포넌트가 웹 브라우저 상에 나타난 후 호출하는 메서드

### update

컴포넌트를 업데이트 하는 경우는 아래 4가지다.

1. props가 바뀔때
2. state가 바뀔때
3. 부모컴포넌트가 리렌더링 될때
4. `this.forceUpdate`를 통하여 강제로 렌더링을 트리거할 때

호출되는 메서드들은 아래와 같다.

1. `props`변경 / 부모 컴포넌트가 리렌더링
2. `getDerivedStateFromProps`: `props`에 있는 값을 `state`와 동기화 시키는 메서드다.
3. `state`가 변경
4. `shouldComponentUpdate`: 컴포넌트가 리렌더링 해야하는지 결정하는 메서드다. 여기에서 `false`가 리턴되면 아래 메서드들을 더 이상 호출 하지 않는다.
5. `forceUpdate` 호출
6. `render`
7. `getSnapshotBeforeUpdate`: `Component` 변화를 DOM에 반영하기 전에 호출하는 메서드
8. 웹브라우저의 `dom`이 변화
9. `componentDidUpdate`: 컴포넌트의 업데이트 작업이 끝난 후 호출하는 메서드

### Unmount

컴포넌트를 DOM에서 제거하는 것을 말한다.

1. 언마운트
2. `componentWillUnmount`: 컴포넌트가 웹 브라우저 상에서 사라지기 전에 호출되는 메서드다.

### render() { ... }

컴포넌트의 모양새를 정의한다. 라이프사이클 메서드중 유일하게 필수 메서드이기도 하다. 여기에서 `this.props` `this.state`에 접근할 수 있으며, 리액트 요소를 반환한다. 다만 이 안에서는 절대 state를 변경해서는 안되며, 웹브라우저에 접근해서도 안된다.

### constructor

컴포넌트의 생성자 메서드로 컴포넌트를 만들 때 최초로 실행된다. 여기에서 초기 state를 정할 수 있다.

### getDerivedStateFromProps

v16.3 이후에 등장한 메서드로, `props`로 받아온 값을 `state`에 동기화 시키는 용도로 사용되며, 컴포넌트를 마운트하거나 `props`를 변경할 때 호출된다.

### componentDidMount

컴포넌트를 만들고, 첫 렌더링을 마친 후 실행된다. 이 안에서 다른 자바스크립트 라이브러리나 프레임워크 함수를 호출하거나, 이벤트 등록, `setTimeOut`, `setInterval` 네트워크요청과 같은 비동기 요청을 실행하면 된다.

### shouldComponentUpdate

`props`나 `state`를 변경했을 때, 리렌더링을 시작할지 여부를 결정하는 메서드다. 여기에서는 반드시 `true`, `false`를 반환해야 한다. `false`를 반환하면 이 후의 과정이 모두 멈춘다.

이 안에서 `this.props` `this.states`로 접근할 수 있고, 다음 값을 `this.nextProps` `this.nextState`로 다음 값을 접근할 수 있다. 성능을 최적화 하거나, 알고리즘을 작성하여 리렌더링을 방지하기 위해서 사용하기도 한다.

### getSnapshotBerforeUpdate

v16.3 이후에 등장한 메서드로, `render`를 호출한 후 `DOM`에 변화를 반영하기 바로 직전에 호출하는 메서드다. 여기에서 반환하는 값을 `componentDidUpdate`에서 세번째 파라미터인 `snapshot`으로 값을 전달 받을 수 있다. 주로 스크롤바의 위치와 같이 업데이트하기 직전에 값을 참고할 일이 있을 때 활용한다.

```javascript
getSnapshotBeforeUpdate(prevProps, prevState) {
    if (prevState.array !== this.state.array) {
        const {scrollTop, scrollHeight} = this.list
        return {scrollTop, scrollHeight};
    }
}
```

### componentDidUpdate

리렌더링을 완료한 후에 실행한다. 업데이트가 끝난 이후이므로, DOM관련 처리를 해도된다. 여기에서 `prevProps` `prevState`를 활용하여 이전의 값에 접근할 수 도 있다. `getSnapshotBerforeUpdate`에서 반환한 값이 있다면 여기서 `snapshot`값을 전달 받을 수 있다.

### componentWillUnmount

컴포넌트를 DOM에서 제거할 때 실행한다. `componentDidMount`에서 등록한 이벤트, 타이머, DOM이 있다면 어기에서 제거해야 한다.

![component-life-cycle](https://cdn-images-1.medium.com/max/2400/1*cEWErpe-oY-_S1dOaT1NtA.jpeg)

출처: [https://code.likeagirl.io/understanding-react-component-life-cycle-49bf4b8674de](https://code.likeagirl.io/understanding-react-component-life-cycle-49bf4b8674de)
