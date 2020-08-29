---
title: 리덕스 공부해보기 (1) - 개요
tags:
  - typescript
  - javascript
  - react
published: true
date: 2020-04-26 07:38:25
description: "## 리덕스 공부해보기 1 [리덕스
  공식문서](https://redux.js.org/introduction/getting-started)를 스스로 대충 번역해본
  글입니다.  리덕스는 자바스크립트 앱을 위한 **예측 가능한 상태 관리 컨테이너**다.  리덕스는 일관성 있게 동작하고, 서로 다른 환경
  (클라이언트, 서버, 네이티브)에서 실행되며, 테스트하기 ..."
category: typescript
slug: /2020/04/redux-study-1/
template: post
---
## 리덕스 공부해보기 1

[리덕스 공식문서](https://redux.js.org/introduction/getting-started)를 스스로 대충 번역해본 글입니다.

리덕스는 자바스크립트 앱을 위한 **예측 가능한 상태 관리 컨테이너**다.

리덕스는 일관성 있게 동작하고, 서로 다른 환경 (클라이언트, 서버, 네이티브)에서 실행되며, 테스트하기 쉬운 애플리케이션을 만들 수 있도록 도와준다. 최상단에는, 최상의 개발자 경험을 제공하기 위한 타임 트레블 디버거를 결합한 라이브 코드 편집 등의 기능을 제공한다.

리덕스는 리액트와 함께 사용할 수 있으며, 또한 다른 뷰 라이브러리와 사용할 수 있다. 리액트는 작지만 (디펜던시를 포함하더라도 2kb) 사용가능한 다양한 애드온을 가진 생태계를 가지고 있다.

### 설치

```
# NPM
npm install redux

# Yarn
yarn add redux
```

이외도 또한 글로벌 변수인 `window.Redux`로 선언된 UMD 패키지로도 사용 가능하다. UMD 패키지는 스크립트 태그에 직접적으로 선언해서 사용 가능하다.

### 리덕스 툴 킷

리덕스는 작고 비편향적이다. (대충 특정 플랫폼에 의존적이지 않다는 뜻) 또한 [Redux Toolkit](https://redux-toolkit.js.org/)이라고 하는 패키지를 가지고 있는데, 이는 리덕스를 보다 효과적으로 사용할 수 있는 패키지이며, 공식적으로 리덕스 로직을 사용하는데 이를 활용하기를 추천한다.

리덕스 툴킷 (이하 RTX)는 일반적인 유즈 케이스를 매우 간결하게 하는데 도움을 주는데, 여기에는 [스토어 설정](https://redux-toolkit.js.org/api/configureStore), [리듀서를 만들고 불면한 업데이트 로직을 작성하는법](https://redux-toolkit.js.org/api/createreducer), [심지어 모든 스테이트를 한번에 슬라이스하는 것](https://redux-toolkit.js.org/api/createslice) 도 포함되어 있다.

새로운 프로젝트에서 Redux를 사용하는 사람인이든, 혹은 기존의 애플리케이션을 단순화 하려는 사용자든 간에 리덕스 툴 킷은 리덕스 코드를 더 잘 만들 수 있도록 도와준다.

## 리액트 리덕스 앱 만들기

리액트와 리덕스가 설치된 앱을 만드는 방법으로 추천하는 것은 Create React App의 [오피셜 리액트+JS 템클릿](https://github.com/reduxjs/cra-template-redux)을 사용하는 것이다. 이는 리덕스 툴킷을 사용하는 장점과 리액트 리덕스를 리액트 컴포넌트에 연동하기 쉽게 해준다.

```
npx create-react-app my-app --template redux
```

### 일반적인 예제

**앱의 전체 상태는, 단일 스토어 내의 오브젝트 트리에 저장된다.** 상태 트리에 변화를 주는 유일한 방법은 액션을 emit하는 것인데, 이는 오브젝트에 어떤 변화가 있는지를 알려주는 것이다. 액션이 어떻게 상태 트리를 변화시키는지 지정하기 위해, 순수한 리듀서를 작성한다.

```javascript
import { createStore } from "redux"
/**
 아래 함수는 (state, action) => state로 구성된 순수한 함수인 리듀서 이다.

 state의 구조는 개발자에 따라 달려있다. 원시타입, 배열, 오프젝트, 혹은 Immutabale.js 데이터 구조가 될 수도 있다. 
 여기에서 중요한 것은 상태 객체를 바로 변경하지 말고, 상태가 변경되면 새로운 오브젝트를 반환해야 한다는 것이다. 

 아래 예제에서는, switch와 string을 사용했으며, function map을 사용하는등 다양한 방법을 시도해볼 수 있다. 
 */
function counter(state = 0, action) {
  switch (action.type) {
    case "INCREMENT":
      return state + 1
    case "DECREMENT":
      return state - 1
    default:
      return state
  }
}

// 앱의 상태를 가지고 있을 스토어를 만든다.
// 여기애는 { subscribe, dispatch, getState } 가 있다.
let store = createStore(counter)

// 상태 변화의 응답에 따른 UI를 업데이트 하기 위해서는 subscribe()를 사용해야 한다.
// 보통 개발자들은 뷰 바인딩 라이브러리 (리액트 리덕스)를 subscribe()를 직접적으로 사용하는 것 보다 더 자주 쓸 것이다.
// 그러나 localStorage에서 현재 상태를 유지하는데에도 사용할 수 있다.

store.subscribe(() => console.log(store.getState()))

// 내부 상태값을 바꾸는 유일한 방법은 action을 dispatch하는 것이다.
//액션은 시리얼라이즈 할 수 있으며, 로그를 남기거나, 저장하거나, 이어서 할 수 있다.
store.dispatch({ type: "INCREMENT" })
// 1
store.dispatch({ type: "INCREMENT" })
// 2
store.dispatch({ type: "DECREMENT" })
// 1
```

상태를 직접적으로 수정하는 대신, 개발자가 원하는 변화를 `액션`이라는 플레인 오브젝트로 구체화 해야 한다. 그 다음, 리듀서라고 하는 특수 함수를 작성하여 모든 액션이 전체 애플리케이션을 어떻게 변화 시키는지 결정한다.

전형적인 리액트 앱에는 하나의 단일 스토어와 함께 단일 루트 리듀싱 함수가 존재할 뿐이다. 애플리케이션이 커짐에 다라서, 루트 리듀서를 더 작은 리듀서로 나누고, 각각의 상태 트리를 독립적으로 나누기도 한다. 이것은 리액트 앱에서 하나의 루트 컴포넌트가 있지만, 많은 작은 컴포넌트로 구성되어 있는 것과 비슷하다.

이러한 아키텍쳐는 좀 과해 보일 수 있지만, 이러한 배턴의 미덕은 크고 복잡한 애플리케이션을 잘 확장할 수 있다는 데에 있다. 또한 매우 강력한 개발자도구를 사용할 수 있다. 왜냐하면 이러한 액션으로 인한 모든 변화를 추적할 수 있기 때문이다. 모든 동작을 기록해뒀다가, 다시 재생해서 볼 수 있다.

## 예제 프로젝트

[여기](https://redux.js.org/introduction/examples)를 확인해보자.
