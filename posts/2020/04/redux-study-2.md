---
title: 리덕스 공부해보기 (2) - 리덕스의 탄생, 핵심 개념 그리고 3가지 원칙.
tags:
  - typescript
  - javascript
  - react
published: true
date: 2020-04-28 07:33:32
description: '## 리덕스의 탄생 배경
  https://redux.js.org/introduction/motivation  **자바스크립트 싱글 페이지 애플리케이션에 대한 요구
  사항이 점점 복잡해 짐에 따라서, 우리의 코드는 그 어느 때 보다도 더 많이 상태관리에 대한 필요성을 느끼고 있다.** 여기서 말하는
  상태에는 서버 응답, 캐시된 데이터 뿐만아니라 서버에 아직 요...'
category: typescript
slug: /2020/04/redux-study-2/
template: post
---

## 리덕스의 탄생 배경

https://redux.js.org/introduction/motivation

**자바스크립트 싱글 페이지 애플리케이션에 대한 요구 사항이 점점 복잡해 짐에 따라서, 우리의 코드는 그 어느 때 보다도 더 많이 상태관리에 대한 필요성을 느끼고 있다.** 여기서 말하는 상태에는 서버 응답, 캐시된 데이터 뿐만아니라 서버에 아직 요청되지 않은 로컬로 생성된 데이터도 포함된다. UI 상태 또한 다양한 라우팅, 탭, 스피너, 페이징 등을 관리해야 하므로 그 복잡성이 증가하고 있다.

이러한 상태를 연속적으로 관리하는 것은 더 어렵다. 만약 모델이 다른 모델에 의해서 업데이트 될 수 있고, 뷰가 모델을 업데이트하고 또 다른 모델을 연속적으로 업데이트 하면, 차례대로 다른 뷰를 업데이트 할 수도 있다. 이런식의 복잡성이 증가하다보면, 어느 순간 부터 앱에 무슨일이 일어나는지 이해하지 못하게 된다. 이말은, 개발자가 **언제, 어떻게 , 왜 상태가 업데이트 되는지에 대한 통제력을 잃게 된다.** 시스템이 불투명하고 비결정적일 때, 버그를 재현하거나 새로운 기능을 추가하는 것이 굉장히 어려워진다.

개발자로서, 우리는 서버사이드 렌더링, 라우팅이 일어나기전 데이터를 가져오는 등의 일을 처리해야 된다. 이는 우리가 이전에 다루어 보지 못했던 복잡성을 관리해야 하는 자신을 발견 하게 될 것이다.

이러한 복잡성은 mutation과 비동기라는 두가지 어려운 개념이 혼합되어 있기 때문이다. 나는 이 둘을 멘토스와 콜라라고 부른다. 이 두가지는 분리되어 있을 때는 훌륭하지만, 함께 있게 된다면 엉망진창이 된다. 리액트와 같은 라이브러리는 비동 기 및 직접 돔조작을 모두 뷰단에서 제거함으로서 이 문제를 해결하려고 한다. 하지만 데이터 상태에 대 한 관리는 전적으로 개발자의 몫이다. 여기에서 바로 Redux가 필요해진다.

리덕스는 이러한 상태 변화를, 업데이트가 언제 어떻게 일어날 수 있는지 제한하여 예측가능하게 만드려고 시도한다. 이러한 제한은 리덕스의 세가지 원칙에 반영되어 있다.

## 핵심 개념

https://redux.js.org/introduction/core-concepts

만약 당신의 할일 앱이 상태가 하나의 object로 구성되어 있다고 생각해보자.

```javascript
{
  todos: [{
    text: 'Eat food',
    completed: true
  }, {
    text: 'Exercise',
    completed: false
  }],
  visibilityFilter: 'SHOW_COMPLETED'
}
```

이 객체는 마치 setter가 없는 model처럼 보인다. 이는 상태값을 독단적으로 변경할 수 없기 때문에, 재현하기 어려운 버그를 만든다.

(리덕스에서는) 상태에서 무언가를 바꾸기 위해서는, 액션을 dispatch해야 한다. 여기서 액션은 단순한 자바스크립트 객체이며, 이는 무엇이 일어날지를 묘사한다. 액션의 예를 들어보자.

```javascript
{ type: 'ADD_TODO', text: 'Go to swimming pool' }
{ type: 'TOGGLE_TODO', index: 1 }
{ type: 'SET_VISIBILITY_FILTER', filter: 'SHOW_ALL' }
```

모든 변화가 액션으로 설명하도록 강요하는 것은, 앱에서 무슨일이 일어나고 있는지 명확하게 이해할 수 있도록 도와준다. 만약 무언가 변했다면, 그 이유를 알 수 있다. 액션은 마치 빵가루와 같다. (흔적을 남긴다는 뜻) 그것은 단지 상태와 액션을 argument로 남겨두고, 앱의 다음 상태를 리턴하는 것 뿐이다. 사이즈가 큰 앱의 경우 이러한 기능을 명세하기 어려울 수 있으므로, 상태 값을 작은 함수로 나누어서 작성해야 한다.

```javascript
function visibilityFilter(state = 'SHOW_ALL', action) {
  if (action.type === 'SET_VISIBILITY_FILTER') {
    return action.filter
  } else {
    return state
  }
}

function todos(state = [], action) {
  switch (action.type) {
    case 'ADD_TODO':
      return state.concat([{ text: action.text, completed: false }])
    case 'TOGGLE_TODO':
      return state.map((todo, index) =>
        action.index === index
          ? { text: todo.text, completed: !todo.completed }
          : todo,
      )
    default:
      return state
  }
}
```

그리고 앱의 완전한 전체 상태를 관리하는 리듀서를 작성하여, 두개의 리듀서를 각각의 키로 호출한다.

```javascript
function todoApp(state = {}, action) {
  return {
    todos: todos(state.todos, action),
    visibilityFilter: visibilityFilter(state.visibilityFilter, action),
  }
}
```

이것이 기본적인 리덕스의 전체적인 아이디어다. 리덕스는 어떠한 종류의 API도 사용하지 않는 다는 것을 명심해두자. 이러한 패턴을 유용하게 사용하기 위해 몇가지 유틸리티를 제공하지만, 주된 아이디어는 액션 오브젝트에 대응하여 시간이 지남에 따라 어떻게 상태가 업데이트 관리되는지 묘사하는 것이며, 그리고 당신이 쓰는 코드의 90%는 그저 평범한 자바스크립트 코드이고, 여기에는 어떠한 - 리덕스, API, 꼼수 - 것들도 쓰이지 않았다.

## 3가지 원칙

리덕스는 아래의 3가지 원칙으로 묘사할 수 있다.

### 신뢰할 수 있는 한가지 소스

**애플리케이션의 전채상테는 하나의 스토어 안에 오브젝트 트리 형태로 저장된다.**

이를 통해 서버의 상태를 별도의 코딩 작업 없이 클라이언트로 직렬화 하고, hydrate (plain형태로 되어 있는 state를 읽어내는 행위) 할 수 있으므로 쉽게 유니버설 앱을 만들 수 있다. 또한 하나의 상태 트리를 사용하면, 애플리케이션을 디버깅하거나 검사하기 쉽다. 또한 더 빠른 개발주기 내에서도 애플리케이션의 상태를 관리할 수 있다. 전통적으로 구현하기 어려웠던 일부기능, 예를 들어 Undo/Redo 등도 모든 상태가 단일 트리에 저장되어 있으면 굉장히 구현하기 쉽다.

```javascript
console.log(store.getState())

/* Prints
{
  visibilityFilter: 'SHOW_ALL',
  todos: [
    {
      text: 'Consider using Redux',
      completed: true,
    },
    {
      text: 'Keep all state in a single tree',
      completed: false
    }
  ]
}
*/
```

### 상태는 무조건 읽기전용이다.

**상태를 변경하는 유일한 방법은, 어떤일이 일어나는지 기술하는 action 이다.**

이는 뷰나 네트웤 콜백 모두 직접적으로 상태를 작성하지 못하게 된다. 대신, State를 변화시키려는 목적을 표현하게 된다. 왜냐하면 모든 변화가 중앙에서 이루어지고, 엄격한 순서에 따라서 하나씩 발생하기 때문에, 조심해야할 race condition이 없다. 작업은 단순한 객체일 뿐이므로, 디버깅이나 테스트 목적으로 녹화, 직렬화, 저장 및 재생할 수 있다.

```javascript
store.dispatch({
  type: 'COMPLETE_TODO',
  index: 1,
})

store.dispatch({
  type: 'SET_VISIBILITY_FILTER',
  filter: 'SHOW_COMPLETED',
})
```

### 변화는 순수 함수로 이루어진다.

**액션을 통해서 상태트리가 어떻게 바뀌는지 묘사하기 위해서는, 순수 리듀서를 작성해야 한다.**

리듀서는 이전의 상태값과 액션을 받아다가, 다음 상태를 반환하는 단순한 함수다. 여기에서는 이전의 상태값이 아닌 새로운 상태 오브젝트를 리턴해야 한다. 단일 리듀서로 시작할 수 있으며, 앱이 커지면 상태트리의 일부를 관리하는 작은 리듀러들로 구성할 수 있다. 리듀서는 단순히 함수이기 때문에, 순서를 조절하거나, 추가 데이터를 넘기거나, 페이징같은 기능을 하기 위한 재사용한 리듀서를 만들 수도 있다.

```javascript
function visibilityFilter(state = 'SHOW_ALL', action) {
  switch (action.type) {
    case 'SET_VISIBILITY_FILTER':
      return action.filter
    default:
      return state
  }
}

function todos(state = [], action) {
  switch (action.type) {
    case 'ADD_TODO':
      return [
        ...state,
        {
          text: action.text,
          completed: false,
        },
      ]
    case 'COMPLETE_TODO':
      return state.map((todo, index) => {
        if (index === action.index) {
          return Object.assign({}, todo, {
            completed: true,
          })
        }
        return todo
      })
    default:
      return state
  }
}

import { combineReducers, createStore } from 'redux'
const reducer = combineReducers({ visibilityFilter, todos })
const store = createStore(reducer)
```
