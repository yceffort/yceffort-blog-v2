---
title: 리덕스 공부해보기 (3) - 용어
tags:
  - typescript
  - javascript
  - react
published: true
date: 2020-04-29 06:09:18
description: 'https://redux.js.org/glossary#state ## 용어 모음  ### State
  (상태)  ```typescript type State = any ```  State (State tree라고 도 불리운다)는 Redux
  API에서는 보통 스토어에서 관리하고, `getState()`에 의해 반환되는 단일 값을 가리킨다.  관례적으로, 가장...'
category: typescript
slug: /2020/04/redux-study-3/
template: post
---

https://redux.js.org/glossary#state

## 용어 모음

### State (상태)

```typescript
type State = any
```

State (State tree라고 도 불리운다)는 Redux API에서는 보통 스토어에서 관리하고, `getState()`에 의해 반환되는 단일 값을 가리킨다.

관례적으로, 가장 최상단의 상태는 객체 또는 키값 형태의 맵이지만, 기술적으로는 어떤 타입도 될 수 있다. 여전히, 이 상태값을 직렬화 될 수 있게 관리할 수 있도록 최선을 다해야 한다.

### Action (액션)

```typescript
type Action = Object
```

액션은 순수한 오브젝트로, 상태의 변경을 어떤식으로 할지를 나타낸다. 액션은 스토어에 저장되어 있는 데이터를 꺼내오는 유일한 방법이다. 네트워크 콜백이든, UI 이벤트든, 혹은 웹소캣과같은 다른 어떠한 이벤트 소스에서오든 데이터 이든지 간에, 결국 액션을 통해서 dispatch해야 한다.

액션은 반드시 액션이 실행되는 type을 가르키는 type 필드를 가지고 있어야 한다. Types은 상수또는 다른 모듈에서 가져오는 방법으로 정의될 수 있다. Symbol보다는 string을 타입으로 사용하는 것이 좋은데, 그 이유는 직렬화 될 수 있기 때문이다.

타입 이 외에는, 액션 오브젝트의 구조는 개발자의 손에 달려있다. 만약 액션이 어떻게 구조화되는지 관심있다면, [Flux Standard Action](https://github.com/redux-utilities/flux-standard-action)을 참조하길 바란다.

#### 액션은

```typescript
{
  type: 'ADD_TODO',
  payload: new Error(),
  error: true
}
```

반드시 (MUST)

- 순수 자바스크립트 오브젝트여야 한다.
- `type` 속성을 가지고 있어야 한다.

되도록 (MAY)

- `error`속성을 가지고 있으면 좋다
- `payload`속성을 가지고 있으면 좋다
- `meta`속성을 가지고 있으면 좋다

#### type

액션의 타입은 컨슈머에 액션이 일으키는 속성을 가르킨다. 만약 타입이 똑같다면, 이들은 엄격하게 일치해야한다 (===)

#### payload

`payload`는 어떤 타입의 값이든 가질 수 있다. 이는 액션이 가지는 `payload`를 의미한다. `type`또는 `status`를 제외한 액션의 정보는 모두 `payload`에 있어야 한다. 관례적으로, `error`가 `true`라면, `payload`는 에러 객체를 가지고 있어야 한다. 이는 오류의 promise가 오류시 오류 객체를 리턴하는 것과 같다.

#### error

옵셔널 속성인 `error`는 만약 액션에 에러가 있을 경우 `true`로 세팅된다. error가 true인 액션은 리젝트된 promise와 유사하다. 관례상 payload는 오류 객체가 되어야 한다.

만약 `error`에 `null`을 포함하여 `true`외 에 다른 값이 있는 경우, 해당 액션을 오류로 보지 않는다.

#### meta

옵셔널 속성인 `meta`는 어떤 타입의 값이든 될 수 있다. 여기에는 `payload`의 일부가 될 수 없는 추가적인 정보를 담기위해 설정되어있다.

### Reducer

```typescript
type Reducer<S, A> = (state: S, action: A) => S
```

리듀서는 (리듀싱 함수라고도 불리운다) 파라미터(`accumulation`)를 받아서 새로운 파라미터를 반환하는 함수다. 이는 value의 모음을 하나의 value로 축약하는데 사용된다.

리듀서는 리덕스만의 독특한 것이 아니다. 이는 함수형 프로그래밍의 기초적인 컨셉이다. 심지어 자바스크립트와 같은 비 함수적인 언어에서도 리듀싱을 위한 api가 존재한다. [Array.prototype.reduce()
](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/Reduce)

리덕스에서 누적되는 값 (리듀싱 되는 값)은 상태 오브젝트이며, 여기의 값들은 action에 의해 누적된다. 리듀서는 이전의 상태와 액션을 기반으로 새로운 상태를 만들어 낸다. 이들은 모두 순수함수여야만 한다. 함수들은 주어진 입력값으로 정확히 같은 결과값을 내야 한다. 또한 사이드 이펙트에서 자유로워야 한다. 이러한 전제는 핫 리로딩이나 타임 트래블 (과거의 값을 가져오는)을 가능하게 해준다.

### Dispatching function

```typescript
type BaseDispatch = (a: Action) => Action
type Dispatch = (a: Action | AsyncAction) => any
```

dispatching function (간단히 dispatch function)는 함수가 액션 또는 비동기 액션을 받아드리는 것을 의미한다.

여기에서 일반적인 function dispatching과 어떠한 미들웨어를 거치지 않은 스토어에 제공하는 dispatch function 을 구별해야 한다.

`Base Dispatch function`은 스토어의 리듀서에 동기 액션을 제공해야 하는데, 여기에는 이전의 상태값을 통해서 계산된 새로운 상태값이 포함되어야 한다.

미들웨어는 `dispatch function`을 래핑한다. 이는 `dispatch function`이 비동기 액션을 다룰 수 있도록 한다. 미들웨어는 transform, delay, ignore 등 action을 interpret하는 어떤 것이 될수도 있다. 또한 다음 미들웨어에 넘기기전 비동기 액션이 될 수도 있다. 자세한 것은 하단의 내용을 참고하길 바란다.

### Action Creator

```typescript
type ActionCreator = (...args: any) => Action | AsyncAction
```

`Action Creator`는 간단히 말해서 액션을 만드는 함수다. 액션은 payload정보, `Action Creator`는 액션을 만들기 위한 팩토리 임을 구별해야 한다.

`Action Creator`를 호출한다는 것은 단순히 액션을 만드는 것이지, dispatch하는 것이 아니다. 값에 변화를 주기 위해서는 dispatch를 사용해야 한다. 이 따금 `bound action creator`라는 용어가 나오는데, 이는 action creator를 호출하고 그즉시 그 결과를 특정 스토어에 dispatch하는 것을 의미한다.

만약 `Action Creator`가 현재 상태를 읽어오거나, API호출을 하거나, 라우팅 전환 같은 작업을 수행해야 하는 경우 비동기 액션을 반환해야 한다.

### Async Action

```typescript
type AsyncAction = any
```

비동기 액션은 dispatching function에 내보내 지는 값이지만, 아직 리듀서에서 사용할 준비가 안된 값이다. 이는 dispatch 함수를 통해 보내지기 전에 미들웨어를 통해 처리될 필요가 있다. 비동기 액션은 미들웨어에 따라서 다양한 타입이 될 수 있다. 이것들은 Promise나 thunk같은 비동기 원시타입이 될 수 있는데, 이들은 리듀서에 바로 넘길 수는 없지만, 작업이 끝나게 되면 dispatch할 수 있다.

### Middleware

```typescript
type MiddlewareAPI = {dispatch: Dispatch; getState: () => State}
type Middleware = (api: MiddlewareAPI) => (next: Dispatch) => Dispatch
```

미들웨어는 higher-order function으로 dispatch function 을 compose해 새로운 dispatch function을 만든다. 때로는 비동기 액션을 액션으로 바꾸기도 한다.

미들웨어는 함수 합성으로 만들 수 있다. 미들웨어는 액션에 로그를 남기거나, 라우팅, 비동기 API 호출 등 사이드 이펙트를 발생시키는 것들을 동기 액션 내에서 사용할 때 유용하다.

[여기](https://redux.js.org/api/applymiddleware)를 참고하여 미들웨어가 어떻게 생겼는지 살펴보자.

### Store

```typescript
type Store = {
  dispatch: Dispatch
  getState: () => State
  subscribe: (listener: () => void) => () => void
  replaceReducer: (reducer: Reducer) => void
}
```

store란 애플리케이션의 상태 트리를 가지고 있는 객체다. 리덕스 앱에서는 다양한 리듀서레벨을 합성하여 단하나의 스토어만 둘 수 있다.

- [dispatch(action)](https://redux.js.org/api/store#dispatchaction)은 base dispatch를 의미한다.
- [getState()](https://redux.js.org/api/store#getState)는 현재 스토어의 상태 값을 리턴한다.
- [subscribe](https://redux.js.org/api/store#subscribelistener)는 상태의 변화가 있을때 호출되는 함수를 등록할 수 있다.
- [replaceReducer(nextReducer)](https://redux.js.org/api/store#replacereducernextreducer) 핫리로딩이나 코드 스플리팅에 사용할 수 있다. 대부분의 경우 사용할 일이 없다.

[여기](https://redux.js.org/api/store#dispatchaction)에서 자세한 내용을 확인할 수 있다.

### Store creator

```typescript
type StoreCreator = (reducer: Reducer, preloadedState: ?State) => Store
```

`Store creator`는 리덕스 스토어를 만드는 함수다. dispatching function과 마찬가지로, 리덕스 패키지에서 내보낸 [createStore(reducer, preloadedState)](https://redux.js.org/api/createstore)와 store enhancer에서 반환되는 store creator를 구분해야 한다.

### Store enhancer

```typescript
type StoreEnhancer = (next: StoreCreator) => StoreCreator
```

store enhancer는 higher-order function으로 store creator로 새로운 store creator를 반환하는 함수다. 이는 composable한 방식으로 스토어를 변형할 수 있다는 것에서 미들웨어와 비슷하다.

store enhancer는 리액트의 higher-order 컴포넌트와 매우 비슷한데, 리액트에서도 이를 `component enhancer`로 부른다.

스토어는 인스턴스가 아니라 단순한 객채의 집합인 함수이기 때문에, 원래 스토어를 변형시키지 않고도 복사분을 쉽게 만들고 수정할 수 있다. [여기](https://redux.js.org/api/compose)에서 예제를 찾아볼 수 있다.

그러나 아마 이를 쓸일이 별로 없을 것이다. 그러나 개발 툴에서 제공하는 것을 사용할 수 있다. 일례로 앱에서 일어나는 것을 타임 트레블로 녹화하거나 재생할 때 사용된다. 재밌게도, 리덕스의 미들웨어 구현은 그자체가 store enhancer다.
