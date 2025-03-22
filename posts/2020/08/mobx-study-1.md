---
title: MobX를 공부하자 (1)
tags:
  - javascript
  - MobX
published: true
date: 2020-08-21 15:54:00
description: 'MobX 1페이지 요약에 대한 간단한 번역'
category: MobX
template: post
---

# MobX One Page Summary

[MobX One Page Summary](https://mobx.js.org/README.html)를 번역 및 요약 해보았습니다.

> derive는 적절한 단어가 생각이 안나서 '파생'으로 번역했습니다. 여기서 derive는 state(상태)를 변하게 하는 액션을 의미합니다.

## Table of Contents

## MobX

간단하고, 확장 가능한 상태 관리

## 설치

- 설치
  - 일반: `npm install mobx --save`
  - 리액트: `npm install mobx-react --save`
- CDN:
  - https://unpkg.com/mobx/lib/mobx.umd.js
  - https://cdnjs.com/libraries/mobx

## 브라우저 지원

- 버전 5이상 부터는 [ES6 proxy를 지원하는 모든 브라우저](https://kangax.github.io/compat-table/es6/#test-Proxy)에서 실행 가능하다. IE11, nodejs 6 미만 오래된 자바스크립트 코어를 가진 리액트 네이티브 안드로이드 등에서는 오류가 날 것이다.
- 버전 4는 모든 ES5를 지원하는 브라우저에서 동작하며, 계속해서 유지보수 될 것이다. 4와 5의 api 스펙은 동일하지만, 그러나 4에서는 [몇몇 제한](https://mobx.js.org/README.html#mobx-4-vs-mobx-5)이 있다.

> MobX 5 패키지의 진입지점에서는 모든 빌드 도구와의 역호환성을 위하여 ES5 코드가 함께 제공된다. 그러나 위에서 언급했던 것 처럼, 어차피 MobX 5는 모던 브라우저에서만 작동하므로 더빠르고 가벼운 빌드를 위해서 아래와 같은 웹팩 alias를 추가하기를 권한다.

```javascript
resolve: {
  alias: {
    mobx: __dirname + '/node_modules/mobx/lib/mobx.es6.js'
  }
}
```

## 참고해 볼 만한 것들

- https://egghead.io/courses/manage-complex-state-in-react-apps-with-mobx
- https://mobx.js.org/getting-started
- https://github.com/mobxjs/awesome-mobx#boilerplates
- https://github.com/mobxjs/awesome-mobx#related-projects-and-utilities
- https://github.com/mobxjs/awesome-mobx#awesome-mobx

## 소개

MobX는 함수형 반응형 프로그래밍을 적용하여 전역 상태 관리를 단순하고 확장 가능하게 만드는 라이브러리다. MobX의 철학은 간단하다.

> 애플리케이션 상태에서 파생될 수 있는 것은 모두 자동으로 파생되어야 한다.

![MobX-philosophy](https://mobx.js.org/assets/flow.png)

리액트와 MobX를 같이 쓰는 것은 매우 훌륭한 조합이다. 리액트는 렌더링 가능한 컴포넌트 트리를 변환하는 메커니즘을 바탕으로 애플리케이션 상태를 렌더링한다. MobX는 리액트가 사용하는 애플리케이션 상태를 저장하고, 업데이트 하는 메커니즘을 제공한다.

React와 MobX는 모두 애플리케이션 개발에서 발생하는 공통적인 문제에 대한 각각 고유한 최적의 해결책을 제공한다. 리액트는 비용이 많이드는 DOM 조작을 줄이기 위하여, 가장 DOM을 활용하여 UI를 최적으로 렌더링하는 메커니즘을 재공한다. MobX는 엄격하게 필요할 때만 업데이트 되고, 최신을 유지하는 반응형 가상 종속성 상태 그래프를 사용하여 애플리케이션 상태 값을 리액트 컴포넌트와 함께 최적으로 동기화 하는 메커니즘을 제공한다.

## 핵심 개념

MobX에는 몇가지 핵심 개념이 존재한다.

### Observable state (관찰 가능한 상태)

MobX는 객체, 배열, 클래스 인스턴스와 같은 기존 데이트 구조에 예측 가능한 기능을 추가한다. 이것은 단순히 @observable 데코레이터만 추가하면 된다.

```javascript
import {observable} from 'mobx'

class Todo {
  id = Math.random()
  @observable title = ''
  @observable finished = false
}
```

`observable`을 사용하는 것은 객체의 속성을 마치 스프레드시트 셀로 바꾸는 것과 같다. 이는 수정하면 다른 셀이 자동으로 재계산되거나, 그래프가 다시 렌더링 되거나, 다른 흥미로운 Reaction을 트리거할 수 있다. 스프레드시트 셀과 달리 `observable`한 값은 primitive한 값 뿐만 아니라, 참조 객체 및 배열도 될 수 있다.

만약 개발환경에서 데코레이터 문법을 지원하지 않는다면, [이 글](https://mobx.js.org/best/decorators.html)을 참조해봐도 좋다. 그게 아니라면 MobX는 데코레이터 문법을 지원하지 않아도 decorate 유틸리티를 활용해서 똑같이 구현할 수 있다. 대부분의 MobX 유저들은 데코레이터 문법을 선호하는데, 이는 데코레이터 문법이 조금더 간결하기 때문이다.

```javascript
import {decorate, observable} from 'mobx'

class Todo {
  id = Math.random()
  title = ''
  finished = false
}
decorate(Todo, {
  title: observable,
  finished: observable,
})
```

### Compute Values (자동 값 계산)

MobX를 활용하면, 관련 데이터가 수정될 때 자동으로 파생된 값을 정의할 수 있다. 이는 `@computed` 데코레이터나, 위에서 `observable`를 사용했다면, getter/setter 함수를 활용해서 구현할 수도 있다.

```javascript
class TodoList {
  @observable todos = []
  @computed get unfinishedTodoCount() {
    return this.todos.filter((todo) => !todo.finished).length
  }
}
```

MobX는 todo가 추가되너가 `finished` 값이 수정되면 `unfinishedTodoCount`를 자동으로 계산한다. 이는 마치 엑셀과 같은 스프레드시트 프로그램에서 자동으로 연산이 되는 것과 같다. 이들은 오로지 필요할 때만 자동으로 업데이트 된다.

### Reaction

Reaction은 Compute Values와 비슷하지만 값을 계산하는 대신 콘솔, 네트워크 요청, 리액트 컴포넌트 트리 업데이트 등 다른 부수효과를 만들어낸다. 간단히 말해, Reaction은 반응형과 명령형 프로그래밍 사이의 다리 같은 역할을 한다.

#### React Components

만약 리액트를 사용하고 있다면, `mobx-react` 패키지에 있는 [observer](http://mobxjs.github.io/mobx/refguide/observer-component.html) 함수/데코레이터를 추가하여 반응형 컴포넌트를 만들 수 있다.

```javascript
import React, {Component} from 'react'
import ReactDOM from 'react-dom'
import {observer} from 'mobx-react'

@observer
class TodoListView extends Component {
  render() {
    return (
      <div>
        <ul>
          {this.props.todoList.todos.map((todo) => (
            <TodoView todo={todo} key={todo.id} />
          ))}
        </ul>
        Tasks left: {this.props.todoList.unfinishedTodoCount}
      </div>
    )
  }
}

const TodoView = observer(({todo}) => (
  <li>
    <input
      type="checkbox"
      checked={todo.finished}
      onClick={() => (todo.finished = !todo.finished)}
    />
    {todo.title}
  </li>
))

const store = new TodoList()
ReactDOM.render(
  <TodoListView todoList={store} />,
  document.getElementById('mount'),
)
```

`observer`는 리액트 컴포넌트를 렌더링하는 데이터의 파생 요소로 변환한다. MobX를 사용하면, 모든 컴포넌트들은 스마트하게 렌더링되지만, 멍청한 방식으로 정의된다. MobX는 오직 필요할 때만 컴포넌트를 다시 렌더링하며 그 이상도 그 이하의 작업도 하지 않는다. 따라서 위의 예제에서, `onClick`핸들러는 적절한 `TodoView`를 렌더링 하도록 강제하고, 오직 완료되지 않는 task 숫자가 변경된 경우에 한해서 만 `TodoLIstView`를 렌더링하게 된다. 그러나 `Tasks left` 라인을 삭제하거나 (혹은 다른 컴포넌트로 분리하거나) 하는 경우에는 `TodoListView`는 더 이상 재 렌더링 되지 않는다.

#### Custom Reaction

사용자정의 Reaction은 상황에 맞게 [autorun](http://mobxjs.github.io/mobx/refguide/autorun.html) [reaction](http://mobxjs.github.io/mobx/refguide/reaction.html) [when]()를 사용하면 간단하게 만들 수 있다.

예를 들어 `autorun`을 아래와 같이 활용하면, `unfinishedTodoCount`값이 바뀔 때마다 로그를 찍는다.

```javascript
autorun(() => {
  console.log('Tasks left: ' + todos.unfinishedTodoCount)
})
```

### 무엇이 MobX에서 반응하게 하는가?

왜 `unfinishedTodoCount`가 바뀔 때마다 새로운 메시지가 출력될까?

> MobX는 실제로 추적하는 함수의 실행 중에 읽는 모든 관측가능한 속성에 대해서 반응한다.

더 자세한 내용을 알고 싶다면 [이 글](https://mobx.js.org/best/react.html)을 참고하면 된다.

### Actions

다른 flux 프레임워크와는 다르게, MobX에서는 사용자 이베트를 어떻게 처리해야하는지에 대한 의견이 분분하다.

- Flux와 같은 방식으로 처리하기
- RxJS를 활용
- `onClick`핸들러와 같은 가장 간단하고 직관적인 방식으로

결국 이 모든 것들은 한가지로 요약할 수 있다: 어떻게든 상태를 업데이트해야 한다.

상태를 업데이트 한 이후에, MobX는 효율적이고 결함이 없는 방식으로 나머지 동작을 처리한다. 따라서 아래와 같이 간단한 코드는 인터페이스를 자동으로 업데이트 하기에 충분하다.

이벤트를 트리거하거나, dispatcher를 호출하는 등의 기술적인 필요성은 존재하지 않는다. 리액트 컴포넌트는 결국 상태를 멋있게 표현하는 방식에 지나지 않는다. 이는 MobX가 관리하게 된다.

```javascript
store.todos.push(new Todo('Get Coffee'), new Todo('Write simpler code'))
store.todos[0].finished = true
```

그럼에도 불구하고, MobX에는 선택적으로 활용할 수 있는 빌트인 개념인 [action](https://mobx.js.org/refguide/action.html)을 활용할 수 있다. 비동기 액션을 처리하는 방법에 대해 알고 싶다면, 이 글을 읽어보는 것도 좋다. 이는 매우 쉽고, 코드를 더 잘 구성하고 언제 어디서 수정되어야 하는지에 대한 현명한 결정을 내리는데 도움을 준다.

## MobX: 간단하고, 확장가능한

MobX는 글로벌 상태 관리에 사용할 수 있는, 가장 방해요소가 적은 라이브러리다. 이는 MobX 접근 방식을 단순하게 할 뿐만 아니라, 확장성도 매우 뛰어나게 만든다.

### 클래스와 실제 참조 활용

MobX를 사용하면, 데이터를 정규화 할 필요가 없다. 이는 매우 복잡한 도메인 모델에서도 라이브러리를 적절하게 활용할 수 있다.

### 참조 무결성 보장

데이터는 정규화 될 필요가 없고, MobX는 자동으로 상태와 파생 간의 관계를 추적하기 때문에, 참조 무결성을 공짜로 얻을 수 있다. MobX는 상태를 추적해서, 참조자 중 하나가 바뀔 때마다 다시 렌더링하게 된다. 프로그래머는 컴포넌트에 영향을 미친다는 사실을 잊을 수도 있지만, MobX는 그렇지 않다.

### 간단한 액션은 관리를 쉽게 한다

위에서 설명했듯, MobX를 사용하면 상태를 수정하는 것은 매우 간단해진다. 단순히 의도를 코딩하면 된다. 나머지는 MobX가 알아서 한다.

### 효율적인 세밀한 관측

MobX는 애플리케이션의 모든 파생에 대한 그래프를 구축하여, 무결성을 방지하는데 필요한 최소한의 계산 수를 연산한다. "모든 파생" 이라는 것이 비싸게 들릴 수 있지만, 가상 파생 그래프를 구축하여 데이터를 상태와 동기화 시키는데 필요한 재조합수를 최소화 한다.

### 간편한 상호운용성

Mobx는 순수 자바스크립트 구조로 작동한다. 따라서 MobX가 작동하기 위해서 특정 라이브러리를 필요로 하지 않는다. 따라서 지금 사용하고 있는 다양한 라이브러리와도 호환된다. 같은 이류로 서버와 클라이언트, 리액트 네이티브 등에서도 활용가능하다. 이러한 결과 MobX를 사용할 때 다른 상태 관리 솔루션에 비해 새로운 개념을 덜 알아도 된다.

> 실제로 [dependencies에 아무것도 없다.](https://github.com/mobxjs/mobx/blob/6ec6499fb8b55a17fe65f42b14d1188fd7fa1ba1/package.json#L58)

## MobX 4와 5의 차이

위 두 버전의 차이는, 5에서는 속성값 추적을 위해서 `Proxies`를 활용했다는 것이다. 그 결과 MobX 5에서는 Proxy를 지원하는 브라우저에서만 사용할 수 있고, MobX는 ES5가 작동하는 모든 환경에서 사용 가능하다.

MobX4에서 유념해야할 한계점은

- Observable arrays는 진짜 array가 아니어서, `Array.isArray()`를 통과하지 못한다. 따라서 다른 써드 파티라이브러리에서 이를 활용하기 전에 `slice()`등을 활용할 필요가 있다.
- 기존 observable 객채에 속성을 추가하는 것은 자동으로 선택되지 않는다. 따라서 observable 맵을 활용하거나, 빌트인 유틸리티 함수를 활용해야 한다. [참고](https://mobx.js.org/refguide/object-api.html)
