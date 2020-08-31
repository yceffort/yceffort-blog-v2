---
title: MobX를 공부하자 (4) - React와 Mobx의 10분 요약 글
tags:
  - javascript, MobX
published: true
date: 2020-08-30 19:27:22
description: "React와 MobX에 대한 10분 설명"
category: MobX
template: post
---

[이 글](https://mobx.js.org/getting-started.html)을 번역한 글입니다.

# MobX와 React를 위한 10분 소개글

```toc
tight: true,
from-heading: 2
to-heading: 3
```

## 핵심 아이디어

State는 각 애플리케이션의 핵심이며, 주변에 남아 있는 지역 변수와 일치하지 않는 상태 또는 상태를 생성하는 것만큼 관리 불가능한 버그를 만들 수 있는 더 빠른 방법은 없다. 따라서 많은 국가 관리 솔루션은 예를 들어 상태를 불변하게 함으로써 상태를 수정할 수 있는 방법을 제한하려고 한다. 그러나 이것은 새로운 문제를 야기시킨다; 데이터는 정규화되어야 하고, 참조 무결성은 더 이상 보장될 수 없으며, 프로토타입과 같은 강력한 개념을 사용하는 것은 거의 불가능해진다.

state(상태)는 애플리케이션의 핵심이자 동시에, 지역 변수와 일치하지 않는 상태를 만들거나, 상태 값이 관리 불가능해지는 등 각종 버그를 야기하는 문제점 이기도 하다. 따라서 많은 전역 상태 관리 솔루션은 상태 값을 불면하게 만들어서 상태를 수정할 수 있는 방법을 제한하려고 한다. (Immutable.js) 그러나 이는 또다른 문제를 만드는 원인이 되기도 한다. 데이터를 정규화 해야하고, 참조 무결성은 더잇아 보장되지 않으며, 프로토타입과 같은 강력한 기능을 사용하는 것은 더이상 불가능해 진다.

MobX는 상태관리를 근본적인 문제를 다룸으로써 이러한 상태 관리 문제를 간단하게 한다. MobX에서는 즉 일관되지 않은 상태를 만드는 것을 막는다. 이를 달성하기 위한 전략은 간단하다. 자동적으로 애플리케이션 상태에서 파생될 수 있는 모든 것들이 파생되게 하는 것이다.

MobX는 애플리케이션을 마치 스프레드시트 (엑셀) 처럼 다룬다.

![MobX Overview](https://mobx.js.org/assets/getting-started-assets/overview.png)

1. 먼저, 애플리케이션에는 상태가 있다. 이러한 상태에는 오브젝트, 배열, 원시값, 참고 값등 애플리케이션 모델을 구성하는 것들이 존재한다. 그리고 이 값들은 애플리케이션의 '데이터 셀' 처럼 작동한다.
2. 두번째로, 파생이 있다. 기본적으로, 어떤 값들이든 애플리케이션의 상태에 따라서 자동으로 계산되어 진다. 이러한 파생 (또는 계산된 값) 들은 끝내지 못한 할일 수와 같은 단순한 숫자일 수도 있으며, 혹은 할일 목록을 나타내는 HTML 값이 될 수도 있다. 스프레드시트 용어를 빌리자면, 이들은 수식이자 애플리케이션의 차트가 될 수 있다.
3. 리액션은 파생과 매우 비슷하다. 이 둘의 핵심 차이점은 값을 따로 만들어내지 않는 다는 것이다. 그 대신, 무언가 다른 작업을 자동으로 수행한다. 보통 I/O와 관련된 작업ㅇ르 수행한다. 이들은 DOM이 업데이트 되도록 하거나 또는 적절한 타이밍에 네트워크 요청을 하는 등의 작업을 한다.
4. 마지막으로 액션이 있다. 액션은 상태값을 바꾸는 것들을 말한다. Mobx는 액션으로 인한 애플리케이션 상태의 모든 변경사항이 자동으로 파생과 리액션에 의해 처리되도록 한다. 이는 동기식으로 이루어지며, 버그로부터 자유로울 수 있다.

## 간단한 할일 목록 예제

아래는 간단한 할일 목록을 보여주는 예제다. 보시다시피, 따로 MobX는 포함되어 있지 않ㄴ다. `TodoStore`가 할일 목록을 가지고 있다.

```javascript
class TodoStore {
	todos = [];

	get completedTodosCount() {
    	return this.todos.filter(
			todo => todo.completed === true
		).length;
    }

	report() {
		if (this.todos.length === 0)
			return "<none>";
		const nextTodo = this.todos.find(todo => todo.completed === false);
		return `Next todo: "${nextTodo ? nextTodo.task : "<none>"}". ` +
			`Progress: ${this.completedTodosCount}/${this.todos.length}`;
	}

    addTodo(task) {
		this.todos.push({
			task: task,
			completed: false,
            assignee: null
		});
	}
}

const todoStore = new TodoStore();                     
```

`todos` 목록과 함께 `todoStore`를 만들었다. 이번에는 todoStore에 할일 목록을 넣어보자. 주목해야 할 것은, `report`는 항상 최초의 할일을 프린트 하도록 되어 있다는 것이다. 이것은 약간 인위적인 기능이지만, MobX의 기능을 이해하는데 있어서 유효한 예제다.

```javascript
todoStore.addTodo("read MobX tutorial");
console.log(todoStore.report());

todoStore.addTodo("try MobX");
console.log(todoStore.report());

todoStore.todos[0].completed = true;
console.log(todoStore.report());

todoStore.todos[1].task = "try MobX in own project";
console.log(todoStore.report());

todoStore.todos[0].task = "grok MobX tutorial";
console.log(todoStore.report());
```

## 반응형으로 만들기.

아직까지, 코드에 특별한 것은 없다. 하지만 만약에 명시적으로 `report()`를 호출하는 대신에, 단지 상태값이 바뀔대 마다 자동으로 호출되게 할 수 있을까? 이는 `report()`에 영향을 미치는 코드를 호출하는 책임에 대해서 자유로워질 수 있다. 

이러한 지점이 바로 MobX가 도움이 될 수 있는 부분이다. 상태에만 의존하는 코드를 자동으로 실행하게 해주자. 이는 스프레드시트의 차트처럼 `report`가 자동으로 업데이트 되도록 하는 것이다. 이를 위해, MobX가 TodoStore를 관찰 가능하도록 만들어야 한다.

또한 `completedTodosCount` 의 값은 할일 목록에서 자동으로 파생될 수 있다. `@observable`과 `@computed` 데코레이터를 사용하면 가능하다.


```javascript
class ObservableTodoStore {
	@observable todos = [];
    @observable pendingRequests = 0;

    constructor() {
        mobx.autorun(() => console.log(this.report));
    }

	@computed get completedTodosCount() {
    	return this.todos.filter(
			todo => todo.completed === true
		).length;
    }

	@computed get report() {
		if (this.todos.length === 0)
			return "<none>";
		const nextTodo = this.todos.find(todo => todo.completed === false);
		return `Next todo: "${nextTodo ? nextTodo.task : "<none>"}". ` +
			`Progress: ${this.completedTodosCount}/${this.todos.length}`;
	}

	addTodo(task) {
		this.todos.push({
			task: task,
			completed: false,
			assignee: null
		});
	}
}


const observableTodoStore = new ObservableTodoStore();
```

이것이 전부다. 일부 속성을 `@observable`하게 만들어서, MobX가 해당 값이 변화할 때마다 추적하도록 한다. `@computed`는 이러한 변화한 상태에 따라서 자동으로 값이 계산되도록 한다.

`pendingRequests`와 `assignee`는 아직 사용되고 있지 않지만, 이후 튜토리얼에서 사용 될 것이다. 간결하게 코딩하기 위하여 이 예제에서는 ES6, JSX, 그리고 데코레이터를 사용한다. 그리고 모든 데코레이터들은 이 대신 사용할 ES5 스펙의 기술들이 대응되어 있다.

`constructor`에서 `autorun`으로 감싼 `report`함수를 볼 수 있다. `Autorun`은 한번 실행되는 reaction을 만들어내며, 이는 함수 내부에서 사용된 관찰 가능한 데이터가 변경될 때마다 자동으로 실행된다. `report`는 관찰할 수 있는 `todos`를 사용하고 있기 때문에, 적절한 때에 자동으로 실행될 것이다. 

```javascript
observableTodoStore.addTodo("read MobX tutorial");
observableTodoStore.addTodo("try MobX");
observableTodoStore.todos[0].completed = true;
observableTodoStore.todos[1].task = "try MobX in own project";
observableTodoStore.todos[0].task = "grok MobX tutorial";
```

`report` 함수가 자동으로, 동기적으로, 그리고 중간에 값을 유출하지 않는 형태로 프린트 하는 것을 볼 수 있다. 앞선 예제와는 다르게, 마지막 5번째 로그가 출력되지 않는 것을 볼 수 있다. 왜냐하면 실제로 단순히 이름을 변경한 경우이기 때문이다. 반면, 1번째 이름을 변경하게 되면, 해당 이름이 사용되고 있기 때문에 업데이트 되어 로그가 찍히는 것을 볼 수 있다. 이는 `Autorun`에서 `todos`배열 뿐만 아니라, 항목 내부의 개별 속성도 관찰하고 있음을 알 수 있다.

## 더욱 더 반응형으로 만들어보기

여기까지는 단순히 `report`를 단순하게 반응형으로 만들어 보았다. 이제는 동일한 스토어에 있는 사용자 인터페이스를 더욱더 반응형으로 만들어 볼 차례다. 반응형 컴포넌트는 (그 이름에도 불구하고) 즉시 반응하지 않는다. `mobx-react` 패키지의 `@observable` 데코레이터는 React 컴포넌트 렌더링 함수를 `autorun`으로 감써서, 컴포넌트를 자동으로 상태와 동기화되게 하여 이를 수정한다. 이는 개념적으로 우리가 이전에 했던 `report`와 다르지 않다.

아래 예제 코드는 리액트 컴포넌트를 정의한 예제이다. 여기에서 MobX는 오직 `@observable` 데코레이터만 사용되었다. 단순히 이것 만으로도 상태 값이 변할때마다 각 컴포넌트가 각각 다시 렌더링 하도록 하는데 충분하다. 더이상 `setState`를 호출할 필요가 없으며, 또한 어떤 상태값 혹은 어떤 higher order component가 렌더링하는데 필요한지 알아낼 필요가 없다. 기본적으로, 모든 컴포넌트가 스마트 해진 것이다. 그리고 이들은 여전히 멍청한 기존방식대로 작성되어 있다.

```javascript
@observer
class TodoList extends React.Component {
  render() {
    const store = this.props.store;
    return (
      <div>
        { store.report }
        <ul>
        { store.todos.map(
          (todo, idx) => <TodoView todo={ todo } key={ idx } />
        ) }
        </ul>
        { store.pendingRequests > 0 ? <marquee>Loading...</marquee> : null }
        <button onClick={ this.onNewTodo }>New Todo</button>
        <small> (double-click a todo to edit)</small>
        <RenderCounter />
      </div>
    );
  }

  onNewTodo = () => {
    this.props.store.addTodo(prompt('Enter a new todo:','coffee plz'));
  }
}

@observer
class TodoView extends React.Component {
  render() {
    const todo = this.props.todo;
    return (
      <li onDoubleClick={ this.onRename }>
        <input
          type='checkbox'
          checked={ todo.completed }
          onChange={ this.onToggleCompleted }
        />
        { todo.task }
        { todo.assignee
          ? <small>{ todo.assignee.name }</small>
          : null
        }
        <RenderCounter />
      </li>
    );
  }

  onToggleCompleted = () => {
    const todo = this.props.todo;
    todo.completed = !todo.completed;
  }

  onRename = () => {
    const todo = this.props.todo;
    todo.task = prompt('Task name', todo.task) || todo.task;
  }
}

ReactDOM.render(
  <TodoList store={ observableTodoStore } />,
  document.getElementById('reactjs-app')
);
```

```javascript
const store = observableTodoStore;
store.todos[0].completed = !store.todos[0].completed;
store.todos[1].task = "Random todo " + Math.random();
store.todos.push({ task: "Find a fine cheese", completed: true });
// etc etc.. add your own statements here...
```

## 참조된 값과의 작동

지금까지는 단순히 원시값과 배열과 함께 작동되는 모습을 보았다. 하지만 참조값과 작동은 어떻게 될까? 아래의 예제를 살펴보자.

```javascript
const store = observableTodoStore;
store.todos[0].completed = !store.todos[0].completed;
store.todos[1].task = "Random todo " + Math.random();
store.todos.push({ task: "Find a fine cheese", completed: true });
// etc etc.. add your own statements here...
```

이제 두개의 독립된 store를 갖게 되었다. 하나는 `people`이고 다른 하나는 `todos`다. `assignee`에 `people` store에 있는 값을 할당하기 위해, 단순히 참조값을 할당하였다. 이러한 변화는 자동으로 `TodoView`에 반영된다. MobX에서는, 컴포넌트를 업데이트하기 위하여 더이상 자료형을 정규화할 필요가 없다. 사실, 데이터가 어디에 있든지 상관없다. 객체가 `observable`한 이상, MobX는 이를 실시간으로 추적하게 된다. 

```html
<input onkeyup="peopleStore[1].name = event.target.value" />
```

## 비동기 액션 

todo 애플리케이션은 모든 것이 상태에서 파생되기 때문에, 상태가 언제 어떻게 바뀌는지는 정말 문제가 되지 않는다. 이는 비동기 작업을 매우 쉽게 만든다. 

코드는 매우 직관적이다. `pendingRequests`값을 업데이트 하게 되면, 이는 바로 UI의 상태 값에 반영된다. 그리고 로딩이 끝나게 되면, 해당 값을 다시 줄이게 된다. 

```javascript
observableTodoStore.pendingRequests++;
setTimeout(function() {
    observableTodoStore.addTodo('Random Todo ' + Math.random());
    observableTodoStore.pendingRequests--;
}, 2000);
```

## 결론

이것이 전부다. 특별한 보일러플레이트는 필요치 않다. 우리는 UI를 완성시키기 위해 단순히 간단한 선언적인 컴포넌트를 만들었다. 이들은 상태값으로 부터 자동적으로 파생되고, 반응하게 된다. 지금까지 배운것을 요약하면 다음과 같다.

1.  `@observable` 데코레이터 또는 `observable`을 사용하여 MobX가 해당 객체를 추적가능하게 한다.
2.  `@computed` 데코레이터를 사용하여 자동으로 상태값으로 부터 값이 계산되는 함수를 만든다.
3.  `autorun`은 의존하고 있는 `observable`한 값이 변경 될때 마다 자동으로 실행되는 함수다. 이는 로깅, 네트워크 요청 등을 처리할 때 유용하다.
4.  `mobx-react`의 `@observer`를 사용하여 리액트 컴포넌트를 반응형으로 반들 수 있다. 이는 자동적으로 그리고 효율적으로 컴포넌트를 업데이트 한다. 이는 매우 크고 복잡한 데이터를 다루는 어플리케이션에서도 유용하다.

## MobX는 상태 컨테이너가 아니다.

사람들은 종종 MobX를 Redux의 대체재로 생각하고는 한다. MobX는 그러나 단순히 기술적인 문제를 풀기 위한 라이브러리이며, 상태 컨테이너 그 자체 또는 새로운 아키텍쳐가 절대 아니다. 그런 의미에서 위의 예제들이 고안되었으며, 로직의 컵슐화, 스토어 또는 컨트롤러 등의 정리등은 적절한 엔지니어링 관행을 사용하는 것이 좋다.