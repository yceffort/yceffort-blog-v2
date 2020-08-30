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

State is the heart of each application and there is no quicker way to create buggy, unmanageable applications than by producing an inconsistent state or state that is out-of-sync with local variables that linger around. Hence many state management solutions try to restrict the ways in which you can modify state, for example by making state immutable. But this introduces new problems; data needs to be normalized, referential integrity can no longer be guaranteed and it becomes next to impossible to use powerful concepts like prototypes.

MobX makes state management simple again by addressing the root issue: it makes it impossible to produce an inconsistent state. The strategy to achieve that is simple: Make sure that everything that can be derived from the application state, will be derived. Automatically.

Conceptually MobX treats your application like a spreadsheet.

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
