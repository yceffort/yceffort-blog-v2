---
title: "Fiber: 리액트의 새로운 재조정 알고리즘인 Fiber에 대해 살펴보기"
tags:
  - javascript
  - web
  - react
published: false
date: 2020-07-05 03:23:12
description: "[Inside Fiber: in-depth overview of the new reconciliation
  algorithm in
  React](https://indepth.dev/inside-fiber-in-depth-overview-of-the-new-reconcil\
  iation-algorithm-in-react/)을 번역했습니다.  ```toc tight..."
category: javascript
slug: /2020/07/new-reconciliation-algorithm-in-react/
template: post
---
[Inside Fiber: in-depth overview of the new reconciliation algorithm in React](https://indepth.dev/inside-fiber-in-depth-overview-of-the-new-reconciliation-algorithm-in-react/)을 번역했습니다. 

```toc
tight: true,
from-heading: 2
to-heading: 3
```

React는 사용자 인터페이스를 구축하기 위한 자바스크립트 라이브러리다. 이것의 중심에는 컴포넌트의 상태 변화를 추적하고, 업데이트된 상태를 화면에 반영하는 메커니즘(Change Detection)이 존재한다. 리액트에서는 이것을 Reconciliation(이하 재조정)이라고 한다. `setState`메소드를 호출하여 `state`또는 `prop`값이 변화하였는지 확인하고, UI의 컴포넌트를 다시 렌더링 한다.

[리액트는 이 메커니즘에 대한 좋은 설명을 제공한다.](https://reactjs.org/docs/reconciliation.html): 리액트 엘리먼트의 역할, 라이프 사이클 메소드, `render`메소드, 컴포넌트의 자식을 비교 하는 알고리즘 등. render 메소드로 부터 반환되는 불변의 리액트 엘리먼트 트리는 가상 DOM으로 알려져 있다. 이 용어는 초기에 리액트를 설명하는데 많은 도움이 되었지만, 한편으로 많은 혼란을 야기 시켰고 이제 더이상 공식 문서에서 사용되지 않는다. 여기에서는 우리는 가상 DOM 대신 리액트 엘리먼트 트리라고 부르겠다.

리액트 엘리먼트 트리외에도, 프레임워크에는 state를 유지하기 위해 다양한 인스턴스를 사용한다. (컴포넌트, DOM 노드 등) 리액트 버전 16부터는, 내부 인스턴스 트리와 이를 관리하는 알고리즘을 코드명 `Fiber`로 새롭게 구현했다. Fiber 아키텍쳐가 제공하는 이점에 대한 자세한 내용은 [여기](https://indepth.dev/the-how-and-why-on-reacts-usage-of-linked-list-in-fiber-to-walk-the-components-tree/)를 참조하면 된다.

이 아티클은 리액트 내부 구조를 알리기 위한 목적으로 제작된 첫번째 글이다. 이 글에서 알고리즘과 관련된 중요한 개념과, 데이터 구조에 대한 심층적인 개요를 제공한다. 이 후에는, Fiber트리를 횡단하고 처리하는데 사용되는 알고리즘과 주요 기능을 탐구할 것이다. 다음 글에서는 리액트가 알고리즘을 활용하여 초기 렌더링과 `state`를 처리하는 법, 그리고 `props`을 업데이트 하는 법을 볼 것이다. 그 다음에 스케줄러의 세부사항, 재조정과정, 그리고 이를 작성하는 메커니즘을 볼 것이다. 

리액트의 동시성 작업에 숨겨진 마술을 이해하기 위해 이 글을 보기를 권한다. 만약 리액트에 기여할 생각이 있다면, 이 글은 훌륭한 가이드가 될 것이다. 

확실히 많은 내용을 담고 있으니, 이해되지 않더라고 스트레스는 받을 필요가 없다. 모든 가치있는 것을 이해하는데는 시간이 든다. 리액트를 사용하기 위해 이 모든 것을 알 필요가 없다. 다만 내부에서 어떻게 작동하는지를 이해하는데 도움이 된다.

## 배경

이 글에서 다룰 예제를 소개한다. 버튼 하나가 있고, 이버튼이 화면에 나와있는 숫자를 하나씩 업데이트 한다.


```javascript
class ClickCounter extends React.Component {
    constructor(props) {
        super(props);
        this.state = {count: 0};
        this.handleClick = this.handleClick.bind(this);
    }

    handleClick() {
        this.setState((state) => {
            return {count: state.count + 1};
        });
    }


    render() {
        return [
            <button key="1" onClick={this.handleClick}>Update counter</button>,
            <span key="2">{this.state.count}</span>
        ]
    }
}
```

보시다시피, `render`에서 `button`과 `span`을 리턴하는 간단한 컴포넌트다. 버튼을 클릭하면, 핸들러 내부에서 컴포넌트의 `state`를 업데이트 한다. 그 결과가 `<span/>` 태그 내에서 보여진다.

리액트가 재조정을 하는 과정에서는 다양한 작업이 이루어진다. 예를 들어, 여기에는 첫번째 렌더링과 상태 업데이트후 리액트가 수행하는 연산등이 포함되어 있다.

- `ClickCounter` 내부의 `state.count` 를 업데이트
- `ClickCounter`의 자식과 props 값을 비교하는 과정
- `span`태그를 위해 `props`를 업데이트 하는 과정

재조정 과정에서는 라이프 사이클 메소드를 호출하거나, `ref`를 갱신하는 등의 다른 작업도 존재한다. 이러한 모든 활동을 Fiber 아키텍쳐에서는 `work`(이하 작업) 라고 한다. 이러한 작업의 유형은 리액트 엘리먼트의 타입에 따라 달라진다. 예를 들어, 클래스 컴포넌트의 경우 리액트는 인스턴스를 만들어야 하지만, 함수형 컴포넌트는 그렇지 않다. 알다시피, 리액트에는 함수형과 클래스형 컴포넌트, 호스트 컴포넌트 (DOM 노드), 포탈 등이 존재한다. 리액 엘리먼트들의 타입은 `createElemtn`함수에 첫번째로 들어가는 파라미터에 따라서 결정된다. 이 함수는 일반적으로 `render`메소드 내에서 엘리먼트를 만들때 사용된다.

Fiber알고리즘과 여기에서 일어나는 작업들을 살펴보기전에, 리액트에 의해 내부적으로 사요되는 데이터 구조에 대해 알아보자.

## 리액트 엘리먼트에서 Fiber 노드까지

모든 리액트의 컴포넌트는 UI 표현식을 가지고 있으며, 우리는 이를 `render` 메소드를 호출하면 리턴되는 값으로 볼 수 있다. `ClickCounter`의 템플릿을 살펴보자.

```javascript
<button key="1" onClick={this.onClick}>Update counter</button>
<span key="2">{this.state.count}</span>
```

### 리액트 엘리먼트

템플릿이 JSX 컴파일러로 들어가면, 수만은 리액트 엘리먼트가 생기게 된다. 이는 HTML이 아닌, 리액트 컴포넌트의 렌더링 메소드가 실제로 반환한 것이다. JSX는 꼭 필요한 것이 아니므로, `ClickCounter`컴포넌트의 렌더링 방식은 아래와 같이 다시 작성될 수 있다.

```javascript
class ClickCounter {
    ...
    render() {
        return [
            React.createElement(
                'button',
                {
                    key: '1',
                    onClick: this.onClick
                },
                'Update counter'
            ),
            React.createElement(
                'span',
                {
                    key: '2'
                },
                this.state.count
            )
        ]
    }
}
```

`render`메소드 내부의 `React.createElement`는 두가지 데이터 구조를 가지게 된다.

```javascript
[
    {
        $$typeof: Symbol(react.element),
        type: 'button',
        key: "1",
        props: {
            children: 'Update counter',
            onClick: () => { ... }
        }
    },
    {
        $$typeof: Symbol(react.element),
        type: 'span',
        key: "2",
        props: {
            children: 0
        }
    }
]
```

여기서 우리는 리액트가 [$$typeof](https://overreacted.io/why-do-react-elements-have-typeof-property/)를 오브젝트애ㅔ 추가하여 각 리엑트 엘리먼트를 구별하는 것을 볼 수 있다. 여기에는 다시 `type` `key` `props`등 의 프로퍼티를 가지고 있다. 이 값들은 `React.createElement`에서 넘어온 값들이다. 리액트가 어떻게 `span` `button` 등의 하위 노드에서 텍스트를 어떻게 나타내는지 주목하자. 그리고 클릭 핸들러는 `button` 컴포넌트의 props로 구성되어 있다. 그리고 이 문서에서는 다루지 않는 ref와 같은 다른 필드들도 리엑트 엘리먼트에 존재한다.

`ClickCounter`의 리액트 엘리먼트는 별다른 key나 props를 가지고 있지 않다.

```javascript
{
    $$typeof: Symbol(react.element),
    key: null,
    props: {},
    ref: null,
    type: ClickCounter
}
```

### Fiber 노드

재조정과정에서, 리액트 엘리먼트의 render 로 부터 리턴된 모든 데이터들은 fiber 노드들의 트리에 병합된다. 모든 리액트 엘리먼트는 이와 대응하는 Fiber 노드가 있다. 리액트 엘리먼트와는 다르게, 파이버는 모든 `render` 시에 재생성되는 것이 아니다. 이는 컴포넌트 상태와 DOM 정보를 가지고 있는 변이가능한 데이터 구조다. 

앞에서 우리는 리액트 엘리먼트의 유형애 따라 프레임워크가 다른 활동을 할 수 있다고 언급했다. 우리의 샘플 어플리케이션에서 `ClickCounter` 클래스 컴포넌트의 경우에는 라이프 사이클 메소드와 렌더 메소드를 호출하고, `<span />` 호스트 컴포넌트의 경우 (DOM 노드) DOM Mutation(변이)를 수행한다. 따라서 리액트의 각 엘리먼트는 수행해야하는 작업을 각 해당하는 Fiber 노드로 변환 시킨다. [해당 코드](https://github.com/facebook/react/blob/769b1f270e1251d9dbdce0fcbd9e92e502d059b8/packages/shared/ReactWorkTags.js)

**다시 말해 Fiber는 해야할 작업, 해야할 작업의 단위를 나타내는 데이터 구조라고 볼 수 있다. Fiber 아키텍쳐는 또한 작업을 추적, 스케쥴, 중지, 취소하는 편리한 방법을 제공한다.**

리액트 엘리먼트가 처음으로 Fiber 노드로 변환되면, 리액트는 요소의 데이터를 사용하여 [createFiberFromTypeAndProps](https://github.com/facebook/react/blob/769b1f270e1251d9dbdce0fcbd9e92e502d059b8/packages/react-reconciler/src/ReactFiber.js#L414) 함수를 호출하여 Fiber를 생성한다. 리액트는 Fiber노드를 재사용하고 단지 해당 리액트 엘리먼트의 데이터를 사용하여 필요한 속성을 업데이트 한다. 또한 리액트는 `key` prop을 기반으로한 계층 구조에서 노드를 이동하거나, 해당 리액트 노드의 렌더에서 더이상 반환하는 경우 삭제를 할 수도 있다.

> [ChildReconciler](https://github.com/facebook/react/blob/95a313ec0b957f71798a69d8e83408f40e76765b/packages/react-reconciler/src/ReactChildFiber.js#L239)를 확인하면 리액트의 성능및 모든 작업 목록을 확인할 수 있다.

리액트는 각 리액트 엘리먼트에 대해 Fiber를 생성하기 때문에, 그리고 우리는 엘리먼트에 대해 트리를 가지고 있기 때문에 결과적으로 Fiber 노드 트리를 가지고 있는 것이다. 우리의 예제의 경우 다음과 같이 보인다.

![](https://admin.indepth.dev/content/images/2019/07/image-51.png)

모든 파이버 노드는 child, sibling, return이라는 Fiber노드 속성을 활용하여 링크된 리스트에 연결된다. 이러한 방식으로 작동하는 이유와 자세한 내용은 [여기](https://medium.com/dailyjs/the-how-and-why-on-reacts-usage-of-linked-list-in-fiber-67f1014d0eb7)를 참조하면 된다.

## `Current` 및 `Work in process`

첫번째 렌더링 이후, React는 UI를 렌더링하는데 사용한 어플리케이션 상태를 반영하는 fiber 트리를 반환하게 된다. 이를 `work`라고 한다. 리액트가 업데이트 작업을 시작하면, 화면에 미래의 상태를 반영할 `workInProcess` 트리를 만들게 된다.

Fiber에서 수행되는 모든 `work`는 `workInProgress`트리에서 수행된다. 리액트가 `current`트리를 거칠 때마다, 각 각의 존재하는 fiber 노드는 `workInProgress` 트리를 구성하는 대체 노드를 생성한다. 이 노드는 `render`메소드에 의해 반환된 데이터에 의해 만들어진다. 업데이트 작업이 진행되고 모든 작업이 끝나게 되면, 리액트는 화면에 보여줄 대체 트리를 표시한다. `workInProgress`트리가 화면에 렌더링 되면, 이는 현재의 `current` 트리가 된다.

리액트의 핵심 원칙중 한가지는 일관성이다. 리액트는 항상 DOM 업데이트를 한번에 하지, 각각의 결과물을 따로 보여주지 않는다. `workInProgress`트리는 유저에게는 보이지 않는 일종의 임시 트리로 간주된다. 따라서 리액트가 모든 컴포넌트에 대해 해당 작업을 끝내게 되면, 이 변화를 화면에 보여주게 된다.

소스코드 내에서 `currnet`와 `workInProgress`트리에서 fiber 노드를 취하는 많은 함수를 볼 수 있다. 예를 들어보면 아래와 같다.

```javascript
function updateHostComponent(current, workInProgress, renderExpirationTime) {...}
```

각 파이버 노드들은 대체 필드의 다른 트리에서 상대에 대한 참조를 가지고 있다. 현재 트리 노드가 `workinProgress` 트리의 노드를 가리키고, 반대로도 가리키고 있다.

## 부수효과

리액트의 엘리먼트는 state와 prop을 활용하여 UI표현을 연산하는 함수로 볼 수 있다. DOM을 변경하건, 라이프사이클 메소드를 호출하는 것과 같은 다른 모든 작업은 부수효과로 간주되어야 한다. 여기에서 말하는 효과 (effect)는 다음과 같이 정리되어 있다.

> 데이터 fetch, subscriptions 혹은 DOM을 강제로 변경하는 작업을 리액트 컴포넌트에서 해보았을 것이다. 이러한 작업을 `side effect` 혹은 `effect`라고 하는데, 그 이유은 이것이 다른 컴포넌트에 영향을 미칠 수 있고 또한 렌더링 도충에는 수행할 수 없기 때문이다.

대부분의 `state` 또는 `props` 의 변화가 어떤 부수효과를 야기할 수 있을지 알수 있다. 그리고 effect를 적용하는 것은 일종의 작업의 종류이기 때문에, Fiber노드는 이러한 effect를 추적하기 위한 편리한 메커니즘이다. 각 fiber 노드는 그것과 관련된 효과만을 가질 수 있다. 그리고 이것은 `effectTag`에 포함되어있다.


## 효과 목록

리액트는 업데이트를 처리하기 위해 매우 신속하게 빠르고 처리하기 위해, 몇가지 흥미로운 기술을 사용했다. **그 중 하나가 빠른 반복을 위한 Fiber nodes와 effect 의 선형 리스트를 만드는 것이다.** 선형 리스트를 순회하는 것은 트리보다 훨씬 빠르며, 부수효과가 없는 노드에 시간을 할애할 필요도 없다.

이 리스트의 목적은 DOM 업데이트 및 effect가 존재하는 노드를 표시하는 것이다. 이들은 `finishedWork` 트리의 서브셋이며, 이들은 `current` 및 `workInProgress`의 트리 자식 속성을 사용하는 대신 `nextEffect`라고 하는 속성을 사용한다.

