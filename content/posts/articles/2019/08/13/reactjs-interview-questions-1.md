---
title: 리액트 면접 질문 모음 (1)
date: 2019-08-21 01:22:35
published: true
tags:
  - javascript
  - react
description: "[목차](/2019/08/13/reactjs-interview-questions/)  ```toc tight:
  true, from-heading: 2 to-heading: 3 ```  ## Core React  ### What is
  React  리액트는 오픈소스 프론트엔드 자바스크립트 라이브러리로, 특히 싱글 페이지 어플리케이션의 사용자 인터페이스 구축을..."
category: javascript
slug: /2019/08/13/reactjs-interview-questions-1/
template: post
---
[목차](/2019/08/13/reactjs-interview-questions/)


```toc
tight: true,
from-heading: 2
to-heading: 3
```

## Core React

### What is React

리액트는 오픈소스 프론트엔드 자바스크립트 라이브러리로, 특히 싱글 페이지 어플리케이션의 사용자 인터페이스 구축을 위해 사용된다. 웹가 모바일 앱의 뷰단을 다르기 위하여 사용되고 있다. 리액트는 페이스북에서 일아흔 Jordan Walke가 만들었다. 최초로 리액트 기반으로 만들어진 서비스는 2011년에 페이스북 뉴스 피드이며, 2012년에는 인스타그램도 리액트로 만들어 졌다.

### What are the major features of React?

리액트의 주요 기능은 무엇인가?

- RealDOM을 조작하는데 많은 비용이 소모되어 대신 VirtualDOM을 활용하고 있다.
- 서버사이드렌더링을 지원한다
- 단방향 데이터흐름 또는 단방향 데이터 바인딩을 따른다
- 뷰를 개발하는데 있어 재사용 가능한 컴포넌트 사용

### What is JSX?

JSX는 ECMA Script의 XML 신택스 확장 표기법이다. (Javascript XML의 약자다.) 기본적으로, `React.createElement()`함수에 문법 슈가를 제공하며,HTML 스타일의 템플릿 구문화함께 javascript를 표현할 수 있다.

아래 예제에서, `return`안에 있는 `<h1>` 구문이 자바스크립트 함수의 render function 으로 제공된다.

```javascript
class App extends React.Component {
  render() {
    return (
      <div>
        <h1>{"Welcome to React world!"}</h1>
      </div>
    );
  }
}
```

### What is the difference between Element and Component?

`element`는 DOM노드나 컴포넌트 단에서 화면에 보여주고 싶은 요소를 그리는 하나의 오브젝트를 의미한다. `element`는 `element`의 props에서 포함될 수 있다. 리액트에서 `element`를 만드는건 많은 비용이 들지 않는다. 한번 만들고 나면, 더이상 변경이 불가능하다.

리액트에서 `element`를 만드는 예시는 아래와 같다.

```javascript
const element = React.createElement("div", { id: "login-btn" }, "Login");
```

위 함수는 아래와 같은 object를 리턴한다

```javascript
{
  type: 'div',
  props: {
    children: 'Login',
    id: 'login-btn'
  }
}
```

그리고 `ReactDOM.render()`이 아래와 같은 DOM을 만들어 줄 것이다.

```html
<div id="login-btn">Login</div>
```

반면에 컴포넌트는 다양한 방식으로 선언가능하다. 컴포넌트는 `render()`와 함께 쓴다면 클래스가 될 수도 있다. 좀더 단순한 방법으로, 함수로도 선언이 될 수 있다. 두 방식 모두 `props`를 input으로 받으며, `JSX`를 리턴한다.

```javascript
const Button = ({ onLogin }) => (
  <div id={"login-btn"} onClick={onLogin}>
    Login
  </div>
);
```

JSX는 이를 `React.createElement()` 함수로 트랜스파일 시킬 것이다.

```html
const Button = ({ onLogin }) => React.createElement( 'div', { id: 'login-btn',
onClick: onLogin }, 'Login' )
```

### How to create components in React?

두 가지 방법이 존재한다.

1.  함수형 컴포넌트: 컴포넌트를 만드는 가장 심플한 방식이다. `props`를 첫번째 파라미터로 받는 받는 순수 자바스크립트 함수를 만들고, React Element를 반환하면 된다.

```javascript
function Greeting({ message }) {
  return <h1>{`Hello, ${message}`}</h1>;
}
```

1.  클래스 컴포넌트: ES6의 클래스를 활용하여 컴포넌트를 정의할 수도 있다. 위 컴포넌트를 클래스 컴포넌트로 바꾼다면 이렇게 될 것이다.

```javascript
class Greeting extends React.Component {
  render() {
    return <h1>{`Hello, ${this.props.message}`}</h1>;
  }
}
```

### When to use a Class Component over a Function Component?

컴포넌트가 **state나 라이프 사이클 메소드를** 필요로 할 때 클래스 컴포넌트를, 그렇지 않으면 함수형 컴포넌트를 활용하면 된다.

> 근데 요즘은 `useState`을 사용하면 함수형 컴포넌트에서도 state사용이 가능하다

### What are Pure Components?

`React.PureComponent`는 `React.Component`에서 `shouldComponentUpdate`가 없다는 것만 제외하면 동일하다. `props`나 `state`에 변화가 있을 경우, `PureComponent`는 두 변수에 대해서 [얕은 비교](https://reactjs.org/docs/shallow-compare.html)를 한다. 반면 `Component`는 그런 비교를 하지 않는다. 따라서 `Component`는 `shouldComponentUpdate`가 호출 될 때마다 다시 render한다.

### What is state in React?

`state`란 컴포넌트가 살아있는 동안에 걸쳐 변화할 수도 있는 값을 가지고 있는 object다. 따라서 state를 가능한 간단하게, 그리고 state의 구성요소를 최소화하는 노력을 기울여야 한다. 다음은 User Component에 message state를 관리하는 예제다.

```javascript
class User extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      message: "Welcome to React world"
    };
  }

  render() {
    return (
      <div>
        <h1>{this.state.message}</h1>
      </div>
    );
  }
}
```

`state`는 `props`와 비슷하지만, 컴포넌트가 완전히 소유권을 쥐고 있다는 것이 다르다.다른 어떤 컴포넌트도 한 컴포넌트가 소유하고 있는 `state`에 접근할 수 없다.

### What are props in React?

`props`는 컴포넌트의 input 값이다. HTML 태그 속성과 유사한 규칙을 사용하여 ReactComponent에 전달할 수 있는 단일 값 또는 객체 다. 이런 데이터 들은 부모 컴포넌트에서 자식 컴포넌트로 보낼 수 있다.

리액트에서 `props`를 쓰는 주요 목적은 컴포넌트에 아래와 같은 기능을 제공하기 위해서다.

- 컴포넌트에 custom data를 넘기기 위해
- `state`의 변화를 trigger 하기 위해
- Component의 render메소드 안에서 this.props.\*\*\* 로 사용하기 위함

예를 들어, `reactProp` 을 만들어서 쓴다고 가정해 보자.

```javascript
<Element reactProp={"1"} />
```

`reactProp`은 (뭐라고 정의했던 지 간에) React를 사용하여 생성된 component에서 접근이 가능하고, React native props에서 접근하여 사용할 수 있다.

```javascript
props.reactProp;
```

### What is the difference between state and props?

`props`와 `state`는 모두 순수 자바스크립트 오브젝트다. 두 객체 모두 `render`의 output에 영향을 줄 수 있는 정보를 가지고 있지만, 컴포넌트의 기능적인 측면에서는 약간 다르다. `props`는 함수의 파라미터와 비슷한 방식으로 작동하는 반면, `state`는 컴포넌트 내에서 선언된 변수와 비슷하다.

### Why should we not update the state directly?

`state`를 아래와 같이 바로 업데이트 하면 렌더링이 일어나지 않는다.

```javascript
this.state.message = "Hello world";
```

대신에 `setState()` 메서드를 사용하자.이는 `state`의 변경이 있을 때 `component`를 업데이트 해준다. `state`에 변화가 있을 경우, 컴포넌트는 리렌더링으로 응답한다.

```javascript
//Correct
this.setState({ message: "Hello World" });
```

주의: state를 직접 할당할 수 있는 곳은 `constructor` 혹은 자바스크립트 클래스의 필드를 선언하는 syntax 뿐이다.

### What is the purpose of callback function as an argument of `setState()`?

콜백함수는 setState가 끝나고 컴포넌트가 렌더링 된 이후에 실행된다.`setState`는 비동기로 이루어지기 때문에 callback에서는 어떤 액션이든 취할 수 있다.

주의: 콜백함수를 사용하는 것보다 라이프사이클 메서드를 사용하는게 더 좋다.

```javascript
setState({ name: "John" }, () =>
  console.log("The name has updated and component re-rendered")
);
```

### What is the difference between HTML and React event handling?

1. HTML에서는 이벤트명은 소문자로 작성되어야 한다.

```html
<button onclick="activateLasers()"></button>
```

React는 camelCase를 사용한다.

```html
<button onClick="{activateLasers}"></button>
```

2. HTML에서는, `false`를 리턴하면 이후 기본 액션을 막을 수 있다.

```html
<a href="#" onclick='console.log("The link was clicked."); return false;' />
```

하지만 react에서는 `preventDefault()`를 명시적으로 사용해야 한다.

```javascript
function handleClick(event) {
  event.preventDefault();
  console.log("The link was clicked.");
}
```

### How to bind methods or event handlers in JSX callbacks?

1. 생성자에서 바인딩하기: 자바스크립트 클래스에서는, 메소드들이 기본적으로 바인딩 되어 있지 않다. 이는 클래스 메서드로 정의된 리액트 이벤트 핸들러와 마찬가지다. 보통, 생성자에서 바인딩한다.

```javascript
class Component extends React.Componenet {
  constructor(props) {
    super(props);
    this.handleClick = this.handleClick.bind(this);
  }

  handleClick() {
    // ...
  }
}
```

2. 퍼블리기 클래스 필드 구문: 생성자에서 바인딩 되기를 원치 않는다면, 퍼블릭 클래스의 필드 구문을 이용하여 callback을 올바르게 바인딩 할 수 있다.

```javascript
handleClick = () => {
  console.log("this is:", this);
};

<button onClick={this.handleClick}> Click me </button>;
```

> 클래스 필드(class field)
> 클래스 내부의 캡슐화된 변수를 말한다. 데이터 멤버 또는 멤버 변수라고도 부른다. 클래스 필드는 인스턴스의 프로퍼티 또는 정적 프로퍼티가 될 수 있다. 쉽게 말해, 자바스크립트의 생성자 함수에서 this에 추가한 프로퍼티를 클래스 기반 객체지향 언어에서는 클래스 필드라고 부른다.

```javascript
class Foo {
  name = ""; // SyntaxError

  constructor() {}
}
```

> constructor 내부에서 선언한 클래스 필드는 클래스가 생성할 인스턴스를 가리키는 this에 바인딩한다. 이로써 클래스 필드는 클래스가 생성할 인스턴스의 프로퍼티가 되며, 클래스의 인스턴스를 통해 클래스 외부에서 언제나 참조할 수 있다. 즉, 언제나 public이다.
> ES6의 클래스는 다른 객체지향 언어처럼 private, public, protected 키워드와 같은 접근 제한자(access modifier)를 지원하지 않는다.

3. 화살표함수: 콜백에 화살표 함수를 사용할 수도 있다.

```javascript
<button onClick={event => this.handleClick(event)}>{"Click me"}</button>
```

주의: 콜백이 하위 컴포넌트에 `prop`으로 전달된다면, component가 리렌더링 될 수도 있다. 이러한 경우에는, 성능을 고려해서 1, 2번의 예제를 활용하는 것이 낫다.

### How to pass a parameter to an event handler or callback?

이벤트 핸들러와 파라미터 전달을ㅇ 화살표 함수로 감쌀 수 있다.

```html
<button onClick={() => this.handleClick(id)} />
```

이는 `.bind`와 같다.

```html
<button onClick="{this.handleClick.bind(this," id)} />
```

두 방식 이외에도, 아래와 같은 배열 함수 방식으로 정의해서 전달할 수도 있다.

```javascript
<button onClick={this.handleClick(id)} />;
handleClick = id => () => {
  console.log("Hello, your ticket number is", id);
};
```

### What are synthetic events in React?

synthetic event (합성함수) 는 브라우저의 네이티브 이벤트를 위한 크로스 브라우저 래퍼다. 이 api는 브라우저의 네이티브 이벤트와 동일하며, 마찬가지로 `stopPropagation()` `preventDefault()`도 포함하고 있지만, 모든 브라우저에서 동일하게 작동한다는 점이 다르다.

### What is inline conditional expressions?

조건부 렌더 표현을 위해 javascript의 if문이나 삼항연산자를 사용할 수 있다. 이외에도 중괄호로 묶어서 javascript의 논리식인 &&을 붙여서 jsx에서도 사용할 수 있다.

```html
<h1>Hello!</h1>
; { messages.length > 0 && !isLogin ? (
<h2>You have {messages.length} unread messages.</h2>
) : (
<h2>You don't have unread messages.</h2>
); }
```

### What are "key" props and what is the benefit of using them in arrays of elements?

`key`는 특별한 string 속성으로, 배열을 사용할 때 이용해야 한다. `key`는 리액트에서 어떤 item이 변화하고, 추가되고, 삭제되었는지 구별하는데 도움을 준다. 대부분 key로 id를 사용한다.

```html
const todoItems = todos.map(todo =>
<li key="{todo.id}">{todo.text}</li>
);
```

만약 이런 ID가 없다면, index를 사용할 수 있다.

```html
const todoItems = todos.map((todo, index) =>
<li key="{index}">
  {todo.text}
</li>
)
```

주의

1. index를 key로 사용하는 방식은, 아이템의 순서가 바뀌는 경우가 발생할 수 있는 케이스에는 별로 추천할만하지 못하다. 이는 퍼포먼스에 악영향을 미치고, component state에 악영향을 미칠 수 있다.
2. list를 별도 컴포넌트로 뽑아서 사용하는 경우, key를 리스트 컴포넌트가 아닌 `li` 태그에 사용해야 한다.
3. 리스트 아이템에 `key`가 없으면 콘솔에 경고 메시지가 뜬다.

### What is the use of refs?

`ref`는 element의 참조값을 반환한다. 대부분 이러한 경우는 피해야 하지만, DOM이나 component에 다이렉트로 접근해야할 때 유용하다.

### How to create refs?

1. 최근에 추가된 방식으로, `React.createRef()` 메소들를 사용하면, React element는 `ref`를 통해서 접근할 수 있다. `ref`를 컴포넌트에서 접근하기 위해서는, 생성자 안에 `ref`를 instance property로 할당하면 된다.

```javascript
class MyComponent extends React.Component {
  constructor(props) {
    super(props);
    this.myRef = React.createRef();
  }
  render() {
    return <div ref={this.myRef} />;
  }
}
```

2. React 버전과 상관없이 ref 콜백을 활용하는 방식이 있다. 예를 들어, SearchBar 컴포넌트의 인풋 요소들은 아래와 같은 방식으로 접근 가능하다.

```javascript
class SearchBar extends Component {
  constructor(props) {
    super(props);
    this.txtSearch = null;
    this.state = { term: "" };
    this.setInputSearchRef = e => {
      this.txtSearch = e;
    };
  }
  onInputChange(event) {
    this.setState({ term: this.txtSearch.value });
  }
  render() {
    return (
      <input
        value={this.state.term}
        onChange={this.onInputChange.bind(this)}
        ref={this.setInputSearchRef}
      />
    );
  }
}
```

또한 컴포넌트의 함수 내에서 클로져를 `ref`를 사용할 수도 있다.

주의: 추천할만한 방법은 아니지만, 인라인 `ref` callback을 이용하는 방식도 있다.

### What are forward refs?

Ref forwarding은 일부 컴포넌트에서 ref를 받아서 자식 컴포넌트에게 전달하는 것을 의미한다.

```javascript
const ButtonElement = React.forwardRef((props, ref) => (
  <button ref={ref} className="CustomButton">
    {props.children}
  </button>
));

// Create ref to the DOM button:
const ref = React.createRef();
<ButtonElement ref={ref}>{"Forward Ref"}</ButtonElement>;
```

### Which is preferred option with in callback refs and findDOMNode()?

callback ref를 쓰는 것이 더 선호된다. 왜냐하면 `findDOMNode()`는 향후에 있을 리액트의 개선사항이 반영되지 않기 때문이다.

레거시에서 `findDOMNode`를 사용하는 방법이 있다.

```javascript
class MyComponent extends Component {
  componentDidMount() {
    findDOMNode(this).scrollIntoView();
  }

  render() {
    return <div />;
  }
}
```

그래서 선호하는 방법은 다음과 같다.

```javascript
class MyComponent extends Component {
  constructor(props) {
    super(props);
    this.node = createRef();
  }
  componentDidMount() {
    this.node.current.scrollIntoView();
  }

  render() {
    return <div ref={this.node} />;
  }
}
```

### Why are String Refs legacy?

예전에 React를 다뤄보았다면, 옛날 방식인 `ref`를 string으로 쓰는, `ref={'textInput'}` 와 같이 ref속성이 string이고, DOM Node인 `refs.textInput`로 접근하는 방법에 익숙할 것이다. 그러나 이러한 string ref는 하단에서 언급할 문제들 때문에, 레거시로 보는 것이 맞다. 그리고 string ref는 React v16에서 제거 되었다.

1. String ref는 실행중인 component 요소를 추적하도록 강제한다. 그리고 React Module을 stateful하게 만들기 때문에, 이는 번들시 react module이 중복 되는 경우 이상한 오류를 발생시킨다.
2. 라이브러리를 추가하여 String ref를 child component에 전달한다면, 사용자는 다른 ref를 추가할 수 없다. 그러나 callback ref를 사용하면 이런 문제를 해결할 수 있다.
3. Flow와 같은 정적 분석에서는 동작하지 않는다. Flow는 string ref를 this.refs와 같은 형태로 표시하도록 만드는 트릭을 추적할 수 없다. callback ref는 string ref보다 flow에 더 잘맞다.
4. 대부분이 render callback 패턴으로 동작하기를 기대하지만, 그렇게 동작하지 않는다.

```javascript
class MyComponent extends Component {
  renderRow = index => {
    // 동작하지 않는다. ref는 MyComponent가 아닌 DataTable에 연결될 것이다.
    return <input ref={"input-" + index} />;

    // 이거는 동작한다. callback ref가 짱이다.
    return <input ref={input => (this["input-" + index] = input)} />;
  };

  render() {
    return <DataTable data={this.props.data} renderRow={this.renderRow} />;
  }
}
```

### What is Virtual DOM?

Virtual DOM은 메모리 내에서 표현되는 Real DOM 이다. UI는 메모리 상에서 표현되며, 그리고 real DOM과 동기화 된다. 이는 렌더 함수 호출과 화면에 elements 표시 하는 사이에 일어난다. 이 모든 과정을 `reconciliation`이라고 한다.

### How Virtual DOM works?

1. 어디서든 데이터가 편하면, Virtual DOM내에서 전체 UI가 다시 렌덜이 된다.
   ![virtual-dom-1](https://github.com/sudheerj/reactjs-interview-questions/raw/master../../../images/vdom1.png)

2. 그런 다음 이전 DOM과 새로운 DOM을 비교한다.
   ![virtual-dom-2](https://github.com/sudheerj/reactjs-interview-questions/raw/master../../../images/vdom2.png)

3. 계산이 끝나면, Real DOM 중에서 실제로 업데이트가 있었던 부분 만 변경을 가한다.
   ![virtual-dom-3](https://github.com/sudheerj/reactjs-interview-questions/raw/master../../../images/vdom3.png)

### What is the difference between Shadow DOM and Virtual DOM?

Shadow DOM은 web component의 scope및 CSS scope 지정을 위해 설계된 web browser 기술이다. Virtual DOM은 브라우저 API 위에 자바스크립트에서 구현되는 개념이다.

### What is React Fiber?

Fiber는 React v16에서 새로운 reconciliation 엔진, 그리고 코어 알고리즘을 새로 작성한 것으로 볼 수 있다. React Fiber의 목적은 애니메이션, 레이아웃, 제스쳐, 작업일시정지 및 중단, 여려 유형의 업데이트 우선순위 조절, 동시성 등 여러가지 기본 사항에 대한 성능을 높이는 것이다.

### What is the main goal of React Fiber?

React Fiber 의 목표는 애니메이션, 레이아웃, 제스처등의 성능을 높이는 것이다. 렌더링 작업을 chunk별로 작업하고, 여러 프레임 별로 이를 펼치면서 작업하는 점진적 렌더링을 통해 이를 구현했다.

### What are controlled components?

입력요소를 제어하는 component를 controlled components라고 부른다. 모든 상태변경에 연관뢴 handler function이 존재한다.

예를 들어, 모든 이름을 대문자로 쓰기 위해서는, `handleChange`를 아래와 같이 쓰게 된다.

```javascript
handleChange(event) {
  this.setState({value: event.target.value.toUpperCase()})
}
```

### What are uncontrolled components?

uncontrolled components란 내부적으로 자기 자신의 state를 가지고 있는 component다. 현재 필요한 값을 찾기 위해 ref를 사용하여 DOM query를 할 수 있다. 이는 전통적인 HTML 과 비슷하다.

`UserProfile` Component를 아래에서 보자면, `name` input이 ref를 통해서 접근할 수 있다.

```javascript
class UserProfile extends React.Component {
  constructor(props) {
    super(props);
    this.handleSubmit = this.handleSubmit.bind(this);
    this.input = React.createRef();
  }

  handleSubmit(event) {
    alert("A name was submitted: " + this.input.current.value);
    event.preventDefault();
  }

  render() {
    return (
      <form onSubmit={this.handleSubmit}>
        <label>
          {"Name:"}
          <input type="text" ref={this.input} />
        </label>
        <input type="submit" value="Submit" />
      </form>
    );
  }
}
```

대부분의 경우, 폼에서는 controlled component를 사용하기를 추천한다.

### What is the difference between createElement and cloneElement?

JSX는 `React.createElement()` 함수로 UI에 나타낼 React element를 생성한다. 반면 `cloneElement`는 element를 props로 보낼 때 사용한다.

### What is Lifting State Up in React?

여러 component 들이 동일한 변경 데이터를 공유해야하는 경우 가까운 부모 component 로 state를 올리는 것이 좋다. 즉, 두개의 자식 component가 부모에 있는 동일한 데이터를 공유할 때. 두개의 자식 component 들은 local state를 유지하는 대신, 부모로 state를 올려야 한다.

### What are the different phases of component lifecycle?

React lifecycle에는 세 개의 phase가 있다.

1. `mounting`: 컴포넌트가 browser DOM에 마운트 될 준비가 된 상태다. 이 phase에는 `constructor()` `getDerivedStateFromProps()` `render()` `componentDidMount()`가 있다
2. `updating`: 이 단계에서는, 컴포넌트가 두가지 방법으로 업데이트 된다. 새로운 `props`를 보내거나, `setState()` `forceUpdate()`를 통해서 state를 업데이트 하는 방법이 있다. 이 단계에서는, `getDerivedStateFromProps()` `shouldComponentUpdate()` `render()` `getSnapshotBeforeUpdate()` `componentDidUpdate()` 가 포함된다.
3. `unmounting`: 이단계에서는, browser DOM이 더 이 더이상 필요 없어지거나 unmount된다. 여기에는 `componentWillUnmount()`가 포함된다.

DOM에서의 변경을 적용할 때, 내부에서 어떤 과정을 거치는지 알아볼 필요가 있다. 각 단계는 아래와 같다.

1. `Render` 컴포넌트가 어떠한 사이드 이펙트 없이 렌더링 된다. 이는 Pure Component에 적용되며, 이 단계에서는 일시정지, 중단, 렌더 재시작등이 가능하다.
2. `Pre-commit`: 컴포넌트가 실제 변화를 DOM에 반영하기 전에, 리액트가 DOM을 `getSnapshotBeforeUpdate()` 통해서 DOM 을 읽을 수도 있다.
3. `Commit`: React는 DOM과 함께 작동하며, 각각의 라이프 사이클 마지막에 실행되는 것들이 포함된다. `componentDidMount()` `componentDidUpdate()` `componentWillUnmount()`

   16.3 이후

![react-16.3-phases](https://github.com/sudheerj/reactjs-interview-questions/raw/master/images/phases16.3.jpg)

16.3 이전

![before-react-16.3](https://github.com/sudheerj/reactjs-interview-questions/raw/master/images/phases.png)

### What are the lifecycle methods of React?

React 16.3+

- `getDerivedStateFromProps`: 모든 `render()`가 실행되기 바로 직전에 호출된다. props의 변화의 결과로 내부 state 변화를 가능하게 해주는 메서드로, 굉장히 드물게 사용된다.
- `componentDidMount`: 첫렌더링이 다 끝나고, 모든 ajax 요청이 완료, DOM이나 state 변화, 그리고 이벤트 리스너가 모두 설정된 다음에 호출된다.
- `shouldComponentUpdate`: 컴포넌트가 업데이트 될지 말지를 결정한다. default로 true를 리턴한다. 만약 state나 props 업데이트 이후에 컴포넌트가 업데이트 될 필요가 없다고 생각한다면, false를 리턴하면 된다. 컴포넌트가 새로운 props를 받은 후에, 리 렌더링을 방지해서 성능을 향상시키기에 가장 좋은 위치다.
- `getSnapshotBeforeUpdate`: 렌더 결과물이 DOM에 커밋되기 직전에 호출된다. 여기서 리턴된 모든 값은 `componentDidUpdate()`로 넘겨진다. 스크롤 포지션 등, DOM에서 필요한 정보를 사용할 때 유용하다.
- `componentDidUpdate`: prop/state의 변화d의 응답으로 DOM을 업데이트 할 때 필요하다. 이 메소드는 만약 `shouldComponentUpdate()`가 `false`를 리턴하면 호출되지 않는다.
- `componentWillUnmount`: 네트워크 요청을 취소하거나, 컴포넌트와 관련된 이벤트 리스너를 삭제할 때 쓰인다.

> before 16.3은 따로 번역하지 않겠습니다.

- `componentWillMount`: Executed before rendering and is used for App level configuration in your root component.
- `componentDidMount`: Executed after first rendering and here all AJAX requests, DOM or state updates, and set up event listeners should occur.
  componentWillReceiveProps: Executed when particular prop updates to trigger state transitions.
- `shouldComponentUpdate`: Determines if the component will be updated or not. By default it returns true. If you are sure that the component doesn't need to render after state or props are updated, you can return false value. It is a great place to improve performance as it allows you to prevent a re-render if component receives new prop.
- `componentWillUpdate`: Executed before re-rendering the component when there are props & state changes confirmed by shouldComponentUpdate() which returns true.
- `componentDidUpdate`: Mostly it is used to update the DOM in response to prop or state changes.
- `componentWillUnmount`: It will be used to cancel any outgoing network requests, or remove all event listeners associated with the component.

### What are Higher-Order Components?

Higher-order Component (이하 HOC)는 컴포넌트를 받아서 새로운 컴포넌트를 리턴하는 컴포넌트다. 기본적으로, 이러한 패턴은 리액트의 컴포넌트적인 특성에서 유래되었다.

이를 `Pure Component`라고 부르는데, 동적으로 제공되는 하위 component를 그대로 사용하지만, 입력받은 component를 수정/복사하지 않기 때문이다.

HOC는 아래와 같은 use case에서 사용할 수 있다.

- 코드 재사용, 로직 추상화
- render 하이재킹
- state 추상화 또는 조작
- props 조작

### How to create props proxy for HOC component?

`props proxy pattern`을 아래와 같이 사용한다면, 컴포넌트에 넘겨진 props를 추가/수정할 수 있다.

```javascript
function HOC(WrappedComponent) {
  return class Test extends Component {
    render() {
      const newProps = {
        title: "New Header",
        footer: false,
        showFeatureX: false,
        showFeatureY: true
      };

      return <WrappedComponent {...this.props} {...newProps} />;
    }
  };
}
```

### What is context?

Context는 props을 탑다운으로 주지 않고도, 어느 레벨에서든 데이터를 컴포넌트 트리에 넘기는 방법이다. 예를 들어 인증받은 사용자, 언어 설정, UI theme 등 어플리케이션 단위에서 다양한 컴포넌트가 사용해야 하는 데이터를 context를 통해서 줄 수 있다.

```javascript
const { Provider, Consumer } = React.createContext(defaultValue);
```

### What is children prop?

Children은 prop (`this.prop.children`) 으로, 다른 컴포넌트에 컴포넌트를 넘길 수 있는 방법으로, 다른 prop를 사용하는 것과 동일하다. 컴포넌트 트리는 이 children을 여닫는 태그 사이에 두며, 이는 컴포넌트를 `children prop`으로 건내게 된다.

React API에서 이러한 형태로 다양한 prop을 제공하고 있다. `React.Children.map` `React.Children.forEach` `React.Children.count` `React.Children.only` `React.Children.toArray` 사용예제는 아래와 같다.

```javascript
const MyDiv = React.createClass({
  render: function() {
    return <div>{this.props.children}</div>;
  }
});

ReactDOM.render(
  <MyDiv>
    <span>{"Hello"}</span>
    <span>{"World"}</span>
  </MyDiv>,
  node
);
```

### How to write comments in React?

React/JSX의 주석은 자바스크립트의 다중 주석과 비슷하지만, `{ }`에 쌓여있다는 것이 다르다.

한 줄

```html
<div>
  {/* Single-line comments(In vanilla JavaScript, the single-line comments are
  represented by double slash(//)) */} {`Welcome ${user}, let's play React`}
</div>
```

여러 줄

```html
<div>
  {/* Multi-line comments for more than one line */} {`Welcome ${user}, let's
  play React`}
</div>
```

### What is the purpose of using super constructor with props argument?

자식 클래스 생성자는 `super()`메소드가 호출되기 전까지 `this` 레퍼런스를 쓸 수 없다. 이와 동일한것이 es6의 서브 클래스에 구현되어 있다. `super()` 메소드에 props를 파라미터로 호출하는 주요 이유는 `this.props`를 자식 생성자에서 쓰기 위해서다.

props 넘기는 경우

```javascript
class MyComponent extends React.Component {
  constructor(props) {
    super(props);

    console.log(this.props); // prints { name: 'John', age: 42 }
  }
}
```

props 안 넘기는 경우

```javascript
class MyComponent extends React.Component {
  constructor(props) {
    super();

    console.log(this.props); // prints undefined

    // but props parameter is still available
    console.log(props); // prints { name: 'John', age: 42 }
  }

  render() {
    // no difference outside constructor
    console.log(this.props); // prints { name: 'John', age: 42 }
  }
}
```

### What is reconciliation?

컴포넌트의 props나 state에 변경이 있을때, React는 이전에 렌더링 된 element와 새롭게 렌더링된 것을 비교하여 실제 DOM이 업데이트 되어야 할지를 결정한다. 똑같지 않을때, React는 DOM을 업데이트 한다. 이 과정을 `reconciliation`이라고 한다.

### How to set state with a dynamic key name?

JSX코드 내에서 es6또는 바벨 트랜스파일러를 쓰고 있다면, computed property 명을 쓸 수 있다.

```javascript
handleInputChange(event) {
  this.setState({ [event.target.id]: event.target.value })
}
```

### What would be the common mistake of function being called every time the component renders?

함수를 파라미터로 넘기는 과정에서 함수가 호출되지 않는지 확인해야 한다.

### Is lazy function supports named exports?

아니다. 현재 `React.lazy`함수는 default export만 지원한다. named exports된 모듈을 import 하고 싶을 경우에는, 사이에 디폴트로 reexports 하는 모듈을 만들수 있다. 이는 트리쉐이킹을 도와주고, 사용하지 않는 컴포넌트를 pull하지 않을 수 있다. 밑에서 예를 살펴보자.

```javascript
// MoreComponents.js
export const SomeComponent = /* ... */;
export const UnusedComponent = /* ... */;
```

이 컴포넌트 중간에 `IntermediateComponent.js`를 만들어서 다시 export 한다.

```javascript
// IntermediateComponent.js
export { SomeComponent as default } from "./MoreComponents.js";
```

그리고 lazy 함수를 이용해서 아래와 같이 임포트 할 수 있다.

```javascript
import React, { lazy } from "react";
const SomeComponent = lazy(() => import("./IntermediateComponent.js"));
```

### Why React uses `className` over `class` attribute?

`class`는 자바스크립트의 예약어 이고, JSX는 javascript를 확장해 만든 것이다. 따라서 `class`를 쓰면 충돌이 일어나기 자바스크립트 예약어와 충동리 발생하기 때문에 `className`을 사용한다. `className` prop에 `string`을 넘겨 주면 된다.

```javascript
render() {
  return <span className={'menu navigation-menu'}>{'Menu'}</span>
}
```

### What are fragments?

React에서는 하나의 컴포넌트가 여러개의 elements를 리턴하는 것이 일반적인 패턴이다. Fragments는 추가로 DOM 노드를 사용하지 않더라도 여러개의 노드들을 묶을 수 있게 해준다.

```javascript
render() {
  return (
    <React.Fragment>
      <ChildA />
      <ChildB />
      <ChildC />
    </React.Fragment>
  )
}
```

```javascript
render() {
  return (
    <>
      <ChildA />
      <ChildB />
      <ChildC />
    </>
  )
}
```

### Why fragments are better than container divs?

1. Fragment는 실제로 추가적인 DOM을 만들지 않기 때문에 더 빠르고 메모리 사용량도 적다. 이는 매우 크고 깊은 트리를 만들 때 상당한 이점으로 작용한다.
2. CSS Grid나 firefox같은 일부 특수한 CSS 메커니즘은 특별한 부모-자식 관계를 가지고 있는데, div를 중간에 추가하는 것은 원하는 레이아웃을 그리기 어렵게 한다.
3. DOM Inspector를 사용할 때 덜 혼잡스럽다.

### What are portals in React?

portals 은 상위 Component 의 DOM 계층 구조 외부에 존재하는 DOM 노드로, 자식을 render 하는데 권장되는 방법이다.

```javascript
ReactDOM.createPortal(child, container);
```

첫번째 인자는 React Child에서만 렌더링이 가능하며, 여기에는 element, string, fragment 가 포함된다. 두번째 인자는 DOM 엘리먼트다.

### What are stateless components?

컴포넌트의 동작이 state와 독립되어 있다면, 이는 stateless 컴포넌트다. 함수나 클래스를 이용해서 stateless 컴포넌트를 만들 수 있다. 하지만 컴포넌트의 라이프 사이클 훅이 필요하지 않다면, 함수형으로 가는 것이 좋다. 함수형 컴포넌트를 선택한다면 많은 이점을 가져갈 수 있다. 코드 사용 및 이해가 쉽고, 조금더 빠르며, 그리고 `this` 키워드의 충돌을 막을 수 있다.

### What are stateful components?

state의 사용에 종속적인 컴포넌트를 stateful component라고 한다. 이 컴포넌트는 항상 class 컴포넌트로 만들어 져야 하며, `constructor`를 통해서 초기화 되어야 한다.

```javascript
class App extends Component {
  constructor(props) {
    super(props);
    this.state = { count: 0 };
  }

  render() {
    // ...
  }
}
```

### How to apply validation on props in React?

React가 development로 실행한다면, 자동으로 컴포넌트에 있는 props의 타입을 올바르게 체크해 준다. 만약 타입이 올바르지 않다면, React는 콘솔에 경고 메시지를 띄운다. 성능 상의 이슈를 위해 production에서는 이 기능이 꺼져 있다. 필수적인 prop은 `isRequired`다. 사용할 수 있는 prop type의 종류는 아래와 같다.

1. `PropTypes.number`
2. `PropTypes.string`
3. `PropTypes.array`
4. `PropTypes.object`
5. `PropTypes.func`
6. `PropTypes.node`
7. `PropTypes.element`
8. `PropTypes.bool`
9. `PropTypes.symbol`
10. `PropTypes.any`

아래와 같이 쓸수 있다.

```javascript
import React from "react";
import PropTypes from "prop-types";

class User extends React.Component {
  static propTypes = {
    name: PropTypes.string.isRequired,
    age: PropTypes.number.isRequired
  };

  render() {
    return (
      <>
        <h1>{`Welcome, ${this.props.name}`}</h1>
        <h2>{`Age, ${this.props.age}`}</h2>
      </>
    );
  }
}
```

주의: 리액트 v15.5부터 PropType이 `React.PropTypes`에서 `prop-types`로 이동했다.

### What are the advantages of React?

1. Virtual DOM으로 어플리케이션의 성능을 향상시킬 수 있음
2. JSX를 통해 코들르 쉽게 읽고 쓸수 있음
3. 클라이언트와 서버사이드 양쪽에서 렌더링 라능
4. 뷰만 다루는 라이브러리이기 때문에, 다른 프레임워크 (Angular, Backbone) 등과 쉽게 연동 가능
5. Jest와 같은 툴로 쉽게 유닛/인티그레이션 테스트 가능

### What are the limitations of React?

1. 풀 프레임워크가 아니라, view만 다루고 있음.
2. 뉴비 웹 개발자들에게 러닝 커브가 존재
3. 전통적인 MVC 프레임워크와 인터그레이팅을 하기 위해서는 추가적인 설정이 필요
4. inline 템플릿과 JSX로 인해 코드의 복잡성 증가
5. 오버엔지니어링/보일러플레이팅을 야기하는 작은 단위의 컴포넌트가 너무 많이 존재

### What are error boundaries in React v16?

Error boundaries란 하위 component tree 에서 자바스크립트 에러 를 catch 하고, 기록하고, 에러가 발생한 component tree가 아닌 대체 UI를 표현해 주는 component를 말한다.

새롭게 추가된 라이프사이클 메서드인 `componentDidCatch(error, info)`나 `static getDerivedStateFromError()`를 사용한다면, 클래스 컴포넌트는 error boundary가 될 수 있다.

```javascript
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  componentDidCatch(error, info) {
    // 에러 리포틍 서비스를 위해 로그를 기록할 수도 있고
    logErrorToMyService(error, info);
  }

  static getDerivedStateFromError(error) {
    // fallback UI를 표현하기 위해여 state를 업데이트 할 수도 있다.
    return { hasError: true };
  }

  render() {
    if (this.state.hasError) {
      // custom Fallback UI를 그릴 수 있다.
      return <h1>{"Something went wrong."}</h1>;
    }
    return this.props.children;
  }
}
```

그리고 이 컴포넌트는 아래와 같이 사용할 수 있다.

```html
<ErrorBoundary>
  <MyWidget />
</ErrorBoundary>
```

### How error boundaries handled in React v15?

`unstable_handleError` 메서드를 활용한 기본적인 error boundaries만 제공하고 있다. 그리고 v16에서 `componentDidCatch`로 변경되었다.

### What are the recommended ways for static type checking?

보통 `PropTypes`를 많이 사용한다. 그러나 크기가 큰 어플리케이션의 경우에는, Flow나 타입스크립트같은, 컴파일 단계에서 타입체킹을 제공하고 자동완성을 지원해주는 정적 타입 체커를 사용하는 것이 좋다.

### What is the use of `react-dom` package?

`react-dom`은 앱 최 상단 레벨에서 사용되는, DOM을 다루는데 필요한 메서드를 제공한다. 대부분의 컴포넌트는 이 모듈을 필요로 하지 않는다. 여기에 있는 메소드를 몇가지 나열하면

1. `render()`
2. `hydrate()`
3. `unmountComponentAtNode()`
4. `findDOMNode()`
5. `createPortal()`

### What is the purpose of render method of `react-dom`?

render 메서드는 제공된 컨테이너의 DOM에 있는 React element를 render 하고 Component에 대한 참조를 반환하는데 사용된다. React element가 이전에 렌더링 되었다면 update 를 수행하고 최근의 변경사항을 반영하기 위해 필요에 따라 DOM을 변경하기도 한다.

```javascript
ReactDOM.render(element, container[, callback])
```

옵셔널 콜백이 있다면, 컴포넌트가 렌더링/업데이트 된 이후로 실행된다.

### What is ReactDOMServer?

`ReactDOMServer`는 컴포넌트를 정적 마크업으로 렌더링할 수 있게 해준다. (보통 노드 서버에서 많이 사용 된다) 이 오브젝트는 서버사이드 렌더링을 할 때 사용된다. 아래 메서드들은 서버와 브라우저 환경 모두에서 사용할 수 있다.

1. `renderToString()`
2. `renderToStaticMarkup()`

예를 들어, 노드 베이스 웹서버인 Express, Hapi, Koa 등에서 서버를 실행한다면, `renderToString`메서드를 호출하여 이에 대한 응답으로 루트 컴포넌트를 string으로 렌더링할 수 있다.

```jsx
// using Express
import { renderToString } from "react-dom/server";
import MyPage from "./MyPage";

app.get("/", (req, res) => {
  res.write("<!DOCTYPE html><html><head><title>My Page</title></head><body>");
  res.write('<div id="content">');
  res.write(renderToString(<MyPage />));
  res.write("</div></body></html>");
  res.end();
});
```

### How to use innerHTML in React?

browser DOM에서 `innerHTML`대신 `dangerouslySetInnerHTML`를 사용할 수 있다. `innerHTML`과 마찬가지로, 이 속성 또한 크로스 사이트 스크립팅 공격 (XSS)에 취약하다. `__html`을 키로 하고 HTML text를 값으로 가지는 object를 리턴하면 된다.

```javascript
function createMarkup() {
  return { __html: "First &middot; Second" };
}

function MyComponent() {
  return <div dangerouslySetInnerHTML={createMarkup()} />;
}
```

### How to use styles in React?

style 속성은 css 문자열 대신 camelCased속성이 있는 자바스크립트 오브젝트를 허용한다. 이는 DOM 스타일 자바스크립트 속성과 일치하며, 효율적이고, XSS 보안 허점을 막아준다.

### How events are different in React?

React 엘리먼트에서 이벤트를 다루는 것은 문법상 약간의 차이가 있다.

1. 리액트 이벤트 핸들러는 lowerCase가 아닌 camelCase로 써야한다.
2. JSX에서는 문자열이 아닌, 함수 이벤트 핸들러를 파라미터로 보낸다.

### What will happen if you use `setState()` in constructor?

`setState()`를 사용하면, 객체 상태가 할당되고, 자식을 포함한 모든 컴포넌트가 다시 렌더링된다. 그리고 아래와 같은 에러메시지가 나타난다. **Can only update a mounted or mounting component.** 따라서 `this.state`를 사용하여 생성자내에서 변수를 초기화 해야 한다.

### What is the impact of indexes as keys?

키는 리액트에서 엘리먼트를 추적할 수 있도록 안정적이어야 하고, 예측가능해야 하고, 유니크해야 한다.

아래 코드에서 각 엘리먼트의 키는 데이터를 따르는 것이 아니라 단순히 순서에 따라 결정된다. 이는 React가 하는 최적화를 제한한다.

```jsx
{
  todos.map((todo, index) => <Todo {...todo} key={index} />);
}
```

만약 데이터를 유니크 키로 사용한다면 위의 조건을 만족하기 때문에, React는 다시 연산할 필요 없이 재정렬할 수 있다.

```jsx
{
  todos.map(todo => <Todo {...todo} key={todo.id} />);
}
```

### Is it good to use `setState()` in `componentWillMount()` method?

`componentWillMount()`에서 비동기 초기화를 하는 것은 피하도록 권장한다. `componentWillMount()`는 마운팅이 일어나기 직전에 바로 실행된다. 이는 `render()`함수가 불리우기 직전이며, 따라서 여기에서 state를 새로 값을 할당 한다 하더라도 리렌더링을 트리거 하지 않는다. 이 메소드 내에서는 사이드 이펙트나 subscription등은 피해야 한다. 따라서 비동기 초기화는 `componentDidMount()`에서 하는 것이 좋다.

```jsx
componentDidMount() {
  axios.get(`api/todos`)
    .then((result) => {
      this.setState({
        messages: [...result.data]
      })
    })
}
```

### What will happen if you use props in initial state?

컴포넌트의 새로고칩 없이 props가 변경된다면, 현재 상태의 컴포넌트는 절대로 업데이트 하지 않기 때문에 새로운 prop값이 화면에 표시되지 않을 것이다. props를 통한 state값의 초기화는 컴포넌트가 딱 초기화 되었을 때만 실행된다.

```jsx
class MyComponent extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      records: [],
      inputValue: this.props.inputValue
    };
  }

  render() {
    return <div>{this.state.inputValue}</div>;
  }
}
```

props를 render 함수 내에서 쓰면 값을 업데이트 한다.

```jsx
class MyComponent extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      record: []
    };
  }

  render() {
    return <div>{this.props.inputValue}</div>;
  }
}
```

### How do you conditionally render components?

때로는 어떤 상태값에 따라서 렌더링을 다르게 해야하는 경우가 발생한다. JSX는 `false`나 `undefined`는 렌더링하지 않으므로, 특정 조건에 true를 주는 형식으로 조건부 렌더링을 할 수 있다.

```jsx
const MyComponent = ({ name, address }) => (
  <div>
    <h2>{name}</h2>
    {address && <p>{address}</p>}
  </div>
);
```

if-else도 삼항연산자를 활용하면 아래와 같이 할 수 있다.

```jsx
const MyComponent = ({ name, address }) => (
  <div>
    <h2>{name}</h2>
    {address ? <p>{address}</p> : <p>{"Address is not available"}</p>}
  </div>
);
```

### Why we need to be careful when spreading props on DOM elements?

spread prop를 쓴다면, HTML에 알수없는 속성을 추가할 수 있는 위험이 있기 때문에 좋지 못하다. 대신 `...rest` 연산자를 쓴다면, 필요한 props만 추가해서 넣을 수 있다.

```javascript
const ComponentA = () => (
  <ComponentB isDisplay={true} className={"componentStyle"} />
);

const ComponentB = ({ isDisplay, ...domProps }) => (
  <div {...domProps}>{"ComponentB"}</div>
);
```

### How you use decorators in React?

클래스 컴포넌트에 데코레이터를 쓸 수 있으며, 이는 함수에 컴포넌트를 넘기는 것과 동일하다. 데코레이터는 유연하고 읽기 쉬운 방법으로 컴포넌트를 기능적으로 수정할 수 있도록 한다.

```javascript
@setTitle("Profile")
class Profile extends React.Component {
  //....
}
const setTitle = title => WrappedComponent => {
  return class extends React.Component {
    componentDidMount() {
      document.title = title;
    }

    render() {
      return <WrappedComponent {...this.props} />;
    }
  };
};
```

주의: 데코레이터는 es7 문법에 포함되지 못하고 현재 stage2 단계에 있다.

### How do you memoize a component?

함수형 컴포넌트를 기반으로한 메모이제이션이 가능한 라이브러리가 있다. 예를 들어, `moize`라이브러리를 활용하면, 다른 컴포넌트 내에서 컴포넌트를 메모이제이션 할 수 있다.

```javascript
import moize from "moize";
import Component from "./components/Component"; // this module exports a non-memoized component

const MemoizedFoo = moize.react(Component);

const Consumer = () => {
  <div>
    {"I will memoize the following entry:"}
    <MemoizedFoo />
  </div>;
};
```

### How you implement Server Side Rendering or SSR?

React는 이미 노드 서버에서 렌더링을 다룰 수 있도록 지원되고 있다. 클라이언트 사이드와 동일하게 렌더링할 수 있는 특수한 버전의 DOM renderer가 제공되고 있다.

```javascript
import ReactDOMServer from "react-dom/server";
import App from "./App";

ReactDOMServer.renderToString(<App />);
```

이 메소드는 일반적인 HTML을 string으로 내보내며, 이는 서버의 응답 일부를 페이지 본문 내부에 위치시킬 수 있다. 클라이언트 사이드에서, 리액트는 미리 렌더링된 컨텐츠를 감지하고 나머지를 원활하게 렌더링할 수 있다.

### How to enable production mode in React?

Webpack의 `DefinePlugin` 메서드를 활용하여, `NODE_ENV`를 `production`으로 설정해야 propType의 유효성 검사 같은 추가적인 경고를 제거할 수 있다.

production 모드와 별도로, 주석을 제거하고 코드르 압축시키는 uglify의 dead-code 코드를 사용하여 minify하면 번들링 사이즈를 줄일 수 있다.

### What is CRA and its benefits?

CRA(`create-react-app`)는 특별한 설정없이도 빠르고 간편하게 리액트 어플리케이션을 만들수 있도록 해주는 Cli tool이다.

```
# Installation
$ npm install -g create-react-app

# Create new project
$ create-react-app todo-app
$ cd todo-app

# Build, test and run
$ npm run build
$ npm run test
$ npm start`
```

여기에는 리액트 앱을 만드는데 필요한 모든 것이 담겨져 있다.

1. React, JSX, ES6, 문법 지원을 위한 Flow
2. spread operator와 같은 es6 문법
3. auto prefixed css를 통해, -web-kit` 과 같은 접두어를 붙이지 않아도 됨
4. 빠른 인터렉티브 유닛 테스트 러너와 함께 커버리지 리포팅
5. 일반적인 실수에 대해 경고하는 라이브 dev 서버
6. 배포를 위해 소스맵, 해쉬와 함께 제공되는 JS, CSS, 이미지 번들링 해주는 빌드 스크립트

### What is the lifecycle methods order in mounting?

컴포넌트가 생성되고, DOM에 들어가는 과정에서 아래와 같은 라이프 사이클 메서드가 순서대로 호출된다.

1. `constructor()`
2. `static getDerivedStateFromProps()`
3. `render()`
4. `componentDidMount()`

### What are the lifecycle methods going to be deprecated in React v16?

다음 lifecycle메서드는 안전하지 않은 코딩법이 될 수 있고, 비동기 렌더링시 문제가 발생할 수 있다.

1. `componentWillMount()`
2. `componentWillReceiveProps()`
3. `componentWillUpdate()`

v16.3 부터 `UNSAFE_` prefix가 붙고, v17에서는 삭제된다.

### What is the purpose of `getDerivedStateFromProps()` lifecycle method?

새로운 라이프 사이클 메서드 `getDerivedStateFromProps()`는 component가 인스턴스화 된 후, 다시 렌더링 되기전에 호출된다. object를 반환하여 state를 업데이트 하거나, null을 리턴하ㅕㅇ 새로운 props에서 state update가 필요하지 않도록 나타낼 수도 있다.

```javascript
class MyComponent extends React.Component {
  static getDerivedStateFromProps(props, state) {
    // ...
  }
}
```

이 메서드는 `componentDidUpdate()`와 함께 쓴다면, `componentWillReceiveProps()`의 모든 유즈케이스에 적용할 수 있다.

### What is the purpose of `getSnapshotBeforeUpdate()` lifecycle method?

새로운 메서드 `getSnapshotBeforeUpdate()`는 DOM 업데이트 직전에 호출된다. 이 메서드의 반환값은 `componentDidUpdate()`의 세번째 파라미터로 전달된다.

```javascript
class MyComponent extends React.Component {
  getSnapshotBeforeUpdate(prevProps, prevState) {
    // ...
  }
}
```

이 메서드는 `componentDidUpdate()`와 함께 쓴다면, `componentWillUpdate()`의 모든 유즈케이스에 적용할 수 있다.

### Do Hooks replace render props and higher order components?

render props와 HOC 모두 한개의 자식만 렌더링 하지만, 대부분의 경우 Hooks API를 아용하면 트리에 의존성을 줄이면서 간단하게 구현할 수 있다.

### What is the recommended way for naming components?

`displayName`을 쓰는 것 보다 컴포넌트에 레퍼런스를 주는 방법이 더 좋다.

`displayName`을 쓰는 법 보다

```javascript
export default React.createClass({
  displayName: "TodoApp"
  // ...
});
```

이렇게 하는게 더 좋다.

```javascript
export default class TodoApp extends React.Component {
  // ...
}
```

### What is the recommended ordering of methods in component class?

마운팅에서 렌더링까지 아래와 같은 순서로 나열하길 권장한다.

1. `static` 메서드
2. `constructor()`
3. `getChildContext()`
4. `componentWillMount()`
5. `componentDidMount()`
6. `componentWillReceiveProps()`
7. `shouldComponentUpdate()`
8. `componentWillUpdate()`
9. `componentDidUpdate()`
10. `componentWillUnmount()`
11. 클릭 또는 이벤트 핸들러 `onClickSubmit()` `onChangeDescription()`
12. 렌더를 위한 `getter` 메서드 `getSelectReason()` `getFooterContent()`
13. 옵셔널 렌더 메서드 `renderNavigation()` `renderProfilePicture()`
14. `render()`

### What is a switching component?

스위칭 컴포넌트란 하나 이상의 컴포넌트를 렌더링하는 컴포넌트를 의미한다. prop을 map으로 받아서 해당하는 컴포넌트를 보여주면 된다.

아래 코드 참조.

```javascript
import HomePage from "./HomePage";
import AboutPage from "./AboutPage";
import ServicesPage from "./ServicesPage";
import ContactPage from "./ContactPage";

const PAGES = {
  home: HomePage,
  about: AboutPage,
  services: ServicesPage,
  contact: ContactPage
};

const Page = props => {
  const Handler = PAGES[props.page] || ContactPage;

  return <Handler {...props} />;
};

Page.propTypes = {
  page: PropTypes.oneOf(Object.keys(PAGES)).isRequired
};
```

### Why we need to pass a function to setState()?

그 이유는 `setState()`가 비동기로 작동하는데에 있다. React는 성능상의 문제로 인해, state의 변경작업을 배치로 하는데, 이 때문에 `setState()`를 바로 호출한다고 해서 바로 반영되지 않는다. 이 말은, `setState()`를 호출 할 때 그 당시 `state`의 값에 의존하면 안된다는 뜻이다. 따라서 `setState()`에는 이전 값에 접근할 수 있는 함수를 사용하는 것이 좋다. 이는 사용자가 비동기로 작동하는 `setState()`의 특징으로 인해 이전 값에 접근하는 것을 방지해 준다.

초기 값이 0 이라고 가정하자. 여기 1 씩 올리는 동작을 하는 코드가 세개 있다.

```javascript
// assuming this.state.count === 0
this.setState({ count: this.state.count + 1 });
this.setState({ count: this.state.count + 1 });
this.setState({ count: this.state.count + 1 });
// this.state.count === 1, not 3
```

만약 `setState()`에 함수를 넘겨준다면, 올바르게 동작할 것이다.

```javascript
this.setState((prevState, props) => ({
  count: prevState.count + props.increment
}));
// this.state.count === 3 as expected
```

### What is strict mode in React?

`React.StrictMode`는 어플리케이션의 잠재적인 문제를 하이라이팅 해주는 유용한 컴포넌트다. `<Fragment>`와 마찬가지로, `<StrictMode>`는 추가적으로 DOM을 렌더링하지 않는다. 이는 단지 자식 컴포넌트의 추가적인 체크와 경고를 할 뿐이다. 그리고 이러한 체크는 development 에서만 가능하다.

```javascript
import React from "react";

function ExampleApplication() {
  return (
    <div>
      <Header />
      <React.StrictMode>
        <div>
          <ComponentOne />
          <ComponentTwo />
        </div>
      </React.StrictMode>
      <Footer />
    </div>
  );
}
```

위 예에서, `ComponentOne` `ComponentTwo`만 체크할 것이다.

### What are React Mixins?

`Mixins`은 공통적인 기능을 가질 수 있도록 컴포넌트를 분리하는 방법이다. 그러나 사용하지 말아야 한다. HOC 나 데레이터를 사용하면 된다.

가장 유명한 사용법중 하나는 `PureRenderMixin`이다. 이전 props또는 state와 얕은 비교를 했을 때 일치하는 경우, 리렌더링을 막아주는 역할을 한다.

```javascript
const PureRenderMixin = require("react-addons-pure-render-mixin");

const Button = React.createClass({
  mixins: [PureRenderMixin]
  // ...
});
```

### Why is `isMounted()` an anti-pattern and what is the proper solution?

`isMounted()`의 일반적인 사용사례는 컴포넌트가 언마운트 된 후에 `setState()`를 호출하는 것을 방지하기 위함이다.

```javascript
if (this.isMounted()) {
  this.setState({...})
}
```

`setState()`를 호출하기 전에 `isMounted()`를 검사하면 경고를 없앨수있지만, 경고의 목적을 잃어버리는 꼴이 된다. 컴포넌트의 마운트가 해제된 후에 reference를 가지고 있다고 판단하므로 이는 일종의 코드 스멜이라고 볼 수 있다.

좋은 해결책은 컴포넌트의 마운트가 해제된 후 `setState()`가 호출될 수 있는 위치를 찾아 수정하는 것이다. 이러한 상황은 대게 컴포넌트가 데이터를 기다리고 있다가 데이터의 도착전 마운트가 해제 되는, 콜백 상황에서 많이 발생된다. 콜백은 마운트가 해제되기 전에 `componentWillUnMount`에서 취소되어야 한다.

### What are the Pointer Events supported in React?

포인터 이벤트는 모든 입력 이벤트를 다루는 통일된 방법을 제공한다. 과거에는 마우스 및 각각의 이벤트 리스너를 달았지만, 요즘에는 핸드폰 터치, 서피스, 펜 등 마우스 외에 다양한 입력기기가 나타나기 시작했다. 한가지 명심해야 할 점은 이러한 이벤트들이 포인트 이벤트 명세를 지원하는 브라우저에서만 동작할 것이라는 점이다.

아래의 이벤트 타입들이 React DOM에서 지원하는 것이다.

1. onPointerDown
2. onPointerMove
3. onPointerUp
4. onPointerCancel
5. onGotPointerCapture
6. onLostPointerCaptur
7. onPointerEnter
8. onPointerLeave
9. onPointerOver
10. onPointerOut

### Why should component names start with capital letter?

JSX를 이용해서 렌더링을 하다보면, 컴포넌트의 명이 대문자가 아닐 경우 태그 인식에 실패했다는 에러메시지를 뱉는다. 그 이유는 오직 HTML 엘리먼트와 SGV 태그만이 소문자로 시작하기 때문이다.

```javascript
class SomeComponent extends Component {
 // Code goes here
}`
```

클래스 명을 소문자로 시작하게 컴포넌트를 만들 수 있지만, import 할 때는 대문자로 하면 된다.

```javascript
class myComponent extends Component {
  render() {
    return <div />;
  }
}

export default myComponent;
```

```javascript
import MyComponent from "./MyComponent";
```

### Are custom DOM attributes supported in React v16?

가능하다. 과거 React는 알수없는 DOM 속성을 무시했다. JSX에 리액트가 알수 없는 속성을 넣었다면, 리액트는 이를 무시했다.

예를 들어, 과거에는 아래와 같이 동작했다.

```javascript
<div mycustomattribute={"something"} />
```

```html
<div />
```

그러나 React v16부터는 알수없는 속성도 결국 DOM에 반영된다.

```html
<div mycustomattribute="something" />
```

이는 브라우저에 특화된 비표준 속성, 새로운 DOM api, 서드파티 라이브러리 등을 사용할 때 유용하다.

### What is the difference between constructor and getInitialState?

es6 클래스에서는 `constructor`로 state를 초기화 하고, `React.createClass`를 사용할 때는 `getInitialState()`으로 초기화 한다.

es6

```javascript
class MyComponent extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      /* initial state */
    };
  }
}
```

`React.createClass()`

```javascript
const MyComponent = React.createClass({
  getInitialState() {
    return {
      /* initial state */
    };
  }
});
```

### Can you force a component to re-render without calling setState?

기본적으로, state나 prop의 변화가 있을 때만 컴포넌트가 리렌더링 된다. 만약 `render()` 메서드가 외부의 다른 데이터에 의존적이라면, `forceUpdate()`를 통해서 컴포넌트를 리렌더링 할 수 있다.

```javascript
component.forceUpdate(callback);
```

다만 이러한 방법은 권장되지 않으며, `render()`메소드에서 `this.props`나 `this.state`를 참조하는 것이 권장된다.

### What is the difference between `super()` and `super(props)` in React using ES6 classes?

`constructor()`에서 `this.props`에 접근하고 싶다면, `super()`메서드에 `this.props`를 넘겨야 한다.

```javascript
class MyComponent extends React.Component {
  constructor(props) {
    super(props);
    console.log(this.props); // { name: 'John', ... }
  }
}
```

```javascript
class MyComponent extends React.Component {
  constructor(props) {
    super();
    console.log(this.props); // undefined
  }
}
```

### How to loop inside JSX?

`Array.prototype.map`을 es6의 화살표 함수 문법과 사용하면 된다.

```html
<tbody>
  {items.map(item =>
  <SomeComponent key="{item.id}" name="{item.name}" />)}
</tbody>
```

`for`루프는 사용할 수 없다.

```html
<tbody>
  for (let i = 0; i < items.length; i++) {
  <SomeComponent key="{items[i].id}" name="{items[i].name}" />
  }
</tbody>
```

JSX 태그는 함수호출로 트랜스파일이 되는데, 이 경우 표현식내에 제어문을 사용할 수 없다. 다만 이는 stage1에 있는 [do](https://github.com/tc39/proposal-do-expressions) proposal로 해결 될 수도 있다.

### How do you access props in attribute quotes?

React와 JSX는 속성 값에 string interpolation을 지원하지 않는다. 따라서 아래 코드는 작동하지 않는다.

```html
<img className="image" src="images/{this.props.image}" />
```

하지만 `{}`와 함께 javascript 표현식을 넣으면 가능하다.

```html
<img className='image' src={'images/' + this.props.image} /> <img
className='image' src={`images/${this.props.image}`} />
```

### What is React proptype array with shape?

만약 특정 object를 가진 array를 넘기고 싶다면, `React.PropTypes.arrayOf()`와 함께 `React.PropTypes.shape()`를 쓰면 된다.

```javascript
ReactComponent.propTypes = {
  arrayWithShape: React.PropTypes.arrayOf(
    React.PropTypes.shape({
      color: React.PropTypes.string.isRequired,
      fontSize: React.PropTypes.number.isRequired
    })
  ).isRequired
};
```

### How to conditionally apply class attributes?

따옴표 안에 내용은 모두 string으로 인식하기 때문에 `{}`를 쓸 수 없다.

```html
<div className="btn-panel {this.props.visible ? 'show' : 'hidden'}"></div>
```

다만 `{}`안에 모든 식을 넣으면 가능하다. (공백은 반드시 있어야 한다)

```html
<div className={'btn-panel ' + (this.props.visible ? 'show' : 'hidden')}>
```

템플릿 string도 가능하다

```html
<div className={`btn-panel ${this.props.visible ? 'show' : 'hidden'}`}>
```

### What is the difference between React and ReactDOM?

React 패키지내에는 엘리먼트와 컴포넌트 클래스에 도움을 줄 수 있는 `React.createElement()` `React.Component` `React.children`등을 가지고 있다. React 패키지 내에는 컴포넌트를 만드는데 도움이 되는 이러한 요소들이 있다고 보면 된다. 반면 `React-dom`패키지는 `ReactDOM.render()` 서버사이드 렌더링에 필요한 `react-dom/server`에 속한 `ReactDOMServer.renderToString()` `ReactDOMServer.renderToStaticMarkUp()` 이 있다.

### Why ReactDOM is separated from React?

React 팀은 DOM조작과 관련된 모든 기능을 `ReactDOM` 라이브러리로 옮겼다. 이는 React v0.14에서 처음으로 분리되었다. 이 때 패키지를 보자면, `react-native` `react-art` `react-canvas` `react-three`등 패키지 분리가 깔끔해졌으며, `React`패키지 자체에는 브라우저 DOM 조작과 관련된 라이브러리가 없다는 것이 명확해졌다. React가 다수의 환경에서 렌더링을 지원하기 위해, React팀은 React와 React-dom을 분리할 계획을 수립햇다. 이러한 방법론은 웹 버전에서 쓰이는 React와 React-Native사이에 컴포넌트를 쓰는 방법론을 공유할 수 있도록 해준다.

### How to use React label element?

표준 `for` 속성을 사용하는 `text input`에 바인드된 `<label>`을 사용하려고 하면, 속성이 없는 HTML이 생성되고 콘솔에 경고가 출력된다.

```html
<label for={'user'}>{'User'}</label>
<input type={'text'} id={'user'} />
```

for는 자바스크립트의 예약어이므로, `htmlFor`를 사용해야 한다.

```html
<label htmlFor={'user'}>{'User'}</label>
<input type={'text'} id={'user'} />
```

### How to combine multiple inline style objects?

spread 연산자를 사용하면 된다.

```html
 <button style={{...styles.panel.button, ...styles.panel.submitButton}}>{'Submit'}</button>
```

React Native라면 array를 사용하면 된다.

```html
<button style={[styles.panel.button, styles.panel.submitButton]}>{'Submit'}</button>
```

### How to re-render the view when the browser is resized?

`componentDidMount()`에서 `resize`이벤트를 걸어두고, width와 height를 업데이트 하면 된다. 그리고 이 이벤트는 `componentWillUnmount()`에서 제거해줘야 한다.

```javascript
class WindowDimensions extends React.Component {
  constructor(props){
    super(props);
    this.updateDimensions = this.updateDimensions.bind(this);
  }
   
  componentWillMount() {
    this.updateDimensions()
  }

  componentDidMount() {
    window.addEventListener('resize', this.updateDimensions)
  }

  componentWillUnmount() {
    window.removeEventListener('resize', this.updateDimensions)
  }

  updateDimensions() {
    this.setState({width: window.innerWidth, height: window.innerHeight})
  }

  render() {
    return <span>{this.state.width} x {this.state.height}</span>
  }
}
```

### What is the difference between `setState()` and `replaceState()` methods?

`setState()`는 과거의 state값을 현재 값으로 합친다. 반면 `replaceState()`는 현재 state를 버리고 넘어오는 `state`로 바꾼다. 이전 key를 모두 제거하는 경우가 아니라면 보통 `useState()`를 사용한다. `replaceState()`대신 `setState()`에서 `false/null`을 사용할 수도 있다.

### How to listen to state changes?

아래 라이프사이클 메서드는 state의 변화가 있을 때 호출된다. 여기에서 이전 state와 props과 현재 state/props 값을 비교하여 의미있는 변화가 있었는지 추적할 수 있다.

```javascript
componentWillUpdate(object nextProps, object nextState)
componentDidUpdate(object prevProps, object prevState)
```

### What is the recommended approach of removing an array element in React state?

`Array.prototype.filter()`메서드가 올바른 방법이다.

```javascript
removeItem(index) {
  this.setState({
    data: this.state.data.filter((item, i) => i !== index)
  })
}
```

### Is it possible to use React without rendering HTML?

16.2 이상의 버전에서는 가능하다.

```javascript
render() {
  return false
}
```

```javascript
render() {
  return null
}
```

```javascript
render() {
  return []
}
```

```javascript
render() {
  return <React.Fragment></React.Fragment>
}
```

```javascript
render() {
  return <></>
}
```

`undefined`의 경우에는 작동하지 않는다.

### How to pretty print JSON with React?

`<pre>` 태그안에 `JSON.stringify()`를 사용하면 된다.

```javascript
const data = { name: 'John', age: 42 }

class User extends React.Component {
  render() {
    return (
      <pre>
        {JSON.stringify(data, null, 2)}
      </pre>
    )
  }
}

React.render(<User />, document.getElementById('container'))
```

### Why you can't update props in React?

props은 불변이며, 하향식으로 전달되는 것이 `React`의 철학이다. 이 말인 즉, 부모는 어떤 prop값이든 자식에세 보낼 수 있지만, 자식은 그 prop값을 수정할 수 없다는 것이다.

### How to focus an input element on page load?

`input` 엘리먼트에 ref를 만들고, 이를 `componentDidMount()`에서 쓰면 된다.

```javascript
class App extends React.Component{
  componentDidMount() {
    this.nameInput.focus()
  }

  render() {
    return (
      <div>
        <input
          defaultValue={'Won\'t focus'}
        />
        <input
          ref={(input) => this.nameInput = input}
          defaultValue={'Will focus'}
        />
      </div>
    )
  }
}

ReactDOM.render(<App />, document.getElementById('app'))
```

### What are the possible ways of updating objects in state?

1. state를 병합할 object를 `setState()`에 서 사용하는 법
   - `Object.assign()로 Object의 복사본을 만든다.
```javascript
const user = Object.assign({}, this.state.user, { age: 42 })
this.setState({ user })-
```
   - spread 연산자를 사용하는 법 
```javascript
const user = { ...this.state.user, age: 42 }
this.setState({ user })
```

2. `setState()`와 함수를 사용하는 법
   
```javascript
this.setState(prevState => ({
  user: {
    ...prevState.user,
    age: 42
  }
}))
```

### Why function is preferred over object for `setState()`?

React는 성능의 문제로 인해 여러개의 `setState()`를 배치 형태로 호출하게 된다. 왜냐하면 `this.props`와 `this.state`는 비동기로 업데이트 될 수 있기 때문이다. 다음 state를 계산할 때 이전에 계산된 값을 신뢰하면 안된다.

아래 예제는 제대로 작동하지 않는다.

```javascript
// Wrong
this.setState({
  counter: this.state.counter + this.props.increment,
})
```

이를 위해 함수로 `setState()`를 호출하는 것이 권장된다. 함수로 호출시 이전 state값을 받을 수 있고, 업데이트할 때 사용할 `prop`도 받아올 수 있다.

```javascript
// Correct
this.setState((prevState, props) => ({
  counter: prevState.counter + props.increment
}))
```

### How can we find the version of React at runtime in the browser?

`React.version`을 사용하면 된다.

```javascript
const REACT_VERSION = React.version

ReactDOM.render(
  <div>{`React version: ${REACT_VERSION}`}</div>,
  document.getElementById('app')
)
```

### What are the approaches to include polyfills in your `create-react-app`?

1. `core-js`를 수동으로 임포트하는 법
`polyfills.js`과 같은 파일을 만들고, 이를 루트인 `index.js`에서 임포트 한다. 그리고 `core-js`를 설치하여 필요한 기능을 임포트 한다.
```javascript
import 'core-js/fn/array/find'
import 'core-js/fn/array/includes'
import 'core-js/fn/number/is-nan'
```
2. 폴리필 서비스를 이용하는 방법
`polyfill.io`를 CDN으로 가져와서, `index.html`에 추가하는 방법
```html
<script src='https://cdn.polyfill.io/v2/polyfill.min.js?features=default,Array.prototype.includes'></script>
```
### How to use https instead of http in create-react-app?

환경설정에 `HTTPS=true`를 세팅하면 된다. 

pacakge.json

```json
"scripts": {
  "start": "set HTTPS=true && react-scripts start"
}
```

아니면 `set HTTPS=true && npm start`로 실행하면 된다.

### How to avoid using relative path imports in create-react-app?

루트 디렉토리에 `.env`를 만들고, 임포트 경로를 작성한다.

`NODE_PATH=src/app`

개발서벌르 재시작하면, 상대경로 없이 `src/app`에 있는 파일을 import 할 수 있다.

### How to add Google Analytics for React Router?

history 객체에 리스너를 추가하여 각 페이지 뷰에 달아 둔다.

```javascript
history.listen(function (location) {
  window.ga('set', 'page', location.pathname + location.search)
  window.ga('send', 'pageview', location.pathname + location.search)
})
```

### How to update a component every second?

`setInterval()`에 트리거를 걸어두면 되지만, unmount시에 이를 해제하여 메모리 누수와 에러를 방지해야 한다.

```javascript
componentDidMount() {
  this.interval = setInterval(() => this.setState({ time: Date.now() }), 1000)
}

componentWillUnmount() {
  clearInterval(this.interval)
}
```

### How do you apply vendor prefixes to inline styles in React?

react는 자동으로 vender prefix를 붙여주지 않으므로, 수동으로 붙여야 한다.

```javascript
<div style={{
  transform: 'rotate(90deg)',
  WebkitTransform: 'rotate(90deg)', // note the capital 'W' here
  msTransform: 'rotate(90deg)' // 'ms' is the only lowercase vendor prefix
}} />
```

### How to import and export components using React and ES6?

`default`키워드를 사용하여 컴포넌트를 익스포트 한다.

```javascript
import React from 'react'
import User from 'user'

export default class MyProfile extends React.Component {
  render(){
    return (
      <User type="customer">
        //...
      </User>
    )
  }
}
```

위 예제에서는 MyProfile이 멤버가 되어 모듈로 익스포트 되는데, 이는 다른 컴포넌트에서 굳이 이름을 명세하지 않더라도 임포트 할 수 있게 해준다.

### What are the exceptions on React component naming?

몇가지 예외적인 경우를 제외하고, 컴포넌트 명은 대문자로 시작해야 한다. 소문자와 . (속성 접근자)을 사용하는 경우 유효한 컴포넌트 명이다. 아래의 예가 그러한 유효한 경우다.

```javascript
render(){
   return (
       <obj.component /> // `React.createElement(obj.component)`
      )
}
```

### Why is a component constructor called only once?

React의 reconciliation 알고리즘은 후속 렌더링 과정에서 사용자 정의 컴포넌트가 똒같은 위치에 나타나면, 이전과 동일 한 요소이므로 새로운 인스턴스를 만드는 대신 이전 인스턴스를 재사용한다고 가정한다.

### How to define constants in React?

es7의 static 필드를 사용하여 상수를 정의할 수 있다.

```javascript
class MyComponent extends React.Component {
  static DEFAULT_PAGINATION = 10
}
```

현재 static 필드는 stage3에 있다.

### How to programmatically trigger click event in React?

callback을 통한 ref prop를 사용하여 HTMLInputElement 객체에 대한 참조를 가져와서 class property 로 저장한 다음, 이렇게 저장된 참조를 활용하여 `HTMLElement.click` 메서드를 사용해 이벤트 핸들러에서 클릭 이벤트를 트리거 할 수 있다.

1. render 메서드에서 ref를 생성한다.

```javascript
<input ref={input => this.inputElement = input} />
```

2. 이벤트 핸들러에서 클릭 이벤트를 트리거 한다.
```javascript
this.inputElement.click()
```

### Is it possible to use async/await in plain React?

React 에서 async/await 을 사용하고 싶다면 Babel 과 transform-async-to-generator 플러그인이 필요하다. React Native에서는 기본적으로 지원하고 있다.

### What are the common folder structures for React?

크게 두가지 종류가 있다.

1. 기능 또는 라우팅으로 분류하는 방법

기능과 라우팅에 따라서 css, js, 테스트 코드를 묶는 방법이다.

```
common/
├─ Avatar.js
├─ Avatar.css
├─ APIUtils.js
└─ APIUtils.test.js
feed/
├─ index.js
├─ Feed.js
├─ Feed.css
├─ FeedStory.js
├─ FeedStory.test.js
└─ FeedAPI.js
profile/
├─ index.js
├─ Profile.js
├─ ProfileHeader.js
├─ ProfileHeader.css
└─ ProfileAPI.js
```

2. 파일 타입 별로 분류하는 법
```
api/
├─ APIUtils.js
├─ APIUtils.test.js
├─ ProfileAPI.js
└─ UserAPI.js
components/
├─ Avatar.js
├─ Avatar.css
├─ Feed.js
├─ Feed.css
├─ FeedStory.js
├─ FeedStory.test.js
├─ Profile.js
├─ ProfileHeader.js
└─ ProfileHeader.css
```

### What are the popular packages for animation?

React Transition Group과 React Motion이 React 생태계에서 유명한 애니메이션 패키지다.

### What is the benefit of styles modules?

스타일 값을 하드코딩 하는 것은 권장하지 않는 방식이다. 서로다른 UI 컴포넌트에서 넓게 사용되는 값은 하나의 모듈에서 추출해서 쓰는 것이 좋다.

아래와 같은 방식을 사용하면, 서로다른 컴포넌트에서 동일한 스타일을 가져올 수 있다.

```javascript
export const colors = {
  white,
  black,
  blue
}

export const space = [
  0,
  8,
  16,
  32,
  64
]
```

그리고 각각의 컴포넌트에서 이를 임포트 하면 된다.

```javascript
import { space, colors } from './styles'
```

### What are the popular React-specific linters?

자바스크립트 lint로는 eslint가 유명하다. 코드 스타일을 분석할 수 있는 다양한 플러그인이 있다. React에서 가장 유명한 것은 `eslint-plugin-react`다. 기본적으로 몇가지 베스트 프랙티스를 확인하여, 이 규칙을 바탕으로 iterator의 키에서 부터 prop type까지 확인해 준다. 다른 유명한 플러그인으로는 `eslint-plugin-jsx-a11y`가 있는데, 이는 접근성과 관련된 일반적인 문제를 해결하는데 도움을 준다. JSX는 `alt` `tabindex`와 같은 HTML과 약간 다른 문법을 제공하므로, 일반적인 플러그인으로 는 확인이 어렵다.

### How to make AJAX call and in which component lifecycle methods should I make an AJAX call?

Axios, jQuery Ajax, 브라우저 빌트인 `fetch`등을 활용하여 ajax를 활용할 수 있다. 이렇게 데이터를 가져오는 것은 반드시 `componentDidMount()`내에서 해야 한다. 이는 데이터를 받어온 뒤에 `setState()`로 컴포넌트를 업데이트 할 수 있게 해준다.

예를 들어, 아래 코드에서 employee 목록을 가져오고 state를 업데이트 한다.

```javascript
class MyComponent extends React.Component {
  constructor(props) {
    super(props)
    this.state = {
      employees: [],
      error: null
    }
  }

  componentDidMount() {
    fetch('https://api.example.com/items')
      .then(res => res.json())
      .then(
        (result) => {
          this.setState({
            employees: result.employees
          })
        },
        (error) => {
          this.setState({ error })
        }
      )
  }

  render() {
    const { error, employees } = this.state
    if (error) {
      return <div>Error: {error.message}</div>;
    } else {
      return (
        <ul>
          {employees.map(item => (
            <li key={employee.name}>
              {employee.name}-{employees.experience}
            </li>
          ))}
        </ul>
      )
    }
  }
}
```

### What are render props?

**Render Props**는 값이 함수인 prop을 활용하여 컴포넌트 간에 코드를 share할 수 있게 해주는 방법이다. 아래 컴포넌트는 `render prop`을 활용하여 React element를 리턴한다.

```javascript
<DataProvider render={data => (
  <h1>{`Hello ${data.target}`}</h1>
)}/>
```

React Router 와 DownShift 라이브러리가 이 패턴을 사용한다.

