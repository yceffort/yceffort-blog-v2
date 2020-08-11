---
title: 리액트 고차 컴포넌트 (React Higher Order Component)
tags:
  - javascript
  - react
published: true
date: 2020-07-04 04:06:10
description: "[이 글](https://ko.reactjs.org/docs/higher-order-components.html)이
  한글로 번역이 안되있어서 대충 번역해봅니다. # Higher-Order Components  고차 컴포넌트 (이하 HOC)는 리액트에서
  컴포넌트 로직을 재사용하기 위한 고오급 기술이다. HOC는 리액트 API의 일부분은 아니다. 이는 리액트..."
category: javascript
slug: /2020/07/react-higher-component/
template: post
---
[이 글](https://ko.reactjs.org/docs/higher-order-components.html)이 한글로 번역이 안되있어서 대충 번역해봅니다.

# Higher-Order Components

고차 컴포넌트 (이하 HOC)는 리액트에서 컴포넌트 로직을 재사용하기 위한 고오급 기술이다. HOC는 리액트 API의 일부분은 아니다. 이는 리액트의 컴포넌트 환경에서 자주 나타나는 일종의 패턴이다.

구체적으로, **HOC는 컴포넌트를 받아 새로운 컴포넌트를 반환하는 함수다**

```javascript
const EnhancedComponent = higherOrderComponent(WrappedComponent);
```

컴포넌트의 props가 ui를 바꾼다면, HOC는 컴포넌트를 다른 컴포넌트로 바꿔버린다.

이러한 HOC는 리액트 써드 파티 라이브러리에서 자주사용되는 패턴으로, Redux의 `connect`와 `Relay`의 `createFragmentContainer`에서 볼 수 있다. 

이 문서에서는 왜 HOC패턴이 유용한지, 그리고 어떻게 작성하는지 살펴본다.

## 공통적인 문제를 해결하기 위해 사용하는 HOC

컴포넌트는 리액트 내에서 코드를 재사용할 수 있는 가장 기본적인 유닛이다. 그러나, 일부 패턴은 이러한 전톡적인 컴포넌트로 해결할 수 없다는 것을 알게 된다.

예를 들어, 외부에서 데이터를 받아서 목록을 보여주는 `CommentList`라는 컴포넌트가 아래처럼 있다고 가정해보자.

```javascript
class CommentList extends React.Component {
  constructor(props) {
    super(props);
    this.handleChange = this.handleChange.bind(this);
    this.state = {
      // "DataSource" is some global data source
      comments: DataSource.getComments()
    };
  }

  componentDidMount() {
    // Subscribe to changes
    DataSource.addChangeListener(this.handleChange);
  }

  componentWillUnmount() {
    // Clean up listener
    DataSource.removeChangeListener(this.handleChange);
  }

  handleChange() {
    // Update component state whenever the data source changes
    this.setState({
      comments: DataSource.getComments()
    });
  }

  render() {
    return (
      <div>
        {this.state.comments.map((comment) => (
          <Comment comment={comment} key={comment.id} />
        ))}
      </div>
    );
  }
}
```

그리고 비슷한 패턴으로 블로그 포스트 하나를 보여주는 컴포넌트가 있다고 가정하자.

```javascript
class BlogPost extends React.Component {
  constructor(props) {
    super(props);
    this.handleChange = this.handleChange.bind(this);
    this.state = {
      blogPost: DataSource.getBlogPost(props.id)
    };
  }

  componentDidMount() {
    DataSource.addChangeListener(this.handleChange);
  }

  componentWillUnmount() {
    DataSource.removeChangeListener(this.handleChange);
  }

  handleChange() {
    this.setState({
      blogPost: DataSource.getBlogPost(this.props.id)
    });
  }

  render() {
    return <TextBlock text={this.state.blogPost} />;
  }
}
```

`CommentList`와 `BlogPost`는 동일하지 않다. 이 두 컴포넌트는 서로 다른 메소드에서 `DataSource`를 참조하고 있으며, 서로 다른 결과물을 렌더링한다. 하지만 이들은 공통적으로 구현할 수 있는게 있다.

- mount 시점에, DataSource에 `changeListener`를 단다
- 리스너 내부에서 변경된 데이터에 따라 `setState`를 호출한다.
- unmount 시점에 해당 listener를 해제한다.

만약 이 앱의 크기가 커진다면, 이와 비슷한 패턴이 반복해서 나타날 것이다. 우리는 여기서 이러한 로직을 추상화하여 한 요소에 두고, 서로다른 컴포넌트에서 사용하게 할 수 있다. 이것이 바로 HOC 컴포넌트의 기본 개념이다.

우리는 `CommentList`나 `BlogPost`등의 컴포넌트를 만드는 함수를 만들어, 여기에 공통적으로 `DataSource`를 달아 줄 수 있다. 이 함수는 자식 함수 하나를 argument로 넘겨 받아서, 넘겨받은 데이터를 prop으로 넘길 수 있다. 이러한 함수를 `withSubscription`이라고 해보자.

```javascript
const CommentListWithSubscription = withSubscription(
  CommentList,
  (DataSource) => DataSource.getComments()
);

const BlogPostWithSubscription = withSubscription(
  BlogPost,
  (DataSource, props) => DataSource.getBlogPost(props.id)
);
```

첫번째 파라미터는 컴포넌트고, 두번째 파라미터는 데이터를 받아올 `DataSource`다.

`CommentListWithSubscription`와 `BlogPostWithSubscription`가 렌더링되면, `CommentList`와 `BlogPost`는 `DataSource`로 부터 받은 데이터를 prop으로 넘기게 된다.

```javascript
// This function takes a component...
function withSubscription(WrappedComponent, selectData) {
  // ...and returns another component...
  return class extends React.Component {
    constructor(props) {
      super(props);
      this.handleChange = this.handleChange.bind(this);
      this.state = {
        data: selectData(DataSource, props)
      };
    }

    componentDidMount() {
      // ... that takes care of the subscription...
      DataSource.addChangeListener(this.handleChange);
    }

    componentWillUnmount() {
      DataSource.removeChangeListener(this.handleChange);
    }

    handleChange() {
      this.setState({
        data: selectData(DataSource, this.props)
      });
    }

    render() {
      // ... and renders the wrapped component with the fresh data!
      // Notice that we pass through any additional props
      return <WrappedComponent data={this.state.data} {...this.props} />;
    }
  };
}
```

HOC는 파라미터로 넘어온 컴포넌트를 수정하지도, 복제하지도 않는 다는 것을 염두해 두어야 한다. 그 대신, HOC는 단순히 넘겨 받은 컴포넌트를 감싸는 역할을 하는 것이다. HOC는 순수 함수이며, 어떠한 부수효과도 만들지 않는다.

이게 끝이다. 감싸진 컴포넌트는 모든 props를 넘겨 받을 것이며, 새롭게 받은 prop, `data`를 바탕으로 결과물을 그릴 것이다. HOC는 이 데이터가 어떻게 왜 쓰이는지는 관여하지 않으며, 감싼 컴포넌트도 마찬가지로 이러한 데이터가 어디서 오는지 신경쓰지 않는다.

`withSubscription`은 단지 일반적인 함수이므로, 여기에 많은 arguments를 추가할 수 있다. 예를 들어, `data` prop를 설정가능하게 만들고 싶다면, 또다른 HOC를 만들어서 감쌀 수 있다. 또는 새로운 argument를 받아서 `shouldComponentUpdate`에서 수정할 수도 있다. 이는 모두 HOC가 컴포넌트가 어떻게 제어되는지 전체적으로 관리할 수 있기 때문에 가능하다.

컴포넌트와 마찬가지로, `withSubscription`와 감싸진 컴포넌트는 완전히 `prop`을 기반으로 움직인다. 이는 동일한 `prop`을 사용하는 다른 HOC로 교체하기 용이하게 만든다. 이는 데이터를 fetch하는 라이브러리 등을 바꿀때 유용하게 사용할 수 있다.

## 원본 컴포넌트를 바꾸지마라, 대신 Composition을 사용하라.

HOC 내부에서는 컴포넌트를 수정해서는 안된다.

```javascript
function logProps(InputComponent) {
  InputComponent.prototype.componentDidUpdate = function(prevProps) {
    console.log('Current props: ', this.props);
    console.log('Previous props: ', prevProps);
  };
  // The fact that we're returning the original input is a hint that it has
  // been mutated.
  return InputComponent;
}

// EnhancedComponent will log whenever props are received
const EnhancedComponent = logProps(InputComponent);
```

위 코드에는 여러가지 문제가 있다. 그 중 하나는 `EnhancedComponent`로 부터 분리되어 `inputComponent`를 재사용할 수 없다는 것이다. 더 끔찍한 것은, 기존 컴포넌트의 `ComponentDidUpdate`도 엎어버린다는 것이다. 또한 이는 라이프 사이클 메소드가 없는 함수형 컴포넌트에서는 사용할 수가 ㅇ벗다.

컴포넌트를 변경하는 HOC는 추상화를 누출 시키는 것이다. 다른 HOC와의 충돌을 막기 위해서는, 어떻게 구현되는지 알아야 한다.

직접적으로 변경하는 대신에, HOC는 `InputComponent`를 감싸는 컨테이너 컴포넌트를 정의하여 합성을 해야 한다.

```javascript
function logProps(WrappedComponent) {
  return class extends React.Component {
    componentDidUpdate(prevProps) {
      console.log('Current props: ', this.props);
      console.log('Previous props: ', prevProps);
    }
    render() {
      // Wraps the input component in a container, without mutating it. Good!
      return <WrappedComponent {...this.props} />;
    }
  }
}
```

이런식으로 작성한다면, 기능적으로도 완전히 동일하게 작동하며 잠재적인 충돌 이슈도 피할 수 있다.

이전에 살짝 언급했지만, HOC는 컨테이너 컴포넌트 패턴으로도 불리운다. 컨테이너 컴포넌트란 고차원과 저차원의 관심사를 분리하여 역할을 맡기는 일종의 전략이다. 컨테이너는 state와 데이터 변화를 감지하는 역할을 하고, UI를 렌더링하는 컴포넌트에 이러한 데이터를 넘기는 역할을 한다. HOC는  컨테이너를 일종의 implmentation으로 사용한다. HOC를 일종의 파라미터화 된 컴포넌트 정의로 생각할 수도 있다.

## 규칙: HOC와 관련이 없는 prop을 Wrapped Component에 넘겨라

HOC는 컴포넌트에 일종의 규칙을 더해준다. 따라서 극적인 변화를 가하지는 않는다. HOC에서 반환된 컴포넌트는 기존의 컴포넌트와 비슷한 인터페이스를 가지고 있으리라 예상한다.

HOC는 HOC에서 관여하지 않는 props 에 대해서는 바로 넘겨줘야 한다. 아래의 예제를 살펴보자.

```javascript
render() {
  // 그냥 컴포넌트에 넘길 prop을 분리한다.
  const { extraProp, ...passThroughProps } = this.props;

  // WrappedComponent에 넣을 props을 정의한다.
  const injectedProp = someStateOrInstanceMethod;

  // 이렇게 넘긴다.
  return (
    <WrappedComponent
      injectedProp={injectedProp}
      {...passThroughProps}
    />
  );
```

이러한 컨벤션은 HOC를 더욱 유연하고 재사용할 수 있게 해준다.

## 규칙: 결합성을 극대화 하라.

모든 HOC가 다 똒같은 생김새를 가지고 있는 것은 아니다. argument가 컴포넌트 단 하나인 경우도 있다.

```javascript
const NavbarWithRouter = withRouter(Navbar);
```

보통 HOC는 추가적인 arugment를 받는다. Relay를 예로 들면, 컴포넌트의 데이터 디펜던시를 명세한 config 오브젝트를 추가적으로 받는다.

```javascript
const CommentWithRelay = Relay.createContainer(Comment, config);
```

그러나 일반적인 HOC는 이렇게 생겼다.

```javascript
// React Redux's `connect`
const ConnectedComment = connect(commentSelector, commentActions)(CommentList);
```

생김새가 달라서 당황스럽지만, 나누면 이렇게 구성되어 있다.

```javascript
// connect is a function that returns another function
const enhance = connect(commentListSelector, commentListActions);
// The returned function is a HOC, which returns a component that is connected
// to the Redux store
const ConnectedComment = enhance(CommentList);
```

다시말해, `connect`는 HOC를 리턴하는 HOC인 것이다.

이러한 형태는 혼란스럽고 불필요해보일 수 있지만, 꽤 유용한 면도 가지고 있다. 단일 argument를 받는 connect 함수는 컴포넌트에서 컴포넌트를 반환하는 모양새를 띄고 있다. 출력과 입력이 동일한 함수는 합성하기 매우 편리하다.

```javascript
// 이렇게 하는 대신에..
const EnhancedComponent = withRouter(connect(commentSelector)(WrappedComponent))

// ... 함수를 합성하는 유틸리티를 사용할 수 있다.
// compose(f, g, h) is the same as (...args) => f(g(h(...args)))
const enhance = compose(
  // 여기는 모두 인자를 하나로 받는 HOC들이다.
  withRouter,
  connect(commentSelector)
)
const EnhancedComponent = enhance(WrappedComponent)
```

이러한 유틸리티 함수 `componse` 는 lodash, redux, ramda와 같은 다양한 써드파티 라이브러리에서 지원한다.

## 규칙: 감싼 컴포넌트에 이름을 부여하여 디버깅을 쉽게 하자.

HOC에 의해 생성된 컨테이너 컴포넌트는 React Developer Tool에서 다른 컴포넌트 처럼 보인다. 디버깅을 쉽게 하기 위해서는, 이러한 HOC 에 이름을 부여할 필요가 있다.

```javascript
function withSubscription(WrappedComponent) {
  class WithSubscription extends React.Component {/* ... */}
  WithSubscription.displayName = `WithSubscription(${getDisplayName(WrappedComponent)})`;
  return WithSubscription;
}

function getDisplayName(WrappedComponent) {
  return WrappedComponent.displayName || WrappedComponent.name || 'Component';
}
```


## 주의사항

HOC에는 리액트를 처음 접하는 사람의 경우 헷갈릴 수 있는 몇가지 주의사항이 있다.

### HOC를 렌더링 메소드 내부에서 사용하지 마라

리액트의 비교 알고리즘 ([재조정](https://ko.reactjs.org/docs/reconciliation.html)) 은 현재 존재하는 서브트리에서 컴포넌트 업데이트가 필요한지 혹은 새롭게 마운트 해야하는지 결정한다. 만약 render에서 반환된 요소가 이전 렌더의 요소와 완전히 동일하다면, 리액트는 새로운 렌더와 이전의 서브트리를 재귀적으로 비교하면서 업데이트를 한다. 만약 동일하지 않다면, 이전의 서브트리는 완전히 unmount 된다.

일반적으로는 이러한 것들을 고민할 필요가 없다. 그러나 HOC에서는 문제가 될 수 있는데, 이는 HOC 컴포넌트내에서는 render 메소드를 사용할 수 없기 때문이다.

```javascript
render() {
  // 렌더링이 될때마다 새로운 버전의 EnchanceComponent가 생성된다.
  const EnhancedComponent = enhance(MyComponent);
  // 이는 서브트리가 반복적으로 mount 되는 원인이 된다.
  return <EnhancedComponent />;
}
```

단순히 성능만이 문제가 아니다. 컴포넌트를 새롭게 mount 한다는 것은 하위 컴포넌트들의 상태값이 모두 사라진 다는 것을 의미한다.

이렇게 하지말고, HOC의 결과물을 컴포넌트 밖에서 적용하여 딱 한번만 만들도록 해야 한다.  그러면 렌더링 사이에서 일관성이 유지되는 것이고, 이는 개발자가 원하는 것이다.

HOC를 동적으로 사용해야 하는 매우 드문 경우에는, 컴포넌트의 라이프사이클 혹은 constructor 내부에서 수행해야 한다.

### static 메소드는 반드시 복사 해야 한다.

때때로 리액트 컴포넌트에 정적 메소드를 정의하는 것이 유용할 때가 있다. 예를 들어 Relay컨테이넌 `getFragment`라는 정적 메소드를 활용하여 GraphQL 과의 합성을 용이하게 한다.

그러나 HOC 컴포넌트에 이를 적용할 경우, 원 컴포넌트는 컨테이너 컴포넌트로 감싸지게 된다. 이 말은 원본 컴포넌트가 가지고 있던 static 메소드가 모두 사라진다는 것을 의미한다.

```javascript
// Define a static method
WrappedComponent.staticMethod = function() {/*...*/}
// Now apply a HOC
const EnhancedComponent = enhance(WrappedComponent);

// The enhanced component has no static method
typeof EnhancedComponent.staticMethod === 'undefined' // true
```

이를 해결하기 위해서는, 메소드를 이전에 복사해두었다가 붙이는 방법을 써야 한다.

```javascript
function enhance(WrappedComponent) {
  class Enhance extends React.Component {/*...*/}
  // Must know exactly which method(s) to copy :(
  Enhance.staticMethod = WrappedComponent.staticMethod;
  return Enhance;
}
```

그러나 이런 기법을 사용하기 위해서는, 어떤 메소드가 정의되어있는지 정확히 알아야 한다. 이 때는 [hoist-non-react-statics](https://github.com/mridgway/hoist-non-react-statics)라이브러리를 사용하여 자동으로 복사하게 할 수 있다.

```javascript
import hoistNonReactStatic from 'hoist-non-react-statics';
function enhance(WrappedComponent) {
  class Enhance extends React.Component {/*...*/}
  hoistNonReactStatic(Enhance, WrappedComponent);
  return Enhance;
}
```

다른 방법으로는 static 메서드를 컴포넌트와 분리하여 따로 export 하는 방법이 있다.

```javascript
// Instead of...
MyComponent.someFunction = someFunction;
export default MyComponent;

// ...export the method separately...
export { someFunction };

// ...and in the consuming module, import both
import MyComponent, { someFunction } from './MyComponent.js';
```


### Ref 는 넘어가지 않는다.

HOC가 모든 props을 넘기지만, ref에는 똑같이 적용할 수 없다. 그 이유는 `ref`가 사실 `prop`이라기 보다는 `key`에 가깝기 때문이다. 만약 HOC의 결과로 나온 컴포넌트에 ref를 달게 된다면, ref는 감싼 컴포넌트가 아닌 컨테이너 컴포넌트를 가르키게 된다.

이것을 해결할 수 있는 방법은 `React.forwardRef` API를 사용하는 것이다. [여기](https://ko.reactjs.org/docs/forwarding-refs.html)에서 자세한 내용을 살펴보자.
