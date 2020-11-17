---
title: React 공부하기 3 - Component
date: 2019-05-08 09:03:03
published: true
tags:
  - react
  - javascript
description: "## 컴포넌트 기본적인 컴포넌트를 만들어 보자.  ```javascript import React,
  {Component} from 'react';  class MyComponent extends Component{     render()
  {         return (             <div className='hello'>            ..."
category: react
slug: /2019/05/08/react-study-3-component/
template: post
---
## 컴포넌트

기본적인 컴포넌트를 만들어 보자.

```javascript
import React, {Component} from 'react';

class MyComponent extends Component{
    render() {
        return (
            <div className='hello'>
                brand new componenet
            </div>
        )
    }
}

export default MyComponent;
```

```javascript
import React from 'react';

import MyComponent from './MyComponent';

function App() {
  return (
        <MyComponent/>
  );
}

export default App;

```

### Props

컴포넌트의 속성을 줄때 사용하는 값이다. 

```javascript
import React, {Component} from 'react';

class MyComponent extends Component{
    render() {
        return (
            <div className='hello'>
                brand new componenet is {this.props.name}
            </div>
        )
    }
}

export default MyComponent;
```

아런식으로 부모에서 값을 넘겨 줄 수 있다. 또한 default값을 설정할 수도 있다.

```javascript
import React, {Component} from 'react';

class MyComponent extends Component{
    render() {
        return (
            <div className='hello'>
                brand new componenet is {this.props.name}.
            </div>
            <div>
                age: {this.props.age}
            </div>
        )
    }
}

MyComponent.defaultProps = {
    name: 'yceffort',
}

export default MyComponent;
```

유의할 점은 props에 아무값을 안넘겨줘도 (`nane=""`) 값은 default 가 아닌 "" 가 나온다는 것이다. 왜냐하면 ""는 기본적으로 string으로 인식 되기 때문이다. 여기에 숫자등의 값을 넘기고 싶다면 {}를 써야 한다.

props의 값을 검증하기 위해서는 propTypes를 사용한다.

```javascript
import React, {Component} from 'react';
import PropTypes from 'prop-types';

class MyComponent extends Component{
    render() {
        return (
            <div>
                <div className='hello'>
                    brand new componenet is {this.props.name}
                </div>
                <div>
                    age: {this.props.age}
                </div>
                <div>
                    company: {this.props.company}
                </div>
            </div>
        )
    }
}

MyComponent.defaultProps = {
    name: 'yceffort',
    age: 30
}

MyComponent.propTypes = {
    name: PropTypes.string,
    age: PropTypes.number,
    company: PropTypes.string.isRequired,
}

export default MyComponent;
```

숫자에 문자열을 넣거나, required인데 값을 안넘겨주는 경우에도 렌더링은 된다. 다만 콘솔창에 에러가 출력된다.


### state

props는 부모 컴포넌트가 설정하는 읽기전용 값이다. 컴포넌트 내부에서 값을 또 읽고 업데이트를 하려면 state를 써야 한다. state는 

1. 항상 기본 값이 있어야 하며
2. `this.setState()`메소드로 만 값을 업데이트 해야 한다.

```javascript
class MyComponent extends Component{

    constructor(props) {
        super(props);
        this.state = {
            number: 0
        }
    }
    render() {
        return (
            <div>
                <div className='hello'>
                    brand new componenet is {this.props.name}
                </div>
                <div>
                    age: {this.props.age}
                </div>
                <div>
                    company: {this.props.company} / {this.state.number} 개
                </div>

                <button onClick={()=> {
                        this.setState({
                            number: this.state.number + 1
                        })
                    }
                }>
                    더하기
                </button>
            </div>
        )
    }
}
```

위와 같은 방식으로도 할 수 있지만, `transform-class-properties`문법을 사용하여, constructor 바깥에서도 state와 props를 정의할 수 있다.


```javascript
import React, {Component} from 'react';
import PropTypes from 'prop-types';

class MyComponent extends Component{

    static defaultProps = {
        name: 'yceffort',
        age: 30
    };

    static propTypes = {
        name: PropTypes.string,
        age: PropTypes.number,
        company: PropTypes.string.isRequired,
    }

    state = {
        number: 0
    };

    render() {
        return (
            <div>
                <div className='hello'>
                    brand new componenet is {this.props.name}
                </div>
                <div>
                    age: {this.props.age}
                </div>
                <div>
                    company: {this.props.company} / {this.state.number} 개
                </div>

                <button onClick={()=> {
                        this.setState({
                            number: this.state.number + 1
                        })
                    }
                }>
                    더하기
                </button>
            </div>
        )
    }
}

export default MyComponent;
```

[transform-class-properties](https://babeljs.io/docs/en/babel-plugin-proposal-class-properties)란 기존에 자바스크립트 클래스 syntax에서는 method외에는 아무것도 선언할 수 없게 되어 있는 것을, 조금 더 유연하게 만들어 주는 것이다.

#### before

```javascript
class Cat {
  constructor(name, breed) {
    this.name = name;
    this.breed = breed;
  }
  getBreed() {
    return this.name + ' is a ' + this.breed;
  }
}
```

#### after

```javascript
class Cat {
  name = "Chairman Meow";
  breed = "Sphynx";
  
  getBreed = function() {
    return this.name + ' is a ' + this.breed;
  }
}
```

또한, 화살표 함수가 더이상 그 구문내의 `this`를 생성해 내지 않고, 올바르게 this가 class를 가르킬 수 있도록 도와준다.

```javascript
getBreed = () => {
    return this.name + ' is a ' + this.breed;
  }
```

이렇게 쓰는 것도 가능해 진다는 뜻이다.