---
title: React 공부하기 6 - 컴포넌트 반복
date: 2019-05-21 12:16:08
published: true
tags:
  - react
  - javascript
description: "## 컴포넌트 반복해서 쓰기 ```javascript import React, {Component} from
  'react';  class IterationSample extends Component {     render ()
  {         const names = ['눈사람', '얼음', '눈', '바람']         const nameList ..."
category: react
slug: /2019/05/20/react-study-6-component-repeat/
template: post
---
## 컴포넌트 반복해서 쓰기

```javascript
import React, {Component} from 'react';

class IterationSample extends Component {
    render () {
        const names = ['눈사람', '얼음', '눈', '바람']
        const nameList = names.map(
            (name) => (<li>{name}</li>)
        );

        return (
            <ul>
                {nameList}
            </ul>
        )
    }
}

export default IterationSample;
```

```javascript
class App extends Component {
  render () {
    return (
      <IterationSample/>
    )
  }
}
```

특별한 거는 없지만, 콘솔에서 `key`가 없다는 에러가 발생한다. 가상 DOM을 비교한느 과정에서, Key값을 활용하여 변화가 일어나는지 확인하기 때문에, key값을 지정해줘야한다.

```javascript
class IterationSample extends Component {
    render () {
        const names = ['눈사람', '얼음', '눈', '바람']
        const nameList = names.map(
            (name, index) => (<li key={index}>{name}</li>)
        );

        return (
            <ul>
                {nameList}
            </ul>
        )
    }
}
```

이제 에러가 나지 않는다. 

보통은 이렇게 정적인 데이터를 쓰기보다는, 동적인 데이터를 더 렌더링할 기회가 더 많을 것이다.

```javascript
import React, {Component} from 'react';

class IterationSample extends Component {
    state = {
        names: ['토니안', '강타', '문희준', '이재원', '장우혁'],
        name: ''
    }

    handleChange = (e) => {
        this.setState({
            name: e.target.value
        })
    }

    handleInsert = (e) => {
        this.setState({
            names: this.state.names.concat(this.state.name),
            name: ''
        })
    }

    handleRemove = (index) => {
        // this.state의 레퍼런스
        const {names} = this.state;
        this.setState({
            names: names.filter((item, idx) => {return idx !== index})
        })
    }

    render() {
        const nameList = this.state.names.map(
            (name, index) => (<li onDoubleClick={() => this.handleRemove(index)} key={index}>{name}</li>)
        );

        return (
            <div>
                <input 
                onChange={this.handleChange}
                value={this.state.name}/>

                <button onClick={this.handleInsert}>추가</button>
                <ul>
                    {nameList}
                </ul>
            </div>
        )
    }
}

export default IterationSample;
```