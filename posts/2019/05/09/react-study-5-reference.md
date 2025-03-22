---
title: React 공부하기 5 - Reference
date: 2019-05-09 06:22:14
published: true
tags:
  - react
  - javascript
description:
  '## Reference (Ref) 특정 DOM요소에 작업을 하기 위해서 id를 부여하는 것 처럼, React에서
  DOM에 이름을 다는 방식이 있는데 이것이 바로 ref (Reference)다. 반드시, `DOM에 직접적으로 접근하여 조작이 필요할 때 만
  이용해야 한다.`  ### 컴퍼넌트 내부에서 사용  ```javascript import React, ...'
category: react
slug: /2019/05/09/react-study-5-reference/
template: post
---

## Reference (Ref)

특정 DOM요소에 작업을 하기 위해서 id를 부여하는 것 처럼, React에서 DOM에 이름을 다는 방식이 있는데 이것이 바로 ref (Reference)다. 반드시, `DOM에 직접적으로 접근하여 조작이 필요할 때 만 이용해야 한다.`

### 컴퍼넌트 내부에서 사용

```javascript
import React, {Component} from 'react'
import './ValidationSample.css'

class ValidationSample extends Component {
  state = {
    password: '',
    clicked: false,
    validated: false,
  }

  handleChange = (e) => {
    this.setState({
      password: e.target.value,
    })
  }

  handleButtonClick = () => {
    this.setState({
      clicked: true,
      validated: this.state.password === '0000',
    })
    this.input.focus()
  }

  render() {
    return (
      <div>
        <input
          ref={(ref) => (this.input = ref)}
          type="password"
          value={this.state.password}
          onChange={this.handleChange}
          className={
            this.state.clicked
              ? this.state.validated
                ? 'success'
                : 'failure'
              : ''
          }
        ></input>
        <button onClick={this.handleButtonClick}>Validation</button>
      </div>
    )
  }
}

export default ValidationSample
```

중요하게 봐야할 부분은 바로 여기

```html
<input ref="{(ref)" ="" /> this.input=ref}/>
```

ref 속성을 추가할 때는 props를 설정하듯이 하면 된다. ref 값으로는 콜백 함수를 전달하는데, 이 콜백함수는 ref를 파라미터로 가지며 함수 내부에서 멤버변수에 ref를 담으면 된다. 여기에서는 `this.input`에 담았다.

`this.input.focus()`를 통해서 input 태그에 포커스를 달았다.

### 컴포넌트에 Ref 달기

```javascript
import React, {Component} from 'react'

class ScrollBox extends Component {
  scrollToBottom = () => {
    const {scrollHeight, clientHeight, width} = this.box
    this.box.scrollTop = scrollHeight - clientHeight
  }

  render() {
    const style = {
      border: '1px solid black',
      height: '300px',
      width: '300px',
      overflow: 'auto',
      position: 'relative',
    }

    const innerStyle = {
      width: '100%',
      height: '650px',
      background: 'linear-gradient(white, black)',
    }

    return (
      <div
        style={style}
        ref={(ref) => {
          this.box = ref
        }}
      >
        <div style={innerStyle} />
      </div>
    )
  }
}

export default ScrollBox
```

```javascript
import React, {Component} from 'react'
import ScrollBox from './ScrollBox'

class App extends Component {
  render() {
    return (
      <div>
        <ScrollBox
          ref={(ref) => {
            this.scrollBox = ref
          }}
        />
        <button
          onClick={() => {
            this.scrollBox.scrollToBottom()
          }}
        >
          맨밑으로
        </button>
      </div>
    )
  }
}

export default App
```

ScrollBox에서 scrollToBottom 함수를 정의했다. 그리고 ScrollBox 컴포넌트를 `this.scrollBox`로 ref를 부여하여 다른 DOM에서 해당 함수를 호출 할 수 있었다.
