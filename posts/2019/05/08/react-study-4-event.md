---
title: React 공부하기 4 - Event
date: 2019-05-08 10:53:58
published: true
tags:
  - react
  - javascript
description: "## 이벤트 리액트의 이벤트는 기본적으로 HTML의 이벤트와 비슷하지만, 주의사항이 몇가지 있습니다.  1. 이벤트
  명은 카멜 케이스로 작성해야 한다. `onclick` → `onClick` 2. 이벤트에 실행할 자바스크립트 코드를 전달하는 것이 아니고,
  함수형태의 값을 전달해야 한다. 3. DOM요소에만 설정할 수 있다. Custom Component는..."
category: react
slug: /2019/05/08/react-study-4-event/
template: post
---
## 이벤트

리액트의 이벤트는 기본적으로 HTML의 이벤트와 비슷하지만, 주의사항이 몇가지 있습니다.

1. 이벤트 명은 카멜 케이스로 작성해야 한다. `onclick` → `onClick`
2. 이벤트에 실행할 자바스크립트 코드를 전달하는 것이 아니고, 함수형태의 값을 전달해야 한다.
3. DOM요소에만 설정할 수 있다. Custom Component는 onClick이벤트가 설정되는 것이 아니고, onClick Prop에 값이 넘어가는 것이다.

### 지원하는 이벤트 종류

- Clipboard
- Form
- Composition
- Keyboard
- Selection
- Focus
- Touch
- UI
- Image
- Wheel
- Animation
- Media
- Transition

```javascript
import React, {Component} from 'react';


class MyEventComponent2 extends Component {

    state = {
        message: ''
    }

    handleChange = (e) => {
        this.setState({
            message: e.target.value,
        })
    }

    handleClick = (e) => {
        alert(this.state.message);
        this.setState({
            message: ''
        })
    }

    handleKeyPress = (e) => {
        if (e.key === 'Enter') this.handleClick();
    }

    render() {
        return (
            <div>
                <h1>이벤트 연습</h1>
                <input
                    type="text"
                    name="message" 
                    placeholder="event가 달릴거야"
                    value={this.state.message}
                    onChange={this.handleChange}
                    onKeyPress={this.handleKeyPress}
                />            
                <button
                    onClick={this.handleClick}
                >
                확인
                </button>
            </div>
        )
    }
}

export default MyEventComponent2;
```

이 값을 위 처럼 state에도 넣을 수 있다.