---
title: 'styled-components로 스타일 적용하는법'
category: styledComponents
tags:
  - styledComponents
  - javascript
published: true
date: 2020-10-07 21:29:18
description: 'styled components가 쓰고 싶습니다'
template: post
---

먼저 [styled components](https://styled-components.com/)를 쓰는 방법을 간단하게 살펴보자.

## Table of Contents

## Create

```javascript
import styled from 'styled-components'

const Button = styled.button`
  display: inline-block;
  padding: 6px 12px;
  font-size: 16px;
  font-family: Arial, sans-serif;
  line-height: 1.5;
  color: white;
  background-color: #6c757d;
  border: none;
  border-radius: 4px;
  :not(:disabled) {
    cursor: pointer;
  }
  :hover {
    background-color: #5a6268;
  }
`
```

`styled` 컴포넌트를 임포트하고, `button`과 함께 사용하여 백틱 두개 사이에서 스타일을 적용하였다. `button` 대신에 `h1` `p` 등의 HTML elements를 사용할 수 있으며, 스타일에는 유효한 모든 CSS 문법이 가능하다.

## Usage

이렇게 만든 styled component는 리액트 컴포넌트와 마찬가지로 JSX 문법을 이용하여 사용할 수 있다.

```javascript
const App = () => {
  return (
    <Button onClick={() => alert('clicked!')} type="button">
      Button
    </Button>
  )
}
```

## Internal

이렇게 해서 만들어진 `button`을 한번 살펴보자.

https://codesandbox.io/embed/sweet-forest-1eix4?fontsize=14&hidenavigation=1&theme=dark

```html
<style data-styled="active" data-styled-version="5.2.0">
  .eWMwHd {
    display: inline-block;
    padding: 6px 12px;
    font-size: 16px;
    font-family: Arial, sans-serif;
    line-height: 1.5;
    color: white;
    background-color: #6c757d;
    border: none;
    border-radius: 4px;
  }
  .eWMwHd:not(:disabled) {
    cursor: pointer;
  }
  .eWMwHd:hover {
    background-color: #5a6268;
  }
</style>
<button type="button" class="sc-dlnjPT eWMwHd">Button</button>
```

그렇다면 내부에서는 어떻게 작할까?

1. styled components는 정의된 스타일을 기반으로 유니크한 클래스명을 만든다.
2. HTML `<head>` 영역에 `<style>` 태그를 넣고, 여기에 1번에서 만든 유니크 클래스명과 연결되는 스타일을 넣어둔다.
3. 1번과 2번을 바탕으로 렌더링한다.

## Extend

기존에 존재하는 styled component를 바탕으로 디자인을 확장할 수 있다. 대신 `.button`이 아닌 `.(확장할 컴포넌트명)`을 사용한다.

```javascript
const PrimaryButton = styled(Button)`
  background-color: #007bff;
  :hover {
    background-color: #0069d9;
  }
`
const App = () => {
  return (
    <PrimaryButton onClick={() => alert('clicked!')} type="button">
      Primary
    </Button>
  )
}
```

이와 유사하게, 리액트 컴포넌트를 확장할 수 도 있다.

```javascript
const ButtonComponent = ({className}) => {
  return (
    <Button
      className={className}
      onClick={() => alert('clicked!')}
      type="button"
    >
      Primary
    </Button>
  )
}
const PrimaryButton = styled(ButtonComponent)`
  background-color: #007bff;
  :hover {
    background-color: #0069d9;
  }
`
```

한가지 다른 점은 `className` prop이 추가되었다는 것이다. 이 `className`은 앞서 언급했던 class와 동일하게 동작하며, 반드시 prop으로 넘겨주어야 한다. 그렇지 않으면 작동하지 않는다.

## Compose

기존에 존재하는 스타일과 합성도 가능하다.

```javascript
import styled, {css} from 'styled-components'
const blackFont = css`
  color: black;
`

const WarningButton = styled(Button)`
  background-color: #ffc107;
  :hover {
    background-color: #e0a800;
  }
  ${blackFont}
`

const App = () => {
  return (
    <WarningButton onClick={() => alert('clicked!')} type="button">
      Warning
    </WarningButton>
  )
}
```

## Prop Style

prop을 받아서 스타일을 선택적으로 적용할 수도 있다.

```javascript
import styled, {css} from 'styled-components'
const SuccessButton = styled(Button)`
  ${(props) =>
    props.$success
      ? css`
          background-color: #28a745;
          :hover {
            background-color: #218838;
          }
        `
      : ''}
`
const App = () => {
  return (
    <SuccessButton $success onClick={() => alert('clicked!')} type="button">
      Success
    </SuccessButton>
  )
}
```

여기에 필수는 아니지만 `$` prefix를 붙였는데, 이는 다른 DOM, React 컴포넌트에서 사용하는 props와 명확히 구별하기 위함이다.
