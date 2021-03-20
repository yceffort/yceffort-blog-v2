---
title: '리액트 서버사이드 렌더링과 컴포넌트'
tags:
  - react
  - ssr
  - javascript
published: true
date: 2021-03-19 19:31:14
description: '갈길이 멀다'
---

next + react 로 서버사이드 렌더링 환경을 구축하면서 개발을 하고 있었는데 두 가지 문제에 부딪혔었다.

## 1. `window is not defined`, SSR 환경에서의 컴포넌트

먼저 원래 코드를 보자.

```javascript
import Calendar from '@toast-ui/react-calendar'

export default function Index() {
  return (
    <Calendar
      view="month"
      month={{
        narrowWeekend: true,
      }}
      onBeforeCreateSchedule={(e) => {
        setOpenCreatePopup(true)
        setSelectedDate(e.start.toDate())
      }}
      onClickSchedule={(e) => {
        console.log(e)
      }}
      scheduleView
      calendars={calendars}
      schedules={schedules}
    />
  )
}
```

```bash
Server Error
ReferenceError: window is not defined

This error happened while generating the page. Any console logs will be displayed in the terminal window.
Call Stack
Object.<anonymous>
file:///.../node_modules/tui-calendar/dist/tui-calendar.js (16:4)
```

```javascript
(function webpackUniversalModuleDefinition(root, factory) {
	if(typeof exports === 'object' && typeof module === 'object')
		module.exports = factory(require("tui-code-snippet"), require("tui-date-picker"));
	else if(typeof define === 'function' && define.amd)
		define(["tui-code-snippet", "tui-date-picker"], factory);
	else if(typeof exports === 'object')
		exports["Calendar"] = factory(require("tui-code-snippet"), require("tui-date-picker"));
	else
		root["tui"] = root["tui"] || {}, root["tui"]["Calendar"] = factory((root["tui"] && root["tui"]["util"]), (root["tui"] && root["tui"]["DatePicker"]));
})(window, function(__WEBPACK_EXTERNAL_MODULE_tui_code_snippet__, __WEBPACK_EXTERNAL_MODULE_tui_date_picker__) // 여기에서 에러가 난다.
```

해당 컴포넌트는 최초 시작시에 `window` 가 필요한데, 서버사이드 렌더링 시에는 `window`가 없는 환경이기 때문에 에러가 난다.

아래 코드를 넣고, 최초 페이지 접근시에 새로고침을 하면 이 모듈이 실행되는 환경이 node 임을 알 수 있다. 

```javascript
console.log('node  >> ', globalThis === global) // true
```

결론적으로 이 컴포넌트는 서버사이드 렌더링을 지원하지 않고 있으며, 이를 해결하기 위해서는 `window`가 있는 브라우저 환경에서만 import 해서 사용해야 한다. 이를 nextjs에서 처리하기 위해서는 아래와 같이 하면 된다.

```javascript
// dynamic 만으로는 부족하다. 꼭 ssr을 꺼야 한다.
import dynamic from 'next/dynamic'
const Calendar = dynamic(() => import('@toast-ui/react-calendar'), {
  ssr: false,
})
```

## 2. Next SSR 환경에서의 ref

```javascript
export default function Index() {
  const cal = useRef()

  useEffect(() => {
    console.log(cal.current) 
  }, [cal])

  return <Calendar ref={cal} />
}
```

위의 log 는 아래와 같이 찍힌다.

```bash
{retry: ƒ}
retry: ƒ ()arguments: (...)caller: (...)length: 0name: "bound retry"__proto__: ƒ ()[[TargetFunction]]: ƒ retry()[[BoundThis]]: LoadableSubscription[[BoundArgs]]: Array(0)__proto__: Object
```

> `useEffect`는 SSR에서 절대로 실행되지 않는다. 이를 해결하기 위해 [useServerEffect](https://www.npmjs.com/package/use-server-effect)라고 불리우는(?) 해괴한 effect가 있지만 , 굳이 그럴필요 없이 next의 `getServerSideProps`를 사용하면 된다.

왜 `useRef`는 정상적으로 동작하지 않는 것일까?

> `useRef` returns a mutable ref object whose `.current` property is initialized to the passed argument (initialValue). The returned object will persist for the full lifetime of the component.
> 
> ...
> 
> This works because useRef() creates a plain JavaScript object. The only difference between useRef() and creating a {current: ...} object yourself is that useRef will give you the same ref object on every render.

https://reactjs.org/docs/hooks-reference.html#useref

`useRef`는 순수한 자바스크립트 객체이며, 컴포넌트가 아무리 렌더링이 된다고 해도 같은 `ref`객체를 반환한다. 그런데 현재 `ref.current`에는 `retry`만 존재한다. 이것은 무엇일까?

https://github.com/vercel/next.js/blob/f06c58911515d980e25c33874c5f18ade5ac99df/packages/next/next-server/lib/loadable.js#L219-L260

https://github.com/vercel/next.js/blob/f06c58911515d980e25c33874c5f18ade5ac99df/packages/next/next-server/lib/loadable.js#L161-L173

위 두 코드에 정답이 나와있다. `useImperativeHandle`를 통해서 `ref`를 노출하고 있기 때문에, current에는 현재 세팅되어 있는 `retry`만 보이고 있었던 것이다. `useImperativeHandle`는 [`forwardRef`](https://ko.reactjs.org/docs/react-api.html#reactforwardref)와 사용해야 한다.

> `useImperativeHandle` customizes the instance value that is exposed to parent components when using `ref`. As always, imperative code using refs should be avoided in most cases. useImperativeHandle should be used with `forwardRef`:

https://.reactjs.org/docs/react-api.html#reactforwardref

`forwardRef`는 전달 받은 `ref`속성을 하부트리의 다른 컴포넌트로 전달 할 수 있는 리액트 컴포넌트를 생성한다. 

```javascript
// #components/TuiCalendarWrapper
import React from "react";
import Calendar from "@toast-ui/react-calendar";

export default (props) => (
  // 3. 넘겨받은 `forwardedRef`를 진짜 컴포넌트에 넘긴다.
  <Calendar {...props} ref={props.forwardedRef} />
);
```

```javascript
const TuiCalendar = dynamic(() => import('#components/TuiCalendarWrapper'), { ssr: false });
// 2. forwardRef를 통해서 전달받은 ref를 하위 컴포넌트에 보낸다.
const CalendarWithForwardedRef = React.forwardRef((props, ref) => (
  <TuiCalendar {...props} forwardedRef={ref} />
));

export default function Index() {
  const ref = useRef()
  // 1. ref를 넘겨준다.
  return <CalendarWithForwardedRef ref={ref} />
}
```

Ref를 포워딩 하는 방법은 여기에 더 자세히 나와있다.

https://reactjs.org/docs/forwarding-refs.html
