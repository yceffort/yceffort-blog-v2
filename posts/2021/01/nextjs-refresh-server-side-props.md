---
title: 'Nextjs에서 Server Side props를 새로고침하기'
tags:
  - javascript
  - nextjs
published: true
date: 2021-01-28 09:13:31
description: '항상 감사하십시오 and I also, nextjs 조아'
---

```javascript
import {useRouter} from 'next/router'

export default function IndexPage({time}) {
  const router = useRouter()

  const refreshServerSide = () => {
    router.replace(router.asPath)
  }

  return (
    <>
      <div>Server Side Props Request Time: {time}</div>
      <button onClick={refreshServerSide}>Refresh Server Side</button>
    </>
  )
}

export async function getServerSideProps(context) {
  const currentDateTime = new Date().getTime()
  return {
    props: {time: currentDateTime},
  }
}
```

`getServerSideProps` 의 동작을 상상해본다면 아래와 같을 것이다.

https://yceffort.kr/2020/03/nextjs-02-data-fetching#4-getserversideprops

1. `getServerSideProps`가 있는 사이트 방문
2. nextjs가 `getServerSideProps`를 호출하여 HTML 파일 생성
3. HTML 파일을 사용자가 내려받고, 리액트가 클라이언트에서 처리 (rehydration)

그러나 한가지 nextjs에서 `getServerSideProps`를 클라이언트 사이드에서 처리하는 경우가 있다. 아래와 같은 시나리오를 상상해보자.

1. 유저가 이미 사이트에 있고, next.js의 `Link`를 클릭하여 서버사이드 렌더링 된 페이지에 방문
2. nextjs가 `getServerSideProps`를 서버에 호출하는데, 이전 처럼 HTML파일을 내려주는 대신에 단순히 데이터를 json으로 클라이언트에 전송
3. 리액트가 해당 json을 initial props로 브라우저에서 새 페이지를 렌더링

이 와 같은 과정이 위 예제 코드에서 이루어지고 있다.

최초 페이지 진입시에는 이렇게 완성된 html을 내려준다.

```javascript
<div id="__next">
    <div>Server Side Props Request Time:
      <!-- -->1611794984243</div><button>Refresh Server Side</button>
</div>
```

위 코드로 새로고침하면, 아래와 같은 json데이터만 받는다.

```json
{"pageProps": {"time": 1611795117182}, "__N_SSP": true}
```

> `__N_SSP`는 server side props이라는 뜻이다. https://github.com/vercel/next.js/blob/e819e00d0c0b2d9cd851c2c7215af1211c561932/packages/next/next-server/lib/constants.ts#L39

nextjs의 좋은 점 중 하나는 `getServerSideProps`를 일종의 api 호출처럼 사용할 수도 있다는 것이다. 위에서 사용했던 `router.asPath`는 현재 페이지의 위치를 반환한다. 여기서는 `/` 일 것이다. 이 페이지로 리다이렉트 시켜달라는 뜻은, 즉 page에 해당 데이터를 json으로 내려달라는 말과 동일하다.그리고 `push` 대신 `replace`를 사용하여 히스토리 스택에 쌓이는 것도 방지하였다.

만약 이 경우에 로딩 스피너를 걸어둬야 한다면 어떻게 할까?

```javascript
const [loading, isLoading] = useState(false)

const refreshServerSide = () => {
  router.replace(router.asPath)
  isLoading(true)
}

useEffect(() => {
  isLoading(false)
}, [time])
```

`useEffect` 훅에 서버사이드 props를 넣어주었다.

그리고 이렇게 넘겨 받은 서버사이드 props를 고치고 싶다면, 아래와 같이 처리해주면 된다.

```javascript
export default function IndexPage({time}) {
  const [currentTime, setCurrentTime] = useState(time)
  //...
}
```
