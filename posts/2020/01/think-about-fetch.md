---
title: 자바스크립트에서 http 요청하기 - fetch에 대한 고찰
tags:
  - browser
  - web
  - javascript
  - typescript
published: true
date: 2020-01-22 03:57:45
description: "`toc tight: true, from-heading: 2 to-heading: 3 ` ## 1.
서론 자바스크립트에서 http 요청을 하는 것은 이제 비일비재한 일이 되었다. 서버에서 모든 데이터를 가져와서 static 한 html을
만들어서 보여주고 있는 웹페이지는 아마 찾기 어려울 것이다. 맨 처음 웹을 배울 때, jquery의 ajax ..."
category: browser
slug: /2020/01/think-about-fetch/
template: post
---

## Table of Contents

## 1. 서론

자바스크립트에서 http 요청을 하는 것은 이제 비일비재한 일이 되었다. 서버에서 모든 데이터를 가져와서 static 한 html을 만들어서 보여주고 있는 웹페이지는 아마 찾기 어려울 것이다. 맨 처음 웹을 배울 때, jquery의 ajax 요청을 배우 던 것이 한 5년 전 쯤 되었다. 비동기 http 요청이 비일비재한 요즘, 지금은 그 기술이 어디까지 왔을까? 그리고 어떻게 써야 더 깔끔하게 쓸수 있을까?

## 2. XMLHttpRequest

- [Caniuse: XMLHttpRequest](https://caniuse.com/#search=XMLHttpRequest)
- [MDN: XMLHttpRequest](https://developer.mozilla.org/ko/docs/Web/API/XMLHttpRequest)
- [whatwg: XMLHttpRequest](https://xhr.spec.whatwg.org/)

가장 원초적으로 요청을 날리는 방법이다. 지금이 API를 이용하여 호출하고 있는 사람은 아마 없을 것이다.

```javascript
var xmlHttp = new XMLHttpRequest()

xmlHttp.onreadystatechange = function () {
  if (this.status == 200 && this.readyState == this.DONE) {
    console.log(xmlHttp.responseText)
  }
}

xmlHttp.open('GET', '/yceffort/request.txt', true)

xmlHttp.send()
```

어차피 쓸 일도 거의 없고, 스펙은 위 링크에서 자세히 나와있을 테니 생략한다.

## 3. JQuery Ajax

아직도 많은 곳에서 쓰고 있을 우리 친구 JQuery와 그의 친구 `JQuery.Ajax`다.

- [jquery: ajax](https://api.jquery.com/jquery.ajax/)

```javascript
$.ajax({
  url: '/yceffort/request.txt',
  success: function (data) {
    console.log(data)
  },
})
```

마찬가지로 자세한 스펙 설명은 마찬가지로 생략한다. 물론 여기까지만 안다 하더라도, 왠만한 수준의 request는 처리할 수 있다. 그러나 복잡한 유즈케이스에서는 조금 더 이야기하기 피곤해진다.

만약 1번 request의 정보를 받아서 2번 request를 날리고, 3번 request를 날려야 하면 어떻게 될까?

```javascript
$.ajax({
  url: "/yceffort/request1.json",
  success: function(data) {
    const result = JSON.parse(data);
    $.ajax({
        url: `/yceffort/request2.json?data=${result.data}`
        success: function(data2){
            const result2 = JSON.parse(data2);
            $.ajax({
                url: `/yceffort/request2.json?data=${result2.data}`
                success: function(data3){
                    ......
                }
            })
        }
    })
  },
})
```

[Promise의 callback hell](http://callbackhell.com/)의 지옥도가 여기서도 보이게 된다. 물론 이래저래 callback을 풀어내는 방법도 있지만, 여전히 then(success)의 체이닝 콤보를 벗어날 수가 없다.

## 3. async await & fetch

> fetch는 물론 promise로도 쓸 수 있다.

es7 에서 추가된 async await과 fetch api를 활용한다면, 위의 코드를 조금더 깔끔 하게 쓸 수 있다.

- [MDN: async](https://developer.mozilla.org/ko/docs/Web/JavaScript/Reference/Statements/async_function)
- [MDN: await](https://developer.mozilla.org/ko/docs/Web/JavaScript/Reference/Operators/await)
- [MDN: fetch](https://developer.mozilla.org/ko/docs/Web/API/Fetch_API)
- [wahtwg: fetch](https://fetch.spec.whatwg.org/)

```javascript
const response1 = await fetch('/yceffort/data1.json')
const result1 = await response1.json()

const response2 = await fetch(`/yceffort/data2.json?${result1.data}`)
const result2 = await response2.json()

const response3 = await fetch(`/yceffort/data3.json?${result2.data}`)
const result3 = await response3.json()
```

fetch api는 XMLHttpRequest와 비슷하지만, 조금더 강력하고 유연한 조작이 가능하다. 또한 CORS, http origin header에 관한 개념도 정리되어 있다.

```javascript
fetch("/yceffort/data1.json", {
  method: "POST",
  mode: 'cors',
  cache: 'no-cache',
  headers:  {"Content-Type", "application/json"},
  credentials: "same-origin",
  body: JSON.stringify(bodyData)
});
```

이 외에도 다양한 옵션들이 있으니, 스펙을 참고해보자. 그러나 이 fetch api에는 단점이 존재한다. 바로 우리가 사랑하는 익스플로러를 지원하지 않는다는 것이다.

[Caniuse: Fetch](https://caniuse.com/#search=fetch)

아쉽게도, fetch를 바로 쓸 수는 없다. (이미 async, await을 쓴 시점 부터 글렀지만)

## 4. fetch polyfill

여러가지 fetch Polyfill이 존재하지만, 그 중에서 가장 많이 사용되는 것은 [isomorphic-fetch](https://github.com/matthew-andrews/isomorphic-fetch)와 [axios](https://github.com/axios/axios)가 있는 것 같다. 둘 중에 뭘 써야 되는 글이 [여기](https://gist.github.com/jsjoeio/0fd8563bc23ef852bc921836512992d9) [저기](https://stackoverflow.com/questions/40844297/what-is-difference-between-axios-and-fetch) 많이 존재한다. 대충 요약하면, isomorphic-fetch은 polyfill이 필요한 대신 원래 fetch와 가장 비슷하고(이름부터가 `isomorphic`다!) 가볍다. 반면에 axios는 사용법은 조금 다르지만 무겁고 더 여러가지 기능을 제공하는 것 같다. 취향 껏 쓰자. 여기서는 `isomorphic-fetch`를 기준으로 쓴다.

## 5. deep dive to fetch

데이터를 제공하는 api 서버가 존재하고, 여기에서 모든 응답을 json으로 내려 준다고 가정하자. 어떠한 경우에도 사용자에게 에러를 보여주지 않고 (100% 커버할 순 없지만) 최대한 자연스럽게 fetch를 해야 한다면 어떻게 해야할까?

### 5-1. 에러 처리

```javascript
const response = await `/yceffort/data1`

// 200이 아닐 경우의 처리
if (!response.ok) {
  captureException(`failed to fetch /yceffort/data1. [${response.code}]`)
}

try {
  const result = await response.json()
} catch (e) {
  // json 으로 파싱을 못할때의 처리
  captureException(`failed to parse /yceffort/data1, ${e}`)
}
```

### 5-2. Abortable Fetch

[참고](https://developers.google.com/web/updates/2017/09/abortable-fetch?hl=ko)

몇몇 fetch 요청은 그 시간이 오래 걸리거나, 사용자의 요청으로 취소를 할 수도 있어야 하는 경우가 발생한다. 그 경우 사용하는 것이 [AbortController](https://developer.mozilla.org/en-US/docs/Web/API/AbortController)다.

```javascript
const controller = new AbortController()
const signal = controller.signal

setTimeout(() => controller.abort(), 5000)

fetch(url, {signal})
  .then((response) => {
    return response.text()
  })
  .then((text) => {
    console.log(text)
  })
```

5초 뒤에 자동으로 abort 되는 코드이다. fetch를 abort하게되면, request와 response 모두 취소된다. 따라서, `response.text()`도 취소된다.

```
DOMException: The user aborted a request.
```

fetch시에 발생한 exception이 abort인지를 구별하기 위해서는 아래와 같이 처리하면 된다.

```javascript
fetch(url, {signal})
  .then((response) => {
    return response.text()
  })
  .then((text) => {
    console.log(text)
  })
  .catch((err) => {
    if (err.name === 'AbortError') {
      console.log('Fetch aborted by user')
    } else {
      console.error('other error', err)
    }
  })
```

### 5-3. fetch in react

이렇게 복잡한 fetch를 리액트스럽게 처리하는 라이브러리가 여기저기 있다.

- [use-http](https://github.com/alex-cory/use-http)
- [useSWR](https://github.com/zeit/swr)

대충 여기서 얘기 한 것 들을 기준으로, `useFetch`를 만들어 보자.

```javascript
const useFetch = (url, options) => {
  const [response, setResponse] = useState(null);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const controller = new AbortController()
  const signal = controller.signal

  useEffect(() => {
    const fetchData = async () => {
      setIsLoading(true);
      try {
        const response = await fetch(url, {...options, {signal});
        const result = await res.json();
        setResponse(result);
        setIsLoading(false)
      } catch (error) {
        setError(error);
      }
    };
    fetchData();
  }, []);
  return { response, error, isLoading, signal };
};
```

## 6. 결론

잘 만들어진 걸 가져다 쓰자.
