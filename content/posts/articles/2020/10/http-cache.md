---
title: 'HTTP Cache로 불필요한 네트워크 요청 줄이기'
tags:
  - browser
published: true
date: 2020-10-20 23:59:46
description: 'HTTP Cache에 대한 이해'
---

네트워크를 통해서 리소트를 가져오는 것은 느리고 비싸다.

- 리소스가 많아지면 서버와 브라우저 사이에서 라운드트립이 잦아진다.
- 중요 리소스가 다운로드 될 때 까지 페이지가 로딩되지 않는다.
- 만약 누군가 모바일로 제한된 데이터로 접근하려 할 경우, 불필요한 리소스 호출은 그들에게 돈장비로 이어진다.

이러한 불필요한 네트워크 요청을 줄이는 가장 좋은 방법은 HTTP Cache다.

## 브라우저 호환성

HTTP Cache라고 불리우는 하나의 API가 존재하는 것은 아니다. 보통 HTTP Cache라고 한다면, 아래와 같은 것들을 의미한다.

- [Cache-control](https://developer.mozilla.org/docs/Web/HTTP/Headers/Cache-Control#Browser_compatibility)
- [Etag](https://developer.mozilla.org/docs/Web/HTTP/Headers/ETag#Browser_compatibility)
- [Last-modified](https://developer.mozilla.org/docs/Web/HTTP/Headers/Last-Modified#Browser_compatibility)

위 기술들은 모든 브라우저에서 작동한다.

## HTTP Cache는 어떻게 작동하는가?

브라우저가 시도하는 모든 HTTP 요청은 먼저 브라우저 캐시로 라우팅되어, 요청을 수행하는데 사용할 수 있는 유효한 캐시가 있는지를 먼저 확인한다. 만약 유효한 캐시가 있으면, 이 캐시를 읽어서 불필요한 전송으로 인해 발생하는 네트워크 대기시간, 데이터 비용을 모두 상쇄한다.

HTTP 캐시의 동작은 [request header](https://developer.mozilla.org/en-US/docs/Glossary/Request_header)와 [response header](https://developer.mozilla.org/en-US/docs/Glossary/Response_header) 의 조합으로 제어된다. 이상적인 시나리오에서는 웹 어플리케이션의 코드(requset header)와 웹서버의 구성(response header) 모두를 제어할 수 있다.

https://developer.mozilla.org/ko/docs/Web/HTTP/Caching

## Request Header: 일반적으로 기본값을 유지

웹 애플리케이션의 request 요청에 포함되어야 하는 중요한 헤더들이 많지만, 브라우저는 요청을 할 때 거의 항상 사용자를 대신에 헤더를 생성한다. [If-None-Match](https://developer.mozilla.org/docs/Web/HTTP/Headers/If-None-Match), [If-Modified-Since](https://developer.mozilla.org/docs/Web/HTTP/Headers/If-Modified-Since) 와 같이 캐시의 신선도(?)를 확인하는 요청 헤더의 경우에는, 브라우저가 현재 값을 기준으로 요청을 날리게 된다.

이는 개발자에게는 희소식이다. 단순히 HTML에서 `<img src="my-image.png" />` 만 쓰더라도, 브라우저는 알아서 캐시에 필요한 요청을 날려준다.

> 물론 fetch의 헤더를 직접 작성하여 cache를 커스터마이징 할 수 있다.

## Response Header: 웹 서버 설정 변경

- [Cache-control](https://developer.mozilla.org/docs/Web/HTTP/Headers/Cache-Control#Browser_compatibility): 서버는 직접적으로 `Cache-Control`의 값을 리턴해서 어떻게, 그리고 얼마나 캐시할지를 직접적으로 개별 요청에 대해서 지시를 내릴 수 있다.
- [Etag](https://developer.mozilla.org/docs/Web/HTTP/Headers/ETag#Browser_compatibility): 만약 브라우저가 만료된 캐시 응답을 찾을 경우, 작은 토큰(일반적으로 파일 컨텐츠의 해쉬)를 서버로 보내서 파일이 변경되었는지를 확인할 수 있다. 만약 서버가 같은 토큰을 리턴한다면 파일이 변경되지 않았다는 뜻이므로, 다시 다운로드 할 필요가 없다.
- [Last-modified](https://developer.mozilla.org/docs/Web/HTTP/Headers/Last-Modified#Browser_compatibility): `Etag`와 같은 목적으로 만들어졌으며, 여기서는 대신에 시간을 기준으로 판단하게 된다.

일부 웹서버에는 기본적으로 이러한 헤더를 설정하는 기능이 내장되어 있으며, 다른 웹서버의 경우에는 명시적으로 구성하지 않으면 헤더를 완전히 제어 한다.

설령 `Cache-Control`의 값을 그대로 두어도 Http 캐싱이 비활성화되지 않는다. 대신 브라우저는 특정 유형의 컨텐츠에 가장 적합한 캐싱 동작 유형을 알아서 추측한다. 이를 [Heuristic Freshness](https://www.mnot.net/blog/2017/03/16/browser-caching#heuristic-freshness)라 한다.

## 어떤 Response Header를 사용해야 할까?

### 버전별 URL을 활용한 장기간 캐싱

만약 CSS 파일의 캐싱을 1년으로 설정해두었다고 해보자. 만약 디자이너가 방금 무언가를 고쳐서 다시 업데이트 해야하는 상황이라면? 어떻게 브라우저에게 업데이트 하라고 알려줄 것인가? URL자체를 바꾸지 않는 한 이는 불가능하다. 브라우저가 응답을 캐싱해버린 이상, `max-age`나 `expires`로 결정하거나, 사용자가 캐시를 날리지 않는 한 계속해서 남아있게 된다. 결과적으로, 새로 들어온 사용자와 기존 사용자가 다른 페이지를 보는 꼴이 되어 버린다. 이러한 경우를 방지하기 위해, 파일명에 버전명을 두는 방법을 사용한다.

만약 요청 URL에 특별한 지문이 있거나 버전 관리 정보를 포함하고, 데이터가 결코 변경될 일이 없다면 `Cache-Control: max-age=3153600` (1년) 을 응답에 추가한다.

이는 브라우저에 1년이 지나지 않는 한 (1년이 최대 값이다) 같은 URL에 대해서는 즉시 네트워크 요청없이 캐싱된 응답을 리턴하게 된다. [웹팩을 활용하여](https://webpack.js.org/guides/caching/#output-filenames)이러한 과정을 자동화 할 수 있다.

> `immutable`을 지정하여 절대로 바뀌지 않는 다는 것을 명시할 수도 있지만, 아쉽게도 모든 브라우저에서 작동하지는 않는다.

### 버전 없는 URL에 대해 서버에서 재검증

안타깝게도 모든 URL의 버전이 관리되는 것이 아니다. 예를 들어 www.naver.com/pay.html 에 대해서 URL 버저닝을 한다면 www.naver.com/pay.1cde52.html이 될텐데, 이렇게 하게되면 ...

HTTP 캐싱 만으로는 네트워크 요청을 피해가면서 캐싱하기에는 부족하다. 하지만 네트워크 요청을 가장 빠르고 최소화하여 캐싱할 수 있는 방법이 몇가지 있다.

아래의 `Cache-Control` 값은 버저닝 되지 않는 URL에 대한 최적화를 진행할 수 있다.

- `no-cache`. 이는 캐시된 버전의 URL을 사용하기 전에 서버에서 재검증을 해야 함을 브라우저에 지시할 수 있다.
- `no-store`. 브라우저 및 기타 중간 과정의 캐시 (`CDN` 같이)가 파일의 어떤 버전도 저장하지 않도록 지시한다.
- `private` 브라우저는 파일을 캐시할 수 있지만 중간 캐시를 할 수 없다.
- `public` 모든 응답이 캐시에 저장할 수 있다.

![cache-control flowchart](https://webdev.imgix.net/http-cache/flowchart.png)

### ETag

`ETag`나 `Last-Modified`를 사용하면, 조금더 재검증을 효과적으로 할 수 있다. 이들은 결국 요청 헤더에서 언급했던 `If-Modified-SInce`, `If-None-Match`를 트리거 하게 된다.

적절하게 구성된 웹서버가 이러한 요청 해더를 보게 되면, 브라우저가 이미 HTTP 캐시에 있는 리소스의 버전이 웹서버의 최신 버전과 일치하는지 확인 할 수 있다. 일치하는 항목이 있으면 서버는 304 not modified로 응답할 수 있다. 이는 '그냥 갖고 있는 것을 써라' 와 같다. 이러한 유형의 응답을 주고 받게 되면 실제 원본 데이터를 보내는 것 보다 데이터의 양을 확연히 줄일 수 있다.

![304](https://webdev.imgix.net/http-cache/http-cache.png)

## 요약

HTTP 캐시는 불필요한 네트워크 요청을 줄이기 때문에 웹페이지 로딩 성능을 향상 시킬 수 있는 좋은 방법이다.

## 더 많은 팁

- 일관된 URL을 사용하라. 다른 URL에서 동일한 콘텐츠를 제공하는 경우 해당 콘텐츠를 여러번 가져와서 저장한다.
- 리소스의 일부가 자주 업데이트되고, 나머지 파일은 업데이트가 잘 안되는 경우, 각각 파일을 나눠 캐시 전략을 따로 가져가는 것이 좋다.

## 예제

#### 1. Immutable content + Long max-age

```bash
Cache-Control: max-age=31536000
```

- 이 URL의 컨텐츠는 절대 변할일이 없다
- 따라서 브라우저/CDN은 이 리소스를 1년 간 캐싱해둘 것이다
- `max-age` 를 넘지 않는 리소스에 대해서 서버에 따로 요청하지 않고도 쓸 수 있다.

변경이 필요하면 URL의 컨텐츠를 바꾸는게 아니고 URL 자체를 바꿔야 한다.

일반적인 웹 서버들은 이 기능을 손쉽게 사용할 수 있도록 기능을 제공한다.

그러나 이러한 패턴을 아티클이나 블로그 포스트에 쓰면 안된다. URL은 버저닝 될 수 없지만, 컨텐츠는 변할 가능성이 존재하기 떄문이다.

### 2. Mutable content, always server-revalidated

```bash
Cache-Control: no-cache
```

- 이 URL의 컨텐츠는 변동 가능성이 있다.
- 따라서 로컬에 캐시되어 있는 정보는 믿을 수 없어서, 서버에 요청을 해봐야 한다.

`no-cache`는 캐시를 안한다는 뜻이 아니다. 이는 캐시된 리소스를 사용하기전에 서버의 체크를 거쳐야 한다는 것이다. `no-store`는 브라우저가 캐시를 저장하지 않는 다는 것이다. 마찬가지로, `must-revalidate` 또한 무조건 재검증을 한다는 것이 아니다. `max-age`에 아직 도달하지 않았다면 로컬 리소스를 사용하고, 그렇지 않다면 재검증을 한다는 것이다.

이러한 경우 `ETag`나 `Last-Modified`를 응답 헤더에 추가할 수 있다. 다음에 클라이언트가 리소스를 요청할 경우, 방금 받았던 값을 `If-None-Match`나 `If-Modified-Since`에 넣어서 사용할 수 있는데, 이 경우 서버는 HTTP 304를 리턴하여, 그냥 가지고 있는 것을 쓰라고 응답할 수 있다.

`ETag`나 `Last-Modified`가 없다면, 서버는 항상 컨텐츠를 내려준다.

설명에도 나와있듯, 이 방법은 항상 네트워크 요청을 수반한다.

### 변경 가능한 content에 max-age를 세팅하는 것은 잘못된 선택일 수도 있다.

- `/article/`
- `styles.css`
- `/script.js`

가

```bash
Cache-Control: must-revalidate, max-age=600
```

로 제공된다고 가정해보자.

이는

- URL 내의 데이터가 변경될 수 있다.
- 만약 브라우저가 10분 이전의 데이터를 가지고 있다면, 서버에 요청하지 안흔ㄴ다.
- 그 외의 경우 네트워크 요청을 한다. `If-None-Match`나 `If-Modified-Since`를 함께 사용할 수 있다.

를 의미한다.

테스트시에는 잘 동작하는 것처럼 보일 수도 있지만, 실제 사용시에 문제를 야기할 수 있으며, 문제를 추적하기도 어렵다. 위 예제에서, 만약 CSS 만 서버에서 업데이트 되었다면, 버전 불일치가 발생하게 된다. 이 리소스들은 서로 상호 읜존적이지만 캐싱 헤더는 이를 표현할 수 있다. 사용자는 리소스 중 한두개만 새거를, 그리고 나머지는 오래된 리소스를 사용할 수 있다.

`max-age`는 응답 시간과 관련이 있으므로, 모든 리소스가 동일한 내비게이션의 일부로 요청되면 거의 동시에 만료될 수 있지만 여전히 경주의 가능성이 존재한다. 그러나 이경우, 사용자가 해결할 수 있는 방법이 있긴 한다.

- 새로고침: 페이지가 새로고침으로 다시 로드 되는 경우, 브라우저는 항상 서버에서 다시 유효성 검사를 한다. 물론 사이트가 사용자에게 이런걸 강요할 수는 없다.

그러나 그렇다고 해서 `max-age`가 항상 잘못된 것은 아니다. 페이지 별로 종속성이 존재하지 않는다면, race condition은 문제 되지 않을 수 있다.
