---
title: '태그에서 DOM으로의 여정'
tags:
  - javascript
  - web
  - browser
published: true
date: 2021-11-30 12:30:37
description: '네트워크도 공부해야하는데'
---

## Introduction

[이전 글](/2021/11/journey-from-server-to-client)에서는 브라우저에서 서버로 URL이 전송되었을 때 어떻게 처리하는지, 그리고 관련 리소스 전달을 위해 어떻게 최적화 되고 있는지 등에 대해 알아보았다. 이제 데이터가 왔으니, 브라우저 엔진이 이 리소스를 렌더링하여 웹 페이지로 만들어야 한다. 어떻게 하면 HTML을 화면에 만들 수 있는 페이지로 만드는지 살펴보자.

## Parsing

네트워크를 통해서 서버에서 클라이언트로 리소스가 전달되면 이를 변환하는 작업이 필요하다. 가장 첫번째로 일어나는 일은 HTML 파서로, 여기에서 인코딩, pre-parsing, 토큰화 (tokenization), 트리 구조 변환 등을 처리한다.

### 1. Encoding

http 응답은 HTML 텍스트에서 이미지에 이르기까지 모든 것이 될 수 있다. 파서가 첫번째로 해야 하는 일은 방금 응답으로 밭은 바이트를 해석하는 방법을 알아내는 것이다. HTML 문서를 처리한다고 가정해보자. HTML 문서를 처리하기전에, 디코더는 텍스트 문서가 어떻게 바이트로 변환되었는지 확인해야 한다.

> 텍스트도 사실 컴퓨터에서 바이너리로 변환을 해야 컴퓨터가 읽을 수 있다는 사실을 기억해야 한다.

텍스트를 어떻게 디코딩 해야 하는지 알아내는 것은 브라우저가 해야할 일이다. 서버는 `Content-Type` 헤더로 브라우저에 이 콘텐츠에 대한 힌트를 줄 수 있으며, [BOM](https://en.wikipedia.org/wiki/Byte_order_mark)을 통해서 맨 앞 비트를 가지고 분석을 할 수도 있다. 그럼에도 브라우저가 인코딩 할 수 없을 경우에는, 브라우저는 휴리스틱을 활용하여 최선의 인코딩을 분석할 수 있다. 혹은 html 태그에서 `<meta />` 태그로 인코딩된 콘텐츠에서 발견할 수도 있다. 최악의 경우, 브라우저가 일단 추측을 한다음, 파싱이 본격적으로 시작된 후로 이 인코딩에 대한 정보가 담겨있는 `<meta />` 태그를 발견할 수도 있다. 이러한 경우에는 이전까지 디코딩한 콘텐츠를 다 버리고 다시 처음부터 시작해야 한다. 브라우저는 종종 오래된 웹 콘텐츠 (레거시 인코딩으로 된)를 처리할 때가 있는데, 이러한 페이지들이 이런 방식으로 처리되고 있다.

### 2. Pre-parsing 및 scanning

인코딩을 확인하게 되면, 추가 리소스에 대한 왕복 딜레이를 최소화 하기 위해, 콘텐츠를 스캔하기 위한 initial pre-parsing을 시작하게 된다. 이 pre-parser는 완전한 파서로 보기는 어렵다. 왜냐하면 HTML이 얼마나 중첩되어 있는지, 그리고 부모-자식 관계는 무엇인지 확인하지 못하기 때문이다. 하지만 특정 HTML 태그의 속성 등을 파악할 수는 있다. 예를 들어 HTML 콘텐츠 어딘가에 

```html
<img src="https://somewhere.example.com/images/dog.png" alt=">
```

가 있다고 가정하자.

이 pre-parser는 이 `src`의 값을 확인하고 이 리소스 값을 리소스 요청 대기열에 집어 넣어준다. 이렇게 함으로써 이미지를 최대한 빨리 요청할 수 있고, 이미지가 도착하는데 까지 걸리는 시간을 최소화 할 수 있다. 이외에도 [preload](https://developer.mozilla.org/en-US/docs/Web/HTML/Preloading_content)나 [pre-fetch 지시자](https://developer.mozilla.org/en-US/docs/Web/HTTP/Link_prefetching_FAQ)와 같은 것들을 확인하여 대기열에 집어 넣어줄 수 있다.


#### Tokenization

토큰화는 HTML 파싱 과정 중 하나로, 마크업을 `begin tag` `end tag` `text run` `comment` 등과 같은 개별 토큰으로 변환하여, 파서의 다음 상태로 만들어 준다. `tokenizer`는 상태 머신으로, HTML 언어의 서로다른 상태를 처리해준다. `|`를 이 상태 머신이 처리하는 과정이라고 간주해보자.

- `<|video controls>`: 태그가 열려 있는 상태임
- `<video con|trols>`: 태그의 `controls`이라고 하는 속성을 파악
- `<video controls|>`: 태그가 닫혀있음.

이렇듯 `tokenizer`는 문자를 읽을 때 마다 반복적으로 태그의 상태를 파악하는 역할을 한다.

![tokenization](https://i0.wp.com/alistapart.com/wp-content/uploads/2018/10/fig2.png?w=960&ssl=1)

[HTML 스펙 문서](https://html.spec.whatwg.org/multipage/parsing.html)를 살펴보면 `tokenizer`를 위해서 대략 80여개의 상태를 정의해둔다. 텍스트의 내용이 유효한 HTML 콘텐츠가 아니라도, 텍스트 컨텐츠를 처리하고 HTML 문서로 변환할 수도 있다. 이와 같은 탄력성은 개발자들이 쉽게 웹 개발을 할 수 있도록 해주는 특징이다. 그러나 이러한 탄력성이 예상치 못한 결과를 야기할 수도 있으며, 이로 인해 미묘한 버그가 발생할 수도 있다. HTML validator로 한번 검사하면 이러한 실수를 사전에 방지 할 수 있다.

마크업 언어 정확성에 엄격하게 대응하기 위해, 어떠한 실패라도 렌더링하지 못하게 막는 메커니즘이 있다. 이 parsing module은 [HTML을 처리할 때 XML규칙을 사용](https://en.wikipedia.org/wiki/XHTML)하며, 문서를 `application/xhtml+xml` 유형으로 브라우저에 전송하면 된다.

브라우저는 이 pre-parser단계와 tokeniaztion 단계를 최적화를 위해서 한꺼번에 수행할 수도 있다.

### 3. 파싱 및 트리 구조화