---
title: '서버에서 클라이언트로의 여정'
tags:
  - javascript
  - web
  - server
published: true
date: 2021-11-23 23:12:01
description: ''
---

## Table of Contents

## Introduction

브라우저에서 웹사이트를 보여주기 위해 무언가를 하기전에, 먼저 브라우저가 어디로 가는지 알아야 한다. 주소 표시줄에 URL을 입력하거나, 페이지 또는 다른 앱의 링크를 클릭하거나, 즐겨찾기를 클릭하는 등 다양한 방법으로 웹사이트에 접근할 수 있다. 어떤 경우든, 결국 `navigation`이라는 과정이 일어나게 된다. 소위 이 탐색이라는 과정은, 웹 사이트를 상호작용하는 과정의 첫번째 단계이며, 이 후에 웹 페이지 로드에 필요한 이벤트가 연쇄적으로 일어나게 된다.

## 최초 요청

브라우저에 로딩해야할 URL이 주어지면, 아래 몇가지 일이 일어난다.

### HSTS 확인

> https://en.wikipedia.org/wiki/HTTP_Strict_Transport_Security

먼저 쁘라우저는 URL이 HTTP 방식을 지정하는지를 확인해야 한다. 만약 HTTP 요청이라면, 브라우저는 도메인이 [HSTS](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Strict-Transport-Security) 목록에 있는지 확인 해야 한다. 이 목록은 사전에 로딩한 목록과 HSTS를 사용하도록 선택되어진 이전에 방문한 사이트 목록으로 구성되어 있으며, 두 사이트 목록 모두 브라우저에 저장된다. 요청된 HTTP 호스트가 HSTS 목록에 저장되어 있는 경우, HTTP 대신 HTTPS 버전의 URL로 요청이 이루어진다. 그렇기 때문에 브라우저에 http://yceffort.kr 를 입력해도 https://yceffort.kr 로 대신 보내진다.

### 서비스워커 확인

다음 부터는, 브라우저는 [서비스 워커](https://developer.mozilla.org/en-US/docs/Web/API/Service_Worker_API)가 요청을 처리할 수 있는지 확인해야 한다. 이 서비스 워커는 사용자가 오프라인 상태이고, 네트워크 연결이 없을 때 특히 중요하다. 서비스 워커는 비교적 최근에 나온 기능이다. (라고 하기엔 나온지는 꽤 되었지만) 서비스 워커는 오프라인에서도 웹 사이트를 사용할 수 있도록 네트워크 요청을 차단하고 [캐시](https://developer.mozilla.org/en-US/docs/Web/API/Cache)에서 처리할 수 있도록 도와준다.

서비스 워커는 페이지를 방문했을 때, [서비스 워커 등록 및 로컬 데이터 베이스에 URL 매핑을 기록할 수 있다.](https://www.w3.org/TR/service-workers-1/#dfn-scope-to-registration-map) 서비스 워커가 설치되었는지 여부를 확인하는 것은 데이터베이스에서 이전에 탐색한 적이 있는 URL을 조회하는 것 만큼이나 간단하다. 지정된 URL에 서비스 워커가 있는 경우, 요청에 대한 응답을 처리할 수 있다. 브라우저에서 [Navigation Preload](https://developers.google.com/web/updates/2017/02/navigation-preload#the-solution)를 사용할 수 있고, 사이트가 이 기능을 활용할 수 있는 경우, 브라우저는 초기 네비게이션 요청을 위해 네트워크를 동시에 참조한다. 이는 브라우저가 서비스 워커가 느려서 요청을 차단하지 않도록 하기 때문에 유용하다.

초기 요청을 처리한 서비스 워커가 없는 경우 (또는 Navigation Preload가 이미 사용 중인 경우) 브라우저는 네트워크 계층을 참조하기 시작한다.

### 네트워크 요청 확인

브라우저는 네트워크 계층을 통해 캐시에 새로운 응답이 있는지 확인 한다. 일반적으로 이는 응답의 [Cache-Control](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Cache-Control) 헤더에 의해 정의된다. `max-age` 으로 캐시된 항목이 얼마나 유효한지 정의할 수 있으며, `no-store`로 저장되지 않는 캐시를 정의할 수도 있다. 그리고 물론, 브라우저가 네트워크 요청의 캐시에서 아무것도 확인할 수 없는 경우에는, 네트워크 요청이 필연적으로 필요하다. 이후에 약 캐시에 새로운 응답이 있을 경우, 페이지를 로드하기 위해 리턴된다. 리소스가 발견되었지만, 굳이 새로운 리소스가 필요하지 않은 경우, 브라우저는 이 요청을 조건부 재평가 요청 (conditional revalidation request) 으로 반환할 수 있다. 여기에는 브라우저가 캐시에 이미 있는 콘텐츠 버전을 서버에 알리는 `If-Modified-Since` `If-None-Match` 헤더가 포함된다. 서버는 응답 없이 HTTP 304를 반환하여 사본이 여전히 유효하다는 것을 알리거나, 새 버전의 리소스와 함께 200 응답을 반환하여 사본이 오래된 것임을 브라우저에 알릴 수도 있다.