---
title: Chrome Samesite 쿠키 정책
tags:
  - browser
  - web
  - javascript
published: true
date: 2020-01-09 09:09:03
description:
  '# 문제의 시작 지난 주말, 엄청나게 급하게 빠른 속도로 프로젝트를 heroku에 올릴 일이 있었다. DB도
  새로만들어야하고, 로그인도 필요한 사이트라 DB는 Heroku의 Clean DB를, 로그인은 [google sign-in for
  websites](https://developers.google.com/identity/sign-in/web)을 사용하...'
category: browser
slug: /2020/01/chrome-cookie-same-site-secure/
template: post
---

# 문제의 시작

지난 주말, 엄청나게 급하게 빠른 속도로 프로젝트를 heroku에 올릴 일이 있었다. DB도 새로만들어야하고, 로그인도 필요한 사이트라 DB는 Heroku의 Clean DB를, 로그인은 [google sign-in for websites](https://developers.google.com/identity/sign-in/web)을 사용하였다. 처음에는 [passport google auth](https://github.com/jaredhanson/passport-google-oauth2)를 사용하려다가, 그냥 하는 김에 직접 api 도큐먼트를 보면서 진행하였다.

진행 하다보니, 로그인이 안되는 문제가 발생하였다. Chrome beta를 기본 브라우저로 사용하고 있었는데, 문제는 다음과 같았다.

https://github.com/anthonyjgrove/react-google-login/issues/261

https://github.com/google/google-api-javascript-client/issues/561

https://bugs.chromium.org/p/chromium/issues/detail?id=1019168#c26

생각해보니 네이버 페이에서도 아래와 같은 메일을 받은 기억이 난다.

```
안녕하세요. 네이버페이입니다.
구글에서 서비스하고 있는 크롬 브라우저 80버전에서부터 변경될 새로운 쿠키 정책 ( SameSite Cookie ) 에 따라
네이버페이 Javascript SDK PC 결제창 호출 방식 중 레이어 타입 지원을 종료하게 될 예정입니다.

개별 가맹점에서는 크롬 브라우저 정책 내용등을 확인하시어
향후 서비스 제공을 위해 레이어 타입 지원 종료 이전 페이지 이동 또는 팝업 형태의 호출로 변경 진행 부탁 드립니다.

■ 관련 내용 : 구글 크로미엄 블로그 ( https://blog.chromium.org/2019/10/developers-get-ready-for-new.html )

■ 서비스 종료 예정일 :

  - 2020년 2월 4일 Chrome 80 배포 예정 ( https://www.chromestatus.com/features/schedule )
  - 해당 일자 전에 변경 적용 필요

■ 변경 내용 : Chrome 80 SameSite Cookie 정책 변경에 따른 naverpay javascript sdk layer 타입 지원 종료
```

보안을 위한 Cookie 정책 업데이트가 크롬에서 있었는데 (베타), 정작 google signin에서 대응을 못하고 있는 것이었다. 대략 1월 쯤에 조치를 해줄 것으로 보인다. 아직도 스레드에서 이야기 되는 것으로 보아하니, 조만간 업데이트가 될 것 같다.

# SameSite=None, Secure Cookie Settings은 무슨 정책일까?

## Samesite Cookie는 무엇인가?

기본적으로 CSRF(Cross site request forgery)공격을 막기 위해 추가된 정책이다. CSRF란 사이트간 요청 위조라는 뜻의 웹사이트 취약점 공격 방식 중 하나로, 사용자의 의지와는 관련없이 공격자가 의도한 행위를 웹사이트에 요청하는 것을 의미한다. 대표적으로 예전에 옥션이 이 공격을 이용해서 털렸다.

## 쿠키란 무엇인가?

쿠키는 키=값 이라는 쌍으로 이루어져있으며, 쿠키 유효 기간, 도메인 등을 정보로 가지고 있다. 예를 들어, 웹사이트에서 '새로운 상품' 을 알려주는 팝업이 있다고 가정하자. 대게 이런 웹사이트는 X일간 해당 정보를 표시하지 않는다, 라는 옵션을 사용자에게 선택할 수 있게 해준다. 웹사이트에서는 보통 이 정보를 쿠키를 이용해서 저장하며, HTTPS를 통해 전달할 것이다. 그리고 아마도 헤더는 아래와 같이 생겼을 것이다.

```
Set-Cookie: visited=true; Max-Age=2600000; Secure
```

만약 사용자가 이전에 보지 않음 체크를 한 유저라면, 그리고 보안연결 상태이고 n일가나 미만이라면, 브라우저가 페이지 요청시 다음 헤더를 전송한다.

```
Cookie: visited=true
```

이러한 쿠키는 사이트내 javascript 에서도 `document.cookie`를 통해 관리할 수 있다.

```javascript
document.cookie = 'visited=true; Max-Age=2600000; Secure'
document.cookie
```

당장 네이버만 가보더라도, 온갖 쿠키들이 주렁주렁 달려 있는 것을 알 수 있다.

## 쿠키 생성 주체에 따른 차이점

SameSite 정책으로 돌아와서, 사이트 방문시 현재 방문한 사이트의 쿠키 뿐만 아니라, 다양한 도메인의 쿠키가 존재한다. 현재 사이트의 도메인과 일치하는 쿠키, 즉 브라우저 주소 표시줄에 표시되는 쿠키를 First party Cookie라 한다. 그리고 현재 사이트 이외의 도메인 쿠키를 Third Party Cookie라 한다. 즉, 동일한 쿠키라 하더라도 내가 방문하고 있는 사이트에 따라 쿠키의 속성이 달라진다.

예를 들어, 내 사이트에 쩌는 힙합곡이 있어서, 다른 사이트에서도 내 음원을 다이렉트로 사용하고 있다고 가정해보자. `/blog/fucking-awesome-music.mp3`. 만약 사용자가 내 사이트에 방문한 적이있고, 또 내 사이트에서 쿠키를 받아간 적이 있다면, 다른 사이트에서 내 쩌는 힙합곡을 요청할 때, 해당 쿠키도 같이 딸려 들어갈 것이다. 다른 사이트에서는 내 쿠키를 쓰는 곳이 아무곳도 없지만, 아무튼 내사이트에서 쿠키를 받아간적이 있고, 내사이트로 다이렉트로 요청을 하고 있으므로 해당 쿠키가 다른 사이트에서 사용되는 것이다.

## Third party 쿠키의 유용성과 위험성

이런 기능은 언제 유용할까? 이러한 메카니즘은 제3자 컨텍스트에서도 상태를 유지할 수 있도록 해준다. 예를 들어, 내 사이트에서 embedded 된 A라는 유튜브 비디오를 보고 있는 사용자가 있다고 가정하자. 이 사용자가 이미 유튜브에 로그인 되어 있다면, 해당 세션은 제3자 쿠키로 만들어질 수 있다. 즉, 로그인된 사용자가 현재 보는 비디오에 '나중에보기' 버튼을 누른다면, 현재 비디오의 시청상태를 쿠키에 저장할 수 있는 것이다.

웹의 특성상 많은 부분이 개방적이지만, 반대로 이로인해 보안과 사생활 침해 우려가 있는 것도 사실이다. 앞서 말한 CSRF공격은, 쿠키 요청을 날린 사람이 누구든, 쿠키가 주어진 origin으로 요청이 간다는 것이다. 예를 들어 누군가 `evil.com` 을 로그인한다면, 내 웹사이트에 요청을 보낼 수 있고, 브라우저는 자동으로 이와 관련된 쿠키를 첨부해서 보낸다는 것이다. 그 요청은 악의적인 데이터 수집, 수정, 삭제등이 될 수 있다.

## SameSite 속성을 이용한 쿠키 사용 현황을 명시

여기에서, 앞서말한 `SameSite` 속성이 빛을 발한다. 즉, 쿠키를 first party 또는 same-site context 내에서만 사용되도록 제한 하는 것이다. 즉 완전히 동일한 사이트에서 생성된 쿠키만 사용할 수 있는 것이다. 예를 들어 `www.yceffort.kr` 도메인은 `yceffort.kr`의 일부이므로, SameSite다. 마찬가지로 `static.yceffort.kr`도 SameSite 다.

이 `SameSite` 에는 행동을 제어할 수 있는 두가지 속성도 존재한다. `strict`와 `Lax` 가 그것이다.

### Strict

`SameSite=Strict`는 쿠키 전송을 first-party cookie로만 제한한다. 이 경우, 쿠키의 사이트가 브라우저 URL 표시줄에 일치하는 경우에만 전송한다.

```
Set-Cookie: visited=true; SameSite=Strict
```

즉, 다른사이트에서 또는 이메일을 통해 사이트 링크를 따라 갈 때, 초기 요청에 쿠키가 전송되지 않는다.

### Lax

아까 힙합곡 예시로 돌아가보자.

```
Set-Cookie: visited=true; SameSite=Lax
```

```html
<audio controls>
  <source src="http://yceffort.kr/static/fucking-awesome-music.mp3" />
</audio>
<p>
  Listen the
  <a href="https://yceffort.kr/fucking-awesome-music.html">article</a>.
</p>
```

Lax 설정시 embedded된 음성 파일 요청시에는 쿠키가 들어가지 않는다. 하지만, .html로 페이지를 방문시에는, 해당 요청을 쿠키와 함께 보내게 된다.

### None

마지막으로, 값을 지정하지 않는 방식이 있다. 이는 Third party context에서도 쿠키를 사용해도 된다는 것을 의미한다.

### 정리

![설명](https://web-dev.imgix.net/image/tcFciHGuF3MxnTr1y5ue01OGLBn2/1MhNdg9exp0rKnHpwCWT.png?auto=format&w=1600)

## 그래서, 크롬은?

- [Chrome80부터 기본값을 `SameSite=Lax` 로 바꿨다.](https://blog.chromium.org/2019/10/developers-get-ready-for-new.html)

- [이는 2020년 2월 4일에 배포될 예정이다.](https://www.chromestatus.com/features/schedule)

- `SameSite=None`을 쓰고 싶다면 Secure 플래그를 활성화 해야 한다.

```
   > Rejected | Set-Cookie: widget_session=abc123; SameSite=None
   > Accepted | Set-Cookie: widget_session=abc123; SameSite=None; Secure
```

- 설정을 끄고 싶다면 chrome://flags/#same-site-by-default-cookies 로' 가면 된다.

![](./images/samesite.png)
