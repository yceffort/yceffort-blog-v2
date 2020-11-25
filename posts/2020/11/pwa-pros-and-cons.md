---
title: 'PWA 적용 후기 및 장단점'
tags:
  - web
  - PWA
published: true
date: 2020-11-25 20:50:54
description: '노력은 나만 하고 즐기는것도 나만 즐긴다'
---

PWA가 나타난 이후로, [이제 모든 웹 프로젝트는 PWA로 이루어져야 한다](https://alistapart.com/article/yes-that-web-project-should-be-a-pwa/), 라든지 PWA가 웹 프로젝트의 미래라든지, 이제 모든 애플리케이션이 PWA로 만들어질 것이라든지, 더이상 일렉트론은 사라질 것이라든지의 많은 글을 봐왔다. 귀에 딱지가 앉도록 많이 들었기 때문에, 한번 해보고 싶은 마음이 들었다. 어차피 막나가는 토이 프로젝트 블로그에 PWA 하나 추가된다고 별일 없을 것이다, 라는 생각으로 블로그에 PWA를 적용해 보았다.

## HOW TO

정석대로 하는 방법은 https://web.dev/progressive-web-apps/ 여기를 참고 하는게 좋다. 원래 번역까지 해보려고 했는데, 굳이 그렇게 까지 안해도 될 것 같아서 읽고 공부해보기만 했다.

### next-pwa

고맙게도 nextjs 환경에서 pwa를 적용해줄 수 있는 라이브러리가 존재한다. 바로 [next-pwa](https://github.com/shadowwalker/next-pwa)이다. 별다른 설정을 해주지 않아도 next를 pwa환경으로 바꾸어 주었다. 그외에 설정에 맞게 `manifest.json` 추가, `<meta/>` 를 추가했다.

### pwa-asset-generator

pwa를 ios나 안드로이드에서 제대로 보여주기 위해서는 아이콘과 splash이미지를 잘 준비해야 한다. 특히 ios의 splash 이미지는 [apple launch screen 가이드에 따라서 모든 사이즈에 대응할 수 있는 이미지](https://developer.apple.com/design/human-interface-guidelines/ios/visual-design/adaptivity-and-layout/#device-screen-sizes-and-orientations)가 준비되어 있어야 해서 조금 귀찮다. 그런 작업들을 모두 [pwa-asset-generator](https://github.com/onderceylan/pwa-asset-generator)가 도와주었다. 아이콘과 스플래시 이미지, 그리고 그에 따른 태그를 알아서 만들어준다. 감사합니다

## 결과

![pwa1](./images/pwa1.png)

lighthouse 검사결과에서 PWA로 잘 인식되는 것을 볼 수 있다. `start_url`이 응답을 안한다고 하는 것은 [light house의 버그로 보인다.](https://github.com/shadowwalker/next-pwa/issues/107)

![pwa-offline](./images/pwa-offline.png)

네트워크가 오프라인으로 되어 있어도 서비스 워커를 통해서 서비스가 잘 작동하는 것을 볼 수 있다.

![pwa2-1](./images/pwa2-1.png)

![pwa2-2](./images/pwa2-2.png)

![pwa2-3](./images/pwa2-3.png)

pwa로 등록되면 크롬 주소창에 앱으로 등록할 수 있다는 뜻의 작은 아이콘이 뜬다. 이를 등록해두면 애플리케이션처럼 사용가능해진다.

![pwa-splash](./images/pwa-splash.png)

안드로이드가 없어서 테스트 해보지는 못했지만, ios에서는 splash이미지가 잘나오고 있었다.

![pw3-1](./images/pwa3-1.png)

![pw3-2](./images/pwa3-2.png)

ios에서 이렇게 실제 앱과 비슷하게 사용할 수 있었다.

## 단점

장점은 여기저기에 나열되어 있으니, 단점과 개인적인 소회를 적어보려고 한다.

### 네이티브 인터페이스를 사용할 수 없다는 한계

네이티브 인터페이스가 없기 때문에 당연히 일반 모바일 애플리케이션 대비 할 수 있는 액션이 제한되어 있다. iOS의 경우 푸쉬 알림이 불가능하며, 사용자가 PWA를 앱 아이콘으로 등록하지 않으면, 7일이 넘는 캐시데이터를 자동으로 삭제한다.

### (모바일 애플리케이션을 대체할 수 있다는 관점에 한해서) 높은 진입 장벽

PWA를 정말 일반 애플리케이션 처럼 쓰려면 모바일 기준으로, 일반사용자가 한다고 했을 때 아래와 같은 과정을 거쳐야 한다.

- PWA로 되어 있는 해당 사이트 방문 (사파리만 가능)
- 화면 중앙 하단의 공유 버튼 클릭
- 스크롤을 조금 더 내려서 홈 화면에 추가

일반적인 사용자가 절대로 경험해본적이 없을 UX이며, 설령 이 과정을 toast 든 팝업이든 알려준다 치더라도 실제로 이를 실행에 옮길 사용자가 몇이나 될까 싶다.

대부분의 사용자들은 애플리케이션 설치를 곧 앱스토어에 접속한다는 행위로 인식하고 있기 때문에, 앱스토어에 PWA를 등록하게 해주지 않는 이상 - 이러한 장점은 거의 없다고 봐야할 것이다.

## PWA의 의미

https://medium.com/iquii/progressive-web-app-pwa-what-they-are-pros-and-cons-and-the-main-examples-on-the-market-318f4538c670 의 글을 빌리자면, PWA의 장점은 아래와 같다(고한다.)

- Progressive
- Responsive
- AppLike
- Updated
- Secure
- Searchable
- Reactivable
- Installable
- Linkable
- Offline

이에 대해 의견을 달아보면

- ~~Progressive~~: 브라우저에 관계없이 사용할 수 있다는 것은, 그냥 웹의 장점이다.
- ~~Responsive~~: 반응형은 모든 웹사이트에서 구현 가능한 것이다.
- ~~AppLike~~: 위에서 언급한 이유 때문에, 앱과 비슷한 경험을 준다는 것은 사용자에게 장점으로 어필하기 쉽지 않다.
- ~~Updated~~: 서비스워커로든, http fetch로든 언제든 데이터를 제공할 수 있다. 앱과 다르게 배포가 필요하지 않다는 장점도 있지만 이는 비단 PWA만의 장점은 아니다.
- ~~Secure~~: https를 반드시 이용해야 PWA를 사용가능하지만, secure는 장점이 아니고 필수다.
- ~~Searchable~~: 웹 도 충분히 검색엔진에서 검색될 수 있다.
- Reactivable
- ~~Installable~~: AppLike와 같음
- ~~Linkable~~: 웹 사이트도 링크를 제공할 수 있다.
- Offline

PWA의 장점은 앱과 비슷한 경험을 웹으로도 줄 수 있다는 정도로 볼 수 있을 것 같다. 물론 그 비슷한 경험에도 명확하게 한계가 있기 떄문에 100% 호환이 된다고 하긴 어렵지만, iOS와 안드로이드에서 PWA를 얼마나 밀어주느냐에 따라서 미래가 달려있다고 볼 수 있을 것 같다. 물론 이 또한 긍정적이지는 않다. 인앱결제도 못하는 PWA를 자사 앱스토어 시장을 그대로 둔채로 밀어줄 것 같지는 않다.

그렇기 때문에 여전히, 모바일 서비스의 대부분은 지금처럼 앱/플레이 스토어가 중심이 될 것이다. 그렇다고 웹이 들어갈 자리가 없는 것은 아니다. 언제든 업데이트가 가능하다는점, 브라우저 스펙을 잘 따르면 거의 비슷한 사용자 경험을 줄 수 있다는 장점 때문에 앱처럼 보이는 웹, 그러니까 하이브리드 앱이 지금처럼 계속 대세를 이룰 것 같다. (그에 대한 방증으로 대부분의 회사에서 프론트 엔드 인력을 필요로 하고 있다.) 멀티플랫폼 지원, iOS, 안드로이드 의 공통 리소스 확보, 상시 업데이트를 위해 대다수의 사이즈가 큰 애플리케이션들은 이미 하이브리드 앱으로 구동되고 있다.

## 요약

- PWA로 일반 모바일 애플리케이션과 비슷한 경험을 제공할 수 있다
- 그러나 그 경험을 일반 사용자가 느끼기엔 너무 힘들다
- PWA로 느낄 수 있는 향상된 경험은 일반 웹에서도 충분히 고민해볼만한 것들이다 (secure, responsive, progressive...)
- 하이브리드 앱은 계속해서 대세가 될 것 같다
- 블로그에 있었던 PWA 버그를 제보해주신 존경하는 개발자 분께 감사의 말씀을 드린다.