---
title: '파티셔닝 캐시 (partitioning cache)'
tags:
  - browser
published: true
date: 2020-12-17 23:44:19
description: 'Google Font 를 써도 이제 캐시 효과는 못받겠네요'
---

일반적으로 캐싱은 데이터를 저장해두었다가, 다시 요청할 때 이를 요청하지 않고 저장한 데이터를 불러와서 성능을 향상시켜 요청이 더 빨리 처리되도록 한다. 예를 들어, 네트워크의 캐시된 리소스는 서버로 왔다리 갔다리 하는 일을 피할 수 있다. 캐시된 결과는 동일한 계산을 수행하는 것을 막을 수 있다. 크롬에서는, 이러한 캐싱을 위한 여러가지 메커니즘이 있으며, HTTP 캐시를 그중 하나로 예를 들 수 가 있다.

## Chrome 85 까지의 캐싱 동작

크롬 85 버전 까지는, 크롬은 네트워크로 부터 요청한 데이터를 캐싱하는데 있어서 각 리소스 URL을 캐시키로 활용했다. 아래 예를 살펴보자.

1. `https://a.example`에서 `https://x.example/doge.png`를 요청했다. 이 미지는 `https://x.example/doge.png`를 캐시키로 캐싱이 된다.
2. 1과는 다른 `https://b.example`에서 1과 동일한 이미지인 `https://x.example/doge.png`를 요청했다. 브라우저는 HTTP 캐시를 확인할 때 해당 미지 URL 키를 기준으로 검사하여 이 리소스가 캐시되었는지 확인한다. 1에서 이미 캐시를 했었으므로 캐시된 리소스 버전을 사용한다.
3. `https://c.example`에서 iframe 으로 `https://d.example`를 요청하고, 이 `https://d.example`에서 `https://x.example/doge.png` 를 요청했다고 가정해보자. 이 경우에도 마찬가지로 리소스 URL을 캐시키로 활용하기 때문에 캐싱된 이미지를 사용한다.

이러한 매커니즘은 성능이라는 관점에서 굉장히 잘 작동하므로 오랫동안 이용되어 왔다. 그러나 HTTP 요청에 응답하기 위해 웹 사이트가 필요로 하는 이 시간은, 과거 브라우저가 동일한 리소스에 액세스 했다는 것을 식별할 수 이으며, 이는 브라우저를 다음과 같은 보안 공격에 취약하게 만든다.

1. 유저가 특정한 사이트를 방문했는지를 식별할 수 있음: 공격자는 캐시에 특정 사이트 또는 사이트 cohort에 해당하는 리소스가 있는지 확인하여 사용자의 검색 기록을 탐지할 수 있다.
2. [크로스 사이트 검색 공격](https://portswigger.net/daily-swig/new-xs-leak-techniques-reveal-fresh-ways-to-expose-user-information): 공격자는 특정 웹 사이트에 사용하는 '검색 결과 없음' 이미지가 브라우저의 캐시에 있는지 확인하여, 사용자의 검색 결과에 특정 문자열이 있는지 여부를 확인할 수 있다.
3. 크로스 사이트 트래킹: 이 캐시를 사용하여 쿠키와 유사한 식별자를 크로스 사이트 트래킹 매커니즘으로 악용할 수 있다.

이런 이유 때문에, 크롬에서는 86부터 (사파리는 이미 적용된듯) 파티셔닝 HTTP 캐시 전략을 사용하기로 결정했다.

## 캐시 파티셔닝이 크롬 HTTP 캐시에 미치는 영향

캐시 파티셔닝을 사용하면, 캐시된 리소스는 리소스 URL 외에 추가로 'Network Isolation Key'를 활용하여 키를 만든다. 이 키는 최상위 사이트와 현재 프레임 사이트로 구성된다.

이 전략을 사용하면, 위의 예제가 아래처럼 바뀌게 된다.

1. `https://a.example`에서 `https://x.example/doge.png`를 요청한다. 이 경우 캐시키는 아래오 같이 구성될 것이다. (위에서 부터 top-level site, current-frame site, resource url)

```bash
{
  https://a.example,
  https://a.example,
  https://x.example/doge.png
}
```

2. `https://b.example`에서 `https://x.example/doge.png`를 요청한다. 1에서 같은 이미지를 요청했음에도 불구하고, 캐시 키가 일치하지 않기 때문에 캐시를 불러오지 않는다.

```bash
{
  https://b.example,
  https://b.example,
  https://x.example/doge.png
}
```

3. `https://a.example`에서 iframe으로 embedded 된 `https://a.example`를 불러오는 경우. 여기에서 같은 이미지 `https://x.example/doge.png` 를 요청했다고 가정해보자. 이 경우에는 캐시를 불러올 수 있다.

```bash
{
  https://a.example,
  https://a.example,
  https://x.example/doge.png
}
```

4. `https://a.example`에서 iframe으로 embedded 된 `https://c.example`를 불러오는 경우. 여기에 같은 이미지 리소스를 불러온 다 하더라도, frame이 다르기 때문에 캐시키를 불러올 수 없다.

```bash
{
  https://a.example,
  https://c.example,
  https://x.example/doge.png
}
```

5. `https://a.example`의 서브 도메인 `https://sub.a.example`에서 iframe으로 embedded 된 `https://c.example:8080`를 불러오는 경우. 이 경우엔 키는 `"scheme://eTLD+1"` 전략으로 생성되기 때문에, 4번에서 생성한 캐시를 가져올 수 있게 된다.

```bash
{
  https://a.example,
  https://c.example,
  https://x.example/doge.png
}
```

6. `https://a.example`에서 `https://b.example`를 embedded하고 또 이것이 `https://c.example`를 불러오는 경우. 이 경우에는 top-site 가 기준이므로 캐시키는 아래와 같이 생성되고, 이경우에도 4번의 캐시키를 히트 할 수 있다.

```bash
{
  https://a.example,
  https://c.example,
  https://x.example/doge.png
}
```

## 적용되었는지 확인하는 법

https://www.zdnet.com/article/chromes-new-cache-partitioning-system-impacts-google-fonts-performance/

1. `chrome://net-export/`로 진입해서 `Start Logging to Disk`를 누른다.
2. 로그를 저장할 위치를 고른다
3. 크롬에서 웹 서핑을 한다
4. 1번으로 돌아가서 로깅을 멈춘다.
5. https://netlog-viewer.appspot.com/#import 로 들어간다
6. 저장한 로그 파일을 업로드 한다.

`SplitCacheByNetworkIsolationKey`에서 `Experiment_`로 되어 있다면 파티셔닝 캐시가 활성화 되어 있는 것이다. `Control_`나 `Default_`는 활성화 되지 않은 것이다.

```bash
SplitCacheByNetworkIsolationKey:Experiment_Triple_Key_20201210
```

내 경우엔 활성화 되어 있었다.

## 테스트 하는법

크롬을 `--enable-features=SplitCacheByNetworkIsolationKey`와 [함께 실행하면 된다.](https://www.chromium.org/developers/how-tos/run-chromium-with-flags)

## 개발자가 취해야할 조치

breaking change는 아니지만, 일부 웹서비스의 성능에 대한 고려를 해볼 수 있다. 글꼴이나 인기 있는 스크립트를 제공하는 CDN 사이트의 경우 트래픽 요청이 증가할 수 있다.

> 예를 들어 google font를 사용한다고 해서 이제 웹 사이트의 성능적인 이점을 누릴 수 없다. 여기저기서 범용적으로 사용하는 google font를 사용한다 하더라도, 파티셔닝 캐시로 인해서 캐시를 히트하지 못하고, 이는 사이트 방문시에 같은 폰트를 다른 사이트에서 쓴적이 있다 하더라도 새롭게 받을 것이다. 따라서 그냥 google font를 쓰는 것보다 폰트나 리소스를 self-hosting 하는게 낫다.

## 성능에 미치는 영향

전체 캐시 누락률이 약 3.6% 정도 증가하고, FCP (First Contentful Paint)가 0.3%, 네트워크에서 로드되는 전체 바이트 비율이 4%정도 증가할 수 있다.

## 브라우저 별 차이

[HTTP cache partitions은 fetch 표준](https://fetch.spec.whatwg.org/#http-cache-partitions)이다. 그러나 브라우저 별로 약간의 차이가 있다.

- Chrome: top-level 과 프레임에 `scheme://eTLD+1`를 사용한다.
- Safari: top-level `scheme://eTLD+1`를 사용한다. [참고](https://webkit.org/blog/8613/intelligent-tracking-prevention-2-1/)
- Firefox: [적용 예정](https://bugzilla.mozilla.org/show_bug.cgi?id=1536058) top-level `scheme://eTLD+1`를 사용하며, 크롬처럼 2번째 키 사용을 고려중

출처: https://developers.google.com/web/updates/2020/10/http-cache-partitioning
