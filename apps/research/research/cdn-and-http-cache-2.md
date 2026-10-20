---
title: 'CDN과 HTTP 캐시 2: 전략과 문제 해결'
marp: true
paginate: true
theme: midnight
tags:
  - caching
  - networking
  - web-performance
date: 2026-09-11
description: 'HTML·JS·이미지·폰트·API의 캐시 정책을 정하고, 배포 순서와 DevTools로 캐시 문제를 진단하는 실무 강의'
published: true
art:
  undraw: connected-world
---

# CDN과 HTTP 캐시 2

전략과 문제 해결: 서비스에 캐시 적용하기

<!-- _class: invert deck-cover -->

@yceffort

---

## 2편의 목표와 순서

**파일마다 캐시 정책을 정하고, 배포 후 옛 화면이 보이는 이유를 찾아 고칠 수 있게** 되는 것.

1. **자산별 전략** : HTML·JS·이미지·폰트·API, URL 버저닝과 CDN 무효화
2. **함정과 도구** : 개인화 응답, 새로고침, DevTools, 여러 종류의 캐시
3. **실무 적용** : 배포 순서, 코드 리뷰, 장애 신고 진단

중간 점검을 풀고, 마지막에 **실무 상황 네 문제**로 연결해 본다.

[1편: 원리와 규칙](/slides/cdn-and-http-cache)에서 배운 내용을 다음 한 장으로 복습한다.

<!-- 기준은 1편과 같다. HTTP 캐시는 RFC 9111, immutable은 RFC 8246, stale-while-revalidate는 RFC 5861을 따른다. yceffort.kr의 Next.js 16.3.1/Vercel 실측은 2026-09-11, 우리 서비스 실측은 2026-09-16 기준이다. 브라우저는 Chrome을 기준으로 하고 차이가 있는 곳은 별도로 표시한다. -->

---

## 1편 복습 : 이 규칙을 가지고 시작한다

**브라우저와 CDN은 서로 다른 캐시다.** 같은 응답도 각자의 규칙과 설정으로 판단한다.

| 확인할 것        | 기억할 규칙                                                 |
| ---------------- | ----------------------------------------------------------- |
| 저장 여부        | `no-store`는 저장 금지, `private`은 공유 캐시 저장 금지     |
| 재사용 전에 확인 | `no-cache`는 저장할 수 있지만 매번 확인                     |
| 신선 기간        | `max-age`에서 이미 지난 시간인 `Age`를 고려                 |
| 변경 여부        | ETag를 `If-None-Match`로 보내고, 같으면 `304`로 본문 재사용 |
| 같은 사본의 범위 | `Vary`와 CDN 캐시 키가 사본을 구분                          |

이제 **변경 빈도·URL 버저닝·개인화 여부**에 따라 이 규칙을 조합한다.

---

## Part 1 : 자산별 캐시 전략

<!-- _class: invert -->

---

## 왜 파일 종류마다 다르게 다루는가

캐시 정책을 정할 때 보는 축은 셋이다.

| 자산             | 얼마나 자주 바뀌나     | URL이 내용을 식별하나   | 개인화되나  |
| ---------------- | ---------------------- | ----------------------- | ----------- |
| HTML             | 배포마다, 또는 매 요청 | 아니오 (`/`는 항상 `/`) | 때때로      |
| JS / CSS         | 배포마다               | **예** (해시 파일명)    | 아니오      |
| 빌드 포함 이미지 | 거의 안 바뀜           | 예 (해시 파일명)        | 아니오      |
| 업로드 이미지    | 사용자가 바꿈          | 대개 아니오             | 아니오      |
| 폰트             | 거의 안 바뀜           | 예                      | 아니오      |
| API 응답         | 매 요청                | 아니오                  | **대개 예** |

"URL이 내용을 식별하는가"가 가장 중요한 축이다. 그렇다면 **무한히 캐시**해도 되고, 아니라면 **바뀔 수 있는 주기 안에서** 잘라야 한다.

---

## HTML : 짧게, 그러나 CDN은 길게

HTML은 진입점이다. 개인정보가 없는 공통 HTML은 **최신 여부를 확인하되, 같은 본문을 반복 전송하지 않게** 한다.

```text
Cache-Control: no-cache            (또는 max-age=0, must-revalidate)
ETag: "..."                        ← 응답 서버가 HTML의 버전을 구분해 생성
```

- 브라우저는 매번 확인하되, 안 바뀌었으면 `304`로 본문을 아낀다
- **ETag 생성과 조건부 요청 처리는 응답 서버에 맡긴다.** 정적 파일 서버의 기본 기능이나 프레임워크의 재검증 기능을 우선 사용한다

공개 HTML의 갱신 지연을 허용한다면 CDN용 정책을 따로 줄 수도 있다.

```text
Cache-Control: public, max-age=0, must-revalidate,
               s-maxage=300, stale-while-revalidate=86400
```

이 예시는 CDN에서 5분간 신선하게 쓰고, 이후에는 최대 하루 동안 낡은 응답을 주며 백그라운드에서 갱신하도록 허용한다.

<!-- no-cache + ETag는 브라우저가 매번 확인하면서 본문 전송을 줄이는 기본 예시다. 아래 코드는 공유 캐시에 갱신 지연을 허용하는 별도 선택지이며 위 no-cache 예시에 덧붙이는 지시어가 아니다. s-maxage=300은 오리진 요청을 정확히 5분에 한 번으로 보장하지 않는다. 엣지·캐시 키·퇴출·CDN 정책에 따라 달라진다. ETag 값은 본문 버전과 일치해야 하며, 배포 ID만 같아도 HTML의 데이터나 import map이 달라질 수 있다면 동일한 태그를 써서는 안 된다. -->

---

## ETag는 HTML을 응답하는 곳에서 만든다

| HTML을 제공하는 방식      | 권장하는 설정·구현                                           |
| ------------------------- | ------------------------------------------------------------ |
| Nginx가 정적 HTML 제공    | `etag on;`으로 자동 생성. 기본값도 `on`                      |
| S3가 정적 HTML 제공       | S3가 ETag와 조건부 GET을 제공. `Cache-Control`을 별도로 설정 |
| 서버가 HTML을 조립·렌더링 | 프레임워크의 ETag·조건부 GET 기능을 우선 사용                |

Nginx가 `index.html`을 직접 제공하는 경우의 예시다.

```nginx
location = /index.html {
    etag on;
    add_header Cache-Control "no-cache";
}
```

직접 구현한다면 **완성된 HTML의 해시 등으로 ETag를 만들고**, `If-None-Match`가 현재 값과 일치할 때 **본문 없는 `304`**를 반환해야 한다.

**검증:** 같은 본문 + 받은 ETag로 요청 → `304`. HTML 변경 후 옛 ETag로 요청 → `200` + 새 본문·새 ETag.

<!-- 위 Nginx 예시는 서버의 root 등이 이미 설정되어 정적 index.html을 직접 제공하는 경우다. etag on은 프록시 뒤에서 동적으로 생성된 HTML에 ETag를 자동으로 만들어 주는 설정이 아니다. /kkb/home이 어느 계층에서 HTML을 조립하는지는 헤더만으로 확인하지 않았으므로 운영 서버 종류를 먼저 확인한다. 직접 구현할 때는 고정 문자열이나 모든 응답에 같은 배포 ID를 무조건 쓰지 않는다. 최종 내용과 언어·인코딩 등 표현의 차이를 식별할 수 있어야 한다. If-None-Match의 약한 비교, 태그 목록, * 처리와 304의 ETag/Cache-Control/Vary 등 메타데이터까지 직접 다루기보다 검증된 서버 기능을 우선 사용한다. Next.js의 generateEtags는 기본 활성화지만, 로컬 Next.js 16.3.1의 send-payload.js는 result.isDynamic인 스트리밍 응답에서 payload를 null로 두고 해당 ETag 생성 경로를 건너뛴다. 설정 하나로 모든 HTML에 ETag가 생긴다고 보장하지 않는다. ETag 때문에 스트리밍 전체를 버퍼링하면 첫 바이트 전송이 늦어질 수 있다. 출처: https://nginx.org/en/docs/http/ngx_http_core_module.html#etag , https://docs.aws.amazon.com/AmazonS3/latest/API/API_GetObject.html , https://httpwg.org/specs/rfc9110.html#field.etag , https://nextjs.org/docs/app/api-reference/config/next-config-js/generateEtags -->

---

## JS / CSS : 내용 해시 + 1년 + immutable

```text
/_next/static/immutable/chunks/2kzkiswgqacx4.js
Cache-Control: public, max-age=31536000, immutable
```

파일명 `2kzkiswgqacx4`는 내용의 해시다. 이 한 가지 사실에서 전략 전체가 나온다.

- 내용이 바뀌면 이름이 바뀐다. 그러니 **같은 URL의 내용은 영원히 같다**
- 영원히 같으니 1년(HTTP가 권고하는 상한) 동안 신선하다고 해도 되고, 새로고침 때 확인도 필요 없다(`immutable`)
- **무효화가 필요 없는 캐시**다. 바꾸고 싶으면 새 이름으로 새 파일을 올리면 된다

번들러(webpack, Vite, Turbopack)가 해시 파일명을 만드는 이유가 이것이다. 해시 없는 `app.js`에 1년 캐시를 주면 배포가 사용자에게 도달하지 않는다.

<!-- 1년은 RFC 9111이 "1년을 넘는 값은 1년으로 간주해도 된다"고 하는 데서 온 관행이다. -->

---

## 쿼리스트링 버저닝은 반쪽짜리다

해시 파일명 대신 `app.js?v=42`를 쓰는 프로젝트가 아직 많다.

- 브라우저는 쿼리스트링을 캐시 키에 넣으니 **브라우저 캐시는 갱신된다**
- 그러나 CDN 중에는 쿼리스트링을 무시하도록 설정된 곳이 있다(CloudFront의 관리형 정책 CachingOptimized가 그렇다). 그러면 `?v=42`도 `?v=41`도 같은 사본이다
- 반대로 CDN이 쿼리를 키에 넣으면 `?utm_source=...` 값마다 사본이 갈라져 HIT율이 떨어진다

파일명에 해시를 넣으면 이 양쪽 문제가 모두 없다. 쿼리 버저닝을 유지해야 한다면 CDN 캐시 정책에서 **버전 파라미터만** 키에 포함하도록 지정한다.

---

## 해시 파일명이 만드는 배포 순서 문제

```text
t0  사용자가 HTML(v1)을 열어 둔 상태. 화면은 app.a1b2.js를 로드해 두었다
t1  v2 배포. HTML(v2)은 app.c3d4.js를 참조. 서버에서 app.a1b2.js는 삭제됨
t2  사용자가 v1 화면에서 페이지 이동 → 아직 안 받은 청크 chunk-7.a1b2.js 요청
    → 404 → ChunkLoadError
```

첫 슬라이드의 두 번째 질문("일부 사용자만 ChunkLoadError")의 정체다. 새로고침하면 HTML(v2)을 받아 사라지니 재현이 안 된다.

해법은 세 겹으로 둔다.

1. **자산을 먼저 올리고 HTML을 나중에 바꾼다.** 새 HTML이 참조하는 파일이 없는 순간을 없앤다
2. **이전 배포의 자산을 한동안 남긴다.** Vercel은 이전 배포를 지우지 않고, S3 배포라면 정리 주기를 며칠로 둔다
3. **버전 불일치를 감지하면 새로고침을 유도한다.** 청크 로드 실패 시 한 번 `location.reload()`

<!-- Vercel의 "skew protection"은 3번을 플랫폼이 대신 해 주는 기능이다. 요청에 배포 ID를 실어 옛 배포의 자산을 옛 배포에서 가져온다. -->

---

## 이미지 : 두 종류를 나눠서 본다

**빌드에 포함된 이미지** (로고, 아이콘, `import logo from './logo.png'`)

- 번들러가 해시 파일명을 붙인다. JS와 같은 전략: 1년 + `immutable`

**런타임에 정해지는 이미지** (사용자 업로드, CMS)

- URL이 `/uploads/profile.jpg`처럼 고정이면 교체해도 캐시가 옛것을 준다. 첫 슬라이드의 세 번째 질문이다
- 해법은 둘 중 하나. **업로드마다 새 키**(`/uploads/8f3a...jpg`)를 만들거나, URL을 고정할 거면 **짧은 TTL + ETag**

```text
Next.js 16의 /_next/image 기본값:
  정적 import 이미지  → public, max-age=31536000, immutable   (실측)
  원격 URL 이미지     → minimumCacheTTL 14400초(4시간) 기준으로 max-age 결정
```

---

## 이미지 : 크기와 포맷마다 사본이 갈라진다

이미지 CDN(next/image, Cloudinary, imgix)은 한 원본에서 여러 변형을 만든다.

```text
/_next/image?url=/media/profile.jpeg&w=48&q=75    Accept: image/avif → avif 735B
/_next/image?url=/media/profile.jpeg&w=48&q=75    Accept: */*        → jpeg 975B
/_next/image?url=/media/profile.jpeg&w=384&q=75   ...
```

- 캐시 키는 `URL(원본, w, q) + Accept`다. `Vary: Accept`가 그 마지막 조각을 키에 넣는다
- 변형 수가 많을수록 **첫 사용자는 매번 MISS**를 맞는다. 1편에서 잰 MISS 372ms가 그런 사례다. `sizes`와 `deviceSizes`를 줄여 변형 수를 통제하는 것이 HIT율을 올리는 방법이다

---

## 이미지 도메인 분리 : 장점과 비용

`img.example.com`처럼 호스트를 나누면 **독립적으로 운영하기 쉬워지지만, 연결과 설정의 비용이 늘 수 있다.**

| 관점 | 장점                                       | 비용·조건                               |
| ---- | ------------------------------------------ | --------------------------------------- |
| 운영 | 이미지 CDN·배포·캐시 정책을 따로 관리      | DNS·인증서·장애 감시 대상이 늘어남      |
| 쿠키 | 메인 서비스 쿠키의 전송 범위에서 제외 가능 | 쿠키의 `Domain` 설정도 맞춰야 함        |
| 출처 | 사용자 업로드를 메인 앱과 다른 출처에 둠   | CSP 허용 목록, 폰트·canvas의 CORS 점검  |
| 연결 | 별도 이미지 CDN을 선택할 수 있음           | DNS 조회·연결·TLS 비용이 추가될 수 있음 |

---

## 도메인을 나눠도 쿠키는 따라올 수 있다

**쿠키는 자동으로 분리되지 않는다.** `Domain=example.com`인 쿠키는 `img.example.com`에도 갈 수 있다. 호스트 전용 쿠키나 별도 도메인으로 범위를 나눠야 한다.

**선택 기준:** 별도 CDN이나 업로드 출처 격리가 필요하면 분리한다. TTL만 다르면 같은 호스트의 `/images/*` 경로별 정책도 검토한다.

HTTP/2에서는 분리 자체가 속도 향상을 보장하지 않는다. 분리한다면 **주요 이미지 호스트에만 `preconnect`를 적용하고 실제 LCP를 비교**한다.

<!-- Domain 속성이 없는 쿠키는 발급한 호스트에만 전송된다. 부모 도메인의 Domain 쿠키는 조건이 맞는 하위 호스트에도 전달되므로 서브도메인 분리만으로 쿠키가 없어지지는 않는다. 우리 예시의 kakaopaysec.com 쿠키는 별도 도메인인 t1.kakaocdn.net에는 전달되지 않지만, kakaocdn.net 자체의 쿠키까지 없다는 뜻은 아니다. HTTP/2는 한 연결에서 여러 요청을 다중화하고 조건이 맞으면 출처 간 연결도 재사용할 수 있으므로 추가 연결 비용이 항상 발생한다고 단정하지 않는다. 일반 img 표시에 CORS가 항상 필요한 것은 아니며, 폰트 로드나 canvas 픽셀 읽기 등에서 별도 점검한다. preconnect는 비용을 없애는 것이 아니라 미리 치르게 하는 힌트이며 남발하면 연결 자원을 낭비한다. 정책만 분리하는 목적이라면 CloudFront의 경로별 Behavior 등으로 같은 호스트에서도 가능하다. 출처: https://httpwg.org/specs/rfc6265.html#attribute-domain , https://httpwg.org/specs/rfc9113.html#reuse , https://developer.chrome.com/docs/lighthouse/performance/uses-rel-preconnect , https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/CORS -->

---

## 폰트 : immutable에 CORS까지

```text
$ curl -sI https://yceffort.kr/_next/static/immutable/media/\
03bda585a99c6450-s.p.0glcd8n2pdab0.woff2
cache-control: public,max-age=31536000,immutable
access-control-allow-origin: *
content-type: font/woff2
```

- 폰트 파일은 해시 파일명이고 거의 안 바뀐다. JS와 같은 전략이다
- `@font-face`의 폰트 요청은 명세상 **anonymous CORS 모드**다. 다른 출처의 폰트는 응답의 `Access-Control-Allow-Origin`이 허용해야 사용할 수 있다
- `<link rel="preload" as="font" crossorigin>`는 실제 폰트 요청과 모드를 맞춘다. **같은 출처의 폰트를 preload할 때도** 붙인다
- S3에서 CORS 응답을 만든다면 버킷에 허용 출처를 설정한다. **출처별로 응답이 달라지면** CloudFront 캐시 키에도 `Origin`을 넣는다

<!-- @font-face의 anonymous 모드는 credentials: same-origin에 해당한다. 같은 출처의 폰트에는 ACAO가 필요하지 않지만, preload는 실제 폰트 요청과 CORS·credentials 모드가 일치해야 재사용된다. 공개 폰트에 항상 ACAO: *를 붙이는 구성이라면 CORS 응답을 나누기 위해 Origin을 캐시 키에 넣을 필요는 없다. S3가 Origin에 따라 CORS 헤더를 생성하는 구성과 구분한다. 출처: https://www.w3.org/TR/css-fonts-3/#font-fetching-requirements , https://html.spec.whatwg.org/multipage/links.html#link-type-preload -->

---

## 폰트는 왜 서버의 사용 허락을 받을까?

**남의 사이트가 내 폰트 URL을 그대로 가져다 쓰면, 내 서버가 그 트래픽까지 감당한다.**

- 초기 웹폰트 표준화에서는 이런 **무단 직접 연결(hotlinking)** 과 대역폭 비용이 논의됐다. 특정 사이트에만 허용한 폰트의 라이선스 문제도 함께 다뤄졌다
- 그래서 기본은 같은 출처에서 사용하고, **다른 출처에서 쓰려면 폰트 서버가 CORS 헤더로 허락**하도록 정했다. 지금은 관행이 아니라 CSS Fonts 명세의 요구사항이다

```text
폰트 응답: Access-Control-Allow-Origin: https://shop.example

shop.example에서 사용   → 허용
other.example에서 사용  → CORS 검사 실패, 폰트 적용 안 됨
```

**복제 방지 장치는 아니다.** 파일을 내려받아 다른 서버에 올리는 것까지 막지는 못한다.

<!-- 2011년 W3C WebFonts WG 논의에는 bandwidth 비용과 single-site 라이선스를 위한 가벼운 제한이라는 관점이 함께 나온다. 2014년 논의는 same-origin/CORS 요구사항이 WOFF에서 모든 웹폰트에 적용되는 CSS Fonts로 옮겨졌고, deep linking과 bandwidth hijacking 방지가 배경이었다고 설명한다. 개별 메일을 단일한 공식 도입 사유로 단정하지 않는다. CORS는 서버에 요청 자체가 도달하지 않게 하는 기능이 아니며, 허용되지 않은 출처에서 응답을 폰트로 사용하는 것을 브라우저가 막는다. 출처: https://lists.w3.org/Archives/Public/public-webfonts-wg/2011Feb/0066.html , https://lists.w3.org/Archives/Public/public-webfonts-wg/2014Mar/0004.html , https://www.w3.org/TR/css-fonts-3/#font-fetching-requirements -->

---

## API 응답 : 기본은 저장 금지, 예외만 열어 준다

```text
GET /api/me                  → Cache-Control: private, no-store
GET /api/products?page=1     → Cache-Control: public, s-maxage=60,
                               stale-while-revalidate=600
```

- 사용자별로 다른 응답은 **`private` 또는 `no-store`** 다. CDN에 남으면 다른 사용자에게 나간다
- 모든 사용자에게 같은 공개 데이터(상품 목록, 환율, 공지)는 CDN에 짧게 두고 SWR로 갱신한다. 오리진 부하가 1/N로 준다

RFC 9111의 함정: `Authorization` 요청의 응답은 공유 캐시가 저장하지 않는 것이 기본이지만, 응답에 **`public`, `s-maxage`, `must-revalidate` 중 하나라도 있으면 저장한다.** 인증 API에 "일단 public" 습관이 위험한 이유다.

CloudFront의 함정: `Authorization`을 오리진 요청 정책으로 전달하면서 캐시 키에 넣지 않으면, 문서의 경고 그대로 "인증된 사용자와 안 된 사용자에게 같은 사본"이 나간다. 인증 API는 CachingDisabled 동작으로 뺀다.

---

## 자산별 전략 한 장 요약

| 자산                          | Cache-Control                         | 검증자 | 갱신 방법           |
| ----------------------------- | ------------------------------------- | ------ | ------------------- |
| HTML                          | `no-cache` (+ CDN용 `s-maxage`, SWR)  | ETag   | 다음 요청에 자동    |
| JS / CSS / 빌드 이미지 / 폰트 | `public, max-age=31536000, immutable` | 불필요 | 새 해시 파일명      |
| 업로드 이미지                 | 새 키, 또는 `max-age=짧게` + ETag     | ETag   | 새 키 또는 TTL 만료 |
| 개인화 API                    | `private, no-store`                   | 없음   | 캐시 없음           |
| 공개 API                      | `public, s-maxage=N` + SWR            | ETag   | TTL 또는 CDN purge  |

한 문장으로 줄이면: **URL이 내용을 식별하면 영원히, 아니면 짧게 확인하며, 개인화면 저장하지 않는다.**

---

## 무효화(purge)와 버저닝의 결정적 차이

"CDN 캐시를 비우면 되지 않나요?"

```text
purge      CDN API로 엣지의 사본을 지운다      →  브라우저 캐시는 그대로다
버저닝     URL을 바꾼다                        →  브라우저도 새 URL을 새로 받는다
```

- CDN은 내 것이라 지울 수 있지만, **사용자 브라우저의 캐시는 내 손이 닿지 않는다.** 만료되거나 URL이 바뀌어야만 갱신된다
- 그래서 "HTML은 짧게, 자산은 URL 버저닝으로 길게" 조합이 표준이다. 브라우저 캐시를 지울 방법이 없다는 전제 위에 세운 설계다
- purge는 HTML처럼 URL을 못 바꾸는 응답을 **CDN 쪽에서만** 급히 갱신할 때 쓴다. 브라우저 쪽은 `no-cache`가 알아서 확인한다

<!-- "사용자에게 캐시 지워 달라고 안내"는 해법이 아니다. 그 안내가 필요한 시점에 설계가 틀린 것이다. -->

---

## CloudFront 무효화(invalidation)

```bash
aws cloudfront create-invalidation --distribution-id E1ABC2DEF3 \
  --paths "/index.html" "/images/*"
```

- 엣지의 사본을 지운다. 다음 요청은 오리진으로 간다. 반영까지 보통 수십 초에서 몇 분
- 요금은 **경로 수** 기준이다. 한 달 1,000경로까지 무료, 넘으면 경로당 과금. `/images/*`처럼 와일드카드 한 줄은 파일이 수천 개여도 **1경로**로 센다
- 그래서 `--paths "/*"` 한 줄이 파일 열 개를 나열하는 것보다 싸다

AWS 문서도 무효화보다 **버전 파일명을 우선 쓰라**고 권한다. 이유는 앞 슬라이드와 같다. 무효화는 사용자 브라우저와 회사 프록시의 사본을 못 건드린다. 배포 파이프라인에서 무효화가 필요한 대상은 `index.html` 같은 고정 URL뿐이다.

---

## S3 설정 : 파일에 캐시 규칙을 붙여 저장한다

**S3는 파일 저장소, CloudFront는 그 앞에서 사본을 제공하는 CDN이다.** 먼저 원본 파일을 올릴 때 응답에 사용할 헤더를 정한다.

```bash
aws s3 cp build/static/ s3://my-bucket/static/ --recursive \
  --cache-control "public, max-age=31536000, immutable"
aws s3 cp build/index.html s3://my-bucket/index.html \
  --cache-control "no-cache"
```

- 첫 명령: **해시 파일명인 JS/CSS**를 `/static/`에 올리면서 1년 캐시를 지정한다
- 둘째 명령: **고정 이름인 HTML**은 저장하되, 다시 쓸 때 최신인지 확인하게 한다
- `--cache-control` 값은 **S3 객체의 메타데이터**로 저장되고, S3가 파일을 응답할 때 HTTP 헤더로 나간다

**아직 절반이다.** 이제 CloudFront도 파일 종류에 맞는 정책을 쓰도록 설정해야 한다.

<!-- build/static/에는 내용을 바꾸면 파일명도 달라지는 공개 자산만 있다고 가정한다. 두 명령은 예시이며 실제 AWS에 실행하지 않는다. S3 메타데이터를 바꿔도 이미 CloudFront나 브라우저에 저장된 사본이 즉시 바뀌지는 않는다. 고정 URL은 기존 TTL과 필요 시 CDN 무효화를 함께 고려한다. 출처: https://docs.aws.amazon.com/cli/latest/reference/s3/cp.html , https://docs.aws.amazon.com/AmazonS3/latest/userguide/UsingMetadata.html -->

---

## CloudFront 설정 : URL 경로별로 규칙을 연결한다

CloudFront의 **Behavior(동작)는 URL 경로별 규칙**이다. 요청을 보낼 원본 서버와 적용할 캐시 정책을 여기서 고른다.

| 요청 경로        | 원본 서버     | 연결할 캐시 정책과 이유                                     |
| ---------------- | ------------- | ----------------------------------------------------------- |
| `/static/*`      | S3            | `CachingOptimized` — 해시 파일의 긴 캐시 활용               |
| `/api/*`         | API 서버(ALB) | `CachingDisabled` — 개인화 응답의 CDN 캐시 끄기             |
| `*` (기본, HTML) | S3            | 사용자 지정: Min TTL `0`, Default TTL `0` — `no-cache` 존중 |

예를 들어 `/index.html`은 기본 규칙 → S3로 가고, S3의 `no-cache`에 따라 **CDN도 사본을 다시 쓰기 전에 확인**한다.

**헤더가 빠졌을 때는 정책의 Default TTL이 적용된다.** `CachingOptimized`는 24시간, 위 HTML 정책은 0초다. 항상 24시간인 것은 아니다.

<!-- 공개 S3 정적 HTML용 사용자 지정 정책 예시: Min TTL 0, Default TTL 0, Max TTL 31536000, 쿠키·쿼리스트링·뷰어 Host는 캐시 키에서 제외, 필요 시 gzip/brotli 활성화. 쿼리나 헤더별로 내용이 달라지는 응답에는 이 예시를 그대로 쓰지 않는다. 만료 정보가 없는 응답의 CDN 신선도는 Default TTL로 정하며, 브라우저는 별도의 HTTP 캐시 규칙을 따른다. Behavior는 먼저 일치하는 경로가 적용되고 기본 동작은 마지막이다. CachingDisabled는 CDN 캐시만 끄므로 개인화 API 응답의 no-store와 필요한 인증 정보의 오리진 전달 설정도 별도로 유지한다. 정적 폰트에 출처별 CORS 응답을 쓴다면 CachingOptimized를 그대로 쓰지 말고 Origin을 포함한 캐시 정책을 검토한다. 출처: https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/using-managed-cache-policies.html , https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/DownloadDistValuesCacheBehavior.html -->

---

## Part 1 정리

- 정책은 세 축으로 정한다: 변경 빈도, **URL이 내용을 식별하는가**, 개인화 여부
- HTML은 `no-cache` + `ETag`로 매번 확인하되 본문은 아끼고, CDN에는 `s-maxage` + SWR로 길게 둔다
- JS/CSS/폰트/빌드 이미지는 해시 파일명 + 1년 + `immutable`. 무효화 대신 새 이름을 쓴다
- 업로드 이미지는 새 키를 만들거나 짧은 TTL, 이미지 변형은 `Vary: Accept`로 키가 갈라진다
- 개인화 API는 `private, no-store`. `public`은 인증 응답까지 CDN에 남긴다
- purge는 CDN만 비운다. 브라우저 캐시는 URL 변경이나 만료로만 갱신된다
- S3 + CloudFront는 업로드 때 `--cache-control`, 경로별 동작, 무효화는 `/*` 한 경로. AWS 문서도 버전 파일명을 우선하라고 한다

---

## 중간 점검 1-1

어제 받은 `/main.js`가 브라우저 캐시에 남아 있다. 오늘 **같은 URL에 새 JS를 덮어쓰고 CDN purge도 완료**했다.

```text
Cache-Control: public, max-age=31536000, immutable
```

사용자가 일반 방문으로 다시 들어왔다. 캐시는 아직 신선하다. 어떤 일이 생길 수 있는가?

1. CDN을 비웠으므로 모든 사용자가 새 JS를 받는다
2. 서버 파일이 바뀌었으므로 브라우저가 자동으로 알아챈다
3. 브라우저가 요청을 보내지 않고 어제 JS를 그대로 쓴다
4. 브라우저가 반드시 ETag로 확인하고 새 JS를 받는다

<!-- 같은 캐시 파티션이고 다른 Vary 조건도 같으며 강력 새로고침, DevTools 캐시 비활성화, 서비스 워커 개입은 없다고 가정한다. immutable 자체가 새로고침의 모든 형태를 막는다고 설명하지 않는다. -->

---

## 중간 점검 1-1 정답 : ③

- 브라우저의 사본은 아직 신선하다. **CDN까지 요청을 보내지 않을 수 있으므로** purge 여부도 알 수 없다
- ETag는 확인 요청을 보낼 때 쓰는 검증자다. 서버의 변경을 브라우저에 알려 주는 알림 장치가 아니다

**이렇게 바꾸자**

- 새 JS는 **`main.새해시.js`**로 올리고, HTML이 그 URL을 참조하게 한다. 새 URL이면 브라우저도 새 파일을 받는다
- HTML은 **`no-cache` + ETag**로 최신 참조를 확인하게 한다. 이전 JS는 이미 열린 화면을 위해 보존한다

같은 이름을 꼭 유지해야 한다면 다음부터 짧은 TTL이나 재검증 정책을 쓴다. **지금 서버 헤더만 바꿔도 이미 저장된 1년짜리 사본은 갱신되지 않는다.**

---

## 중간 점검 1-2

`shop.example`이 `cdn.example`의 폰트를 불러온다. 응답에 **`Access-Control-Allow-Origin`이 없고**, 콘솔에는 CORS 오류가 나온다.

```text
HTTP/2 200
Content-Type: font/woff2
Cache-Control: public, max-age=31536000, immutable
```

CDN에는 파일이 있는데 폰트가 적용되지 않는다. **이 응답에서 먼저 고칠 것은?**

1. `max-age`를 0으로 줄인다
2. 폰트 응답에 `shop.example`을 허용하는 CORS 헤더를 넣는다
3. `immutable`을 지운다
4. HTML에 폰트 preload만 추가한다

---

## 중간 점검 1-2 정답 : ②

**파일을 받았는지와 그 출처에서 사용할 수 있는지는 별개다.** `200`이나 CDN HIT만으로 CORS 검사까지 통과한 것은 아니다.

**이렇게 바꾸자**

```text
Access-Control-Allow-Origin: https://shop.example
```

- 폰트 응답에 허용 헤더를 넣는다. 모든 사이트에 공개할 폰트라면 `*`도 가능하다
- 허용 출처별로 응답이 달라지면 **`Vary: Origin`과 CloudFront의 `Origin` 캐시 키**도 맞춘다. 서로 다른 허용 응답이 섞이는 것을 막는다
- CDN에 헤더 없는 사본이 남아 있다면 갱신하고 다시 확인한다. **해시 파일의 긴 TTL은 유지**해도 된다

<!-- CORS 헤더는 응답 헤더 정책이나 오리진에서 설정할 수 있다. 요청 Origin을 무조건 반사하지 말고 허용 목록을 확인한다. CDN purge는 브라우저의 기존 캐시를 지우지 않으므로, 저장 상태에 따라 검증용 캐시 비활성화나 새 파일 URL이 필요할 수 있다. 출처: https://www.w3.org/TR/css-fonts-3/#font-fetching-requirements , https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/controlling-the-cache-key.html -->

---

## 중간 점검 1-3

공개 HTML인 `/index.html`을 S3에 올렸고, 매번 최신인지 확인하게 하고 싶다.

```text
S3 응답                 Cache-Control: no-cache
CloudFront HTML 정책    Minimum TTL: 60초
```

배포 직후에도 CDN이 확인 없이 이전 HTML을 줄 수 있다. **어떤 설정이 목적에 맞는가?**

1. S3의 `no-cache`를 `immutable`로 바꾼다
2. CloudFront의 Minimum TTL을 1년으로 늘린다
3. 배포할 때마다 purge만 하고 정책은 그대로 둔다
4. HTML의 Minimum TTL을 0으로 바꾸고 `no-cache`를 유지한다

---

## 중간 점검 1-3 정답 : ④

- **Minimum TTL이 60초면**, CloudFront는 오리진의 `no-cache`보다 그 최소 신선 기간을 우선한다
- **Minimum TTL을 0으로 바꿔야** HTML의 `no-cache`에 따라 재사용 전에 확인할 수 있다

**이렇게 바꾸자**

- HTML 전용 정책은 **Min TTL 0, Default TTL 0**으로 둔다. 헤더가 빠졌을 때도 의도치 않은 긴 캐시를 피한다
- **ETag를 유지**하면 내용이 그대로일 때 `304`로 확인하고 기존 본문을 재사용할 수 있다
- 해시 JS/CSS는 별도 경로에서 긴 캐시를 유지한다. HTML 때문에 모든 자산의 캐시를 끌 필요는 없다

**확인:** 배포 후 HTML이 새 JS URL을 참조하는지 보고, 변경이 없을 때 조건부 요청이 `304`를 받는지도 확인한다.

<!-- Minimum TTL 0 자체가 재검증을 강제하는 것은 아니다. 이 문제는 no-cache 응답과 결합했을 때를 묻는다. Default TTL은 응답에 유효 기간 정보가 없는 경우의 기본값이며 no-cache를 덮어쓰는 값이 아니다. 기존 캐시가 정책 변경만으로 즉시 모두 사라진다고 가정하지 말고 필요 시 CDN 무효화로 전환한다. 출처: https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/Expiration.html , https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/using-managed-cache-policies.html -->

---

## Part 2 : 함정과 도구

<!-- _class: invert -->

---

## 함정 1 : 개인화 응답이 공유 캐시에 남는다

2015년 12월 25일, Steam 상점이 DDoS 대응 중 캐시 설정을 잘못 적용해 약 한 시간 동안 **다른 사용자의 계정 페이지(이메일, 주소 일부, 구매 내역)** 를 보여 준 사고가 있었다. Valve가 공식 설명을 냈다.

메커니즘은 단순하다.

```text
GET /account  (사용자 A, 쿠키 있음)  →  오리진 응답, 헤더에 private 없음
                                     →  CDN이 저장
GET /account  (사용자 B)              →  CDN HIT, A의 페이지가 B에게
```

막는 규칙도 단순하다.

- 사용자별 응답에는 **반드시 `private` 또는 `no-store`**. 프레임워크가 세션을 읽는 페이지에 자동으로 붙여 주는지 확인한다
- CloudFront에서 이 사고가 나는 조합은 **오리진 요청 정책으로 쿠키를 전달하면서 캐시 정책에는 쿠키를 안 넣는 것**이다. 캐시는 되는데 키에 사용자가 없다

---

## 함정 2 : 새로고침은 한 종류가 아니다

같은 페이지를 다시 여는 방법이 넷이고, 브라우저는 각각 다르게 요청한다.

| 동작                        | 메인 HTML                 | JS/CSS/이미지              |
| --------------------------- | ------------------------- | -------------------------- |
| 링크 클릭, 주소창 Enter     | 캐시 규칙대로             | 캐시 규칙대로              |
| 일반 새로고침 (Cmd+R)       | 재검증 (`max-age=0`)      | 캐시 규칙대로 (Chrome 54+) |
| 강력 새로고침 (Cmd+Shift+R) | `Cache-Control: no-cache` | 전부 `no-cache`, 캐시 무시 |
| DevTools "Disable cache"    | 캐시 무시                 | 캐시 무시 (열려 있는 동안) |

- "제 브라우저에선 되는데요"의 절반은 DevTools를 열어 둔 채 테스트해서 그렇다. 사용자는 DevTools가 없다
- Chrome이 일반 새로고침에서 서브리소스 재검증을 끊은 것이 `immutable`을 구현하지 않은 이유다. 대신 Firefox/Safari는 `immutable`이 있어야 이 동작을 한다

<!-- 재현할 때는 시크릿 창 + DevTools 닫고 + 일반 새로고침으로 사용자 조건을 맞춘다. -->

---

## 함정 3 : DevTools Network 탭 읽기

| Status | Size 열            | 뜻                                                        |
| ------ | ------------------ | --------------------------------------------------------- |
| 200    | `(memory cache)`   | 이 탭이 이미 메모리에 들고 있던 것. 네트워크 안 나감      |
| 200    | `(disk cache)`     | 디스크 캐시에서 읽음. fresh라 서버에 묻지 않았다          |
| 304    | 수백 B             | 서버에 물었고 "같다"고 해서 본문은 캐시에서 씀            |
| 200    | 전체 크기          | 실제로 다시 받았다. 캐시 없음, 만료, 또는 검증자 없음     |
| 200    | `(prefetch cache)` | `<link rel=prefetch>`나 라우터 프리페치로 미리 받아 둔 것 |

확인하는 순서를 정해 두면 빠르다.

1. Size 열이 캐시 표시인가 → 그렇다면 헤더는 **저장 당시** 것이다
2. 200 전체 크기인데 캐시를 기대했다면 → 응답 헤더의 `Cache-Control`, `Age`, 검증자 유무
3. CDN이 개입했나 → `x-vercel-cache`, `cf-cache-status`, `x-cache`

---

## 함정 4 : 브라우저가 보는 헤더가 CDN이 따른 규칙은 아니다

같은 사이트의 HTML이다.

```text
$ curl -sI https://yceffort.kr/
cache-control: public, max-age=0, must-revalidate
age: 2945
x-vercel-cache: HIT
x-nextjs-stale-time: 300
```

- 브라우저 입장: `max-age=0`이니 받자마자 낡았다. 다음 방문 때 반드시 서버에 확인한다
- 그런데 `age: 2945`, `HIT`. CDN은 49분째 이 사본을 쓰고 있다

오리진(Next.js)이 내보낸 헤더에는 `s-maxage=300, stale-while-revalidate`가 있었다. Vercel CDN은 그 값을 **자기 규칙으로 쓰고 브라우저에는 지운 뒤** 보낸다(벤더 문서에 명시된 동작).

> DevTools의 `Cache-Control`은 **브라우저용 최종본**이다. CDN이 따른 규칙은 `Age`와 HIT/MISS 헤더로 역추적한다.

---

## 함정 5 : 공용 CDN의 라이브러리는 더 이상 공짜가 아니다

"jQuery를 `cdnjs`에서 불러오면 다른 사이트에서 이미 받아 둔 캐시를 재사용한다"는 조언이 오래 돌았다.

- Chrome 86(2020), Firefox 85, Safari(더 이전)부터 브라우저 캐시 키에 **최상위 사이트**가 들어간다(캐시 파티셔닝). `a.com`에서 받은 `cdnjs.../jquery.js`와 `b.com`에서 받은 같은 파일은 **다른 캐시 항목**이다
- 이유는 프라이버시다. 캐시 적중 여부로 사용자가 어떤 사이트를 방문했는지 알아낼 수 있었다

남는 건 비용뿐이다. 외부 도메인이니 DNS + TLS가 한 번 더 들고, 그 CDN 장애가 내 사이트 장애가 된다. 라이브러리는 **번들에 넣어 내 정적 자산 도메인에서** 서빙한다.

---

## 함정 6 : bfcache는 HTTP 캐시가 아니다

뒤로가기를 눌렀을 때 페이지가 즉시 뜨는 것은 HTTP 캐시가 아니라 **bfcache(back/forward cache)** 다. 페이지 전체(DOM, JS 힙)를 메모리에 얼려 두었다가 그대로 되살린다.

- HTTP 캐시가 아무리 잘 돼 있어도 bfcache에 못 들어가면 뒤로가기가 느리다
- 못 들어가는 대표 원인: `unload` 이벤트 핸들러, 열린 `IndexedDB` 트랜잭션, 그리고 **`Cache-Control: no-store`가 붙은 HTML** (Chrome은 최근 일부 조건에서 허용하기 시작했지만 여전히 보수적이다)
- 확인은 DevTools Application 탭 → Back/forward cache → Test back/forward cache

HTML에 습관적으로 `no-store`를 붙이면 HTTP 캐시가 아니라 bfcache를 잃는다. 개인정보가 없는 페이지라면 `no-cache`가 맞다.

---

## 함정 7 : 서비스 워커는 세 번째 캐시다

```text
[페이지] ──▶ [서비스 워커 fetch 핸들러] ──▶ [HTTP 캐시] ──▶ [CDN] ──▶ [오리진]
```

- 서비스 워커가 등록돼 있으면 모든 요청이 **HTTP 캐시보다 먼저** 워커의 `fetch` 이벤트를 지난다. 워커가 `caches.match()`로 응답하면 HTTP 캐시 규칙은 아예 적용되지 않는다
- 반대로 워커 안에서 `fetch()`를 호출하면 그때는 HTTP 캐시를 정상적으로 탄다. 그래서 워커의 Cache API에 넣은 응답이 **HTTP 캐시에서 온 낡은 사본**일 수 있다
- 워커 스크립트(`sw.js`) 자체는 브라우저가 24시간마다 강제로 재검증한다. 여기에 1년 캐시를 줘도 24시간 뒤에는 확인한다

"캐시를 다 지웠는데 옛 화면"이라면 Application 탭에서 서비스 워커를 먼저 본다. 이 주제는 별도 시리즈(서비스 워커 캐싱 딥다이브)에서 다룬다.

---

## 함정 8 : Vary와 쿠키가 캐시를 조용히 죽인다

의도치 않게 HIT율을 0으로 만드는 헤더들이다.

```text
Vary: User-Agent          UA 문자열 수만큼 사본이 갈라진다. 실질적으로 캐시 불가
Vary: Cookie              사용자마다 사본. 공유 캐시가 무의미해진다
Vary: *                   "어떤 요청도 같지 않다". 명세상 캐시 불가
Set-Cookie: ...           대부분의 CDN이 이 응답을 저장 대상에서 뺀다
```

- 정적 자산 응답에 세션 쿠키가 실려 나가는 서버 설정(세션 미들웨어가 전 경로에 걸린 Express 앱 등)은 CDN HIT율을 그대로 깎는다
- 디바이스별 응답이 필요하면 `Vary: User-Agent` 대신 CDN의 디바이스 분류 헤더(예: CloudFront의 `CloudFront-Is-Mobile-Viewer`)로 키를 두세 가지로 제한한다

---

## Part 2 정리

- 개인화 응답에 `private`/`no-store`가 빠지면 공유 캐시가 다른 사용자에게 준다. Steam 2015가 그 사례다
- 새로고침은 네 종류다. 일반 새로고침은 Chrome에서 서브리소스를 재검증하지 않고, DevTools "Disable cache"는 사용자 조건이 아니다
- Network 탭은 Size 열부터 본다. `(disk cache)`면 헤더는 저장 당시 것이다
- 공용 CDN 캐시 공유는 파티셔닝으로 끝났다. bfcache는 HTTP 캐시와 별개고 `no-store`가 막는다
- 서비스 워커는 HTTP 캐시 앞에 선다. `Vary: User-Agent`, `Vary: Cookie`, 정적 자산의 `Set-Cookie`는 HIT율을 없앤다

---

## 중간 점검 2

사용자가 "배포했는데 옛날 화면"이라고 한다. 다음 중 **가장 먼저** 확인할 것은?

1. CDN purge를 실행한다
2. 사용자에게 강력 새로고침을 안내한다
3. 사용자 브라우저에서 HTML 응답의 Size 열과 `Cache-Control`, 그리고 서비스 워커 등록 여부를 본다
4. `max-age`를 전부 0으로 바꿔 재배포한다

---

## 중간 점검 2 정답 : ③

- ①은 브라우저 캐시를 못 지우고, ②는 그 사용자 한 명만 고치며, ④는 문제를 못 찾은 채 성능만 버린다
- 진단 순서: HTML이 `(disk cache)`면 HTML 캐시 정책 문제(휴리스틱 캐시 가능성), `200` 전체 크기인데 옛 화면이면 CDN 사본(`Age`, HIT) 문제, 둘 다 아니면 서비스 워커가 응답한 것

> 원칙: 캐시 문제는 **어느 캐시**인지 먼저 특정한다. 브라우저, CDN, 서비스 워커 중 하나다.

---

## Part 3 : 실무에서는 이렇게

<!-- _class: invert -->

---

## 새 프로젝트를 시작할 때 정하는 기본값

**먼저 프레임워크와 번들러가 이미 해 둔 것을 확인한다.** Next.js, Vite, CRA는 전부 해시 파일명을 만들고, Vercel이나 Amplify 같은 플랫폼은 헤더까지 붙인다. 이걸 덮어쓰는 설정이 없는지부터 본다.

직접 배포한다면 이 네 줄을 인프라 코드(CloudFront 동작, nginx 설정)에 처음부터 넣는다.

```text
해시 파일명 자산  (/static/*, /_next/static/*)   public, max-age=31536000, immutable
HTML                                             no-cache  + ETag
API 기본값                                        private, no-store
공개 API만 예외                                   public, s-maxage=N,
                                               stale-while-revalidate=M
```

- 정적 자산과 API는 **경로 접두사로 분리**해 둔다. CDN 동작을 경로 패턴으로 나누는 것이 전부 여기서 시작된다
- 캐시 정책의 **키에 무엇이 들어가는지**(쿼리스트링, 쿠키, 헤더)를 문서에 한 줄로 적어 둔다. 6개월 뒤의 내가 가장 먼저 찾는 정보다

---

## 배포 파이프라인에 넣는 순서

```text
1. 자산 업로드      해시 파일명 파일을 먼저 올린다. 이전 배포 파일은 지우지 않는다 (--delete 금지)
2. HTML 교체        새 자산을 참조하는 HTML을 올린다
3. 무효화           고정 URL만 (index.html, /). 자산은 무효화할 필요가 없다
4. 검증             curl -sI 로 HTML, JS, 이미지 하나씩 헤더를 찍어 파이프라인 로그에 남긴다
5. 정리             N일 지난 이전 배포 자산을 별도 작업으로 지운다
```

배포 스크립트 끝에 이런 확인을 붙여 두면 "헤더가 어느 순간 사라졌다"를 그날 안다.

```bash
curl -sI https://example.com/ | grep -i cache-control     # no-cache 여야 한다
curl -sI https://example.com/static/main.abc123.js \
  | grep -i cache-control                                 # immutable 이어야 한다
```

앱 쪽에는 **청크 로드 실패 시 한 번 새로고침**하는 코드를 둔다. 배포 순서를 지켜도 오래 열어 둔 탭은 남는다.

---

## 코드 리뷰에서 보는 것

캐시 헤더가 들어간 PR에서 묻는다. 하나라도 "예"면 머지 전에 고친다.

| 질문                                           | 왜                                         |
| ---------------------------------------------- | ------------------------------------------ |
| 해시 없는 파일명에 1년 캐시를 붙였나           | 배포가 사용자에게 도달하지 않는다 (Part 1) |
| 사용자별 응답에 `public`이나 `s-maxage`가 있나 | 다른 사용자에게 나간다 (Part 2)            |
| `Vary: User-Agent`, `Vary: Cookie`를 붙였나    | HIT율이 0에 가까워진다                     |
| 정적 자산 응답에 세션 쿠키가 실려 나가나       | CDN HIT율이 무너진다                       |
| 업로드 이미지 URL이 고정인데 교체가 가능한가   | 교체해도 옛 이미지가 보인다                |
| HTML에 습관적으로 `no-store`를 붙였나          | bfcache를 잃는다 (Part 2)                  |

헤더 값을 외울 필요는 없다. **"이 URL의 내용이 바뀔 수 있는가"** 하나만 물으면 대부분 걸린다.

---

## "옛날 화면이 보여요"를 받았을 때

순서를 지키면 30분 안에 어느 캐시인지 특정된다. 순서를 건너뛰고 purge부터 하면 원인을 영영 모른다.

```text
1. 재현 조건 맞추기     시크릿 창, DevTools 닫기, 일반 새로고침. 사용자와 같은 조건인지 확인
2. 브라우저 캐시        Network 탭 Size 열. (disk cache) 면 브라우저,
                     HTML 응답의 Cache-Control 확인
3. CDN 캐시             curl -sI 로 Age 와 HIT/MISS. Age 가 크고 HIT 면 CDN 사본
4. 서비스 워커          Application 탭 → Service Workers.
                     등록돼 있으면 여기가 응답했을 가능성
5. 그 다음에 purge      어느 캐시인지 알았을 때만. purge 는 CDN 만 비운다는 것을 기억
```

- 사용자에게 "캐시를 지워 보세요"라고 안내하게 됐다면, 그 자체가 **설계가 틀렸다는 신호**다. 안내가 아니라 URL 버저닝을 고친다
- 재발 방지는 헤더 수정이 아니라 **배포 파이프라인의 검증 단계**(앞 슬라이드)에 넣는다

---

## 유지하기 : 무엇을 보고, 누구에게 무엇을 묻나

**대시보드에서 보는 숫자**

- **CDN HIT율** : 정적 자산은 95% 이상이 정상이다. 갑자기 떨어지면 캐시 키가 넓어졌다(쿼리스트링, 헤더 추가)는 뜻
- **오리진 요청 수** : HIT율의 거울. 배포 직후 잠깐 오르는 것은 정상, 계속 높으면 TTL이 짧거나 키가 쪼개진 것
- **이미지 변형 수** : `sizes`, `deviceSizes`를 늘릴 때마다 첫 사용자가 MISS를 맞는다

**인프라 담당에게 묻는 것**

- 캐시 정책의 **키에 무엇이 들어가나** (쿼리스트링을 통째로 무시하나, 특정 파라미터만 보나)
- **`Set-Cookie`가 있는 응답**을 저장하나, 쿠키를 오리진에 전달하면서 키에는 안 넣는 동작이 있나
- **무효화 권한과 자동화** : 배포 파이프라인이 `index.html`만 무효화할 수 있나
- 브라우저에 가는 `Cache-Control`을 **CDN이 고쳐서 보내나** (Vercel처럼 `s-maxage`를 벗기는지)

---

## 종합 퀴즈

<!-- _class: invert -->

이미지 교체, 배포 직후 오류, 인증 응답, 캐시 진단.

**네 문제**로 정책 선택과 문제 해결을 연습한다.

---

## 퀴즈 1 : 이미지 교체 (Part 1)

프로필 이미지 URL이 `/uploads/user-42.jpg`로 고정이고 응답은 `Cache-Control: public, max-age=86400`이다. 사용자가 이미지를 바꿨는데 하루 동안 옛 이미지가 보인다. **가장 근본적인** 해법은?

1. `max-age`를 60으로 줄인다
2. 업로드할 때 CDN purge를 호출한다
3. 업로드마다 새 파일 키(`/uploads/user-42-8f3a1c.jpg`)를 만들고 프로필 데이터에 그 URL을 저장한다
4. 응답에 `Vary: Cookie`를 붙인다

---

## 퀴즈 1 정답 : ③

- ①은 CDN HIT율을 깎으면서도 최대 1분은 여전히 옛 이미지다
- ②는 CDN만 비운다. 사용자 브라우저의 하루짜리 사본은 남는다
- ④는 사용자마다 사본을 쪼개 캐시를 무력화할 뿐 갱신과 무관하다
- ③은 URL이 내용을 식별하게 만든다. 그러면 `max-age`를 1년으로 늘려도 안전하다

> 브라우저 캐시를 갱신하는 방법은 만료와 URL 변경뿐이다.

---

## 퀴즈 2 : 배포 직후 오류 (Part 1)

SPA를 S3 + CloudFront로 배포한다. 빌드 스크립트가 `aws s3 sync --delete`로 이전 파일을 전부 지우고 새 파일을 올린다. HTML은 `no-cache`, JS는 해시 파일명 + `immutable`이다.

배포 직후 **이미 사이트를 열어 둔** 사용자에게 무슨 일이 생기는가? 헤더는 올바른데 왜 그런가?

---

## 퀴즈 2 정답 : 페이지 이동 시 `ChunkLoadError`

- 열어 둔 화면은 옛 HTML이 참조하는 옛 청크 이름을 알고 있다. 아직 안 받은 청크를 요청하는 순간 `--delete`로 지워진 파일이라 404다
- 헤더는 옳다. 문제는 **옛 배포의 자산을 지운 것**이다. 해시 파일명은 "새 이름을 추가"하는 전략이지 "옛 이름을 지워도 되는" 전략이 아니다
- 해법: `--delete`를 빼고 N일 뒤 정리, 자산 먼저 올리고 HTML 나중에, 청크 로드 실패 시 새로고침 유도

---

## 퀴즈 3 : 공유 캐시와 인증 (Part 1, 2)

`GET /api/orders`는 `Authorization: Bearer ...` 헤더로 인증한다. 응답 헤더는 다음과 같다.

```text
Cache-Control: public, max-age=60
```

CDN이 이 응답을 저장하는가? 저장한다면 무슨 일이 생기는가?

---

## 퀴즈 3 정답 : 저장한다. 다른 사용자의 주문 내역이 나간다

- RFC 9111은 `Authorization` 요청의 응답을 공유 캐시가 저장하지 않도록 하지만, **`public`, `s-maxage`, `must-revalidate`가 있으면 예외**다. 여기엔 `public`이 있다
- 캐시 키는 URL이다. `Authorization` 값은 `Vary`에 없으니 키에 안 들어간다. 60초 동안 첫 사용자의 응답이 모두에게 간다
- 고치는 법: `private, no-store`. 정말 CDN에 두고 싶다면 `Authorization`을 캐시 키에 넣고 HIT율을 포기한다
- CloudFront라면 GET의 `Authorization`이 기본으로 제거돼 오리진이 401을 내고, 그 401이 60초 캐시된다. 유출은 아니지만 장애다

> `public`은 "공개해도 되는 데이터"라는 뜻으로 붙이는 단어가 아니다. "공유 캐시에 저장하라"는 지시다.

---

## 퀴즈 4 : 진단 (Part 2)

DevTools Network 탭에서 `main.a1b2c3.js` 행이 이렇게 보인다.

```text
Status 200   Size 412 kB   Time 850 ms
응답 헤더: cache-control: public, max-age=31536000, immutable
           age: 0
           x-vercel-cache: MISS
```

이 사용자는 두 번째 방문이고, 파일은 어제 배포된 것 그대로다. 세 개의 캐시 중 무엇이 빠졌고, 무엇이 빠지지 않았나?

---

## 퀴즈 4 정답 : 브라우저 캐시도 CDN 캐시도 비어 있었다

- Size가 `(disk cache)`가 아니라 실제 크기이니 **브라우저 캐시에 없었다.** 두 번째 방문인데도 없다면 DevTools "Disable cache"가 켜져 있거나, 시크릿 창이거나, 사용자가 캐시를 지웠거나, 어제 방문 때 이 청크를 로드하지 않은 경로였다
- `MISS`, `age: 0`이니 **이 엣지에도 없었다.** 어제 배포 이후 이 지역에서 아무도 이 파일을 요청하지 않았거나, 엣지가 용량 정책으로 내보냈다
- 헤더 자체는 정상이다. 이 요청이 사본을 채웠으니 다음부터는 양쪽 다 HIT다

> 850ms 중 얼마가 캐시 부재 때문인지는 Timing 탭의 Waiting(TTFB)이 답한다. 헤더만 보고 "캐시가 안 된다"고 결론내지 않는다.

---

## 2편 정리 : 정책부터 진단까지

1. **URL이 내용을 식별하는 자산**은 해시 파일명 + 긴 `max-age` + `immutable`로 둔다
2. **고정 URL의 공통 HTML**은 `no-cache` + ETag로 최신 참조를 확인한다. CDN의 TTL 정책도 맞춘다
3. **개인화 응답**은 저장 요구사항에 따라 `private`이나 `no-store`를 선택하고, 공유 캐시 정책을 함께 점검한다
4. **배포는 자산 먼저, HTML 나중에.** 이전 자산을 보존하고, CDN purge가 브라우저 캐시까지 지우지는 않는다는 점을 기억한다
5. **옛 화면 신고는 어느 캐시인지부터.** DevTools의 Size·응답 헤더와 서비스 워커 개입 여부를 보고 원인을 좁힌다

---

## 참고 자료

- [RFC 9111 : HTTP Caching](https://www.rfc-editor.org/rfc/rfc9111) (+ [RFC 8246 immutable](https://www.rfc-editor.org/rfc/rfc8246), [RFC 5861 stale-while-revalidate](https://www.rfc-editor.org/rfc/rfc5861)) : 저장 조건, 신선도, `Authorization` 예외의 원문
- [MDN : HTTP caching](https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/Caching) : 지시어별 브라우저 동작 정리
- Chrome : [Reload, reloaded](https://blog.chromium.org/2017/01/reload-reloaded-faster-and-leaner-page_26.html), [Partitioning the HTTP cache](https://developer.chrome.com/blog/http-cache-partitioning), [bfcache](https://web.dev/articles/bfcache)
- [CloudFront : Manage how long content stays in the cache](https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/Expiration.html) : TTL 합성 표와 Min TTL 경고문
- CloudFront : [Managed cache policies](https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/using-managed-cache-policies.html), [Request and response behavior](https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/RequestAndResponseBehaviorCustomOrigin.html), [Invalidate files](https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/Invalidation.html)
- [Vercel Docs : Edge Cache](https://vercel.com/docs/edge-cache) : `s-maxage`를 브라우저 응답에서 제거하는 동작
- [Steam : Update on Christmas Issues (2015)](https://store.steampowered.com/news/post/1978855) : 캐시 설정 오류 사고의 공식 설명

---

# 감사합니다

<!-- _class: invert -->

[1편: 원리와 규칙 다시 보기](/slides/cdn-and-http-cache)

@yceffort
