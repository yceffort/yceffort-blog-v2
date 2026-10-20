---
title: 'CDN과 HTTP 캐시 1: 원리와 규칙'
marp: true
paginate: true
theme: midnight
tags:
  - caching
  - networking
  - web-performance
date: 2026-09-11
description: 'CDN의 동작과 HTTP 캐시 규칙을 익히고, 우리 서비스의 응답 헤더로 저장·신선도·재검증을 해석하는 강의'
published: true
art:
  undraw: connected-world
---

# CDN과 HTTP 캐시 1

원리와 규칙: 응답 헤더를 읽는 법

<!-- _class: invert deck-cover -->

@yceffort

---

## 배포 직후에 자주 받는 질문 세 가지

1. "배포했는데 사용자 화면은 아직 옛날 버전이에요"
2. "일부 사용자만 `ChunkLoadError`가 나요. 새로고침하면 돼요"
3. "이미지를 교체했는데 계속 옛 이미지가 나와요. 파일명은 같아요"

세 질문의 답은 전부 같은 곳에 있다. **파일이 사용자에게 오는 길 어딘가에 사본이 남아 있다.**

<!-- 손들기: 셋 중 하나라도 직접 겪어 본 사람? 거의 전원이 든다. 그 사본이 어디에, 왜, 얼마나 남는지가 오늘 주제. -->

---

## 1편의 목표

```text
200  (disk cache)  main-2kzkiswg.js  age: 151772  x-vercel-cache: HIT
304  0 B           /                 cache-control: max-age=0, must-revalidate
```

DevTools Network 탭의 이런 한 줄을 읽고 **어느 캐시가 응답했고, 왜 재사용하거나 확인했는지 설명할 수 있게** 되는 것.

마지막에는 **우리 서비스의 HTML·JS·API·이미지 헤더**를 직접 해석한다.

자산별 정책을 정하고 배포 문제를 해결하는 방법은 [2편](/slides/cdn-and-http-cache-2)에서 이어 간다.

---

## 1편에서 다루는 것

1. **CDN** : 왜 필요하고, 요청이 어떻게 가까운 서버로 가고, 그 서버가 무엇을 하는가
2. **HTTP 캐시 규칙** : `Cache-Control` 지시어, 신선도 계산, 재검증, `Vary`

각 부 끝에 **중간 점검**이 있다. 2부에서는 개념 확인 두 문제를 풀고, **우리 서비스 헤더 퀴즈 네 문제**로 마무리한다.

[2편: 자산별 전략과 문제 해결](/slides/cdn-and-http-cache-2)에서는 이 규칙을 실제 배포와 운영에 적용한다.

---

## 기준

- 캐시 규칙은 **RFC 9111 (HTTP Caching)** 기준. `immutable`은 RFC 8246, `stale-while-revalidate`는 RFC 5861
- `yceffort.kr`(Next.js 16.3.1, Vercel) 실측은 **2026-09-11**, 중간 점검 2의 우리 서비스 실측은 **2026-09-16**에 `curl`로 직접 GET 요청한 결과다. 각 예시에 응답 상태와 확인 범위를 표시한다
- AWS 동작은 **CloudFront 개발자 가이드**에서 확인했고, 실측 헤더는 `awsstatic.com`(S3 + CloudFront)에서 받았다
- 브라우저 동작은 **Chrome** 기준으로 설명하고, Firefox/Safari가 다른 지점은 따로 표시한다
- 다루지 않는 것: 서비스 워커의 Cache API, react-query 같은 애플리케이션 캐시, CDN 벤더별 설정 화면

<!-- 구두 보충: CDN 벤더(Cloudflare, CloudFront, Vercel)는 설정 UI만 다르고 아래 규칙은 전부 같은 RFC를 따른다. 규칙을 알면 어느 벤더 문서든 읽힌다. -->

---

## Part 1 : CDN은 무엇을 해결하는가

<!-- _class: invert -->

---

## 요청 하나가 이동하는 거리

서울에서 미국 버지니아까지 약 11,000km, 광섬유 속 빛은 약 200,000km/s다.

```text
편도 이론값     55ms
왕복(RTT) 이론값 110ms
실제 RTT       180 ~ 200ms   (라우터, 경로 우회, 큐잉)
```

게다가 HTTPS는 첫 바이트까지 왕복이 한 번이 아니다.

```text
TCP 핸드셰이크   1 RTT   유실과 순서 꼬임에 대비해 양쪽이 순번을 맞춘다
TLS 핸드셰이크   1 RTT   서버 인증서를 확인하고 암호 키를 합의한다 (TLS 1.3)
HTTP 요청/응답   1 RTT   그제야 실제 요청이 나간다
─────────────────────
첫 바이트까지    3 RTT ≈ 550ms
```

이게 HTML 한 장 값이고, 그 HTML이 부르는 JS와 CSS가 다시 낸다.

---

## 같은 요청을 가까운 곳에서 받으면

서울 사무실에서 `yceffort.kr`의 JS 청크 하나를 요청한 `curl` 타이밍이다.

```text
dns=0.002  connect=0.007  tls=0.025  ttfb=0.065  total=0.144
```

- TCP 연결 **7ms**, TLS까지 **25ms**. 서버가 물리적으로 가깝다는 뜻이다
- 이 서버는 내 파일을 만든 서버(오리진)가 아니라, 오리진의 **사본을 들고 있는 서울의 서버**다

CDN(Content Delivery Network)은 이렇게 **전 세계에 흩어진 캐시 서버들의 네트워크**다. 각 서버를 엣지(edge) 또는 PoP(Point of Presence)라고 부른다.

---

## 요청은 어떻게 가까운 엣지로 가는가

같은 도메인인데 서울 사용자는 서울 엣지로, 도쿄 사용자는 도쿄 엣지로 간다. 방법은 둘이다.

- **DNS 기반** : DNS는 도메인 이름을 IP 주소로 바꿔 주는 단계다. 이때 물어본 리졸버의 위치를 보고 지역마다 다른 IP를 돌려준다
- **Anycast** : IP 하나는 보통 서버 한 대를 가리키는데, 여러 엣지가 **같은 IP**를 동시에 광고해 인터넷 라우팅(BGP)이 그중 가장 가까운 곳으로 보내게 한다. Cloudflare, Vercel이 이 방식이다

어느 엣지에 도착했는지는 응답 헤더에 남는다.

```text
$ curl -sI https://yceffort.kr/ | grep x-vercel-id
x-vercel-id: icn1::icn1::kb2hv-1789123883690-2280c10516e7

$ curl -s https://www.cloudflare.com/cdn-cgi/trace | grep colo
colo=ICN
```

`icn`, `ICN`은 인천공항 코드다. CDN 업계는 PoP 이름에 공항 코드를 쓴다.

---

## 엣지가 하는 일

```text
[브라우저] ──▶ [엣지 icn1] ──▶ [오리진]
                  │
                  ├─ 캐시에 있고 아직 유효하다 ─▶ HIT : 사본을 바로 준다
                  ├─ 캐시에 없다 ─────────────▶ MISS : 오리진에서 받아 저장하고 준다
                  └─ 있지만 만료됐다 ─────────▶ STALE : 오리진에 확인하거나 일단 준다
```

- 캐시를 찾는 열쇠(**캐시 키**)는 기본적으로 `호스트 + 경로 + 쿼리스트링`이다. 여기에 `Vary`로 지정한 요청 헤더가 추가된다(Part 2)
- 그래서 `/posts/1`과 `/posts/1?utm_source=x`는 **다른 키**다. 내용이 같아도 엣지에는 사본이 둘 생긴다
- 단 CloudFront는 기본 캐시 키가 **`도메인 + 경로`뿐**이라 위 둘이 같은 키가 된다. 쿼리스트링을 넣으려면 캐시 정책에 직접 지정한다
- 결과는 벤더마다 다른 이름의 헤더로 알려준다: `x-vercel-cache`, `cf-cache-status`, CloudFront는 `x-cache`

<!-- HIT/MISS/STALE 외에 벤더별로 BYPASS, REVALIDATED, PRERENDER 같은 값이 더 있지만 셋만 알면 나머지는 문서로 읽힌다. -->

---

## AWS로 보면 : CloudFront의 구조

```text
[브라우저] ──▶ [엣지 ICN80] ──▶ [리전 엣지 캐시] ──▶ [오리진 : S3 버킷, ALB, EC2]
```

- **배포(distribution)** 하나가 CDN 한 벌이다. 만들면 `d1234.cloudfront.net` 주소가 나오고, `cdn.example.com` 같은 내 도메인을 대체 도메인 이름으로 등록한 뒤 DNS에서 이 주소로 연결한다
- **오리진** : 사본의 출처. 정적 파일은 S3 버킷, 동적 응답은 ALB나 EC2
- **동작(behavior)** : 경로 패턴별 규칙. `/static/*`는 S3로, `/api/*`는 ALB로 보내고 각각 다른 캐시 정책을 붙인다
- **리전 엣지 캐시** : 엣지와 오리진 사이의 2차 캐시다. 엣지보다 용량이 커서 인기가 식어 엣지에서 밀려난 사본도 여기 남고, 한 리전의 엣지들이 이 캐시를 함께 쓴다
- 그래서 엣지에서 빠져도 오리진까지 안 가는 경우가 많다. 단 동적 요청, `POST` 같은 프록시 메서드, 오리진 S3가 같은 리전에 있는 요청은 여기를 건너뛴다

---

## 엣지 사본은 응답 헤더로 확인한다

```text
$ curl -sI \
  https://a0.awsstatic.com/libra-css/images/logos/aws_logo_smile_1200x630.png
cache-control: max-age=31536000
server: AmazonS3
x-cache: Hit from cloudfront
x-amz-cf-pop: ICN80-P3
age: 4273830            ← 49일째 서울 엣지에 있는 사본
```

---

## MISS와 HIT의 차이를 직접 재 보면

아직 서울 엣지에 없는 이미지 변형(384px)을 네 번 연속 요청했다.

| 순서 | `x-vercel-cache` | TTFB  |
| ---- | ---------------- | ----- |
| 1    | MISS             | 372ms |
| 2    | HIT              | 261ms |
| 3    | HIT              | 60ms  |
| 4    | HIT              | 50ms  |

- 첫 요청은 오리진(이미지 최적화 함수)까지 다녀왔다
- 두 번째는 HIT인데도 느린데, 엣지 안에서 사본이 아직 퍼지는 중이라고 이해하면 된다(벤더 내부 구현이라 단정하지 않는다)
- 이후는 **6~7배** 빠르다. 이 차이가 전 세계 사용자에게, 모든 파일에 곱해진다

---

## 캐시는 하나가 아니다

```text
[브라우저 캐시]  ──▶  [CDN 엣지 캐시]  ──▶  [오리진]
  사설(private)          공유(shared)
  한 사용자 전용          모든 사용자가 공유
```

- **사설 캐시** : 브라우저. 내 컴퓨터에만 있고 나만 쓴다
- **공유 캐시** : CDN, 회사 프록시. 다른 사용자의 응답을 나에게 줄 수 있다

같은 `Cache-Control` 헤더 하나로 둘 다 지시한다. 그래서 지시어에 `private`, `s-maxage`처럼 **어느 캐시를 향한 말인지** 구분하는 것들이 있다.

로그인한 사용자의 페이지가 공유 캐시에 저장되면 다른 사용자에게 그대로 나간다. [2편의 함정과 도구](/slides/cdn-and-http-cache-2)에서 실제 사고를 본다.

---

## 사본은 어디서 오는가 : pull과 push

- **Pull CDN** : 엣지가 오리진 앞에 서 있다가, 요청이 오면 그때 오리진에서 받아 저장한다. Cloudflare, Vercel, CloudFront가 기본으로 이렇게 동작한다
- **Push CDN** : 빌드 시점에 파일을 스토리지(S3 같은)에 올려 두고, CDN이 그 스토리지를 오리진으로 삼는다

실무에서는 보통 둘을 섞는다.

```text
HTML(동적, 자주 바뀜)  ──▶  애플리케이션 서버를 오리진으로 pull
JS/CSS/이미지(정적)    ──▶  빌드 때 스토리지에 push, CDN이 앞에 섬
```

Next.js를 Vercel에 올리면 이 구분이 자동으로 된다. 직접 배포(nginx, S3 + CloudFront)할 때는 이 구조를 손으로 만들어야 한다.

---

## CDN이 캐시 말고 하는 일

이 덱은 캐시만 다루지만, 실무에서 CDN을 켜면 같이 따라오는 것들이다.

- **TLS 종료** : 사용자의 암호 연결이 오리진이 아니라 엣지에서 끝난다는 뜻이다. 인증서를 엣지가 들고 핸드셰이크를 대신 처리한 뒤, 오리진과는 따로 연결한다
- **압축** : Brotli/gzip을 엣지에서 적용한다
- **HTTP/2, HTTP/3** : 오리진이 HTTP/1.1이어도 사용자와는 최신 프로토콜로 통신한다
- **DDoS 완화, WAF** : 오리진에 닿기 전에 걸러낸다
- **이미지 변환** : 크기와 포맷을 요청마다 만들어 캐시한다
- **엣지 함수** : 리다이렉트, A/B 분기 같은 가벼운 로직을 엣지에서 실행한다

---

## "국내 서비스인데 CDN이 왜 필요한가요"

**CDN은 멀어서 쓰는 게 아니다.** 서울에서 서울 리전 S3까지 TCP 연결이 10ms 안팎이다(실측). 거리를 줄이는 장치가 아니라 **오리진 앞에 두는 공유 캐시**다.

1. **오리진 부하 분리** : 같은 파일에 요청이 몰려도 오리진은 엣지마다 TTL에 한 번 응답하면 된다. 트래픽이 튈 때 서버 대신 엣지가 흡수한다
2. **응답을 만드는 비용** : 앞서 잰 MISS 372ms 대 HIT 50ms는 둘 다 서울 엣지에서 잰 값이다. 거리가 아니라 오리진 코드를 실행했느냐의 차이다
3. **전송 비용** : S3에서 CloudFront로 가는 전송은 무료이고, CloudFront 발신이 EC2나 ALB 발신보다 GB당 싸다
4. **가용성** : DDoS 완화와 WAF가 오리진에 닿기 전에 막고, 오리진이 5xx를 뱉어도 엣지가 옛 사본으로 버틸 수 있다(Part 2)

반대로 사내 도구처럼 트래픽이 작고 응답이 전부 개인화돼 캐시할 것이 없다면 CDN은 비용만 든다. 그때는 "커지면 붙인다"가 답이다.

---

## Part 1 정리

- 거리는 RTT로 바뀌고, HTTPS 첫 바이트까지는 RTT가 세 번 든다. 서울에서 미국은 0.5초다
- CDN은 사본을 사용자 가까이 두는 캐시 서버들의 네트워크다. 엣지 도착은 DNS 또는 Anycast로 정해진다
- 엣지는 캐시 키로 사본을 찾아 HIT/MISS/STALE로 응답한다. 실측으로 HIT는 MISS보다 6~7배 빨랐다
- 캐시는 브라우저(사설)와 CDN(공유) 둘이며, 하나의 헤더로 둘 다 지시한다

---

## 중간 점검 1-1

서울 사용자가 `https://shop.example.com/app.js`를 요청했다. 응답 헤더에 `x-cache: HIT`가 있고 TTFB는 40ms였다. 이때 확실히 말할 수 있는 것은?

1. 이 파일은 사용자의 브라우저 캐시에서 왔다
2. 오리진 서버가 서울에 있다
3. 서울 엣지가 이 파일의 사본을 이미 갖고 있었다
4. 이 파일은 앞으로 1년간 갱신되지 않는다

---

## 중간 점검 1-1 정답 : ③

- ① 브라우저 캐시에서 왔다면 네트워크 요청 자체가 없어서 `x-cache` 헤더를 새로 받을 일이 없다. DevTools에는 `(disk cache)`로 표시된다
- ② 엣지가 가까운 것이지 오리진 위치는 알 수 없다
- ④ 얼마나 유지되는지는 `Cache-Control`이 정한다. `HIT`는 "지금 있었다"는 뜻일 뿐이다

> 핵심: CDN 헤더는 **엣지에 있었는지**만 말한다. 브라우저 캐시와 유효 기간은 별개다.

---

## 중간 점검 1-2

마케팅팀이 `?utm_source=kakao`, `?utm_source=x` 같은 파라미터를 붙인 링크를 대량으로 뿌렸다. 어떤 링크로 들어와도 페이지 내용은 똑같은데, 오리진 부하가 눈에 띄게 늘었다.

이 CDN은 쿼리스트링을 캐시 키에 포함한다. 무슨 일이 벌어진 것인가?

---

## 중간 점검 1-2 정답 : 파라미터마다 캐시 키가 달라 전부 MISS였다

- 엣지는 본문을 비교하지 않는다. `?utm_source=kakao`와 `?utm_source=x`는 응답이 같아도 **서로 다른 객체**다
- 링크 종류만큼 MISS가 나고 그만큼 오리진이 실행된다. 엣지에는 같은 내용의 사본이 그 수만큼 쌓인다
- 캐시 키에 넣을 파라미터를 지정할 수 있다면 추적용 파라미터는 빼는 것이 답이다. CloudFront는 반대로 기본이 제외이므로, 응답이 파라미터에 따라 달라지는 페이지에서 반대 방향 사고가 난다

---

## 중간 점검 1-3

서울 사용자 A가 `/app.js`를 처음 요청해 `MISS`를 받았다. 잠시 뒤 도쿄 사용자 B가 같은 파일을 처음 요청했는데 역시 `MISS`였다.

A가 이미 받아 갔는데 왜 B도 MISS인가?

---

## 중간 점검 1-3 정답 : 캐시는 엣지마다 따로다

- A의 요청으로 사본이 생긴 곳은 **서울 엣지**다. 도쿄 엣지에는 아직 없다
- 엣지가 수십 곳이면 배포 직후 오리진은 같은 파일을 그만큼 여러 번 내보낸다. 배포 직후 오리진 부하가 튀는 이유다
- CloudFront처럼 리전 엣지 캐시가 있으면 같은 리전의 엣지끼리 사본을 나눠 쓰므로 오리진까지 가는 횟수가 줄어든다

> 핵심: `MISS`는 "이 엣지에 없었다"는 뜻이지 "어디에도 없었다"가 아니다.

---

## Part 2 : HTTP 캐시 규칙

<!-- _class: invert -->

---

## 캐시가 응답마다 던지는 세 가지 질문

```text
① 저장해도 되는가?              → no-store, private, public
② 저장한 것을 지금 써도 되는가?   → max-age, s-maxage, Age  (신선도)
③ 낡았다면 다시 받지 않고
   확인만 할 수 있는가?          → ETag, Last-Modified   (재검증)
```

브라우저와 CDN 둘 다 이 순서로 판단한다. `Cache-Control` 지시어는 전부 이 세 질문 중 하나에 대한 답이다.

---

## Cache-Control 지시어 : 저장과 신선도

| 지시어       | 뜻                                                            |
| ------------ | ------------------------------------------------------------- |
| `max-age=N`  | 응답 시점부터 N초 동안 신선(fresh)하다                        |
| `s-maxage=N` | 공유 캐시(CDN)에서만 쓰는 max-age. 있으면 CDN은 이쪽을 따른다 |
| `public`     | 공유 캐시가 저장해도 된다 (인증 헤더가 있는 응답까지도)       |
| `private`    | 브라우저만 저장. CDN은 저장하면 안 된다                       |
| `no-store`   | 어디에도 저장하지 마라                                        |
| `no-cache`   | 저장은 해도 되지만, **쓰기 전에 매번 서버에 확인**하라        |

```text
Cache-Control: public, max-age=0, s-maxage=300
   브라우저: 받자마자 낡음 → 매번 확인    CDN: 5분간 신선
```

브라우저가 매번 보내는 확인 요청은 **엣지에서 끝난다.** 엣지가 5분 동안 자기 사본으로 답하므로 오리진까지 가는 요청은 5분에 한 번이다.

---

## no-cache는 캐시를 끄지 않는다

가장 흔한 오해라 따로 한 장을 쓴다.

```text
Cache-Control: no-cache    저장한다. 다만 매번 서버에 "아직 같나요?"를 묻는다
Cache-Control: no-store    저장하지 않는다
```

- `no-cache`는 사실상 `max-age=0, must-revalidate`와 같은 효과다. 파일이 안 바뀌었으면 서버는 `304`만 보내고 본문은 캐시에서 쓴다
- "캐시 때문에 안 바뀌어요"의 해법으로 `no-cache`를 붙이면 **정상 동작**이다. 저장은 되지만 확인을 거치니 항상 최신이다
- 정말 저장을 막아야 하는 것은 개인정보가 담긴 응답이다. 그때만 `no-store`

> 둘 다 사용자는 늘 최신을 본다. 차이는 **매번 다시 받는 양**이다. `no-cache`는 안 바뀌었으면 `304` 헤더만 받고 본문은 캐시에서 쓰고, `no-store`는 매번 본문을 통째로 다시 받는다.

<!-- must-revalidate 단독은 "낡은 뒤에는 반드시 확인하라"이고, 낡기 전까지는 확인 없이 쓴다. no-cache는 낡았든 아니든 확인한다. -->

---

## 신선도 계산 : max-age와 Age

`max-age=300`을 받았다고 300초가 남은 것이 아니다. CDN이 이미 들고 있던 사본일 수 있고, 그 시간이 `Age`에 적혀 온다.

```text
Cache-Control: max-age=300     캐시해도 되는 총 시간은 300초
Age: 120                       그중 120초는 CDN이 이미 썼다
──────────────────────────
브라우저 몫 = 300 - 120 = 180초
```

- 이 180초 동안은 서버에 묻지 않고 쓴다. 브라우저가 들고 있는 시간이 쌓여 180초를 넘기면 그 사본은 **낡았다(stale)**
- 낡아도 버리지 않는다. 서버에 "아직 같나요"를 묻고(재검증), 같다면 받아 둔 본문을 그대로 쓴다

```text
$ curl -sI https://yceffort.kr/_next/static/immutable/chunks/2kzkiswgqacx4.js
cache-control: public,max-age=31536000,immutable
age: 151772                ← 서울 엣지가 1.76일 전에 받아 둔 사본
x-vercel-cache: HIT
```

<!-- RFC 9111 용어로는 총 시간이 freshness_lifetime, 지금 나이가 current_age이고 current_age < freshness_lifetime이면 신선하다. -->

---

## CloudFront의 함정 : no-store만으로는 부족하다

캐시 정책의 **Min TTL > 0이면 `no-store`도 무시된다.** `CachingOptimized`는 Min 1초, Default 24시간, Max 1년이다.

```text
오리진이 max-age=60을 보내면   →  60초. Min과 Max 사이라 그대로
오리진이 헤더를 안 보내면      →  24시간. Default가 대신 정한다
오리진이 no-store를 보내도     →  1초. Min TTL만큼은 캐시된다
```

**피하는 법: CloudFront 배포 → Behaviors(동작)에서 경로별 Cache policy를 지정한다.**

1. **개인화 API `/api/me`** : 별도 동작에 `CachingDisabled`(TTL 모두 0)를 연결하고, 오리진도 `Cache-Control: no-store`를 보낸다. CDN과 브라우저 양쪽의 저장을 막는다
2. **헤더로 캐시할 공개 응답** : `UseOriginCacheControlHeaders`(Min 0, Default 0, Max 1년)를 연결한다. TTL은 오리진 헤더로 정하고, 헤더가 없으면 캐시하지 않는다
3. **설정 배포 후 검증** : A/B 계정으로 같은 `/api/me`를 번갈아 반복 요청한다. 각자 자기 정보가 나오고, `X-Cache: Hit from cloudfront`가 나오지 않는지 확인한다

<!-- 캐시 키가 사용자를 구분하지 않으면 A의 개인 응답이 B에게 나갈 수 있다. 개인화 경로는 겹치는 /api/* 동작보다 위에 둔다(먼저 일치하는 동작이 적용됨). CachingDisabled는 인증 정보 전달을 설정하지 않으므로, 필요한 Cookie/Authorization을 오리진 요청 정책으로 전달해야 한다. 공개 응답도 쿼리에 따라 내용이 다르면 UseOriginCacheControlHeaders-QueryStrings 또는 그에 맞는 사용자 지정 정책을 쓴다. 출처: https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/using-managed-cache-policies.html , https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/DownloadDistValuesCacheBehavior.html -->

---

## 재검증 : 서버는 같은 버전인지 어떻게 알까?

캐시가 낡았다는 것은 **다시 확인할 시간이 됐다**는 뜻이다. 서버의 내용은 그대로일 수 있다. 본문을 다시 받지 않고 같은 버전인지 확인하려면?

서버가 응답에 **현재 버전을 구분하는 값 `ETag`**를 붙여 준다. 브라우저는 본문과 함께 저장했다가, 재검증할 때 `If-None-Match`에 그 값을 넣어 보낸다.

```text
처음 응답     200 OK + 본문 + ETag: "v1"
재검증 요청   GET /app.js + If-None-Match: "v1"
              "내가 가진 v1과 다를 때만 본문을 보내 주세요."

같으면 → 304 Not Modified, 본문 없이 → 브라우저는 저장한 본문 재사용
다르면 → 200 OK + 새 본문 + ETag: "v2" → 브라우저는 캐시 갱신
```

수정 시각으로도 확인할 수 있다. 서버가 `Last-Modified`를 주면 브라우저는 `If-Modified-Since`로 돌려보낸다. 이렇게 **같은 응답인지 확인할 근거**를 **검증자**라고 부른다.

<!-- ETag는 서버가 정하는 값이며 반드시 내용의 해시일 필요는 없다. 여기의 v1/v2는 설명용 식별자다. 2026-09-11 yceffort.kr 실측에서는 curl -s -o /dev/null -w "%{http_code} %{size_download}\n" -H 'If-None-Match: "_next/static/immutable/chunks/2kzkiswgqacx4.js"' https://yceffort.kr/_next/static/immutable/chunks/2kzkiswgqacx4.js 요청이 304 0을 반환했다. 이 응답의 ETag는 경로 문자열이었다. 출처: https://httpwg.org/specs/rfc9110.html#field.etag , https://httpwg.org/specs/rfc9110.html#field.if-none-match -->

---

## 검증자가 없으면 재검증도 없다

`yceffort.kr`의 HTML 응답 헤더에는 `ETag`도 `Last-Modified`도 없다.

```text
cache-control: public, max-age=0, must-revalidate
content-type: text/html; charset=utf-8
(etag 없음, last-modified 없음)
```

- `max-age=0`이라 매번 서버에 가야 하는데, 확인할 근거가 없으니 **매번 전체 본문(약 340KB)을 다시 받는다**
- 서버 입장에서는 `304`를 만들 재료가 없다

`max-age=0` 또는 `no-cache`를 쓸 거라면 검증자를 같이 주어야 그 설정이 싸진다. HTML처럼 매번 확인해야 하는 응답일수록 `ETag`가 중요하다.

<!-- 이 사이트의 경우 HTML 재검증이 안 되는 것이 실제 비용인지는 별개 문제다(스트리밍 RSC 응답이라 ETag 계산이 어렵다). 여기서는 "검증자 없는 max-age=0은 매번 풀 다운로드"라는 규칙만 가져간다. -->

---

## 헤더가 아무것도 없으면 : 휴리스틱 캐시

"캐시 헤더를 안 줬는데 왜 캐시되죠?"

- RFC 9111은 `Cache-Control`도 `Expires`도 없고 `Last-Modified`만 있으면, 캐시가 **스스로 신선도를 추정해도 된다**고 허용한다
- 권고 값은 `(응답 시각 - Last-Modified) × 10%`. Chrome과 Firefox가 그렇게 구현한다

```text
Last-Modified: 100일 전   →  10일 동안 확인 없이 캐시에서 쓴다
Last-Modified: 어제       →  약 2.4시간
```

nginx나 S3에서 정적 파일을 그냥 서빙하면 `Last-Modified`가 자동으로 붙는다. 그래서 헤더를 안 정해도 "가끔 안 바뀌는" 현상이 생긴다.

> 캐시 헤더를 정하지 않는 것은 "캐시 안 함"이 아니라 **"브라우저 마음대로"** 다.

---

## Vary : 같은 URL, 다른 응답

`Vary`는 서버가 보내는 **응답 헤더**다. 값에는 **응답을 달라지게 한 요청 헤더의 이름**을 적는다. 정해진 목록은 없으며 `X-Theme` 같은 사용자 정의 헤더도 가능하다.

| Vary에 넣는 이름  | 구분할 요청 헤더 값의 예                 | 응답이 달라지는 이유  |
| ----------------- | ---------------------------------------- | --------------------- |
| `Accept`          | `image/avif`, `image/webp`               | 지원하는 이미지 포맷  |
| `Accept-Encoding` | `br`, `gzip`                             | 지원하는 압축 방식    |
| `Accept-Language` | `ko`, `en`                               | 선호하는 언어         |
| `Origin`          | `https://a.example`, `https://b.example` | 출처별 CORS 허용 응답 |

---

## Vary에 무엇을 넣느냐가 HIT율을 가른다

```text
Vary: Accept, Accept-Encoding
→ 같은 URL이어도 두 요청 헤더 값이 모두 맞아야 재검증 없이 재사용
```

- `User-Agent`, `Cookie`도 가능하지만, 값이 다양하면 사본이 잘게 나뉘어 HIT율이 낮아진다
- **특수 값 `*`** : 어떤 요청과도 일치하지 않아 재검증 없이 재사용할 수 없다. 저장 금지는 `no-store`로 지시한다

<!-- Vary에는 image/avif 같은 값 자체가 아니라 Accept라는 요청 헤더 이름을 쓴다. 실제로 응답 선택에 영향을 주는 헤더만 지정한다. 2026-09-11 yceffort.kr의 동일한 /_next/image URL에 기본 Accept로 요청하면 image/jpeg(975 B), Accept: image/avif,image/webp,*/*로 요청하면 image/avif(735 B)가 왔고 두 응답 모두 Vary: Accept를 포함했다. Vary: Cookie는 쿠키 값별 구분이지 저장 금지 선언은 아니다. CDN별 지원과 설정은 별도 확인하며 다음 페이지에서 CloudFront의 예외를 설명한다. 출처: https://httpwg.org/specs/rfc9110.html#field.vary , https://httpwg.org/specs/rfc9111.html#caching.negotiated.responses -->

---

## CloudFront에서는 Vary만으로 부족하다

이미지 서버가 같은 `/photo`에서 `Accept`에 따라 AVIF나 JPEG를 보내고, 응답에 `Vary: Accept`를 붙인다고 하자. CloudFront에서는 **캐시 정책에도 `Accept`를 넣어야 두 응답을 구분한다.**

| Accept를 넣는 곳                             | CloudFront가 하는 일                                     |
| -------------------------------------------- | -------------------------------------------------------- |
| **캐시 정책** (Cache policy)                 | Accept 값별로 캐시를 구분하고, 오리진에도 전달           |
| **오리진 요청 정책** (Origin request policy) | 캐시 MISS 때 오리진에 전달만 함. 캐시 구분에는 쓰지 않음 |

```text
Accept를 오리진에 전달만 하고, 캐시 키에서는 빼면:
A: Accept: image/avif → MISS → AVIF 응답을 /photo의 사본으로 저장
B: Accept: image/jpeg → HIT → A에게 줬던 AVIF를 받음!
```

**해결:** 캐시 정책의 **Headers에 `Accept`를 포함**하고 해당 경로의 동작(Behavior)에 연결한다. 그러면 A와 B는 서로 다른 사본을 쓴다.

오리진의 `Vary: Accept`도 유지한다. **브라우저의 캐시 구분에는 여전히 필요하다.**

<!-- 예시는 캐시 가능한 공개 응답이고, Accept 외의 캐시 키 요소는 같으며 A의 응답이 아직 신선하다고 가정한다. 캐시 키는 이해를 돕기 위해 /photo로 단순화했다. 캐시 정책에 포함한 값은 오리진에 자동 전달되므로 오리진 요청 정책에 중복 지정할 필요가 없다. 캐시 정책에 없어도 오리진 요청 정책으로 전달할 수 있다. 출처: https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/controlling-the-cache-key.html , https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/controlling-origin-requests.html -->

---

## stale-while-revalidate : 낡아도 일단 주고 뒤에서 갱신

```text
Cache-Control: max-age=60, stale-while-revalidate=600
```

```text
0 ~ 60초        fresh. 그냥 준다
60 ~ 660초      stale이지만 일단 옛 사본을 주고, 백그라운드에서 새로 받아 둔다
660초 이후      stale. 새로 받을 때까지 기다린다
```

- 사용자는 재검증을 기다리지 않는다. 대신 최대 10분 낡은 응답을 볼 수 있다
- CDN에서 특히 효과적이다. 사용자 한 명이 옛 사본을 받는 대신 **그 뒤 모든 사용자가 새 사본**을 받는다
- 형제 지시어 `stale-if-error=N`은 오리진이 5xx일 때 N초까지 옛 사본으로 버틴다

Chrome, Firefox 모두 지원한다. Next.js가 ISR 페이지에 내보내는 헤더가 바로 이 조합이다.

---

## immutable : 만료 전엔 확인조차 하지 마라

```text
Cache-Control: public, max-age=31536000, immutable
```

- `max-age`만 있으면 사용자가 **새로고침**할 때 브라우저가 "혹시 바뀌었나" 확인 요청(`If-None-Match`)을 보낸다. 304가 돌아와도 왕복은 든다
- `immutable`은 "이 URL의 내용은 절대 바뀌지 않으니 그 확인도 하지 마라"는 선언이다
- Firefox와 Safari가 구현했다. Chrome은 지시어를 무시하는 대신 **새로고침 동작 자체를 바꿨다**: 일반 새로고침 때 메인 리소스만 재검증하고 서브리소스는 캐시 규칙대로 쓴다(2편에서 자세히 다룬다)

이 지시어를 붙일 수 있는 조건은 하나다. **URL이 내용을 식별해야 한다.** 파일명에 내용 해시가 들어간 빌드 산출물이 그 조건을 만족한다.

---

## 지시어 한 장 요약

| 지시어                   | 브라우저                 | CDN                   |
| ------------------------ | ------------------------ | --------------------- |
| `max-age=N`              | N초 신선                 | `s-maxage` 없으면 N초 |
| `s-maxage=N`             | 무시                     | N초 신선              |
| `private`                | 저장                     | 저장 안 함            |
| `public`                 | 저장                     | 저장 (인증 응답도)    |
| `no-store`               | 저장 안 함               | 저장 안 함            |
| `no-cache`               | 저장 후 매번 재검증      | 저장 후 매번 재검증   |
| `must-revalidate`        | 낡으면 반드시 재검증     | 같음                  |
| `stale-while-revalidate` | 낡아도 주고 뒤에서 갱신  | 같음 (효과 큼)        |
| `immutable`              | 새로고침에도 재검증 생략 | 대부분 무시           |

---

## Part 2 정리

- 캐시는 매 응답에 세 질문을 한다: 저장해도 되나, 지금 신선한가, 낡았다면 확인만 할 수 있나
- 신선도는 `max-age`(CDN은 `s-maxage`)에서 `Age`를 뺀 값이다. `no-cache`는 저장하되 매번 확인이고, 저장 금지는 `no-store`다
- 재검증은 `ETag`/`Last-Modified`가 있어야 가능하고, 없으면 `max-age=0`은 매번 전체 다운로드다
- 헤더가 없으면 휴리스틱(Last-Modified 경과의 10%)으로 캐시된다. `Vary`는 캐시 키를 넓히는 선언이다
- CloudFront는 캐시 정책의 Min/Default/Max TTL로 오리진 헤더를 자르고, 키에는 정책에 넣은 것만 들어간다. Min TTL > 0이면 `no-store`도 무시된다

---

## 개념 확인 1 : 헤더 해석 (Part 2)

```text
HTTP/2 200
Cache-Control: private, max-age=0, no-cache
ETag: "x1"
```

이 응답에 대한 설명으로 **틀린** 것은?

1. 브라우저는 이 응답을 저장한다
2. CDN은 이 응답을 저장하지 않는다
3. 다음 요청에서 브라우저는 서버에 묻지 않고 캐시를 쓴다
4. 서버가 `304`를 돌려주면 브라우저는 캐시된 본문을 쓴다

---

## 개념 확인 1 정답 : ③

- `no-cache`는 저장은 하되 매번 확인이다. ①은 맞고 ③은 틀렸다
- `private`이 있으니 CDN은 저장하지 않는다(②)
- `ETag`가 있으니 확인 요청에 `If-None-Match: "x1"`가 실리고, `304`면 캐시 본문을 쓴다(④)

> `no-cache`는 "캐시 끔"이 아니다. "쓰기 전에 물어봄"이다.

---

## 개념 확인 2 : 신선도 계산 (Part 2)

브라우저가 12:00:00에 다음 응답을 받았다.

```text
Cache-Control: max-age=300
Age: 120
```

12:03:30에 같은 URL을 요청하면 브라우저는 네트워크에 나가는가?

---

## 개념 확인 2 정답 : 나간다

- 신선 기간 300초에서 `Age` 120초를 뺀 **180초**가 브라우저 몫이다. 12:03:00까지만 신선하다
- 12:03:30은 30초 지났으니 stale이다. 검증자가 없으므로 확인 요청이 아니라 **전체 재요청**이다

> `Age`를 빼고 계산한 사람은 "300초 안이니 캐시"라고 답한다. CDN 뒤에서 이 실수가 자주 나온다.

---

## 중간 점검 2-1

우리 [HTML](https://sandbox-app.kakaopaysec.com/kkb/home)과 [API](https://sandbox-api-gw-neo.kakaopaysec.com/faye-bank/v1/api/section/section-list)의 **GET 200 응답**이다. (2026-09-16 실측)

```text
HTML  Cache-Control: no-store
API   Cache-Control: no-cache, no-store, max-age=0, must-revalidate
```

브라우저의 HTTP 캐시는 이 두 응답의 본문을 저장해도 되는가?

1. 둘 다 저장하고, 재사용 전에 서버에 확인한다
2. HTML은 저장하지 않고, API는 저장 후 매번 확인한다
3. 둘 다 저장하면 안 된다
4. HTML은 저장하고, API만 저장하면 안 된다

<!-- HTML은 2026-09-16 08:51:08 GMT에 GET 200, Content-Type: text/html, Content-Length: 79000, Cache-Control: no-store. API는 Kps-User-Agent: KAKAOBANK_APP;spec=1을 넣어 08:57:55 GMT에 GET 200, Content-Type: application/json, Content-Length: 7950, 위 Cache-Control과 Pragma: no-cache, Expires: 0을 확인했다. 두 요청에 Authorization/Cookie/Cache-Control 요청 헤더를 넣지 않았다. HTTP 캐시만을 묻고 SPA 내부 이동, bfcache, 서비스 워커는 가정에서 제외한다. -->

---

## 중간 점검 2-1 정답 : ③

- 두 응답 모두 **`no-store`**가 있으므로 HTTP 캐시에 저장하면 안 된다
- API에 `no-cache`가 함께 있어도 **`no-store`의 저장 금지**는 사라지지 않는다

**우리 서비스에서는 이렇게 검토하자**

- **HTML** : 개인정보가 없는 공통 앱 셸이라면 **`no-cache` + `ETag`**를 검토한다. 매번 최신 여부는 확인하면서, 안 바뀐 HTML의 전체 재전송을 줄일 수 있다
- **API** : 사용자별 데이터라면 **현재 `no-store`를 유지**한다. 채널별 공통 설정만 내려준다면, 채널을 캐시 키에 넣는 짧은 캐시를 검토할 수 있다

**헤더만으로 개인화 여부는 알 수 없다.** 실제 응답 내용과 저장 요구사항을 확인한 뒤 바꾼다.

<!-- HTML에 민감한 사용자 데이터가 포함된다면 no-store를 유지한다. 사용자마다 다르지만 브라우저 저장은 허용할 수 있는 응답은 private와 재검증 정책을 함께 검토할 수 있다. API는 Kps-User-Agent 채널 헤더가 필요했으므로 공통 응답이어도 채널별 차이를 확인해야 한다. 응답 본문의 사용자별 차이나 저장 요구사항은 이번 헤더 측정으로 검증하지 않았다. 따라서 위 제안은 조건부다. no-cache의 재검증 규칙: https://httpwg.org/specs/rfc9111.html#cache-response-directive.no-cache -->

---

## 중간 점검 2-2

우리 [JS 파일](https://sandbox-remote-app.kakaopaysec.com/home/20260915-64a935ecb135/index.js)의 **GET 200 응답**이다. (2026-09-16 실측)

```text
Cache-Control: max-age=31536000, immutable
Vary: Origin
```

5분 뒤 같은 파일이 다시 필요하다. **사본은 아직 신선하고 브라우저 캐시에 남아 있으며, URL과 요청의 Origin도 같다.** 일반 요청에서 가능한 동작은?

1. `304`를 받아야만 저장한 본문을 쓸 수 있다
2. 서버에 요청하지 않고 저장한 본문을 쓸 수 있다
3. `immutable`이 있으므로 본문을 항상 새로 받아야 한다
4. `Vary`가 있으므로 캐시에 저장할 수 없다

<!-- 최초 GET은 2026-09-16 08:51:09 GMT에 200, Content-Type: application/javascript, Content-Length: 311375, ETag: "a263a94c496084807a70bf17fadd545e", Last-Modified: Tue, 15 Sep 2026 04:54:40 GMT, X-Cache: Miss from cloudfront, X-Amz-Cf-Pop: ICN53-P1, Vary: Origin이었다. 첫 응답에 Age는 없었다. curl 자체에는 브라우저 HTTP 캐시가 없으므로 브라우저 동작은 헤더로 해석한 것이다. 문제에서는 강력 새로고침과 DevTools 캐시 비활성화를 하지 않으며, 같은 캐시 파티션에서 다른 Vary 조건도 일치한다고 가정한다. -->

---

## 중간 점검 2-2 정답 : ②

- **신선 기간 1년** 안이고 `Vary: Origin` 조건도 같으므로, 서버 확인 없이 저장한 본문을 쓸 수 있다
- `immutable`은 신선한 동안 내용이 바뀌지 않는다는 약속이다

**우리 서비스에서는 현재 방향을 유지하자 — 조건은 같은 URL의 내용을 바꾸지 않는 것**

- 새 배포는 **새 버전 경로**에 올리고 참조 URL도 바꾼다. 같은 경로에 덮어쓰면 기존 브라우저는 예전 JS를 계속 쓸 수 있다
- **이전 버전 파일도 보존**한다. 이미 열린 화면이 나중에 옛 청크를 요청해도 404가 나지 않아야 한다

이 조건을 지키면 **반복 다운로드를 줄이면서 새 배포도 새 URL로 전달**할 수 있다.

<!-- 실측 URL에는 버전 경로가 있지만 내용 해시인지, 덮어쓰기가 금지되는지, 이전 파일을 얼마나 보존하는지는 확인하지 않았다. 헤더의 방향은 적절하되 배포 규칙까지 점검해야 한다는 제안이다. https://httpwg.org/specs/rfc8246.html , https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/UpdatingExistingObjects.html -->

---

## 중간 점검 2-3

우리 [메뉴 SVG](https://t1.kakaocdn.net/kakaopay_redbull/resources/home/bank_menu/kp_service_ic_collect_solid_orange_24.svg)의 **GET 200 응답**이다. (2026-09-16 실측)

```text
Cache-Control: max-age=21600
Age: 12880
```

이 응답을 받은 브라우저는 **앞으로 몇 초 동안 서버 확인 없이** 이 사본을 쓸 수 있는가?

전송 지연은 무시하고, 사본은 캐시에 남아 있다고 가정한다.

<!-- 2026-09-16 08:51:08 GMT 실측. Content-Type: image/svg+xml, Content-Length: 392, 위 Cache-Control/Age와 Last-Modified: Thu, 16 Jul 2026 06:48:18 GMT, Expires: Wed, 16 Sep 2026 11:16:28 GMT를 확인했다. ETag는 없었다. Age는 이 시점의 고정 스냅샷이며 다시 요청하면 달라진다. 정확한 신선도 계산에는 Date, 전송 지연, 로컬 체류 시간도 들어가지만 문제에서는 전송 지연을 무시한다. https://httpwg.org/specs/rfc9111.html#calculating.age -->

---

## 중간 점검 2-3 정답 : 약 8720초

```text
총 신선 기간     21600초 = 6시간
이미 지난 시간  12880초
──────────────────────────
남은 신선 기간   8720초 = 2시간 25분 20초
```

- 브라우저에 6시간이 새로 주어지는 것이 아니라, **이미 지난 시간을 뺀 나머지**가 남는다

**우리 서비스에서는 아이콘 교체 방식에 맞추자**

- 파일 교체 때 URL도 바꿀 수 있다면 **내용 해시 파일명 + 긴 `max-age` + `immutable`**을 추천한다. 같은 아이콘의 반복 다운로드를 줄이고, 변경은 새 URL로 바로 구분할 수 있다
- 같은 URL을 꼭 유지해야 한다면 **허용할 수 있는 갱신 지연에 맞춰 TTL을 정한다.** 6시간 안에 교체가 보여야 하는 요구라면 현재 TTL을 줄여야 한다

<!-- 고정 URL의 현재 사본에는 약 2시간 25분이 남았지만, 일반적으로 새 사본은 최대 6시간 동안 확인 없이 쓰일 수 있다. "6시간이면 무조건 잘못"이 아니라 요구하는 갱신 주기와의 일치를 판단한다. TTL을 줄여 배포해도 이미 브라우저에 저장된 사본의 남은 시간은 즉시 바뀌지 않는다. 즉시 전환이 필요하면 참조 URL을 새 경로로 바꿔야 한다. https://httpwg.org/specs/rfc8246.html#introduction -->

---

## 중간 점검 2-4

같은 [SVG 응답](https://t1.kakaocdn.net/kakaopay_redbull/resources/home/bank_menu/kp_service_ic_collect_solid_orange_24.svg)에는 아래 헤더가 있었고, `ETag`는 없었다.

```text
Last-Modified: Thu, 16 Jul 2026 06:48:18 GMT
```

이제 캐시가 낡았다. 브라우저가 **그때 받은 파일이 아직 같은지 확인**하려면, 다음 GET 요청에 어떤 헤더를 넣으면 되는가?

<!-- 직전 문제에서 사용한 2026-09-16 SVG 실측과 같은 응답이다. 검증자 이름만 답하는 데서 끝내지 말고, 실제 요청 헤더 이름과 값을 함께 말하게 한다. -->

---

## 중간 점검 2-4 정답 : If-Modified-Since

```text
If-Modified-Since: Thu, 16 Jul 2026 06:48:18 GMT
```

- 응답에서 받은 **`Last-Modified`의 시각**을 요청의 `If-Modified-Since`로 돌려보낸다
- 변경이 없고 서버가 조건을 처리하면 `304`로 본문을 재사용할 수 있지만, 이번 실측은 **`200`과 본문 392 B**였다

**우리 서비스에서는 조건부 요청이 실제로 처리되는지 점검하자**

- 고정 URL을 유지한다면 **본문 변경 여부와 CDN·오리진의 조건부 요청 처리**를 함께 확인한다. 변경이 없을 때 `304`를 반환하면 본문 전송량을 아낄 수 있다
- 다만 이 아이콘은 **392 B**다. `304`도 왕복은 필요하므로, 이 건의 개선 우선순위는 요청량과 실제 절감 효과를 보고 정한다

검증자 헤더의 존재만으로 최적화가 끝났다고 판단하지 말고, **응답 코드와 전송량까지 확인**한다.

<!-- 2026-09-16 08:52:38 GMT에 위 If-Modified-Since를 보낸 별도 GET은 200과 392 B 본문을 반환했고 응답 Last-Modified도 동일했다. 어느 계층에서 조건이 처리되지 않았는지는 이 응답만으로 확정하지 않는다. 같은 시각 JS의 ETag를 If-None-Match로 보낸 별도 GET은 304, 본문 0 B, X-Cache: Hit from cloudfront, Age: 59였다. https://httpwg.org/specs/rfc9110.html#field.if-modified-since -->

---

## 1편 정리 : 헤더를 읽는 순서

1. **어느 캐시인가?** 브라우저 캐시와 CDN 캐시는 서로 다른 사본이다
2. **저장해도 되는가?** `no-store`는 저장 금지, `private`은 공유 캐시 저장 금지다
3. **확인 없이 써도 되는가?** `max-age`와 `Age`로 신선도를 보고, `no-cache`면 재사용 전에 확인한다
4. **같은 응답인가?** `Vary`와 캐시 키로 사본을 구분하고, ETag나 Last-Modified로 변경 여부를 확인한다
5. **CDN 설정도 맞는가?** 오리진 헤더와 CloudFront의 캐시 키·TTL 정책을 함께 본다

이제 [2편](/slides/cdn-and-http-cache-2)에서 **어떤 파일에 어떤 규칙을 붙일지** 결정한다.

---

## 참고 자료

- [RFC 9111 : HTTP Caching](https://www.rfc-editor.org/rfc/rfc9111) : 저장 조건, 신선도, 재검증의 원문
- [RFC 8246 : immutable](https://www.rfc-editor.org/rfc/rfc8246), [RFC 5861 : stale-while-revalidate](https://www.rfc-editor.org/rfc/rfc5861)
- [MDN : HTTP caching](https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/Caching) : 지시어별 브라우저 동작 정리
- [CloudFront : Manage how long content stays in the cache](https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/Expiration.html) : TTL 합성 표와 Min TTL 경고문
- CloudFront : [Managed cache policies](https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/using-managed-cache-policies.html), [Control the cache key](https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/controlling-the-cache-key.html)

---

# 감사합니다

<!-- _class: invert -->

다음 편에서는 HTML·JS·이미지·API의 정책을 정하고, 배포 후 캐시 문제를 진단한다.

[2편: 전략과 문제 해결 →](/slides/cdn-and-http-cache-2)

@yceffort
