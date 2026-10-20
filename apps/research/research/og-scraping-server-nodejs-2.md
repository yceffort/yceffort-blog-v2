---
title: 'OG 스크래핑 서버를 Node.js로 짓는다면 2부: SSRF는 어떻게 뚫리는가'
marp: true
paginate: true
theme: midnight
tags:
  - security
  - networking
  - nodejs
date: 2026-08-18
post: https://yceffort.kr/2026/08/og-scraping-server-2
description: '사용자가 준 URL을 서버가 대신 여는 일의 위험, 화이트리스트를 뚫는 우회 여섯 가지와 그것을 막는 방어 원리 다섯 개'
published: true
art:
  undraw: firewall
---

# OG 스크래핑 서버를 Node.js로 짓는다면

SSRF는 어떻게 뚫리는가

<!-- _class: invert deck-cover -->

2부 · 전 2부

@yceffort

---

## 1부에서 여기까지 왔다

- 링크 미리보기의 "에러율 10%"는 증상이고, 성질이 다른 실패 다섯 종류가 뭉쳐 있었다
- 이 워크로드는 I/O 바운드에 저 TPS라, 런타임 후보가 좁혀졌다
- 런타임이 갈리는 첫 번째 지점으로 **"소켓 직전까지의 통제권"**을 꼽았다

그 통제권이 왜 필요한가. **이 부가 그 답이다.**

> 1부를 읽지 않아도 된다. 필요한 전제는 이 한 줄이다.
> **우리 서버는 사용자가 준 URL을 대신 열어 준다.**

---

## 이 부가 답하려는 것

1. 사용자 입력 URL을 서버가 대신 여는 일이 **왜 그렇게 위험한가**
2. 화이트리스트로 막았다고 믿는 코드는 **어떤 방법으로 뚫리는가** (여섯 가지)
3. 원리대로 막으려면 **무엇을 검증해야 하는가** (다섯 가지)
4. 애플리케이션 코드로 막을 수 없는 것은 무엇인가

> 순서가 중요하다. **막는 법보다 뚫리는 법을 먼저 본다.**
> 어떤 우회를 막는지 모르는 방어 코드는 방어가 아니라 선언이다.

---

<!-- _class: invert -->

# 보안: SSRF를 원리부터

여기가 본론이다

---

## 한 문장 정의

> **SSRF(Server-Side Request Forgery)는 공격자가 서버로 하여금
> 공격자가 원하는 곳에 요청을 보내게 만드는 취약점이다.**

핵심은 "요청을 보낸다"가 아니라 **"서버가 보낸다"**에 있다.

---

## 왜 "서버가 보낸다"가 위험한가

공격자의 브라우저와 우리 서버는 **네트워크 위치가 다르다.**

```text
  공격자 브라우저                    우리 서버
       │                                │
       │  ✗ 방화벽에 막힘                │  ✓ 내부망 안에 있다
       ▼                                ▼
 ┌──────────────────────────────────────────────┐
 │  내부망                                       │
 │   • 169.254.169.254  (클라우드 메타데이터)    │
 │   • 10.0.x.x         (내부 어드민, DB, Redis) │
 │   • 127.0.0.1        (같은 호스트의 서비스)   │
 └──────────────────────────────────────────────┘
```

SSRF는 **공격자에게 우리 서버의 네트워크 위치를 빌려준다.**
방화벽은 아무 역할도 하지 못한다. 요청이 안에서 나가기 때문이다.

---

## 그리고 스크래핑은 SSRF의 교과서적 조건이다

SSRF가 성립하려면 두 가지가 필요하다.

1. 서버가 **외부에서 온 URL**로 요청을 보낸다
2. 그 URL을 **공격자가 정할 수 있다**

링크 미리보기는 이 둘을 **기능 명세로 갖고 있다.**

> 취약점이 실수로 생기는 게 아니라, **기능 자체가 취약점의 모양**이다.
> 그래서 "조심해서 짠다"로는 안 되고, 구조로 막아야 한다.

---

## 공격 시나리오: 단계별로

가장 흔한 AWS 환경의 예다.

**① 공격자가 게시글에 링크를 붙여넣는다**

```text
http://169.254.169.254/latest/meta-data/iam/security-credentials/
```

**② 우리 서버가 미리보기를 만들려고 그 URL을 연다**

**③ 응답에 IAM 역할 이름이 들어 있다**

```text
og-scraper-role
```

---

## 공격 시나리오: 마무리

**④ 공격자가 한 번 더 붙여넣는다**

```text
http://169.254.169.254/latest/meta-data/iam/security-credentials/og-scraper-role
```

**⑤ 응답**

```json
{
  "AccessKeyId": "ASIA...",
  "SecretAccessKey": "...",
  "Token": "..."
}
```

**⑥ 이 크레덴셜로 S3, DynamoDB에 직접 접근한다**

미리보기 카드에 에러 메시지만 떠도, 응답 본문이 어딘가에 로깅되면 끝이다.

---

## 왜 하필 169.254.169.254인가

이 대역(`169.254.0.0/16`)은 **link-local**이다. 라우팅되지 않고, 같은 링크 안에서만 유효하다.

클라우드 사업자들은 여기에 인스턴스 메타데이터 서비스를 올린다.

| 사업자  | 주소              | 추가 조건                           |
| ------- | ----------------- | ----------------------------------- |
| AWS     | `169.254.169.254` | IMDSv2면 PUT으로 토큰 선취득        |
| GCP     | `169.254.169.254` | `Metadata-Flavor: Google` 헤더 필요 |
| Azure   | `169.254.169.254` | `Metadata: true` 헤더 필요          |
| Alibaba | `100.100.100.200` |                                     |

---

## IMDSv2가 있으면 안전한가

**절반만 그렇다.**

IMDSv2는 토큰을 먼저 받아야 한다.

```text
PUT /latest/api/token
X-aws-ec2-metadata-token-ttl-seconds: 21600
```

스크래핑은 GET만 보내므로 **단순 SSRF로는 뚫리지 않는다.**

그러나,

- IMDSv1이 아직 켜져 있는 인스턴스가 많다 (기본값이 오래도록 v1 허용이었다)
- SSRF가 메서드나 헤더를 조작할 수 있는 형태라면 여전히 가능하다
- **메타데이터만이 표적이 아니다.** 내부 어드민, Redis, Elasticsearch, k8s API가 남아 있다

> `hop limit = 1`과 IMDSv2 강제는 **반드시** 해두되, 그것을 방어의 전부로 삼지 말 것.

---

## 그래서 화이트리스트면 되는가

가장 흔한 대답이 이것이다.

> "허용 도메인 목록을 두고 그 안에서만 스크래핑하면 됩니다"

**방향은 맞다. 그런데 그것만으로는 뚫린다.**

지금부터 우회 기법 여섯 가지를 본다. 이걸 알아야 방어 원리가 왜 그런 모양인지 이해할 수 있다.

---

## 우회 1: IP 표기 변형

위험한 주소를 **문자열로** 걸러내는 검사가 있다고 하자.

```ts
if (rawUrl.includes('127.0.0.1')) throw new Error('blocked')
```

아래는 이 검사를 **전부 통과한다.** 그리고 전부 `127.0.0.1`로 연결된다.

| 붙여넣는 URL              | 표기                    |
| ------------------------- | ----------------------- |
| `http://2130706433/`      | 32비트 십진수           |
| `http://0x7f000001/`      | 16진수                  |
| `http://0177.0.0.01/`     | 8진수                   |
| `http://127.1/`           | 축약형 (중간 옥텟 생략) |
| `http://127.000.000.001/` | 0 패딩                  |

---

## 우회 1: 그런데 이건 가벼운 편이다

`new URL()`로 **파싱하는 것만으로 사라진다.** WHATWG 파서가 파싱 단계에서 정규화한다.

```ts
new URL('http://2130706433/').hostname // '127.0.0.1'
new URL('http://0x7f000001/').hostname // '127.0.0.1'
new URL('http://127.1/').hostname // '127.0.0.1'
```

**교훈:** IP 표기법을 외우라는 게 아니다. **검증은 원본 문자열이 아니라 파싱한 값으로** 한다.

> 이걸 맨 앞에 둔 이유는 여기서부터다. **남은 다섯은 파싱만으로 막히지 않는다.**

---

## 우회 2: DNS로 사설 IP 가리키기

그럼 파싱해서 얻은 `hostname`을 검사하면 되는가. **그것만으로는 안 된다.**

도메인 이름은 아무 IP나 가리킬 수 있다. 공격자가 자기 도메인의 A 레코드를 이렇게 두면,

```text
evil.example.com.   IN  A   127.0.0.1
```

`evil.example.com`은 `hostname`을 아무리 뜯어봐도 **정상적인 공인 도메인**이다. 그런데 연결되는 곳은 로컬이다.

도메인을 살 필요조차 없다. 이름에 적힌 IP를 그대로 돌려주는 공개 서비스가 있다.

```text
127.0.0.1.nip.io      →  127.0.0.1
10.0.0.1.nip.io       →  10.0.0.1
```

**교훈:** **이름은 목적지에 대해 아무것도 말해주지 않는다.** 봐야 하는 건 해석된 IP다.

---

## 우회 3: DNS rebinding

그럼 이름 대신 **해석된 IP**를 검사하면 되는가. **그것도 뚫린다.**

여섯 중 가장 교묘하고, 가장 많이 놓치는 것이다. 방어 코드가 보통 이렇게 생겼다고 하자.

```ts
const ip = await dns.resolve(url.hostname) // ① 검사용 조회
if (isPrivate(ip)) throw new Error('blocked')

await fetch(url) // ② 실제 요청 (또 조회한다!)
```

①과 ②는 **각각 DNS를 조회한다.** 그 사이에 응답이 바뀔 수 있다.

이 틈을 **TOCTOU(Time-Of-Check to Time-Of-Use)**라고 부른다.

---

## rebinding 타임라인

공격자는 TTL을 0으로 둔 도메인을 준비한다.

```text
t=0ms   ① 검사용 DNS 조회
        evil.com  →  93.184.216.34  (공인 IP)     ✓ 통과

t=1ms   isPrivate() 검사 통과

t=2ms   ② 실제 요청 직전 DNS 재조회
        evil.com  →  169.254.169.254            ✗ 이미 검사 끝났다

t=3ms   메타데이터 서비스로 요청이 나간다
```

공격자가 이걸 할 수 있는 이유는 **자기 도메인의 DNS 서버를 자기가 들고 있기** 때문이다.
TTL이 0이면 캐시되지 않으므로, **두 번의 조회가 다른 답을 받는 것이 정상 동작**이다. 규칙을 어긴 것이 하나도 없다.

**교훈:** 검사한 것과 사용한 것이 같아야 한다. 이름을 검사하면 안 되고, **주소를 검사한 뒤 그 주소를 써야 한다.**

---

## 우회 4: 리다이렉트

여기까지를 전부 지켰다고 하자. 파싱했고, 해석된 IP를 검사했고, 그 IP로 연결했다.
그런데 검사한 것은 **첫 번째 요청 하나뿐**이다.

허용 목록에 `example.com`이 있다고 하자. 공격자가 아래를 붙여넣는다.

```text
https://example.com/redirect?to=http://169.254.169.254/
```

- 첫 요청의 호스트는 `example.com`이다. **검사를 통과한다**
- 서버가 `302 Location: http://169.254.169.254/`를 응답한다
- HTTP 클라이언트가 **자동으로** 따라간다
- 이때는 아무도 검사하지 않는다

Open redirect가 있는 사이트는 흔하다. 그리고 공격자가 **자기 도메인**을 화이트리스트에 넣게 만들 필요도 없다. 이미 허용된 아무 사이트의 open redirect면 된다.

**교훈:** 리다이렉트 자동 추적을 끄고, **홉마다 처음부터 다시 검사**해야 한다.

---

## 우회 5: 검증한 값과 요청하는 값이 다를 때

앞의 넷은 **무엇을 검사할까**의 문제였다. 이건 다르다.
제대로 검사하고도 **검사한 것과 다른 것을 요청에 넘기는** 경우다.

```ts
if (!rawUrl.includes('allowed.com')) throw new Error('blocked')

await request(rawUrl) // 이 요청은 어디로 가는가
```

실제로 넣어 보면 이렇게 갈린다.

| 입력                              | `allowed.com` 문자열 | 실제 호스트   |
| --------------------------------- | -------------------- | ------------- |
| `http://allowed.com@evil.com/`    | 있다                 | `evil.com`    |
| `http://evil.com#@allowed.com/`   | 있다                 | `evil.com`    |
| `http://allowed.com%2f@evil.com/` | 있다                 | `evil.com`    |
| `http://allowed.com\@evil.com/`   | 있다                 | `allowed.com` |

---

## 우회 5: 그래서 파싱은 한 번만

- 1행은 URL의 `user@host` 문법이다. `allowed.com`은 **호스트가 아니라 사용자 이름**이다
- 2행은 `#` 뒤라 **프래그먼트**이고 목적지와 무관하다
- 3행은 `%2f`까지 포함해 사용자 이름으로 들어간다

문자열 검사는 넷 다 통과시키지만, **실제 목적지는 셋이 `evil.com`**이다.

4행은 방향이 반대라 더 눈여겨볼 만하다. 백슬래시가 들어가면 WHATWG 파서는 그것을 **경로 구분자**로 보고 호스트를 `allowed.com`으로 준다. 즉 **같은 문자열을 두고 파서마다 답이 갈린다.**

**교훈: 파싱은 한 번만 하고, 파싱한 그 객체를 그대로 요청에 넘긴다.**

---

## 우회 6: IPv6

마지막은 **잊어버려서** 생기는 구멍이다. IPv4 사설 대역만 검사하면 아래가 전부 통과한다.

| 주소               | 의미                                                |
| ------------------ | --------------------------------------------------- |
| `::1`              | 루프백                                              |
| `::ffff:127.0.0.1` | **IPv4-mapped.** 실제로는 127.0.0.1                 |
| `::ffff:a9fe:a9fe` | **`169.254.169.254`의 16진 표기.** 점 표기가 아니다 |
| `fe80::/10`        | link-local                                          |
| `fc00::/7`         | ULA (사설 대역에 해당)                              |
| `64:ff9b::/96`     | NAT64 (IPv4로 변환된다)                             |

**IPv4-mapped**가 특히 까다롭다. IPv6 형태지만 커널은 IPv4로 연결한다.

---

## 우회 6: 표기가 하나가 아니다

점 표기로 적어 넣어도, **파서는 16진 표기로 돌려준다.**

```ts
const u = new URL('http://[::ffff:169.254.169.254]/')
u.hostname // '[::ffff:a9fe:a9fe]'
```

`a9fe:a9fe`가 `169.254.169.254`를 16진수로 쓴 것이다 (`0xa9 = 169`, `0xfe = 254`).
그러니 `::ffff:169.254.169.254`를 문자열로 막아둔 코드는 **이 주소를 그냥 놓친다.**
우회 1과 같은 이야기가 IPv6에서 한 번 더 반복되는 셈이다.

> 그리고 위 출력에 하나가 더 보인다. **`hostname`에 대괄호가 남아 있다.**
> 이걸 주소로 착각하고 검사에 넘기면 어떻게 되는지는 뒤에서 본다.

---

## 방어 원리 다섯 줄

여섯 가지 우회를 관통하는 원리는 이렇게 정리된다.

1. **이름이 아니라 주소를 검증한다** (우회 1, 2, 6 대응)
2. **검증한 주소로 직접 연결한다** (우회 3 대응)
3. **리다이렉트는 홉마다 1~2를 반복한다** (우회 4 대응)
4. **한 번 파싱한 객체만 쓴다** (우회 5 대응)
5. **애플리케이션 방어를 믿지 않는다** (전부 대응)

하나씩 코드로 옮긴다.

---

## 원리 1: 주소를 검증한다

Node에는 `net.BlockList`가 내장되어 있다. 직접 비트 연산할 필요가 없다.

```ts
import {BlockList, isIP, isIPv4} from 'node:net'

const blocked = new BlockList()

blocked.addSubnet('0.0.0.0', 8, 'ipv4') //  this network
blocked.addSubnet('10.0.0.0', 8, 'ipv4') //  RFC1918
blocked.addSubnet('100.64.0.0', 10, 'ipv4') //  CGNAT
blocked.addSubnet('127.0.0.0', 8, 'ipv4') //  loopback
blocked.addSubnet('169.254.0.0', 16, 'ipv4') //  link-local (메타데이터)
blocked.addSubnet('172.16.0.0', 12, 'ipv4') //  RFC1918
blocked.addSubnet('192.168.0.0', 16, 'ipv4') //  RFC1918
blocked.addSubnet('192.0.0.0', 24, 'ipv4') //  IETF 프로토콜 할당 (DS-Lite 등)
blocked.addSubnet('198.18.0.0', 15, 'ipv4') //  벤치마크
blocked.addSubnet('224.0.0.0', 4, 'ipv4') //  멀티캐스트
blocked.addSubnet('240.0.0.0', 4, 'ipv4') //  reserved
```

---

## IPv6도 같이 막는다

```ts
blocked.addAddress('::', 'ipv6') //  unspecified
blocked.addAddress('::1', 'ipv6') //  loopback
blocked.addSubnet('64:ff9b::', 96, 'ipv6') //  NAT64 (RFC 6052)
blocked.addSubnet('64:ff9b:1::', 48, 'ipv6') //  NAT64 local-use (RFC 8215)
blocked.addSubnet('100::', 64, 'ipv6') //  discard
blocked.addSubnet('fc00::', 7, 'ipv6') //  ULA
blocked.addSubnet('fe80::', 10, 'ipv6') //  link-local
blocked.addSubnet('fec0::', 10, 'ipv6') //  site-local (deprecated, 구형 스택 대비)
blocked.addSubnet('ff00::', 8, 'ipv6') //  멀티캐스트
```

> 이 목록이 막아야 할 IPv6 대역의 **전부는 아니다.** IPv4 주소를 IPv6 안에 숨겨 넣는
> 방식이 여럿 있어서(바로 위 NAT64가 그런 예다), IPv6를 실제로 쓰는 환경이라면
> "IANA IPv6 Special-Purpose Address Registry"(예약 대역 공식 목록)를 한 번 보고 빠진 것을 더한다.

---

## 검사 함수는 짧게, 단 fail-closed로

검사 함수는 **이렇게 짧아야 한다.**

```ts
export function isPublicAddress(ip: string): boolean {
  if (isIP(ip) === 0) return false // ⚠️ IP 형식이 아니면 일단 막는다
  return !blocked.check(ip, isIPv4(ip) ? 'ipv4' : 'ipv6')
}
```

**첫 줄이 왜 필요한가.** `blocked.check()`는 IP가 아닌 문자열(`"localhost"`, `""`,
`"999.1.1.1"` 등)을 받으면 에러를 내는 게 아니라 **"막지 않음"으로 통과**시킨다.
그래서 IP 형식이 아닌 값은 우리가 먼저 잘라낸다. 애매하면 막는 쪽(fail-closed)이 안전하다.

그런데 여기서 **한 걸음 더 나가서 주소를 직접 쪼개 변환하고 싶어지는데, 그게 함정이다.**
다음 장에서 왜인지 본다.

---

## 손으로 언랩하면 오히려 뚫린다

앞 장의 IPv4-mapped 설명을 읽고 나면 이런 코드를 쓰고 싶어진다.

```ts
// ⚠️ 이 코드는 취약하다
const addr = ip.startsWith('::ffff:') ? ip.slice(7) : ip
return !blocked.check(addr, isIPv4(addr) ? 'ipv4' : 'ipv6')
```

Node 24에서 실제로 돌려본 결과다.

| 입력                     | `slice(7)` 결과   | 판정        |
| ------------------------ | ----------------- | ----------- |
| `::ffff:169.254.169.254` | `169.254.169.254` | 차단 ✅     |
| `::ffff:a9fe:a9fe`       | `a9fe:a9fe`       | **통과** ❌ |
| `::ffff:7f00:1`          | `7f00:1`          | **통과** ❌ |

16진 표기를 벗기면 **IPv4도 IPv6도 아닌 문자열**이 된다.
`BlockList.check()`는 이런 입력에 예외를 던지지 않고 **`false`(차단 안 함)를 반환한다.**

---

## `BlockList`는 이미 알아서 한다

같은 환경에서 **언랩하지 않고** 넣어보면,

| 입력                         | 결과    |
| ---------------------------- | ------- |
| `::ffff:169.254.169.254`     | 차단 ✅ |
| `::ffff:a9fe:a9fe`           | 차단 ✅ |
| `::ffff:7f00:1`              | 차단 ✅ |
| `64:ff9b::a9fe:a9fe` (NAT64) | 차단 ✅ |

Node의 `BlockList`는 **`::ffff:…` 형태(IPv6 껍데기만 씌운 IPv4, "IPv4-mapped")는 알아서 IPv4로 보고 검사한다.**
점 표기든 16진이든 가리지 않는다.

---

## 다만 자동인 것은 `::ffff:`뿐이다

**`64:ff9b::…` 형태(NAT64)는 알아서 풀어주지 않는다.** 위 표에서 이게 막힌 건
앞의 코드에 `64:ff9b::/96` 한 줄을 직접 넣어줬기 때문이다. 그 줄을 빼면 `64:ff9b::a9fe:a9fe`는
그냥 통과한다. 정리하면 **`::ffff:`만 자동이고, 나머지 "IPv4를 안에 품은 IPv6"는 대역마다 직접 막아야 한다.**

> **교훈이 둘이다.**
> ① 주소 정규화는 직접 하지 말고 **런타임이 제공하는 것을 쓴다**
> ② 보안 코드는 **반드시 우회 케이스로 테스트한다.** 위 표가 그 테스트다

---

## 그래서 테스트가 명세다

보안 코드는 "잘 동작한다"가 아니라 **"이것들을 막는다"**로 표현해야 한다.

```ts
describe('isPublicAddress', () => {
  const blockedCases = [
    '127.0.0.1',
    '169.254.169.254',
    '10.1.2.3',
    '172.16.0.1',
    '192.168.1.1',
    '100.64.0.1',
    '0.0.0.0',
    '::1',
    '::',
    'fe80::1',
    'fd00::1',
    'ff02::1',
    '::ffff:127.0.0.1',
    '::ffff:169.254.169.254',
    '::ffff:a9fe:a9fe',
    '::ffff:7f00:1', // 16진 표기
    '64:ff9b::a9fe:a9fe', // NAT64
    'localhost', // IP가 아닌 값이라 막아야 한다
    '', // 빈 문자열
    '999.1.1.1', // IP처럼 보이지만 유효하지 않다
    '127.0.0.1 ', // 뒤에 공백
  ]
```

---

## 허용 케이스도 같이 적는다

```ts
const allowedCases = [
  '93.184.216.34',
  '1.1.1.1',
  '2606:4700::1',
  '172.32.0.1',
  '100.128.0.1',
  '11.0.0.0', // 경계 바로 바깥
]

it.each(blockedCases)('%s 를 차단한다', (ip) =>
  expect(isPublicAddress(ip)).toBe(false),
)
it.each(allowedCases)('%s 를 허용한다', (ip) =>
  expect(isPublicAddress(ip)).toBe(true),
)
```

**경계 바로 바깥(`172.32.0.1`)을 넣는 이유:** 과차단도 버그다.

---

## 원리 2: 검증한 주소로 직접 연결한다

여기가 rebinding을 막는 지점이다. `undici`의 `lookup` 훅을 쓴다.

```ts
import {Agent} from 'undici'
import {lookup as dnsLookup} from 'node:dns'

export const scrapeAgent = new Agent({
  connect: {
    lookup(hostname, options, callback) {
      dnsLookup(hostname, {all: true}, (err, addresses) => {
        if (err) return callback(err, '', 0)

        // 하나라도 사설이면 전부 거부한다.
        // 공격자는 A 레코드를 여러 개 줄 수 있다.
        if (!addresses.every((a) => isPublicAddress(a.address))) {
          return callback(new Error('BLOCKED_PRIVATE_ADDRESS'), '', 0)
        }

        // ⚠️ undici는 이 콜백을 "주소 배열"로 답하길 기대한다(Node 20+ 기본 동작).
        //    { all: true } 결과를 그대로 넘기면 된다. 주소 하나만 넘기면
        //    첫 요청부터 "Invalid IP address: undefined"로 죽는다. (실측)
        if (options.all) return callback(null, addresses)
        callback(null, addresses[0].address, addresses[0].family)
      })
    },
  },
})
```

---

## 왜 이게 rebinding을 막는가

`lookup`이 반환한 주소로 **소켓이 곧바로 연결**되기 때문이다.

```text
기존 방식                          lookup 훅 방식
─────────                          ──────────────
① DNS 조회 → 검사                  ① DNS 조회
② fetch()                         ② 검사
   → 내부에서 DNS 재조회 ⚠️         ③ 검사한 그 IP로 소켓 연결 ✓
   → 다른 답이 올 수 있다              (재조회 없음)
```

**검사 시점과 사용 시점 사이에 DNS 조회가 한 번도 없다.** TOCTOU 창이 닫힌다.

> 이때도 요청에 실리는 호스트 이름(`Host` 헤더, HTTPS 인증서 확인용 이름)은 원래대로 유지된다.
> 즉 **한 IP에 여러 사이트가 얹힌 경우나 인증서 검증이 다 정상 동작한다.**

> 한 가지 더: 이 검사는 **새 연결을 열 때만** 돈다. 이미 열린 연결을 재사용(keep-alive)하면
> 다시 조회하지 않는데, 그 IP는 조금 전 검사를 통과한 것이라 안전하다.

---

## 왜 `every`인가, `find`가 아니라

공격자가 A 레코드를 이렇게 줄 수 있다.

```text
evil.com.  IN  A  93.184.216.34    ← 공인
evil.com.  IN  A  169.254.169.254  ← 사설
```

`find(isPublic)`으로 공인 하나만 골라 쓰면 당장은 안전해 보인다.

그러나 **정상적인 도메인이 사설 IP를 함께 반환할 이유가 없다.** 그런 응답 자체가 공격 신호다.

`every`로 전부 거부하는 편이 **의도를 정확히 표현**하고, 조회 순서나 캐시 상태에 따른 흔들림도 없앤다.

---

## 원리 3: 리다이렉트를 직접 따라간다

```ts
const MAX_HOPS = 3

async function fetchGuarded(startUrl: URL) {
  let url = startUrl

  // 리다이렉트 체인 전체에 거는 데드라인. 홉이 몇 번이든 이 시간은 넘기지 않는다.
  const overall = AbortSignal.timeout(8_000)

  for (let hop = 0; hop <= MAX_HOPS; hop++) {
    assertAllowedUrl(url) // 스킴, 포트, 도메인 화이트리스트

    const res = await request(url, {
      dispatcher: scrapeAgent, // redirect 인터셉터를 붙이지 않는다
      // 홉 하나당 5초 + 체인 전체 8초. 둘 중 먼저 걸리는 쪽에서 끊긴다
      signal: AbortSignal.any([overall, AbortSignal.timeout(5_000)]),
    })

    if (res.statusCode < 300 || res.statusCode >= 400) return res

    const location = res.headers.location
    if (!location) return res

    await res.body.dump() // 소켓을 반드시 비운다
    url = new URL(location, url) // 상대 경로도 처리된다
  }

  throw new Error('TOO_MANY_REDIRECTS')
}
```

---

## 진짜 위험한 건 `fetch()` 쪽이다

`undici.request()`는 **기본적으로 리다이렉트를 따라가지 않는다.** 따라가려면 인터셉터를 명시적으로 붙여야 한다.

```ts
new Agent().compose(interceptors.redirect({maxRedirections: 3})) // 이래야 따라간다
```

반면 **전역 `fetch()`는 기본이 `redirect: 'follow'`다.**

```ts
await fetch(userUrl) // ⚠️ open redirect를 그대로 따라간다
await fetch(userUrl, {redirect: 'manual'}) // 이렇게 꺼야 한다
```

---

## 실측: 같은 로컬 서버에 302를 물었다

| 호출                                 | 302 응답에 대해                      |
| ------------------------------------ | ------------------------------------ |
| `undici.request(url)`                | 302를 그대로 반환 ✅                 |
| `fetch(url)`                         | **따라가서 내부 본문을 가져왔다** ❌ |
| `fetch(url, { redirect: 'manual' })` | 302를 그대로 반환 ✅                 |

> 구버전 문서에 나오는 `maxRedirections: 0`은 undici 8에서 **0만 허용되고 그 외 값은 예외**다.
> 어차피 기본값이 추적 안 함이므로, **`fetch()`를 쓰지 않는 것**이 실질적인 방어다.

---

## 이 코드에서 놓치기 쉬운 두 줄

**`await res.body.dump()`**

본문을 읽지 않고 버리면 소켓이 커넥션 풀에 반납되지 않는다. 리다이렉트가 잦으면 커넥션이 고갈된다. `dump()`는 본문을 버리고 소켓을 정리한다.

**`new URL(location, url)`**

`Location` 헤더는 상대 경로일 수 있다 (`/login`, `../other`). 두 번째 인자로 기준 URL을 주면 표준대로 해석된다. 그리고 **이렇게 만든 `URL` 객체가 다음 루프에서 다시 검사받는다.**

---

## 원리 4: 스킴과 포트도 잠근다

`assertAllowedUrl`이 실제로 해야 하는 일이다.

```ts
const ALLOWED_PROTOCOLS = new Set(['http:', 'https:'])
const ALLOWED_PORTS = new Set(['', '80', '443'])

function assertAllowedUrl(url: URL) {
  if (!ALLOWED_PROTOCOLS.has(url.protocol)) {
    throw new BlockedError('SCHEME_NOT_ALLOWED')
  }
  if (!ALLOWED_PORTS.has(url.port)) {
    throw new BlockedError('PORT_NOT_ALLOWED')
  }
  if (url.username || url.password) {
    throw new BlockedError('CREDENTIALS_IN_URL') // user@host 우회 차단
  }
  if (!isAllowedHost(url.hostname)) {
    throw new BlockedError('HOST_NOT_ALLOWED')
  }
}
```

---

## 스킴을 왜 막는가

`http`, `https`가 아닌 스킴은 전혀 다른 공격면을 연다.

| 스킴        | 위험                                           |
| ----------- | ---------------------------------------------- |
| `file://`   | `/etc/passwd`, 애플리케이션 시크릿 파일        |
| `gopher://` | 임의 TCP 페이로드 전송 (Redis, SMTP 명령 주입) |
| `dict://`   | 내부 서비스 배너 수집                          |
| `ftp://`    | 내부 FTP 접근                                  |
| `data:`     | 파서 혼동 유발                                 |

`undici`는 `http`/`https`만 지원하므로 상당 부분 자동으로 막히지만, **명시적으로 거부하는 편이 안전하다.** 나중에 클라이언트를 교체해도 방어가 남는다.

---

## 포트는 왜 막는가

80, 443 외의 포트는 대개 **내부 서비스**다.

```text
6379  Redis
9200  Elasticsearch
5432  PostgreSQL
27017 MongoDB
8080  내부 어드민
2375  Docker daemon
```

IP 대역을 막았다면 이미 상당 부분 커버되지만, **DNS가 공인 IP를 가리키면서 그 뒤에 프록시가 있는 경우** 등에서 한 겹 더 필요하다.

> 보안은 겹으로 쌓는다. 한 겹이 뚫리는 것을 전제로 설계한다.

---

## 도메인 화이트리스트, 매칭에서 실수가 난다

앞의 `assertAllowedUrl`이 부르는 `isAllowedHost`는 생각보다 틀리기 쉽다.

```ts
const ALLOWED_HOSTS = new Set(['example.com'])

function isAllowedHost(hostname: string): boolean {
  if (ALLOWED_HOSTS.has(hostname)) return true // 정확히 일치
  // 서브도메인 허용 시 반드시 "." 경계를 붙인다
  return [...ALLOWED_HOSTS].some((d) => hostname.endsWith('.' + d))
}
```

**가장 흔한 실수:** `.` 없이 `endsWith("example.com")`로 검사하는 것.
그러면 `notexample.com`, `evil-example.com`이 통과한다. 공격자가 이런 도메인 하나만 사면 뚫린다.

서브도메인을 허용한다면 더 조심할 것: 아무도 관리 안 하는 서브도메인을 **공격자가 가로채는 경우**(subdomain
takeover)와, `exаmple.com`처럼 **눈엔 같지만 다른 글자**(키릴 `а`)를 쓴 도메인.

---

## 원리 5: 애플리케이션 방어를 믿지 않는다

지금까지 쓴 코드는 전부 **애플리케이션 레이어**에 있다.

- 라이브러리를 교체하면 사라진다
- 다른 개발자가 `fetch()`를 직접 쓰면 우회된다
- 우리가 모르는 우회 기법이 내일 나올 수 있다

**네트워크 레이어에서 한 번 더 막아야 한다.**

| 계층             | 통제                                             |
| ---------------- | ------------------------------------------------ |
| 인스턴스         | IMDSv2 강제, hop limit 1                         |
| 보안 그룹 / NACL | 아웃바운드에서 내부 대역 차단                    |
| Egress 프록시    | 모든 외부 요청을 한 지점으로 강제, 거기서 필터링 |
| VPC 설계         | 스크래핑 워크로드를 별도 서브넷으로 격리         |

---

## 다층 방어를 그림으로

```text
사용자 URL
    │
    ▼
┌───────────────────────────┐
│ ① 스킴/포트/자격증명 검사  │  ← 파싱된 URL 객체
├───────────────────────────┤
│ ② 도메인 화이트리스트      │  ← 정책 계층
├───────────────────────────┤
│ ③ DNS 해석 + IP 대역 검사  │  ← 보안 계층 (핵심)
├───────────────────────────┤
│ ④ 검사한 IP로 소켓 연결    │  ← rebinding 차단
├───────────────────────────┤
│ ⑤ 리다이렉트마다 ①~④ 반복 │
└───────────────────────────┘
    │
    ▼
┌───────────────────────────┐
│ ⑥ 보안그룹 아웃바운드 차단 │  ← 코드가 뚫려도 남는다
└───────────────────────────┘
```

---

## 화이트리스트: IP 기반 대 도메인 기반

여기서 흔한 오해 하나를 짚는다.

> "IP 화이트리스트가 도메인보다 안전하다"

**스크래핑에서는 그렇지 않다.**

| 방식        | 장점                  | 실제 문제                                                  |
| ----------- | --------------------- | ---------------------------------------------------------- |
| IP 기반     | 이름 조작에 영향 없음 | 대상이 CDN 뒤에 있으면 **그 CDN의 모든 사이트가 허용된다** |
| 도메인 기반 | 정책 표현이 정확함    | 해석된 IP를 검사하지 않으면 무의미하다                     |

---

## 둘은 목적이 다르다

| 목적                                                   | 맞는 도구                  |
| ------------------------------------------------------ | -------------------------- |
| **콘텐츠 정책** (어떤 사이트를 미리보기 허용할 것인가) | 도메인 화이트리스트        |
| **네트워크 보안** (내부망으로 못 나가게 한다)          | IP 대역 차단 + egress 통제 |

**둘 중 하나를 고르는 게 아니라, 둘 다 필요하다.**

도메인 화이트리스트는 "리딩방 링크를 막는다" 같은 **정책**을 표현한다.
IP 대역 차단은 "메타데이터를 못 읽는다"는 **보안**을 표현한다.
서로를 대체하지 못한다.

---

## 응답 크기 상한: 잊기 쉬운 DoS 벡터

공격자가 이런 URL을 준다.

```text
https://evil.com/infinite   →  응답이 끝나지 않는다
https://evil.com/10gb.html  →  거대한 HTML
```

`.text()`나 `.body()`로 전부 읽으면 **메모리가 터진다.**
크기를 세면서 읽다가, 상한을 넘으면 소켓을 끊어야 한다.

---

## 세면서 읽고, 넘으면 끊는다

```ts
const MAX_BYTES = 512 * 1024 // og 태그는 <head>에 있다. 512KB면 충분하다

async function readCapped(body: Readable): Promise<Buffer> {
  const chunks: Buffer[] = []
  let total = 0

  for await (const chunk of body) {
    total += chunk.length
    if (total > MAX_BYTES) {
      body.destroy() // 소켓을 끊는다
      throw new BlockedError('RESPONSE_TOO_LARGE')
    }
    chunks.push(chunk)
  }
  return Buffer.concat(chunks)
}
```

---

## `Content-Length`를 믿으면 안 된다

```ts
if (Number(res.headers['content-length']) > MAX_BYTES) throw ... // 불충분
```

이 검사만으로는 안 되는 이유가 셋이다.

1. `Transfer-Encoding: chunked`면 헤더가 **아예 없다**
2. 헤더 값이 **거짓일 수 있다** (상대 서버가 공격자의 것이다)
3. 압축된 경우 헤더는 압축 크기이고, **압축 해제 후 폭증**할 수 있다 (zip bomb)

**실제로 읽은 바이트를 세는 것만이 확실하다.** 헤더 검사는 조기 차단용 보조 수단으로만 쓴다.

---

## 타임아웃은 하나가 아니다

`timeout: 5000` 하나로 끝내면 안 된다. 단계별로 나눠야 원인 분류가 된다.

```ts
new Agent({
  connectTimeout: 2_000, // TCP + TLS 핸드셰이크
  headersTimeout: 3_000, // 요청 후 응답 헤더까지
  bodyTimeout: 3_000, // 본문 청크 사이의 최대 간격
})

// 그리고 전체 상한을 따로 건다
request(url, {signal: AbortSignal.timeout(8_000)})
```

**`bodyTimeout`은 청크(응답 조각) 사이의 간격이지 전체 시간이 아니다.** 응답을 1바이트씩 아주 느리게 흘려 연결을 오래 붙잡는 응답(slowloris형)은 `bodyTimeout`만으로 못 막는다. 그래서 전체 상한이 따로 필요하다.

그리고 이 전체 상한은 **요청 하나가 아니라 리다이렉트 체인 전체에 하나만** 건다. 앞의 `fetchGuarded`에서 루프 밖에 둔 `overall`이 그 역할이다. 홉마다 새로 5초를 주면, 리다이렉트가 3번 이어질 때 최악 20초까지 늘어난다.

---

## 보안 체크리스트

| 항목                           | 막는 것                                  |
| ------------------------------ | ---------------------------------------- |
| 스킴을 http/https로 제한       | `file://`, `gopher://`                   |
| 포트를 80/443으로 제한         | 내부 서비스 직접 접근                    |
| URL 자격증명 거부              | `user@host` 파서 혼동                    |
| 도메인 화이트리스트            | 콘텐츠 정책                              |
| **DNS 해석 IP 대역 검사**      | 사설망, 메타데이터                       |
| **IPv4-mapped / NAT64**        | `::ffff:a9fe:a9fe` (직접 언랩하지 말 것) |
| **검사한 IP로 연결 (pinning)** | DNS rebinding                            |
| **리다이렉트 홉별 재검증**     | open redirect 경유                       |

---

## 보안 체크리스트 (이어서)

| 항목                        | 막는 것                               |
| --------------------------- | ------------------------------------- |
| 응답 크기 상한              | 메모리 고갈                           |
| 단계별 타임아웃 + 전체 상한 | slowloris                             |
| 보안그룹 아웃바운드 차단    | 위가 전부 뚫렸을 때                   |
| IMDSv2 강제, hop limit 1    | 크레덴셜 탈취                         |
| 요청자별 rate limit / 인증  | 우리 서버를 익명 프록시·중계기로 악용 |

---

## 2부의 주장은 실측했다

글을 쓰고 나서 코드를 전부 돌려봤고, **이 부에서만 셋이 틀렸다.**

| 처음에 쓴 것                             | 실측 결과                                                                    |
| ---------------------------------------- | ---------------------------------------------------------------------------- |
| lookup 훅에서 주소를 **하나만** 돌려준다 | **첫 요청부터 죽는다.** undici는 `{ all: true }`로 부르고 배열 답을 기대한다 |
| `::ffff:` 를 손으로 벗겨 검사한다        | **16진 표기에서 뚫린다.** `BlockList`에 그대로 넘겨야 한다                   |
| `maxRedirections: 3` 으로 제한한다       | undici 8에서 **예외.** redirect 인터셉터를 써야 한다                         |

검증 환경은 Node `v24.14.1`, undici `8.10.0`이다.

> **이게 이 시리즈가 하려는 말 그 자체다.**
> 그럴듯한 보안 코드와 동작하는 보안 코드는 다르다. 차이는 **돌려봤는가**이다.

---

## 직접 재현하려면

```bash
npm i undici

# IPv4-mapped 16진 표기 우회
node -e "const{BlockList,isIPv4}=require('net');
const b=new BlockList();b.addSubnet('169.254.0.0',16,'ipv4');
const bad=ip=>{const a=ip.startsWith('::ffff:')?ip.slice(7):ip;
  return !b.check(a,isIPv4(a)?'ipv4':'ipv6')};
const good=ip=>!b.check(ip,isIPv4(ip)?'ipv4':'ipv6');
console.log('손수 언랩:',bad('::ffff:a9fe:a9fe'),'(true면 취약)');
console.log('그대로   :',good('::ffff:a9fe:a9fe'),'(false면 안전)')"
```

손으로 언랩한 쪽만 통과시킨다. **그게 이 부의 요지다.**

---

## 되게 만드는 법은 1부에 있다

2부는 **뚫리지 않는 법**이었다. 반대쪽인 **되게 만드는 법**은 1부 3~5장에서 다뤘다.

- 왜 User-Agent 하나가 에러율을 가르는가
- 한국어 사이트의 인코딩은 왜 아직도 깨지는가 (여기서 Node가 불리하다)
- 그렇게 가져온 값을 그대로 화면에 그려도 되는가 (이 부와 반대 방향의 위험)
- 캐시는 무엇을 고치고 무엇을 못 고치는가, 스탬피드는 왜 특히 나쁜가
- "P95 1초 미만"이라는 목표를 히트율에서 역산하는 법

**두 부가 이 시리즈의 전부다.**

---

## 참고

**보안**

- OWASP, _Server Side Request Forgery Prevention Cheat Sheet_
- AWS, _Use IMDSv2_ (EC2 User Guide)
- RFC 1918 (사설 주소), RFC 6598 (CGNAT), RFC 4193 (IPv6 ULA)

**표준**

- WHATWG URL Standard

**라이브러리**

- `undici` Dispatcher / Agent 문서 (`connect.lookup`)
- Node.js `net.BlockList`
