---
title: '방문이 드문 블로그의 <em>핵심 웹 지표</em>를 직접 수집하기'
tags:
  - web-performance
  - devops
  - debugging
  - backend
published: true
date: 2026-10-23 13:31:00
updated: 2026-10-23 20:34:50
description: '공용 관측 인프라를 기준으로 만든 핵심 웹 지표 수집을 개인 블로그로 옮겼다. 방문이 드물어 비어 있던 대시보드를 고치고, 일반 브라우저에 맞춰 수집 시점을 바꿨다. 10월 17일부터 22일까지의 변경 과정과 23일에 확인한 조회 누락, 닷새치 데이터로 판단할 수 있는 범위를 기록했다.'
series: 'Core Web Vitals 직접 수집하기'
seriesOrder: 4
art:
  undraw: no-data
---

## Table of Contents

## 첫날 대시보드가 비어 있었다

블로그에 수집 코드를 붙이고 몇 번 방문한 뒤 Mimir 대시보드를 열자 모든 패널이 비어 있었다. 재현 환경에서 페이지 로드 300번을 흘려 넣었을 때는 숫자가 잘 나오던 대시보드였고, 수집 경로도 그대로 옮긴 것이었다. 달라진 것은 방문의 빈도였다. 경로를 고친 뒤 Mimir를 비우고 다시 쌓기 시작해서 두 시간 가까이 들어온 페이지 로드가 7건이었다.

3편까지는 공용 관측 인프라의 설정을 바꿀 수 없었고, 앱 웹뷰는 떠날 때 신호를 주지 않는 경우가 있었다. 블로그에서는 월 \$12짜리 서버 한 대에 수집 서버를 직접 띄우고 collector부터 대시보드까지 설정을 바꿀 수 있었다. 방문자도 일반 브라우저로 들어온다. 웹뷰에서 값을 놓치지 않으려고 넣었던 장치가 여기서도 필요한지, 방문이 드물어도 같은 집계 방식이 맞는지 다시 확인해야 했다.

이 글은 2026년 10월 17일과 18일에 수집 서버를 블로그에 붙이고, 22일에 배포와 입력 검사를 보완한 기록이다. 23일에는 닷새치 데이터를 조회하다 생긴 Tempo 오류의 원인도 확인했다. 번호를 붙인 소제목은 커밋 순서를 따른다. 마지막에는 약 121시간 동안 쌓인 값으로 앞의 선택을 다시 살펴본다.

처음에는 표본이 적어도 건수가 맞게 나오도록 저장과 조회 방식을 고쳤다. 이어서 quiet LCP 확정과 주기적인 metric 전송을 블로그에 맞게 바꾸고, 손실을 확인할 패널을 만들었다. 이 과정에서 수집 코드뿐 아니라 대시보드와 배포 경로도 검증해야 한다는 문제가 남았다.

> 수집 서버의 구성은 블로그 저장소의 [`infra/otel`](https://github.com/yceffort/blog/tree/e02c600a7ff73eaf8d507d153a3a21d26a3670c8/infra/otel)에, 브라우저 쪽 수집 코드는 [`apps/blog/src/utils/telemetry`](https://github.com/yceffort/blog/tree/e02c600a7ff73eaf8d507d153a3a21d26a3670c8/apps/blog/src/utils/telemetry)에 있다. 링크는 10월 18일의 마지막 커밋으로 고정했다. 22일에 추가한 입력 검사와 배포 수정은 [`e9f75f77`의 `infra/otel`](https://github.com/yceffort/blog/tree/e9f75f77a6d391e7eae0d9da38a49c8ff5fb9388/infra/otel)을 기준으로 설명한다.
>
> 버전은 web-vitals 6.2.3, OTel JS 2.12.0(OTLP exporter 0.223.0), OpenTelemetry Collector contrib 0.163.1, Tempo 3.1.0, Mimir 3.2.2, Grafana 13.2.4, Caddy 2.11.7이다. 서버는 Amazon Lightsail 서울 리전의 Debian 13 인스턴스(2GB 메모리)다.

## 1. 수집 서버를 블로그 저장소로

수집 서버는 처음에 재현 환경 저장소의 배포용 덮어쓰기 설정으로 띄웠다가, 실험과 운영을 섞지 않으려고 블로그 저장소의 `infra/otel`로 옮겼다. 서버는 블로그 저장소에서 이 디렉터리만 sparse checkout으로 받아 `docker compose up`으로 돈다. 구성은 Caddy, collector, Tempo, Mimir, Grafana 다섯 개다.

호스팅은 서버와 디스크, 공인 IPv4, 월 전송량이 한 요금에 포함된 Lightsail을 골랐다. 작은 서버 한 대의 비용을 예상하기 쉬웠다. 스냅샷 같은 부가 기능과 전송량 초과분은 별도 요금이다([Lightsail 요금표](https://aws.amazon.com/lightsail/pricing/)).

| 선택지                       | 한 달 비용              | 걸린 점                                                                                                                         |
| ---------------------------- | ----------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| Lightsail `small_3_0`        | \$12                    | 2GB 메모리, vCPU 2개, 디스크 60GB, 전송 3TB, 공인 IPv4 1개가 들어 있다                                                          |
| 같은 AWS의 EC2               | 인스턴스 값에 따로 더함 | 디스크(EBS)와 공인 IPv4(시간당 \$0.005, 한 달 약 \$3.6)가 따로 붙는다                                                           |
| Oracle Cloud Always Free(A1) | 무료                    | 7일 동안 CPU 사용률의 95번째 백분위, 네트워크, 메모리가 모두 20% 미만이면 유휴로 보고 회수할 수 있다[^1]. 수집 서버는 거의 쉰다 |

> Lightsail 가격은 서울 리전에서 Lightsail API(`get-bundles`)로 확인한 값이고, EC2의 공인 IPv4 요금은 AWS의 VPC 요금표[^2]에서 확인했다. 2026년 10월 기준이고, 10월 22일에 다시 조회해도 같았다.

첫날 오후의 메모리는 Mimir 115MiB, Grafana 243MiB, Tempo 158MiB, collector 86MiB, Caddy 19MiB로, 다섯 개를 합쳐 2GB 중 약 620MiB였다. 이 중 Grafana는 기본 설치 그대로 띄웠을 때 가장 컸다. Grafana 13은 처음 뜰 때 쓰지 않는 앱 플러그인을 받아 설치하고, 데이터소스 플러그인을 쓸 때마다 프로세스를 따로 띄운다. 그래서 미리 설치하는 플러그인을 끄고(`GF_PLUGINS_PREINSTALL_DISABLED`), Mimir와 Tempo 말고는 내장 데이터소스 플러그인 11개를 꺼 두었다(`GF_PLUGINS_DISABLE_PLUGINS`). 같은 데이터소스와 대시보드를 넣어 띄우고 한 번씩 조회한 뒤 재 보면 차이는 이 정도였다.

| Grafana 13.2.4  | 5분 뒤 | 6분 뒤 | 데이터소스 플러그인 프로세스 | 미리 설치된 플러그인 |
| --------------- | ------ | ------ | ---------------------------- | -------------------- |
| 기본 설치       | 222MiB | 261MiB | 13개                         | 5개                  |
| 플러그인을 줄임 | 159MiB | 201MiB | 2개                          | 0개                  |

> 로컬 Docker에서 두 대를 같은 시각에 띄워 쟀다. 메모리는 1분 사이에도 40MiB 안팎으로 오르내렸다.

줄여도 아주 작아지지는 않는다. 운영 서버의 Grafana는 플러그인을 줄인 상태로 그날 오후에 243~292MiB였다. 2GB 서버에서 가장 큰 컨테이너인 것은 그대로다.

### 받는 쪽은 누구나 보낼 수 있는 주소다

브라우저가 직접 보내는 수집 주소는 인터넷에 열어 두었다. 브라우저 코드에 공통 비밀키를 넣어도 누구나 복사할 수 있으므로, 그 키로 실제 블로그 방문을 구별할 수는 없다. 이 구성에서는 방문자 인증을 추가하지 않고 받는 요청의 범위를 좁혔다. Caddy는 OTLP/HTTP의 두 경로만 collector로 넘긴다.

```text
# infra/otel/caddy/Caddyfile
# OTLP/HTTP 두 경로만 collector 로 넘긴다. CORS 는 collector 가 처리한다.
{$OTEL_DOMAIN} {
    @otlp path /v1/traces /v1/metrics
    handle @otlp {
        # 떠날 때의 keepalive fetch 는 본문 합계가 64KiB 로 제한된다(fetch 명세).
        request_body {
            max_size 256KB
        }
        reverse_proxy collector:4318
    }
    respond 404
}
```

다른 경로는 404이고, 256KB를 넘는 본문은 413으로 끊는다. CORS는 collector에서 블로그 origin만 허용하지만, curl로 직접 보내는 요청은 막지 못한다. 이때 라벨을 제한한 곳은 브라우저 수집 코드뿐이어서, 직접 만든 OTLP 요청에는 임의의 metric 이름과 라벨을 넣을 수 있었다. 서버의 입력 검사는 10월 22일에 보완한다.

Grafana는 서버의 127.0.0.1에만 붙여 SSH 터널로 연다. Caddy의 접근 로그는 켜지 않았고 collector도 접속 IP를 속성으로 추가하지 않는다. 이 수집 경로에서는 방문자 IP를 따로 기록하지 않도록 구성했다.

### Tempo의 기본 조회 범위로는 표본이 부족했다

Tempo 3.1.0을 기본값으로 두면 trace 검색은 7일, TraceQL metrics는 24시간보다 긴 구간을 거부한다.

```text
range specified by start and end exceeds 168h0m0s. received start=1791535470 end=1792226640
metrics query time range exceeds the maximum allowed duration of 24h0m0s
```

> 기본 설정의 Tempo 3.1.0에 8일 구간의 검색과 2일 구간의 TraceQL metrics를 보내 받은 응답이다.

방문이 드문 블로그에서는 라벨별로 나누면 하루치 표본이 몇 건 되지 않았다. 그래서 보존 기간과 조회 범위를 함께 1년으로 늘렸다(`max_search_duration`, `max_metrics_duration`, `block_retention`). 보존 기간만 늘리면 데이터는 남아 있는데 조회는 막힌다.

## 2. 블로그에 붙이다

브라우저 쪽은 재현 환경의 수집 코드를 블로그 구조에 맞게 옮겼다. production에서만 켜고, OTel SDK는 첫 화면에 필요 없으므로 동적 import로 따로 받는다. 이 청크는 10월 22일의 빌드(커밋 466ad1b1)에서 raw 100,571B, gzip 29,365B다. 수집 코드와 SDK 버전은 둘째 날 이후 바뀌지 않았다. web-vitals는 늦게 붙어도 buffered 항목으로 이미 발생한 LCP와 CLS를 수집할 수 있지만, Event Timing의 buffered 항목은 104ms 이상인 것만 남으므로 SDK를 받기 전의 짧은 입력은 INP 후보에서 빠진다. 화면은 홈, 글, 태그, 시리즈, 그 밖의 다섯 가지(`page.route`)로만 나눈다.

## 3. 대시보드가 비어 있었다

서론의 빈 대시보드가 이때다. 원인을 찾으려고 방문이 드문 사이트를 흉내 내어, 같은 LCP 값 8건을 수집 경로 네 가지로 동시에 보내고 대시보드 쿼리가 몇 건으로 세는지 쟀다. 다섯 개의 라벨 조합은 한 번씩, 하나는 7분 간격으로 세 번 보냈다.

| 경로                                                              | 대시보드가 센 값 |
| ----------------------------------------------------------------- | ---------------- |
| 재현 환경 구성(`delta_to_cumulative`, remote write, `increase()`) | 0.00             |
| 위와 같고 누적값 만료(`max_stale`)만 7일                          | 2.43             |
| `max_stale` 7일, OTLP 수신, 시작 시각에 0 샘플 추가               | 10.66            |
| 누적으로 바꾸지 않고 delta를 그대로 저장, `sum_over_time()`       | 8.00             |

> 세 번 실행했고 재현 환경 구성의 첫 실행만 같은 run 라벨을 다시 써서 앞 실행의 상태를 이어받았다. 나머지 두 번은 같았다. `cwv-otel-grafana/experiments/e8-sparse-counts`.

라벨 조합마다 샘플이 하나뿐이면 `increase()`는 그 시리즈를 결과에서 뺀다. 표본이 둘 미만이면 값을 내지 않으며, 표의 0.00은 빈 결과를 패널이 0으로 보인 것이다. Mimir 3.2.2의 기본 엔진에서도 표본 하나짜리 시리즈 20개에 `increase()`를 걸면 결과가 비었다.

누적값 만료를 늘려도 한 번만 온 조합은 여전히 샘플 하나다. 시작 시각에 0 샘플을 넣으면 증가로 잡히지만, `increase()`가 조회 구간 경계까지 외삽하면서 8건을 10.66건으로 셌다. delta histogram은 각 전송 구간에서 기록한 건수와 버킷을 담으므로, 저장된 값을 더해 측정 건수를 셀 수 있다. 브라우저가 한 번에 여러 내비게이션의 값을 보내면 샘플 하나에 여러 건이 들어갈 수도 있다.

그래서 collector에서 `delta_to_cumulative`를 빼고, Mimir의 delta 저장(`-distributor.otel-native-delta-ingestion`)으로 받았다. 대시보드는 `sum_over_time()`으로 건수를 센다. Mimir 3.2.2에서 이 기능은 아직 실험 단계이므로 버전을 올릴 때 저장과 조회 방식의 변경 여부를 확인해야 한다.

시각도 바꿨다. 브라우저마다 시계가 달라서, 같은 시리즈에 먼저 들어온 표본보다 시각이 이른 표본이 오면 Mimir가 받지 않는다. 그래서 collector가 받은 시각으로 바꿔 저장한다.

```yaml
# infra/otel/collector/config.yaml
  transform/receive_time:
    metric_statements:
      - context: datapoint
        statements:
          - set(start_time, Now())
          - set(time, Now())
```

이 방식에는 두 가지 제약이 있다. 값의 시각이 페이지 로드 시각이 아니라 collector가 받은 시각이 되고, 같은 시리즈의 값 두 개가 같은 밀리초에 들어오면 같은 시각의 표본이 되어 Mimir가 하나를 버린다. 시리즈 하나에 초당 10건이면 1.3~1.5%를 잃었다(3편, `cwv-otel-grafana/experiments/e10-concurrent-writers`). 블로그 트래픽에서는 드물 것으로 보았고, 이 손실은 뒤에서 손실 패널을 만들 때 다시 나온다.

## 4. 누구에게, 무엇이 느린가

블로그는 화면이 다섯 가지뿐이라, 화면만으로는 어디가 느린지 좁히기 어려웠다. 그래서 화면과 접속 환경을 구분할 수 있도록 값의 종류가 정해진 metric 라벨을 더했다. 언어(`page.lang`, `/en`으로 시작하면 en), 기기(`device.class`, 화면 폭 768px과 1024px으로 나눈 세 가지), 브라우저 엔진(`browser.engine`)이다. 브라우저 이름만으로 엔진을 구별할 수 없어 엔진 정보도 따로 남긴다. 유입 경로(검색, 소셜, AI 서비스 등), 서비스 워커를 거친 내비게이션인지, 네트워크와 기기 정보는 값의 종류가 많거나 개별 방문을 조사할 때 필요한 정보여서 span 속성에만 싣는다.

LCP 요소의 selector에서는 StyleX의 원자 클래스를 지운다.

```ts
// apps/blog/src/utils/telemetry/mask.ts
    // StyleX 원자 클래스(`x1j9s4mx`)는 스타일 하나씩이라 요소를 구분하지 못하고, 스타일을 고치면 이름이 바뀐다.
    // 남겨 두면 같은 요소가 배포마다 다른 값으로 갈린다. 숫자가 많으면 아래 규칙에 걸려 `x{n}` 이 되기도 한다.
    .replace(/\.x[0-9a-z]{5,8}(?![\w-])/g, '')
```

블로그의 스타일은 StyleX가 만드는 원자 클래스라, web-vitals가 만든 selector에는 그 클래스가 줄줄이 붙는다. 그대로 두면 Tempo 대시보드의 "LCP 요소별" 패널이 같은 요소를 배포마다 다른 요소로 센다.

## 5. 봇, SPA 이동, bfcache

이때 고친 것은 세 가지다. 첫째는 봇이다. PageSpeed Insights(`Chrome-Lighthouse`), headless 브라우저, 검색 엔진의 렌더러가 블로그를 열면 그 값도 수집됐다. 사람의 방문이 드문 블로그에서는 이런 값이 섞이면 분위수가 쉽게 흔들린다. 그래서 SDK를 받기 전에 user agent와 `navigator.webdriver`로 거른다. 이 판정은 뒤의 Playwright 검증도 걸러 내므로, reader 비교 실험은 별도 데모 페이지에서 진행했다.

둘째는 SPA 이동이다. 블로그 안의 링크 이동은 클라이언트 이동이라, 지금까지는 첫 로드 하나의 값만 있었다. web-vitals의 `reportSoftNavs`를 켜면 Chromium 151 이상에서 이동마다 FCP, LCP, CLS, INP를 다시 잰다. TTFB는 이동에서 0으로 보고되므로 첫 로드에만 잰다. 셋째는 bfcache 복원이다. 첫 로드, SPA 이동, bfcache 복원을 각각 내비게이션 하나로 보고 trace도 따로 만든다. SPA 이동은 web-vitals가 주는 `navigationId`로, bfcache 복원은 `pageshow`의 `persisted`를 센 횟수로 가른다. 대시보드에는 첫 로드와 SPA 이동을 고르는 변수를 두었다.

## 6. 일반 브라우저에 맞춰 수집 시점을 바꾸다

수집 코드와 대시보드를 검토하면서, 웹뷰를 위해 넣은 quiet 확정과 주기 export를 블로그에서도 유지할 이유가 있는지 살펴봤다.

### quiet LCP는 확정값 대신 진단에 쓴다

3편의 quiet LCP는 후보가 3초 동안 바뀌지 않으면 값을 확정하는 장치였다. hidden이 오지 않는 앱 웹뷰에서 LCP를 남기기 위해 넣었다. 수집을 시작한 날 오후까지 블로그의 Tempo에 쌓인 LCP 9건 중 7건도 quiet로 확정됐다. 글을 읽는 동안에는 3초 넘게 입력이 없을 수 있어서, quiet가 web-vitals보다 먼저 끝나는 일이 흔할 것으로 보았다.

일반 브라우저에서는 첫 입력이나 hidden으로 LCP를 확정할 수 있다(web-vitals 6.2.3의 [`onLCP.ts`, `finalizeLCP`](https://github.com/GoogleChrome/web-vitals/blob/v6.2.3/src/onLCP.ts)). 신호를 받지 못할 가능성은 남지만, quiet로 먼저 확정하면 나중에 그려진 큰 요소를 놓친다는 3편의 문제도 그대로 남는다. 블로그에서는 정식 보고를 기다리기로 했다.

그래서 블로그는 web-vitals가 확정한 LCP만 histogram에 넣는다. quiet 타이머는 남겨 두되, 처음으로 3초 멈췄던 후보의 값을 span 속성으로만 싣는다(`cwv.lcp.quiet`가 `none`, `same`, `lower` 중 하나). `lower`는 quiet로 확정했다면 LCP를 작게 기록했을 페이지 로드다. 이 비율이 얼마나 되는지는 "닷새 동안 쌓인 값" 절에서 본다.

### metric은 hidden에서 앞선 응답을 기다리지 않고 보낸다

2편에서는 pagehide의 flush를 빼고 visibilitychange에서만 flush하도록 고쳤다. 그런데 sdk-metrics의 `forceFlush`는 진행 중인 export가 있으면 그 응답을 기다린다. hidden이 오는 순간 10초 주기 export가 나가 있으면 CLS와 INP의 수집도 지연된다. 앱 웹뷰에서는 hidden 없이 값을 보낼 기회가 필요했지만, 블로그에서는 hidden에서의 전송을 우선했다. metric의 주기 export를 없애고, flush할 때 앞선 응답을 기다리지 않고 collect와 export를 시작하는 reader로 바꿨다. span의 5초 주기 export는 유지했다.

```ts
// apps/blog/src/utils/telemetry/flush-reader.ts
export class FlushOnlyMetricReader extends MetricReader {
  constructor(private readonly exporter: PushMetricExporter) {
    super({
      aggregationSelector: exporter.selectAggregation?.bind(exporter),
      aggregationTemporalitySelector:
        exporter.selectAggregationTemporality?.bind(exporter),
    })
  }

  protected async onForceFlush(): Promise<void> {
    const {resourceMetrics} = await this.collect()
    if (resourceMetrics.scopeMetrics.length === 0) return
    await new Promise<void>((resolve) =>
      this.exporter.export(resourceMetrics, () => resolve()),
    )
  }

  protected async onShutdown(): Promise<void> {
    await this.exporter.shutdown()
  }
}
```

delta는 직전 collect 이후 기록한 값을 보내므로, 응답을 기다리지 않고 다음 수집을 시작할 수 있다. 블로그 페이지는 자동화 브라우저를 거르므로, 재현 환경의 데모 페이지에 이 reader를 그대로 복사해 넣고 두 reader를 Chromium에서 비교했다(`cwv-otel-grafana/experiments/e15-flush-only-reader`). 로드 1초 뒤 탭을 숨겨 첫 flush를 보내고, collector 앞의 프록시가 그 응답을 5초 늦춘다. 300ms 뒤 탭으로 돌아와 버튼을 누르고, 1.5초 뒤 다른 페이지로 떠난다. 첫 flush에는 TTFB, FCP, CLS가 실리고, 떠날 때의 두 번째 flush에 LCP와 INP가 실린다.

| 수집 코드                                    | 첫 flush의 응답 지연 | 떠날 때의 두 번째 요청 | Mimir에 도착한 LCP, INP |
| -------------------------------------------- | -------------------- | ---------------------- | ----------------------- |
| 주기 export(`PeriodicExportingMetricReader`) | 5초                  | 오지 않음              | 0건, 0건                |
| 주기 export(`PeriodicExportingMetricReader`) | 0                    | 떠난 뒤 10ms           | 1건, 1건                |
| `FlushOnlyMetricReader`                      | 5초                  | 떠난 뒤 12ms           | 1건, 1건                |
| `FlushOnlyMetricReader`                      | 0                    | 떠난 뒤 11~13ms        | 1건, 1건                |

> 10월 22일에 Playwright 1.64.1의 Chromium으로 조건마다 두 번 실행해 같았다. headless Chromium은 탭을 바꿔도 hidden이 되지 않아, 첫 hidden은 `visibilityState`를 덮어쓰고 `visibilitychange`를 만들어 냈다. 스크립트로 만든 이벤트는 `isTrusted`가 아니라서 web-vitals는 이 hidden으로 LCP를 확정하지 않고(1편), 그래서 LCP가 떠날 때의 요청에 실렸다. 실제 탭 전환에서는 LCP도 첫 flush에 실린다. 측정은 blog-experiments 커밋 [`be34dc4`](https://github.com/yceffort/blog-experiments/tree/be34dc4/cwv-otel-grafana/experiments/e15-flush-only-reader)에 있다.

hidden이 끝내 오지 않으면 metric은 전송되지 않는다. span은 주기 export로 먼저 도착할 수 있다. trace와 metric의 건수 차이를 살펴보기로 했지만, 조회 시점에 아직 열려 있는 탭과 전송에 실패한 탭은 이 차이만으로 구별할 수 없다.

### 같은 id의 갱신과 같은 밀리초의 중복은 재기만 했다

다른 두 건은 고치는 대신 재기만 했다. 하나는 같은 id로 다시 오는 CLS와 INP다. histogram은 첫 보고만 세므로, 탭에 돌아온 뒤에 바뀐 값은 들어가지 않는다. 다른 하나는 앞에서 본 같은 밀리초 중복이다. 3편의 gateway는 측정 id별 최신값을 보관하고 시리즈별로 합쳐 보내므로 두 문제에 대응할 수 있다. 상태를 보관하는 서버를 하나 더 운영하면서 갱신을 반영하는 시간과 재시작 때 잃는 상태를 관리해야 한다. 시간당 몇 건인 트래픽에서는 그 정확도 차이를 데이터로 확인하기도 어렵다. 그래서 다시 온 보고는 histogram 대신 `cwv.cls.update`, `cwv.inp.update` span으로 남기고(첫 값, 등급이 바뀌었는지, 몇 번째 보고인지), Mimir가 버린 표본은 대시보드에 보이게 했다. 둘 다 "닷새 동안 쌓인 값" 절의 값을 보고 다시 판단한다.

[1편](/2026/10/web-vitals-report-timing)에서 찾은 INP 손실도 알고서 그대로 두었다. 블로그는 web-vitals 6.2.3을 쓰고 INP를 기본 설정으로 받는다. 새 상호작용 뒤 1초 안에 떠나면 그 입력이 INP 후보에 반영되기 전에 문서가 사라질 수 있다. INP만 `reportAllChanges`로 받는 우회는 첫 보고만 세는 이 구조와 맞지 않아 넣지 않았고, 수정(#809)이 담긴 버전이 나오면 web-vitals를 올려서 해결할 생각이다. 올린 뒤에는 1편처럼 떠나는 페이지 로드를 만들어 지표별 보고 수를 다시 센다.

## 7. 손실 패널이 손실을 숨기고 있었다

손실 패널은 지표 수집이 끊겨도 0을 보여 주고 있었다. 버린 값이 없는 경우와 손실 지표를 받지 못한 경우를 구별하지 못하는 쿼리였다.

collector는 1분마다 Mimir의 `cortex_discarded_samples_total`과 자기 지표(받지 못한 수, 내보내지 못한 수)를 긁어 Mimir에 넣는다. 처음 패널 쿼리는 이랬다.

```promql
max by (reason) (max_over_time(cortex_discarded_samples_total{job="mimir", user="blog"}[$__range])) or on() label_replace(vector(0), "reason", "없음", "", "")
```

첫째, `or vector(0)`는 손실 지표를 아예 긁지 못한 상태도 0으로 보여 준다. 수집 설정이 빠진 새 서버에서도, collector가 멈춘 뒤에도 패널은 "없음 0"이었다. 둘째, `max_over_time`은 조회 구간의 과거 최댓값이라 Mimir가 재시작해 카운터가 다시 시작해도 재시작 전 값을 보여 준다. 셋째는 손실과 상관없는 버그였다. 대시보드 생성 스크립트가 refId를 매기는 반복문을 마지막 패널에만 돌리고 있었는데, 손실 패널을 그 뒤에 붙이면서 대상이 바뀌어 "지표별 표본 수" 패널의 쿼리 다섯 개가 모두 refId A가 됐다. Grafana는 이 패널을 `400 query.duplicateRefId`로 거부했다. 대시보드를 화면으로 열어 보지 않고 쿼리만 확인한 탓에 놓쳤다.

고친 쿼리는 마지막 수집이 성공했고(`up == 1`) 3분 안일 때만 값을 보여 주고, 카운터의 최신값을 쓴다. 이 값은 조회 범위와 상관없이 Mimir가 시작된 뒤의 누적값이라 패널 제목에도 그렇게 적었다.

```promql
(max by (reason) (cortex_discarded_samples_total{job="mimir", user="blog"}) or on() label_replace(vector(0), "reason", "없음", "", "")) and on() max((up{job="mimir"} == 1) and (time() - timestamp(up{job="mimir"}) < 180))
```

로컬에 같은 구성을 띄우고 fixture로 비교했다. 고친 쿼리는 첫 수집 전에도 "수집 안 됨"이었다.

| 상황                               | 처음 쿼리 | 고친 쿼리  |
| ---------------------------------- | --------- | ---------- |
| collector를 멈추고 4분 뒤, 6분 뒤  | 없음 0    | 수집 안 됨 |
| 버린 표본 10, Mimir 재시작, 다시 5 | 10        | 5          |

> 실제 Grafana 13.2.4의 `/api/ds/query`로 패널 쿼리를 보내 확인했고, 고친 뒤에는 Mimir 패널 30개가 모두 200이었다. 운영 서버의 Grafana에서도 같았다.

## 8. 복원된 페이지의 TTFB가 첫 로드로 세어졌다

다음 날에는 bfcache 복원 지표가 서로 다른 trace에 들어가는 문제를 고쳤다. 복원의 TTFB는 web-vitals가 `pageshow` 리스너 안에서 바로 보고한다(1편의 `onBFCacheRestore`). 수집 코드도 window의 `pageshow`에서 복원 횟수를 세고 그 횟수를 키로 썼는데, web-vitals는 capture로 걸었고 수집 코드는 bubble로 걸었다. 현행 DOM 명세에서는 같은 타깃이라도 capture 리스너가 등록 순서와 상관없이 먼저 돌므로, 복원의 TTFB는 횟수가 늘기 전에 보고되어 다른 지표와 다른 키를 받았다. 프레임 두 개 뒤에 보고되는 FCP와 LCP는 늘어난 횟수로 갔다. 복원 하나가 trace 둘로 갈라진 것이다.

키는 web-vitals가 복원 지표에 넣어 주는 `navigationStartTime`으로 바꿨다. 1편의 코드에서 본 대로 이 값은 복원 `pageshow`의 `timeStamp`라, 같은 복원의 지표가 모두 같은 값을 들고 온다. 수집 코드는 경로만 저장한다. 복원된 문서는 SPA 이동 뒤의 URL일 수 있어서, 같은 `timeStamp`를 키로 그때의 경로를 적어 두고 지표가 올 때 찾는다. 이 리스너도 capture로 바꾸고 web-vitals보다 먼저 등록했다. 같은 타깃의 capture 리스너끼리는 등록 순서대로 돌므로, 복원 직후에 보고되는 지표보다 경로가 먼저 저장된다.

```ts
// apps/blog/src/utils/telemetry/navigation.ts
      if (metric.navigationType === 'back-forward-cache') {
        const start = metric.navigationStartTime ?? 0
        return {
          key: `restore:${start}`,
          kind: 'restore',
          startTime: start,
          pathname: restorePathnames.get(start) ?? initialPathname,
        }
      }
```

키를 고치고 나니 다른 문제가 보였다. 복원의 TTFB는 1편에서 본 대로 web-vitals가 재지 않고 0을 넣는 값인데, 대시보드의 "첫 로드" 변수가 `back-forward-cache`를 포함하고 있었다. 복원의 FCP와 LCP도 처음 로드와 다른 것을 잰 값이다. 48시간 범위에서 복원 2건을 첫 로드에 넣고 뺀 차이는 이랬다.

| 48시간, 첫 로드 p75 | 복원 포함 | 복원 제외 |
| ------------------- | --------- | --------- |
| TTFB                | 375ms     | 383ms     |
| LCP                 | 2.50s     | 2.75s     |
| FCP                 | 1.65s     | 1.80s     |

> 운영 Grafana에서 같은 시각에 변수만 바꿔 읽은 값이다. 첫 로드 TTFB 표본이 40건 안팎이라 복원 2건으로도 p75가 이만큼 움직인다.

그래서 복원의 TTFB는 아예 기록하지 않고(SPA 이동의 TTFB를 재지 않는 것과 같은 이유다), 변수를 첫 로드, SPA 이동, bfcache 복원, 전체의 네 가지로 나눴다. 복원이 얼마나 되는지는 복원마다 보고되는 FCP의 건수로 센다. 이렇게 나눠야 같은 이름이어도 측정 기준이 다른 값을 한 분포에 섞지 않고 볼 수 있다.

## 9. preview 배포도 production 빌드다

Vercel의 preview 배포에서도 수집 코드가 실행되고 있었다. 수집 코드는 `NODE_ENV`가 production일 때만 켜지는데, preview 배포도 production 빌드라 그대로 켜졌다. collector는 블로그 origin만 허용하므로 브라우저가 요청을 막고, 2편의 표대로 exporter는 CORS 거부를 네트워크 오류로 보고 약 8초 동안 5번 시도한다(처음 요청과 재시도 4번). 값이 잘못 들어가지는 않았지만, preview에서 글을 확인할 때마다 SDK를 받고 실패할 요청을 보내고 있었다. SDK를 받기 전에 origin을 비교해 거른다. 같은 자리에서 동적 import의 실패도 잡았다. 청크를 받지 못하면 `unhandledrejection`으로 새던 것을 삼켜서, 계측이 실패해도 독자에게 오류로 보이지 않게 했다.

```tsx
// apps/blog/src/components/analytics/CwvTelemetry.tsx
    // preview 배포도 production 빌드라 여기까지 온다. collector 는 블로그 origin 만 받으므로 보내지 않는다.
    if (window.location.origin !== SiteConfig.url) return
```

## 10. preflight 캐시와 서버 자원 상한을 설정하다

서버에서는 preflight 캐시와 메모리 상한, 로그 보관 설정을 고쳤다. exporter의 cross-origin 요청은 `Content-Type: application/json`을 쓰므로 preflight(OPTIONS)가 필요하다. collector의 CORS 응답에 `Access-Control-Max-Age`가 없으면 결과를 기본값인 5초만 캐시한다[^3]. span의 전송 주기도 5초라 캐시를 오래 활용하기 어려웠다. `cors.max_age`를 Chromium의 상한인 2시간으로 두고, 서버에 OPTIONS 요청을 보내 `access-control-max-age: 7200`을 확인했다. 실제 브라우저에서 OPTIONS가 얼마나 줄었는지는 별도로 세지 않았다.

둘째는 collector의 메모리 상한이다. `memory_limiter`는 `limit_percentage: 80`으로 두었는데, README는 이 값을 컨테이너 환경에서 호스트가 정한 메모리 제한을 기준으로 쓰라고 적는다[^4]. compose에는 collector의 메모리 제한이 없었다. 제한이 없으면 컨테이너가 쓸 수 있는 메모리는 서버 전체라 80%는 2GB의 80%가 되고, 다른 컨테이너가 쓸 메모리를 충분히 남기기 어렵다. 전날 오후의 collector가 86MiB였으므로 상한을 512MiB로 두었다. `memory_limiter`가 받기를 멈추는 선은 80%에서 `spike_limit_percentage` 20%를 뺀 약 307MiB다. 넘으면 전날 고친 손실 패널에 받지 못한 수로 보인다.

셋째는 컨테이너 로그다. Docker의 `json-file` 로그는 기본으로 회전하지 않는다. 1년 보존으로 둔 서버라 로그가 디스크를 채울 수 있어, 컨테이너마다 10MB 파일 3개로 돌린다.

## 11. flush 시점은 수집 코드 하나가 정한다

2편에서 `BatchSpanProcessor`가 document의 `pagehide`에 건 리스너는 불리지 않는다고 썼다. 같은 자리에서 document의 `visibilitychange`에 건 리스너는 불린다. 이 리스너는 SDK를 만들 때 등록되므로 수집 코드의 리스너보다 먼저 돌고, 수집 코드가 뒤이어 부르는 trace의 `forceFlush`는 빈 큐에 대고 부르는 셈이었다. 나가는 값은 같지만, flush 시점을 수집 코드가 정한다는 주석이 span에는 맞지 않았다. 2.12.0의 `disableAutoFlushOnDocumentHide` 옵션으로 SDK의 리스너를 끄고, 수집 코드의 리스너 하나가 span과 metric의 flush를 함께 부르게 했다.

```ts
// apps/blog/src/utils/telemetry/setup.ts
      new BatchSpanProcessor(
        new OTLPTraceExporter({url: `${collectorUrl}/v1/traces`}),
        // 브라우저용 processor 는 기본으로 document 의 visibilitychange 와 pagehide 에 자기 flush 를 건다.
        // flush 시점은 아래 visibilitychange 리스너 하나에서만 정하도록 끈다. 주기 export(5초)는 그대로다.
        {disableAutoFlushOnDocumentHide: true},
      ),
```

## 12. 대시보드를 열자 429가 났다

고친 대시보드를 운영 Grafana에서 열고 맨 아래로 바로 스크롤하자 패널 3개가 `too many outstanding requests`로 깨졌다. Mimir가 429를 돌려준 것이다. 로그에는 15:51:03에 10ms 사이 16건의 요청을 버렸다고 남았고, 그 전 48시간에는 한 건도 없었다. 다시 열거나 천천히 스크롤하면 7일이나 30일 범위에서도 재현되지 않았다.

한꺼번에 늘어난 하위 요청과 Mimir의 대기 상한을 먼저 확인했다. 운영 Mimir의 `/config`로 확인한 값은 쿼리를 24시간 단위로 쪼개고(`split_queries_by_interval`), 쪼갠 조각을 다시 16개로 샤딩하고(`parallelize_shardable_queries`, `query_sharding_total_shards`), 테넌트당 대기 요청을 100개까지만 받는다(`max_outstanding_requests_per_tenant`). 48시간 범위가 24시간 경계에 걸쳐 최대 3구간으로 나뉘면 패널 쿼리 하나가 최대 48개 하위 요청이 되고, 쿼리 다섯 개를 가진 "지표별 표본 수" 패널 하나가 240개다. 여러 querier에 일을 나누려는 설계인데, 이 Mimir는 프로세스 하나에 querier 하나(`max_concurrent: 8`)이고 결과 캐시도 꺼져 있어 이 설정에서는 많은 하위 요청이 하나의 대기 큐로 몰렸다.

```yaml
# infra/otel/mimir/mimir.yaml
frontend:
  split_queries_by_interval: 0
  parallelize_shardable_queries: false
```

패널 쿼리의 시간 분할과 샤딩을 끈 뒤, 서버 안에서 운영 대시보드의 Mimir 쿼리 36개를 한꺼번에 보내는 것을 48시간 범위로 한 번, 30일 범위로 두 번 돌렸다. 108개 요청이 모두 200이었고 Mimir가 버린 요청은 0건이었다. 고치기 전의 같은 실험은 돌리지 않았다. 근거는 실제로 난 429와 기본값의 계산이다.

### trace는 왔는데 metric이 없다

같은 날 넣은 패널 두 개는 Tempo의 첫 로드 trace 56건과 Mimir의 첫 로드 표본 40건을 보여 줬다. 같은 첫 로드를 비교하려던 패널인데 16건이 달랐다. 라벨과 시각을 맞춰 보니 6건은 첫날 Mimir 저장 방식을 바꾼 수정을 배포하기 전의 trace였다. Mimir의 첫 표본은 13:56부터 있었다.

나머지 10건은 17일 저녁부터 18일 오후까지 퍼져 있었고 대부분 데스크톱 Chromium에서 글을 연 방문이었다. span은 주기 export로 먼저 나갈 수 있으므로, 탭이 아직 hidden이 되지 않았거나 hidden에서 시작한 metric 전송이 완료되지 않았을 가능성이 있다. 확인한 Mimir의 폐기 표본과 collector의 수신 거부 카운터는 0이었지만, 그것만으로 원인을 특정할 수는 없었다. 비교 대상 50건 중 10건이라는 차이를 남기고 닷새치 데이터에서 다시 확인했다.

## 13. 설정 반영을 GitHub Actions에 맡기다

지금까지는 서버 설정을 바꿀 때마다 SSH로 들어가 `git pull`과 `docker compose up -d`를 손으로 쳤다. 리뷰 반영처럼 하루에 몇 번씩 바꾸는 날에는 이 단계를 빼먹기 쉽고, 저장소와 서버가 다른 커밋에 있는 채로 글을 쓰게 된다. 그래서 `infra/otel`이 main에 올라가면 GitHub Actions가 서버에 SSH로 들어가 반영하게 했다(`.github/workflows/deploy-otel.yml`, `infra/otel/deploy.sh`).

키는 배포 전용으로 하나 새로 만들었다. 서버의 `authorized_keys`에서 이 키는 forced command에 묶여 있어, 워크플로가 어떤 명령을 보내든 `git pull` 뒤 `deploy.sh`만 실행된다. 포트 포워딩과 pty도 막았다. 이 키로 임의의 셸 명령을 실행하는 대신 정해 둔 배포 명령만 호출하게 했다. 워크플로가 SSH 명령으로 보내는 커밋 해시는 `SSH_ORIGINAL_COMMAND`로 스크립트에 전달되고, pull 뒤의 HEAD와 다르면 경고를 남긴다.

```text
# 서버의 ~/.ssh/authorized_keys
command="cd ~/blog && b=$(git rev-parse HEAD) && git pull --ff-only --quiet && exec bash infra/otel/deploy.sh $b",no-port-forwarding,no-agent-forwarding,no-X11-forwarding,no-pty ssh-ed25519 AAAA... github-actions deploy-otel
```

`deploy.sh`는 `docker compose up -d`를 돌린 뒤, pull 전후의 diff에서 설정 디렉터리가 바뀐 서비스만 다시 시작한다. bind mount한 설정 파일은 바뀌어도 compose가 컨테이너를 다시 만들지 않기 때문이다. compose.yaml이 바뀐 서비스는 `up -d`가 알아서 다시 만든다.

첫 실행에서는 22번 포트가 내 IP에만 열려 있어 GitHub 러너가 연결하지 못했다(`Connection timed out`). GitHub 러너의 IP 대역은 공개되어 있지만 범위가 넓고 바뀌므로 방화벽 규칙으로 좁히지 않고, 22번 포트를 모든 IP에 열었다. 비밀번호 인증은 끈 채 배포 키에 forced command를 적용했지만, 내 키의 셸 접근도 같은 포트로 노출된다. 다시 실행한 워크플로는 서버를 e02c600a에서 f3c6f8ff로 pull했다. 설정 디렉터리에 바뀐 파일이 없어 컨테이너를 다시 시작하지 않고 끝났다. 실제 설정 파일이 바뀌었을 때도 재시작되는지는 다음 변경에서 확인하게 됐다.

## 14. collector가 라벨을 검사하지 않았다

첫날에는 브라우저 코드에서만 라벨을 제한했다. collector는 요청의 `service.name`과 라벨을 그대로 받으므로, 직접 만든 OTLP 요청으로 블로그의 `cwv.lcp`에 임의의 값이나 라벨을 넣을 수 있었다. 지표당 하루 100~180건인 데이터에 가짜 값이 섞이면 p75도 영향을 받는다. 실제 방문인지 확인할 수는 없더라도, 브라우저 코드가 보내지 않는 이름과 라벨까지 받을 필요는 없었다. 당시 Mimir의 상한은 기본값 그대로였다.

| Mimir 3.2.2 기본값 | 값      |
| ------------------ | ------- |
| 테넌트당 시리즈    | 150,000 |
| 시리즈당 라벨 수   | 30      |
| 라벨 값 길이       | 2,048자 |
| 초당 수신 샘플     | 10,000  |

본문 크기를 256KB로 제한해도 요청 횟수와 그 안의 값은 제한하지 못한다. Tempo에 보내는 span에도 블로그가 쓰는 이름과 속성인지 확인하는 절차가 없었다. 그래서 collector에 브라우저 수집 코드의 이름과 라벨을 검사하는 규칙을 추가했다. `filter` 프로세서는 `service.name`이 블로그가 아니거나, metric 이름이 다섯 지표가 아니거나, histogram이 아니거나 delta가 아니거나, 라벨 값이 허용 목록 밖이면 버린다. `transform` 프로세서는 통과한 뒤에도 모르는 라벨 키를 지우고(`keep_keys`), span 속성은 64개와 512자로 자른다.

```yaml
# infra/otel/collector/config.yaml
  filter/schema:
    error_mode: ignore
    metrics:
      metric:
        - 'resource.attributes["service.name"] != "yceffort-blog"'
        - 'not IsMatch(name, "^cwv\\.(ttfb|fcp|lcp|inp|cls)$")'
        - 'type != METRIC_DATA_TYPE_HISTOGRAM'
        - 'aggregation_temporality != AGGREGATION_TEMPORALITY_DELTA'
      datapoint:
        - 'not IsString(attributes["page.route"]) or not IsMatch(attributes["page.route"], "^(home|post|tag|series|other)$")'
```

같은 이미지로 로컬에서 설정을 검증하고, exporter를 debug로 바꿔 요청 12종을 보냈다. metric은 정상 요청과 "모르는 키가 섞인 요청" 두 가지만 살아남았고 섞인 키는 지워져 있었다. 다른 service.name, 모르는 이름, cumulative, gauge, 허용 목록 밖의 라벨 값, 라벨 누락은 모두 버려졌다. trace는 같은 trace의 두 span이 `cwv.*`면 통과했고, 2,000자짜리 속성은 512자로 잘렸다. 이 실험으로 확인한 것은 검사 규칙에 맞지 않는 요청을 거르고, 모르는 datapoint 라벨 키를 지운다는 데까지다.

허용한 라벨로 만든 가짜 값은 여전히 들어온다. 전체 시리즈 수가 수천 개 이하라고 보장할 수도 없다. 이 설정은 남겨 둔 resource 속성의 값이나 histogram의 버킷 경계까지 검증하지 않고, 요청 빈도도 제한하지 않는다. 입력 검사를 추가한 범위와 저장량을 제한하는 문제는 구별해야 한다.

설정은 자동 배포 과정에서도 한 번 더 확인해야 했다. 워크플로는 성공했는데 collector는 여전히 "Up 4 days"였다. `deploy.sh`가 `cd infra/otel` 뒤에 `git diff -- infra/otel`을 실행하면서, 현재 디렉터리 기준으로 `infra/otel/infra/otel`을 찾고 있었다. pathspec을 `.`로 고친 다음 커밋이 올라가자 바뀐 파일 두 개가 출력되고 collector가 다시 시작됐다. 워크플로의 성공 여부만으로 설정 반영까지 확인할 수 없었던 사례다.

## Tempo 조회가 성공해도 빠진 block이 있었다

10월 22일에 닷새치 값을 한 번에 조회하자 Tempo 3.1.0의 TraceQL metrics API가 `end must be greater than start`로 실패했다. 즉시 조회와 구간 조회 모두 같았다. 23일에 원인을 추적해 보니, 조회 구간의 길이보다 저장된 block의 시간 범위가 문제였다.

22일 16:22부터 16:36까지 Tempo 로그에 남은 하위 요청 369건은 모두 같은 block 3개에서 실패했다. 세 block의 메타데이터는 `startTime`이 `endTime`보다 늦었다. 그중 하나에는 INP span 하나만 있었고, span의 끝 시각은 12:23:45인데 block의 시작 시각은 13:32:03이었다.

### span 시각과 block의 조회 범위가 달랐다

live-store의 [`writeHeadBlock`](https://github.com/grafana/tempo/blob/v3.1.0/modules/livestore/instance.go)은 trace에 들어 있는 span들의 시작과 끝으로 시간 범위를 구한 뒤, 수집 시각을 기준으로 그 범위를 보정한다. 이때 바꾸는 것은 block 메타데이터에 반영할 범위다. span 원본의 타임스탬프는 그대로 저장한다.

시작이 수집 시각보다 `ingestion_time_range_slack`(기본 2분) 이상 이르면 하한까지 올린다. 끝은 상한보다 미래일 때만 내린다. 따라서 늦게 도착한 span은 시작만 보정되고 끝은 과거에 남을 수 있다. 앞의 span은 이 규칙으로 보면 13:34쯤 수집된 것으로 추정되며, block 완성 로그는 13:34:13이었다. 이 span 하나만 30초 단위로 닫는 block(`max_block_duration`)에 담기면서 시작과 끝의 역전이 block 메타데이터에도 남았다.

Tempo 3.1.0의 block-builder에는 보정 뒤 끝이 시작보다 이르면 끝을 시작에 맞추는 처리가 있다([`adjustTimeRangeForSlack`](https://github.com/grafana/tempo/blob/v3.1.0/modules/blockbuilder/tenant_store.go)). 같은 버전의 live-store에는 이 처리가 없다. 다만 span의 시각이 수집 시각보다 약 70분 뒤처진 이유는 확인하지 못했다. 브라우저 시계의 편차, 절전 중 `performance.now()`의 동작, 요청 전송 지연 등이 후보로 남았다.

### 짧게 나누면 오류와 함께 표본도 빠졌다

query-frontend는 `block 시작 <= 조회 끝`이고 `block 끝 >= 조회 시작`인 block을 고른다([`blockMetasForSearch`](https://github.com/grafana/tempo/blob/v3.1.0/modules/frontend/frontend.go)). 시간이 역전된 block은 조회 구간이 그 역전 구간을 통째로 포함할 때만 선택된다. 즉시 조회에서는 선택한 block의 범위가 하위 요청에도 역전된 채 들어가 querier가 거부하고, 전체 조회가 500으로 끝났다.

구간 조회에서는 [`TrimToBlockOverlap`](https://github.com/grafana/tempo/blob/v3.1.0/pkg/traceql/engine_metrics.go)이 block의 경계를 step에 맞춘다. 그 결과 시작과 끝이 같아지면 [`buildBackendRequests`](https://github.com/grafana/tempo/blob/v3.1.0/modules/frontend/metrics_query_range_sharder.go)가 그 block의 요청을 만들지 않는다. 보정 뒤에도 시작이 끝보다 크면 오류가 남는다. 따라서 step을 바꿔 200을 받았다고 해서 해당 block을 정상적으로 조회한 것은 아니다.

22일에 확인한 역전 구간은 21일 16:26부터 22일 08:22, 21일 20:31부터 22일 10:01, 22일 12:23부터 13:32였다. 13:00을 기준으로 나눈 2시간 구간에는 어느 역전 구간도 통째로 들어가지 않았다. 이때 조회가 모두 성공한 것은 문제가 있는 block이 선택되지 않았기 때문이다. 2시간이 조회 상한이었던 것은 아니다. 2시간의 결과가 1시간 구간 2개의 합과 같다는 확인도, 양쪽에서 같은 block이 빠질 수 있으므로 완전성을 검증하지는 못한다.

23일에는 기존 block 3개 중 2개가 compaction으로 사라지고 역전된 block 하나가 새로 생겨 있었다. 이 상태에서 확인한 응답은 다음과 같다.

| 조회 조건                              | 응답 |
| -------------------------------------- | ---- |
| 22일 12:00부터 14:00까지, 2시간        | 500  |
| 17일 13:00부터 22일 12:00까지, 119시간 | 200  |
| 121시간 구간, step 1시간               | 200  |
| 위와 같은 121시간 구간, step 5분       | 500  |

아래에는 22일에 조회한 건수를 그대로 남겼다. 이 값은 당시 쿼리가 반환한 범위의 기록이며, 역전된 block까지 복구해 센 전체 저장 건수로 검증한 값은 아니다. 다만 22일에 역전돼 있던 block 3개에는 trace가 1건씩 들어 있었으므로(meta.json과 compactor 로그의 totalObjects), 그때 빠진 것은 그 trace 3건의 span이다. Tempo와 Mimir의 차이를 해석할 때는 전송뿐 아니라 이 조회 누락도 고려해야 한다.

## 닷새 동안 쌓인 값

10월 17일 13:56(Mimir의 첫 표본)부터 10월 22일 15:00까지 약 121시간 동안 쌓인 값이다. Tempo와 Mimir의 건수 비교는 같은 2시간 구간 61개(10월 17일 13:00부터 10월 22일 15:00까지)로 맞췄고, 아래의 다른 수치도 모두 같은 15:00을 끝으로 조회했다. Tempo에서 계산한 건수와 비율은 앞 절의 조회 누락이 있는 상태에서 얻은 결과다.

### trace와 metric의 건수 차이는 지표마다 다르다

| 지표 | Tempo의 span | Mimir의 측정 건수 | 차이        |
| ---- | ------------ | ----------------- | ----------- |
| TTFB | 624          | 526               | 98건(15.7%) |
| FCP  | 954          | 896               | 58건(6.1%)  |
| LCP  | 902          | 880               | 22건(2.4%)  |
| CLS  | 793          | 778               | 15건(1.9%)  |
| INP  | 707          | 702               | 5건(0.7%)   |

Mimir의 측정 건수는 저장된 delta histogram의 count를 합한 값이다. 시계열 샘플 개수와는 다르다. 위 표에는 첫 로드뿐 아니라 SPA 이동과 bfcache 복원도 포함되어 있다.

TTFB와 FCP는 로드 직후 보고되므로, 탭을 숨기기 전에 span만 주기 export로 도착할 수 있다. LCP는 첫 입력이나 hidden에서 확정되고, CLS와 INP도 hidden 외에 SPA 이동 등의 보고 시점이 있다. hidden에서 함께 flush를 시작하더라도 trace와 metric은 별도 HTTP 요청이므로 함께 도착한다고 보장할 수 없다. 보고와 전송 시점뿐 아니라 앞에서 확인한 Tempo의 조회 누락도 차이에 영향을 줄 수 있다. 이 표만으로 각 차이의 원인을 가를 수는 없다.

Mimir의 첫 표본(17일 13:56) 전에 시작한 span은 TTFB 4건, FCP 4건, LCP 3건, CLS 1건, INP 1건이다(18일에 본 6건 중 2건은 비교 창이 시작하는 13:00 전의 것이다). 이 시각을 기준으로 제외해 계산하면 TTFB는 620건 대 526건으로 94건, 약 15.2% 차이이고, FCP는 950건 대 896건으로 54건(5.7%), LCP 19건(2.1%), CLS 14건(1.8%), INP 4건(0.6%)이다. 아직 열려 있는 탭과 전송에 실패한 방문을 구별하지 못하므로 이를 이탈 시 손실률로 해석하지는 않았다. 확인한 Mimir 폐기 표본, collector 수신 거부와 전송 실패 카운터는 모두 0이었다. 다만 이 카운터는 재시작하면 초기화되고 모든 손실 경로를 다루지는 않으므로, 전체 기간의 서버 내 손실이 없었다는 증거로 삼을 수는 없다.

또한 두 저장소의 시각 기준도 다르다. Tempo의 span은 브라우저 시각을 쓰지만, Mimir의 metric은 collector가 받은 시각을 쓴다. 조회 시작과 끝을 같게 두어도 경계 근처의 같은 측정이 서로 다른 구간에 들어갈 수 있다. 앞의 보정 역시 span 시작 시각을 기준으로 한 계산이며, 방문별 대응을 확인한 결과는 아니다.

첫 로드의 Mimir 측정 건수는 TTFB 524건과 FCP 453건이었다. TTFB는 표의 526건에서, 기록을 중단하기 전에 들어온 bfcache 복원 2건을 뺀 값이다. 두 지표의 차이는 71건으로 TTFB 건수의 13.5%다. metric에는 방문별 id를 저장하지 않으므로, 이 차이를 그대로 "TTFB만 있는 방문 71건"으로 확정할 수는 없다. 첫 그리기 전에 페이지가 숨겨지면 web-vitals가 FCP를 보고하지 않는다는 점에서 백그라운드 탭으로 연 방문을 한 후보로 보았지만, 방문 단위로 확인하지는 않았다.

### 라벨 조합 90개 중 38개는 표본이 하나였다

FCP 측정값이 있는 라벨 조합은 90개다. 그중 38개는 닷새 동안 측정값이 하나뿐이었고, 가장 많은 조합은 132건이었다. 조합당 하루 평균은 약 2건이다. 첫날 실험과 같은 누적 변환 설정이었다면 한 번만 기록된 38개 시리즈는 `increase()` 결과에 나오지 않았을 것이다. 방문이 드문 블로그에서 delta 저장을 선택한 이유를 실제 데이터에서도 확인할 수 있었다.

### 수집된 내비게이션의 40%가 SPA 이동이었다

Tempo에서 조회된 내비게이션 1,076건 중 첫 로드가 623건, SPA 이동이 433건(40%), bfcache 복원이 12건(첫 로드의 1.9%)이다(8건은 SPA 이동과 bfcache 복원을 가르기 전의 것이라 속성이 없다). 첫날 `reportSoftNavs`를 켜지 않았다면 이 433건의 이동별 지표는 따로 남지 않았을 것이다. FCP 기준으로 Chromium이 852건(89%), WebKit 62건, Gecko 36건이었고(나머지 4건은 엔진 속성을 넣기 전 것이다), SPA 이동은 Chromium 151 이상에서만 잡히므로 이 40%를 모든 방문의 SPA 이동 비율로 볼 수는 없다. WebKit과 Gecko의 SPA 이동은 이 방식으로 수집되지 않는다.

유입 경로는 첫 로드의 TTFB span을 기준으로 셌다. 속성이 있는 첫 로드 614건(속성을 넣기 전의 8건과 bfcache 복원 2건 제외) 중 직접 277건(45%), 검색 234건(38%), 블로그 안 54건, 그 밖 44건, AI 서비스 5건이었고, 291건(47%)은 서비스 워커를 거친 내비게이션이었다.

### 비교 가능한 LCP의 0.6%에서 quiet 값이 더 작았다

LCP span 902건의 `cwv.lcp.quiet`는 `none` 341건, `same` 546건, `lower` 5건이고, 10건은 속성이 생기기 전 것이다. 속성이 있는 892건 중 quiet 타이머가 정식 보고보다 먼저 끝난 경우는 551건(62%)이었다. 이 중 quiet 값이 정식값보다 작았던 5건은 892건의 약 0.6%이고, 정식값의 등급이 모두 poor였다. quiet로 확정했다면 이 5건을 더 작은 LCP로 기록했을 것이다.

비율이 작더라도 느린 방문의 값을 낮춰 기록한다는 문제가 남는다. 블로그에서는 정식 보고를 기다리고 quiet 값은 진단 속성으로만 남기는 구성을 유지했다. 다만 이 비교는 정식 LCP까지 도착한 표본으로만 가능하다. 정식 보고 없이 끝난 방문을 quiet로 얼마나 더 수집할 수 있었는지는 이 수치로 알 수 없다.

### CLS를 보고한 내비게이션의 4.4%에서 등급이 바뀌었다

같은 id로 여러 번 보고할 수 있으므로 갱신 여부는 trace 단위로 셌다. 이 구성에서 trace 하나는 첫 로드, SPA 이동, bfcache 복원 중 내비게이션 하나에 해당한다. CLS를 보고한 793건 중 갱신이 있는 내비게이션은 77건(9.7%)이었고, 그중 35건(전체의 4.4%)은 등급도 바뀌었다. INP는 갱신이 있는 내비게이션 56건 중 2건에서 등급이 바뀌었다. 갱신 span은 CLS 96건, INP 62건이었다.

첫 보고만 histogram에 넣으면 이후의 CLS 증가가 반영되지 않는다. 4.4%는 등급이 바뀐 내비게이션의 비율이고, 같은 등급 안에서 값이 커진 경우는 포함하지 않는다. 이 비율만으로 p75가 얼마나 달라질지 계산할 수는 없다.

지금은 Mimir의 첫 보고 값으로 개선할 화면을 찾고, 갱신 내용은 Tempo에서 함께 확인하기로 했다. 이미 첫 로드의 CLS p75가 needs improvement 구간이었고, 수정할 대상을 찾는 데는 web-vitals가 남긴 가장 큰 밀림의 요소와 시각이 필요했다. 이 속성은 갱신 span에도 남는다. 다만 이동한 요소가 곧 밀림을 일으킨 원인은 아니므로 화면을 재현해 확인해야 한다.

3편의 gateway를 두면 같은 측정의 최신값으로 교체할 수 있지만, 상태를 보관하는 서버와 30분의 확정 대기, 재시작 때의 손실도 관리해야 한다. 이번에는 이를 추가하지 않고 첫 보고 기준의 집계와 갱신 trace를 함께 보기로 했다. 따라서 이 histogram을 내비게이션 전체 수명 동안의 최종 CLS 분포로 설명해서는 안 된다.

### 블로그의 약점은 CLS다

첫 로드의 p75는 TTFB 455ms, FCP 1,191ms, LCP 1,499ms, INP 75ms, CLS 0.126이다. 등급은 LCP의 90%(445건 중 399건), INP의 96%(339건 중 326건)가 good인데 CLS는 68%(365건 중 250건)에 그쳤고, p75 0.126은 needs improvement 구간이다. LCP는 데스크톱 1,526ms와 모바일 1,473ms로 비슷했고, SPA 이동의 LCP p75는 467ms였다. 이 분위수는 histogram 버킷에서 추정한 값이며, 첫 보고 이후의 갱신은 반영하지 않는다. 지금 수집된 범위에서는 CLS부터 살펴볼 이유가 충분했다. 어느 화면에서 어떤 요소가 이동했는지는 Tempo로 좁혀 보고, 실제 원인 분석은 다른 글에서 다룬다.

### 서버는 627MiB를 쓴다

10월 22일 오후의 메모리는 Grafana 249MiB, Tempo 192MiB, Mimir 93MiB, collector 74MiB, Caddy 19MiB로 합쳐 627MiB다. 첫날 오후의 620MiB와 거의 같고, Tempo가 158MiB에서 192MiB로 늘고 Mimir가 115MiB에서 93MiB로 줄었다. 디스크는 59GB 중 6.5GB를 쓴다. collector의 자기 지표를 Mimir에 1분마다 긁어 둔 값으로는, 10월 18일 16:32에 다시 뜬 뒤 22일 15:00까지 span 4,885개와 metric point 93,495개(1분마다 긁는 손실 지표도 포함된다)를 받았고, 그 전 컨테이너는 span 285개를 받았다. tail sampling이 span 2개 미만으로 버린 trace는 22일에 컨테이너에서 읽은 카운터로 0건이었다. 서버 기본요금은 월 \$12다. 닷새 동안의 사용량이므로 이 결과만으로 1년 보존 시 필요한 디스크 크기를 판단하지는 않았다.

## 적은 표본으로 어디까지 판단할 것인가

블로그로 옮기면서 가장 먼저 필요했던 것은 드문 방문도 건수에 포함하는 일이었다. delta 저장으로 바꾼 뒤, 닷새 동안 값이 하나뿐이었던 라벨 조합 38개도 조회할 수 있었다. 일반 브라우저에서는 quiet로 LCP를 먼저 확정하지 않고, metric은 hidden에서 앞선 전송의 응답을 기다리지 않고 보내도록 바꿨다.

모든 값을 최종값으로 집계하는 구성은 선택하지 않았다. 같은 id의 CLS와 INP 갱신은 trace로 남기고, Mimir는 첫 보고 기준으로 본다. 이 범위에서도 CLS를 먼저 개선해야 한다는 판단은 가능했다. 다만 갱신을 반영한 정확한 분포가 필요해지면 지금의 집계 방식부터 다시 검토해야 한다.

손실 패널과 자동 배포도 확인 대상이었다. 수집이 멈췄을 때 패널이 0을 표시하고, 설정이 반영되지 않았는데 워크플로가 성공하는 경우를 겪었다. 서버 설정을 직접 바꿀 수 있게 된 만큼, 원하는 상황을 만들어 실제 표시와 반영 결과를 확인할 필요가 있었다.

남은 건수 차이를 조사하기 전에 Tempo에서 조회되지 않은 block부터 확인해야 한다. 그 뒤에도 trace와 metric은 시각 기준이 다르고, 아직 열려 있는 탭과 전송 실패가 섞일 수 있다. 닷새 동안의 기록은 현재 집계로 CLS를 먼저 살펴볼 근거가 됐지만, 수집 경로 전체의 손실률을 계산하려면 조회 누락을 해결하고 같은 방문의 값끼리 비교해야 한다.

[^1]: [Oracle Cloud Infrastructure, Always Free Resources](https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm)의 "Reclamation of Idle Compute Instances"

[^2]: [Amazon VPC pricing](https://aws.amazon.com/vpc/pricing/)의 Public IPv4 Address 항목

[^3]: [MDN, Access-Control-Max-Age](https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/Access-Control-Max-Age). 기본값은 5초이고, Chromium은 76부터 2시간(7200초), Firefox는 24시간을 상한으로 둔다고 적는다.

[^4]: [opentelemetry-collector v0.163.0 `processor/memorylimiterprocessor/README.md`](https://github.com/open-telemetry/opentelemetry-collector/blob/v0.163.0/processor/memorylimiterprocessor/README.md)의 "For containerized environments or other similar environments that support setting memory restrictions on the Collector, `limit_percentage` should generally be used, which allows the host environment to determine the Collector's max memory allocation"
