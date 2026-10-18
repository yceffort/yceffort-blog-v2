---
title: '블로그의 <em>핵심 웹 지표</em>를 직접 운영하는 서버로 받기'
tags:
  - web-performance
  - browser
published: false
date: 2026-10-17 09:00:00
description: '사내에서 공용 관측 인프라로 보내던 핵심 웹 지표 수집을 개인 블로그에 올리고, 수집 서버도 직접 띄웠다. 설정을 모두 바꿀 수 있게 되자 방문이 드물다는 것과 바꾼 설정을 모두 직접 확인해야 한다는 것이 새 제약이 됐다. 하루 동안의 커밋 순서로 기록했다.'
series: 'Core Web Vitals 직접 수집하기'
seriesOrder: 4
---

## Table of Contents

## 첫날 대시보드가 비어 있었다

블로그에 수집 코드를 붙이고 몇 번 방문한 뒤 Mimir 대시보드를 열자 모든 패널이 비어 있었다. 재현 환경에서 페이지 로드 300번을 흘려 넣었을 때는 숫자가 잘 나오던 대시보드였고, 수집 경로도 그대로 옮긴 것이었다. 달라진 것은 방문의 빈도였다. 경로를 고친 뒤 Mimir를 비우고 다시 쌓기 시작해서 두 시간 가까이 들어온 페이지 로드가 7건이었다.

3편까지의 제약은 바꿀 수 없는 공용 관측 인프라와 떠날 때 신호를 주지 않는 앱 웹뷰였다. 이번에는 반대다. 수집 서버는 월 $12짜리 서버 한 대에 직접 띄웠고, 방문자는 일반 브라우저로 들어오고, collector부터 대시보드까지 모든 설정을 바꿀 수 있다. 그러자 다른 종류의 제약이 드러났다. 방문이 드물면 서버 트래픽을 기준으로 정한 기본값이 값을 세지 못했고, 설정을 모두 바꿀 수 있는 만큼 바꾼 설정이 맞는지도 모두 직접 확인해야 했다.

이 글은 2026년 10월 17일 하루 동안 수집 서버를 띄우고 블로그에 붙이면서 바꾼 것을, 그날의 커밋 순서대로 적은 기록이다. 절 제목의 시각은 커밋 시각이다. 마지막 절은 몇 주 동안 쌓인 값으로 채운다.

> 수집 서버의 구성은 블로그 저장소의 [`infra/otel`](https://github.com/yceffort/blog/tree/4cc783fd4bae927e6732dc91188e96f711abb422/infra/otel)에, 브라우저 쪽 수집 코드는 [`apps/blog/src/utils/telemetry`](https://github.com/yceffort/blog/tree/4cc783fd4bae927e6732dc91188e96f711abb422/apps/blog/src/utils/telemetry)에 있다. 링크는 이 글을 쓴 시점의 커밋으로 고정했다.
>
> 버전은 web-vitals 6.2.3, OTel JS 2.12.0(OTLP exporter 0.223.0), OpenTelemetry Collector contrib 0.163.1, Tempo 3.1.0, Mimir 3.2.2, Grafana 13.2.4, Caddy 2.11.7이다. 서버는 Amazon Lightsail 서울 리전의 Debian 13 인스턴스(2GB 메모리)다.

## 12:38 수집 서버를 블로그 저장소로

수집 서버는 처음에 재현 환경 저장소의 배포용 덮어쓰기 설정으로 띄웠다가, 실험과 운영을 섞지 않으려고 블로그 저장소의 `infra/otel`로 옮겼다. 서버는 블로그 저장소에서 이 디렉터리만 sparse checkout으로 받아 `docker compose up`으로 돈다. 구성은 Caddy, collector, Tempo, Mimir, Grafana 다섯 개다.

호스팅은 고정 월 요금에 공인 IPv4와 전송량이 들어 있는 Lightsail을 골랐다.

| 선택지                       | 한 달 비용              | 걸린 점                                                                                                                         |
| ---------------------------- | ----------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| Lightsail `small_3_0`        | $12                     | 2GB 메모리, vCPU 2개, 디스크 60GB, 전송 3TB, 공인 IPv4 1개가 들어 있다                                                          |
| 같은 AWS의 EC2               | 인스턴스 값에 따로 더함 | 디스크(EBS)와 공인 IPv4(시간당 $0.005, 한 달 약 $3.6)가 따로 붙는다                                                             |
| Oracle Cloud Always Free(A1) | 무료                    | 7일 동안 CPU 사용률의 95번째 백분위, 네트워크, 메모리가 모두 20% 미만이면 유휴로 보고 회수할 수 있다[^1]. 수집 서버는 거의 쉰다 |

> Lightsail 가격은 서울 리전에서 Lightsail API(`get-bundles`)로 확인한 값이고, EC2의 공인 IPv4 요금은 AWS의 VPC 요금표[^2]에서 확인했다. 2026년 10월 기준이다.

같은 날 오후의 메모리는 Mimir 115MiB, Grafana 243MiB, Tempo 158MiB, collector 86MiB, Caddy 19MiB로, 다섯 개를 합쳐 2GB 중 약 620MiB였다. 이 중 Grafana는 기본 설치 그대로 띄웠을 때 가장 컸다. Grafana 13은 처음 뜰 때 쓰지 않는 앱 플러그인을 받아 설치하고, 데이터소스 플러그인을 쓸 때마다 프로세스를 따로 띄운다. 그래서 미리 설치하는 플러그인을 끄고(`GF_PLUGINS_PREINSTALL_DISABLED`), Mimir와 Tempo 말고는 내장 데이터소스 플러그인 11개를 꺼 두었다(`GF_PLUGINS_DISABLE_PLUGINS`). 같은 데이터소스와 대시보드를 넣어 띄우고 한 번씩 조회한 뒤 재 보면 차이는 이 정도였다.

| Grafana 13.2.4  | 5분 뒤 | 6분 뒤 | 데이터소스 플러그인 프로세스 | 미리 설치된 플러그인 |
| --------------- | ------ | ------ | ---------------------------- | -------------------- |
| 기본 설치       | 222MiB | 261MiB | 13개                         | 5개                  |
| 플러그인을 줄임 | 159MiB | 201MiB | 2개                          | 0개                  |

> 로컬 Docker에서 두 대를 같은 시각에 띄워 쟀다. 메모리는 1분 사이에도 40MiB 안팎으로 오르내렸다.

줄여도 아주 작아지지는 않는다. 운영 서버의 Grafana는 플러그인을 줄인 상태로 그날 오후에 243~292MiB였다. 2GB 서버에서 가장 큰 컨테이너인 것은 그대로다.

### 받는 쪽은 누구나 보낼 수 있는 주소다

브라우저가 직접 보내는 주소라서 수집 주소는 인터넷에 열려 있어야 한다. 인증을 붙일 수도 없다. 브라우저에 실은 비밀은 비밀이 아니기 때문이다. 그래서 막는 것보다 받는 범위를 좁히는 데 집중했다. Caddy는 OTLP/HTTP의 두 경로만 collector로 넘긴다.

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

다른 경로는 404이고, 256KB를 넘는 본문은 413으로 끊는다. CORS는 collector에서 블로그 origin만 허용하지만, CORS는 브라우저가 지키는 규칙이라 curl로 보내는 요청은 막지 못한다. 누군가 가짜 값을 보낼 수 있다는 것은 받아들이고, 그 값이 들어갈 수 있는 자리를 줄였다. metric 라벨은 값의 종류가 정해진 몇 개만 쓰므로 가짜 값이 시리즈를 끝없이 늘리지는 못한다. Grafana는 서버의 127.0.0.1에만 붙여 SSH 터널로만 연다. Caddyfile에 `log` 지시어를 두지 않아 접근 로그가 남지 않고, collector도 요청의 주소를 데이터에 붙이지 않으므로, 방문자의 IP는 어디에도 저장되지 않는다.

### Tempo 3.1은 하루보다 긴 구간을 조회하지 못한다

Tempo 3.1.0을 기본값으로 두면 trace 검색은 7일, TraceQL metrics는 24시간보다 긴 구간을 거부한다.

```text
range specified by start and end exceeds 168h0m0s. received start=1791535470 end=1792226640
metrics query time range exceeds the maximum allowed duration of 24h0m0s
```

> 기본 설정의 Tempo 3.1.0에 8일 구간의 검색과 2일 구간의 TraceQL metrics를 보내 받은 응답이다.

방문이 드문 블로그에서는 하루치로는 분위수를 낼 표본이 모이지 않는다. 그래서 보존 기간과 조회 범위를 함께 1년으로 늘렸다(`max_search_duration`, `max_metrics_duration`, `block_retention`). 보존 기간만 늘리면 데이터는 남아 있는데 조회는 막힌다.

## 12:42 블로그에 붙이다

브라우저 쪽은 재현 환경의 수집 코드를 블로그 구조에 맞게 옮겼다. production에서만 켜고, OTel SDK는 첫 화면에 필요 없으므로 동적 import로 따로 받는다. 이 청크는 지금 빌드에서 gzip으로 약 29KB다. web-vitals는 늦게 붙어도 buffered 항목을 읽어 LCP와 CLS를 놓치지 않지만, Event Timing의 buffered 항목은 104ms 이상인 것만 남으므로 SDK를 받기 전의 짧은 입력은 INP 후보에서 빠진다. 화면은 홈, 글, 태그, 시리즈, 그 밖의 다섯 가지(`page.route`)로만 나눈다.

## 13:42 대시보드가 비어 있었다

서론의 빈 대시보드가 이 시각이다. 원인을 찾으려고 방문이 드문 사이트를 흉내 내어, 같은 LCP 값 8건을 수집 경로 네 가지로 동시에 보내고 대시보드 쿼리가 몇 건으로 세는지 쟀다. 다섯 개의 라벨 조합은 한 번씩, 하나는 7분 간격으로 세 번 보냈다.

| 경로                                                              | 8건을 센 값 |
| ----------------------------------------------------------------- | ----------- |
| 재현 환경 구성(`delta_to_cumulative`, remote write, `increase()`) | 0.00        |
| 위와 같고 누적값 만료(`max_stale`)만 7일                          | 2.43        |
| 위와 같고 Mimir OTLP 엔드포인트가 시작 시각에 0 샘플을 넣음       | 10.66       |
| 누적으로 바꾸지 않고 delta를 그대로 저장, `sum_over_time()`       | 8.00        |

> 세 번 실행했고 재현 환경 구성의 첫 실행만 같은 run 라벨을 다시 써서 앞 실행의 상태를 이어받았다. 나머지 두 번은 같았다. `cwv-otel-grafana/experiments/e8-sparse-counts`.

라벨 조합마다 샘플이 하나뿐이면 `increase()`는 짝을 지을 샘플이 없어 0을 돌려준다. 누적값 만료를 늘려도 한 번만 온 조합은 여전히 샘플 하나이고, 시작 시각에 0 샘플을 넣으면 증가로는 잡히지만 `increase()`가 구간 끝까지 외삽해 8건을 10.66건으로 센다. delta를 그대로 저장하면 샘플 하나가 페이지 로드 하나의 값이라 구간 안의 샘플을 더하면 된다. 그래서 블로그의 metric 경로는 재현 환경과 다르게 갔다. collector에서 `delta_to_cumulative`를 빼고, Mimir의 delta 저장(`-distributor.otel-native-delta-ingestion`)으로 받고, 대시보드는 `sum_over_time()`으로 센다. Mimir 3.2.2는 이 기능을 "Delta support is in an early stage of development. The ingestion and querying process is likely to change over time."이라고 설명하는 실험 기능이라, Mimir를 올릴 때는 릴리스 노트를 확인한다.

시각도 바꿨다. 브라우저마다 시계가 다른데, 같은 시리즈에 시각이 앞선 값보다 이른 값이 들어오면 Mimir가 받지 않는다. 그래서 collector가 받은 시각으로 바꿔 저장한다.

```yaml
# infra/otel/collector/config.yaml
  transform/receive_time:
    metric_statements:
      - context: datapoint
        statements:
          - set(start_time, Now())
          - set(time, Now())
```

대가는 두 가지다. 값의 시각이 페이지 로드 시각이 아니라 collector가 받은 시각이 되고, 같은 시리즈의 값 두 개가 같은 밀리초에 들어오면 같은 시각의 표본이 되어 Mimir가 하나를 버린다. 시리즈 하나에 초당 10건이면 1.3~1.5%를 잃었다(3편, `cwv-otel-grafana/experiments/e10-concurrent-writers`). 블로그 트래픽에서는 드물 것으로 보았고, 이 손실은 뒤의 17:05에서 다시 나온다.

## 13:43 누구에게, 무엇이 느린가

블로그는 화면이 다섯 가지뿐이라, 화면만으로는 어디가 느린지 좁히기 어려웠다. 그래서 값의 종류가 정해진 맥락을 metric 라벨에 더했다. 언어(`page.lang`, `/en`으로 시작하면 en), 기기(`device.class`, 화면 폭 768px과 1024px으로 나눈 세 가지), 브라우저 엔진(`browser.engine`)이다. iOS의 브라우저는 이름과 상관없이 WebKit이라 엔진을 먼저 가른다. 유입 경로(검색, 소셜, AI 서비스 등), 서비스 워커를 거친 내비게이션인지, 네트워크와 기기 정보는 값의 종류가 많거나 질문이 좁아서 span 속성에만 싣는다.

LCP 요소의 selector에서는 StyleX의 원자 클래스를 지운다.

```ts
// apps/blog/src/utils/telemetry/mask.ts
    // StyleX 원자 클래스(`x1j9s4mx`)는 스타일 하나씩이라 요소를 구분하지 못하고, 스타일을 고치면 이름이 바뀐다.
    // 남겨 두면 같은 요소가 배포마다 다른 값으로 갈린다. 숫자가 많으면 아래 규칙에 걸려 `x{n}` 이 되기도 한다.
    .replace(/\.x[0-9a-z]{5,8}(?![\w-])/g, '')
```

블로그의 스타일은 StyleX가 만드는 원자 클래스라, web-vitals가 만든 selector에는 그 클래스가 줄줄이 붙는다. 그대로 두면 Tempo 대시보드의 "LCP 요소별" 패널이 같은 요소를 배포마다 다른 요소로 센다.

## 14:17 봇, SPA 이동, bfcache

이때 고친 것은 세 가지다. 첫째는 봇이다. PageSpeed Insights(`Chrome-Lighthouse`), headless 브라우저, 검색 엔진의 렌더러가 블로그를 열면 그 값도 수집됐다. 사람의 방문이 드문 블로그에서는 이런 값이 섞이면 분위수가 쉽게 흔들린다. 그래서 SDK를 받기 전에 user agent와 `navigator.webdriver`로 거른다. 이 판정 때문에 뒤에서 Playwright로 블로그 페이지를 직접 검증할 수 없게 됐다.

둘째는 SPA 이동이다. 블로그 안의 링크 이동은 클라이언트 이동이라, 지금까지는 첫 로드 하나의 값만 있었다. web-vitals의 `reportSoftNavs`를 켜면 Chromium 151 이상에서 이동마다 FCP, LCP, CLS, INP를 다시 잰다. TTFB는 이동에서 0으로 보고되므로 첫 로드에만 잰다. 셋째는 bfcache 복원이다. 첫 로드, SPA 이동, bfcache 복원을 각각 내비게이션 하나로 보고 trace도 따로 만든다. SPA 이동은 web-vitals가 주는 `navigationId`로, bfcache 복원은 `pageshow`의 `persisted`를 센 횟수로 가른다. 대시보드에는 첫 로드와 SPA 이동을 고르는 변수를 두었다.

## 16:17 리뷰가 찾은 것

여기까지 올린 뒤 수집 코드와 대시보드를 리뷰받았고, 이슈 6건이 나왔다. 6건 모두 코드와 소스로 확인됐다. 그중 블로그에서 판단이 갈린 것을 적는다.

### 일반 브라우저에서 quiet 확정은 해롭다

3편의 quiet LCP(후보가 3초 동안 바뀌지 않으면 확정)는 hidden이 오지 않는 앱 웹뷰를 위한 장치였다. 수집을 시작한 날 오후까지 Tempo에 쌓인 LCP 9건 중 7건이 quiet로 확정됐다. 글을 읽는 동안에는 3초 넘게 아무것도 누르지 않는 경우가 많아서, quiet가 web-vitals보다 먼저 확정하는 일이 흔할 것으로 본다. 그런데 일반 브라우저는 떠날 때 hidden이 오고, 그때 web-vitals가 LCP를 확정한다. quiet의 이득은 거의 없고, 3편에서 본 대로 더 큰 요소를 놓칠 위험만 남는다.

그래서 블로그는 web-vitals가 확정한 LCP만 histogram에 넣는다. quiet 타이머는 남겨 두되, 처음으로 3초 멈췄던 후보의 값을 span 속성으로만 싣는다(`cwv.lcp.quiet`가 `none`, `same`, `lower` 중 하나). `lower`는 quiet로 확정했다면 LCP를 작게 기록했을 페이지 로드다. 이 비율이 얼마나 되는지는 마지막 절에서 본다.

### 주기 export를 없앴다

2편에서 pagehide의 flush를 빼고 visibilitychange에서만 flush하도록 고쳤다. 그런데 sdk-metrics의 `forceFlush`는 어떤 export든 진행 중이면 그 응답을 기다린다. hidden이 오는 순간 10초 주기 export가 나가 있으면, CLS와 INP는 그 응답이 올 때까지 collect되지 않는다. 앱 웹뷰에서는 주기 export가 값을 내보낼 유일한 기회라 끌 수 없었지만, 블로그는 hidden이 오니 끌 수 있다. 그래서 flush할 때만 collect하고 export하며, 앞선 export를 기다리지 않는 reader로 바꿨다.

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

delta라서 flush가 겹쳐도 같은 값을 두 번 보내지 않는다. 블로그 페이지는 자동화 브라우저를 거르므로, 블로그의 수집 코드를 그대로 묶은 시험 페이지를 만들어 Chromium에서 비교했다. 첫 hidden의 flush 응답을 collector가 5초 늦게 주게 하고, 그 사이 클릭한 뒤 다른 페이지로 떠났다.

| 수집 코드                                    | 떠날 때의 두 번째 요청(LCP, INP) |
| -------------------------------------------- | -------------------------------- |
| 주기 export(`PeriodicExportingMetricReader`) | 오지 않음                        |
| `FlushOnlyMetricReader`                      | 떠난 뒤 약 12ms에 도착           |

> Playwright 1.64.1의 Chromium 156에서 두 번 실행해 같았다. 첫 hidden은 `visibilityState`를 덮어써 만들었다. headless Chromium에서는 탭을 바꿔도 hidden이 되지 않았다. [측정 스크립트 공개 위치 필요]

대가는 hidden이 끝내 오지 않는 페이지 로드다. 그런 페이지 로드는 값이 하나도 나가지 않는다. 일반 브라우저에서는 탭을 닫거나 앱을 바꿀 때도 hidden이 오므로 드물 것으로 보지만, 비율을 잴 방법은 없다.

### 같은 id의 갱신과 같은 밀리초의 중복은 재기만 했다

나머지 두 건은 고치는 대신 재기만 했다. 하나는 같은 id로 다시 오는 CLS와 INP다. histogram은 첫 보고만 세므로, 탭에 돌아온 뒤에 나빠진 값은 들어가지 않는다. 다른 하나는 앞에서 본 같은 밀리초 중복이다. 둘 다 3편의 재현 환경처럼 gateway를 두어 측정마다 최신값을 집계하면 풀린다. 다만 상태를 들고 있는 서버를 하나 더 운영해야 하고, 시간당 몇 건인 트래픽에서는 그 정확도 차이를 데이터로 확인하기도 어렵다. 그래서 다시 온 보고는 histogram 대신 `cwv.cls.update`, `cwv.inp.update` span으로 남기고(첫 값, 등급이 바뀌었는지, 몇 번째 보고인지), Mimir가 버린 표본은 대시보드에 보이게 했다. 둘 다 마지막 절의 값을 보고 다시 판단한다.

[1편](/2026/10/web-vitals-report-timing)에서 찾은 INP 손실도 알고서 그대로 두었다. 블로그는 web-vitals 6.2.3을 쓰고 INP를 기본 설정으로 받으므로, 상호작용 뒤 1초 안에 떠난 페이지 로드의 INP는 보고되지 않는다. INP만 `reportAllChanges`로 받는 우회는 첫 보고만 세는 이 구조와 맞지 않아 넣지 않았고, 수정(#809)이 담긴 버전이 나오면 web-vitals를 올려서 해결할 생각이다. 올린 뒤에는 1편처럼 떠나는 페이지 로드를 만들어 지표별 보고 수를 다시 센다.

## 17:05 손실 패널이 손실을 숨기고 있었다

버린 표본을 보이려고 만든 패널을 다시 리뷰받자 3건이 나왔다. 셋 다 맞는 지적이었고, 셋 다 그 패널이 손실을 정상처럼 보이게 하는 문제였다.

collector는 1분마다 Mimir의 `cortex_discarded_samples_total`과 자기 지표(받지 못한 수, 내보내지 못한 수)를 긁어 Mimir에 넣는다. 처음 패널 쿼리는 이랬다.

```promql
max by (reason) (max_over_time(cortex_discarded_samples_total{job="mimir", user="blog"}[$__range])) or on() label_replace(vector(0), "reason", "없음", "", "")
```

첫째, `or vector(0)`는 손실 지표를 아예 긁지 못한 상태도 0으로 보여 준다. 수집 설정이 빠진 새 서버에서도, collector가 멈춘 뒤에도 패널은 "없음 0"이었다. 둘째, `max_over_time`은 조회 구간의 과거 최댓값이라 Mimir가 재시작해 카운터가 다시 시작해도 재시작 전 값을 보여 준다. 셋째는 손실과 상관없는 버그였다. 대시보드 생성 스크립트가 refId를 매기는 반복문을 마지막 패널에만 돌리고 있었는데, 손실 패널을 그 뒤에 붙이면서 대상이 바뀌어 "지표별 표본 수" 패널의 쿼리 다섯 개가 모두 refId A가 됐다. Grafana는 이 패널을 `400 query.duplicateRefId`로 거부했다. 대시보드를 화면으로 열어 보지 않고 쿼리만 확인한 탓에 놓쳤다.

고친 쿼리는 마지막 수집이 성공했고(`up == 1`) 3분 안일 때만 값을 보여 주고, 카운터의 최신값을 쓴다.

```promql
(max by (reason) (cortex_discarded_samples_total{job="mimir", user="blog"}) or on() label_replace(vector(0), "reason", "없음", "", "")) and on() max((up{job="mimir"} == 1) and (time() - timestamp(up{job="mimir"}) < 180))
```

로컬에 같은 구성을 띄우고 fixture로 비교했다. 고친 쿼리는 첫 수집 전에도 "수집 안 됨"이었다.

| 상황                               | 처음 쿼리 | 고친 쿼리  |
| ---------------------------------- | --------- | ---------- |
| collector를 멈추고 4분 뒤, 6분 뒤  | 없음 0    | 수집 안 됨 |
| 버린 표본 10, Mimir 재시작, 다시 5 | 10        | 5          |

> 실제 Grafana 13.2.4의 `/api/ds/query`로 패널 쿼리를 보내 확인했고, 고친 뒤에는 Mimir 패널 30개가 모두 200이었다. 운영 서버의 Grafana에서도 같았다.

## 몇 주 동안 쌓인 값

[측정값 필요: 2026-11-07 이후에 채운다]

- 페이지 로드 수와 지표별 표본 수, 라벨 조합당 하루 표본 수
- `cwv.lcp.quiet`가 `lower`였던 비율, 즉 quiet로 확정했다면 LCP를 작게 기록했을 페이지 로드
- `cwv.cls.update`, `cwv.inp.update`의 빈도와 등급이 바뀐 비율
- Mimir가 버린 표본과 collector가 내보내지 못한 수
- SPA 이동의 비율과 Chromium 151 이상의 비율
- 서버의 메모리 추이와 실제 청구 금액

## 바꿀 수 있는 곳의 제약

3편에서는 바꿀 수 없는 기본값이 값을 버렸다. 블로그에서는 같은 기본값을 모두 바꿀 수 있었고, 실제로 하루 동안 대부분을 바꿨다. 누적 변환을 빼고, 시각을 받은 시각으로 바꾸고, quiet 확정을 진단으로 내리고, 주기 export를 없앴다. 그런데 바꾼 설정마다 새 기본값을 내가 정한 셈이 됐고, 그 기본값이 맞는지 확인하는 일도 함께 내 몫이 됐다. 손실을 보이려고 만든 패널이 손실을 가린 것이 그 예다.

그래서 설정을 바꿀 때마다 쓴 확인 방법은 3편과 같았다. 보낸 수와 받은 수를 세는 것이다. 다만 대상이 하나 늘었다. 값이 사라지는 경로뿐 아니라, 사라진 것을 알려 주는 경로도 그 상황을 일부러 만들어 세어 봐야 했다. 방문이 드물어 운영 데이터로는 그런 상황이 좀처럼 생기지 않으니, 재시작이나 수집 중단 같은 상황을 직접 만들어 보는 것이 가장 확실한 방법이라고 생각한다.

[^1]: [Oracle Cloud Infrastructure, Always Free Resources](https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm)의 "Reclamation of Idle Compute Instances"

[^2]: [Amazon VPC pricing](https://aws.amazon.com/vpc/pricing/)의 Public IPv4 Address 항목
