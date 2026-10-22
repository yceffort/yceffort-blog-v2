---
title: '방문이 드문 블로그의 <em>핵심 웹 지표</em>를 직접 수집하기'
tags:
  - web-performance
  - browser
published: false
date: 2026-10-22 18:47:00
description: '사내에서 공용 관측 인프라로 보내던 핵심 웹 지표 수집을 개인 블로그에 올리고, 수집 서버도 직접 띄웠다. 설정을 모두 바꿀 수 있게 되자 방문이 드물다는 것과 바꾼 설정을 모두 직접 확인해야 한다는 것이 새 제약이 됐다. 이틀 동안의 커밋 순서로 기록하고, 닷새 동안 쌓인 값으로 바꾼 것의 효과를 확인했다.'
series: 'Core Web Vitals 직접 수집하기'
seriesOrder: 4
---

## Table of Contents

## 첫날 대시보드가 비어 있었다

블로그에 수집 코드를 붙이고 몇 번 방문한 뒤 Mimir 대시보드를 열자 모든 패널이 비어 있었다. 재현 환경에서 페이지 로드 300번을 흘려 넣었을 때는 숫자가 잘 나오던 대시보드였고, 수집 경로도 그대로 옮긴 것이었다. 달라진 것은 방문의 빈도였다. 경로를 고친 뒤 Mimir를 비우고 다시 쌓기 시작해서 두 시간 가까이 들어온 페이지 로드가 7건이었다. (참으로 슬픈 일이 아닐 수 없다 😭)

3편까지의 제약은 바꿀 수 없는 공용 관측 인프라와 떠날 때 신호를 주지 않는 앱 웹뷰였다. 이번에는 반대다. 수집 서버는 월 \$12짜리 서버 한 대에 직접 띄웠고, 방문자는 일반 브라우저로 들어오고, collector부터 대시보드까지 모든 설정을 바꿀 수 있다. 그러자 다른 종류의 제약이 드러났다. 웹사이트의 방문이 드물면 서버 트래픽을 기준으로 정한 기본값이 값을 세지 못했고, 설정을 모두 바꿀 수 있는 만큼 바꾼 설정이 맞는지도 모두 직접 확인해야 했다.

이 글은 2026년 10월 17일과 18일 이틀 동안 수집 서버를 띄우고 블로그에 붙이면서 바꾼 것을, 커밋 순서대로 적은 기록이다. 절 제목의 날짜와 시각은 커밋 시각이다. "닷새 동안 쌓인 값" 절만 커밋이 아니라 그 뒤로 쌓인 운영 데이터로 채웠다.

> 수집 서버의 구성은 블로그 저장소의 [`infra/otel`](https://github.com/yceffort/blog/tree/e02c600a7ff73eaf8d507d153a3a21d26a3670c8/infra/otel)에, 브라우저 쪽 수집 코드는 [`apps/blog/src/utils/telemetry`](https://github.com/yceffort/blog/tree/e02c600a7ff73eaf8d507d153a3a21d26a3670c8/apps/blog/src/utils/telemetry)에 있다. 링크는 둘째 날 마지막 커밋으로 고정했다.
>
> 버전은 web-vitals 6.2.3, OTel JS 2.12.0(OTLP exporter 0.223.0), OpenTelemetry Collector contrib 0.163.1, Tempo 3.1.0, Mimir 3.2.2, Grafana 13.2.4, Caddy 2.11.7이다. 서버는 Amazon Lightsail 서울 리전의 Debian 13 인스턴스(2GB 메모리)다.

## 10월 17일 12:38 수집 서버를 블로그 저장소로

수집 서버는 처음에 재현 환경 저장소의 배포용 덮어쓰기 설정으로 띄웠다가, 실험과 운영을 섞지 않으려고 블로그 저장소의 `infra/otel`로 옮겼다. 서버는 블로그 저장소에서 이 디렉터리만 sparse checkout으로 받아 `docker compose up`으로 돈다. 구성은 Caddy, collector, Tempo, Mimir, Grafana 다섯 개다.

Lightsail은 AWS가 EC2 위에 얹어 파는 고정 요금 가상 서버다. 인스턴스, 디스크, 월 전송량, 공인 IPv4가 한 요금에 묶여 있고, 요금 밖에서 붙는 것은 전송량 초과분뿐이다. 호스팅은 그 고정 요금에 공인 IPv4와 전송량이 들어 있는 Lightsail을 골랐다.

| 선택지                       | 한 달 비용              | 걸린 점                                                                                                                         |
| ---------------------------- | ----------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| Lightsail `small_3_0`        | \$12                    | 2GB 메모리, vCPU 2개, 디스크 60GB, 전송 3TB, 공인 IPv4 1개가 들어 있다                                                          |
| 같은 AWS의 EC2               | 인스턴스 값에 따로 더함 | 디스크(EBS)와 공인 IPv4(시간당 \$0.005, 한 달 약 \$3.6)가 따로 붙는다                                                           |
| Oracle Cloud Always Free(A1) | 무료                    | 7일 동안 CPU 사용률의 95번째 백분위, 네트워크, 메모리가 모두 20% 미만이면 유휴로 보고 회수할 수 있다[^1]. 수집 서버는 거의 쉰다 |

> Lightsail 가격은 서울 리전에서 Lightsail API(`get-bundles`)로 확인한 값이고, EC2의 공인 IPv4 요금은 AWS의 VPC 요금표[^2]에서 확인했다. 2026년 10월 기준이고, 10월 22일에 다시 조회해도 같았다.

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

다른 경로는 404이고, 256KB를 넘는 본문은 413으로 끊는다. CORS는 collector에서 블로그 origin만 허용하지만, CORS는 브라우저가 지키는 규칙이라 curl로 보내는 요청은 막지 못한다. 누군가 가짜 값을 보낼 수 있다는 것은 받아들이고, 그 값이 들어갈 수 있는 자리를 줄이기로 했다. 다만 이때 라벨을 몇 개로 제한한 것은 브라우저 쪽 수집 코드뿐이었다. 서버는 라벨을 검사하지 않았으므로 직접 만든 OTLP 요청은 아무 metric 이름과 라벨이든 실을 수 있었고, 이 구멍은 10월 22일에 막는다. Grafana는 서버의 127.0.0.1에만 붙여 SSH 터널로만 연다. Caddyfile에 `log` 지시어를 두지 않아 접근 로그가 남지 않고, collector도 요청의 주소를 데이터에 붙이지 않으므로, 방문자의 IP는 어디에도 저장되지 않는다.

### Tempo 3.1의 TraceQL metrics는 하루보다 긴 구간을 조회하지 못한다

Tempo 3.1.0을 기본값으로 두면 trace 검색은 7일, TraceQL metrics는 24시간보다 긴 구간을 거부한다.

```text
range specified by start and end exceeds 168h0m0s. received start=1791535470 end=1792226640
metrics query time range exceeds the maximum allowed duration of 24h0m0s
```

> 기본 설정의 Tempo 3.1.0에 8일 구간의 검색과 2일 구간의 TraceQL metrics를 보내 받은 응답이다.

방문이 드문 블로그에서는 하루치로는 분위수를 낼 표본이 모이지 않는다. 그래서 보존 기간과 조회 범위를 함께 1년으로 늘렸다(`max_search_duration`, `max_metrics_duration`, `block_retention`). 보존 기간만 늘리면 데이터는 남아 있는데 조회는 막힌다.

## 10월 17일 12:42 블로그에 붙이다

브라우저 쪽은 재현 환경의 수집 코드를 블로그 구조에 맞게 옮겼다. production에서만 켜고, OTel SDK는 첫 화면에 필요 없으므로 동적 import로 따로 받는다. 이 청크는 10월 22일의 빌드(커밋 466ad1b1)에서 raw 100,571B, gzip 29,365B다. 수집 코드와 SDK 버전은 둘째 날 이후 바뀌지 않았다. web-vitals는 늦게 붙어도 buffered 항목을 읽어 LCP와 CLS를 놓치지 않지만, Event Timing의 buffered 항목은 104ms 이상인 것만 남으므로 SDK를 받기 전의 짧은 입력은 INP 후보에서 빠진다. 화면은 홈, 글, 태그, 시리즈, 그 밖의 다섯 가지(`page.route`)로만 나눈다.

## 10월 17일 13:42 대시보드가 비어 있었다

서론의 빈 대시보드가 이 시각이다. 원인을 찾으려고 방문이 드문 사이트를 흉내 내어, 같은 LCP 값 8건을 수집 경로 네 가지로 동시에 보내고 대시보드 쿼리가 몇 건으로 세는지 쟀다. 다섯 개의 라벨 조합은 한 번씩, 하나는 7분 간격으로 세 번 보냈다.

| 경로                                                              | 대시보드가 센 값 |
| ----------------------------------------------------------------- | ---------------- |
| 재현 환경 구성(`delta_to_cumulative`, remote write, `increase()`) | 0.00             |
| 위와 같고 누적값 만료(`max_stale`)만 7일                          | 2.43             |
| 위와 같고 Mimir OTLP 엔드포인트가 시작 시각에 0 샘플을 넣음       | 10.66            |
| 누적으로 바꾸지 않고 delta를 그대로 저장, `sum_over_time()`       | 8.00             |

> 세 번 실행했고 재현 환경 구성의 첫 실행만 같은 run 라벨을 다시 써서 앞 실행의 상태를 이어받았다. 나머지 두 번은 같았다. `cwv-otel-grafana/experiments/e8-sparse-counts`.

라벨 조합마다 샘플이 하나뿐이면 `increase()`는 그 시리즈를 결과에서 아예 뺀다. Prometheus와 같은 규칙으로 표본이 둘 미만이면 값을 내지 않으며, 표의 0.00은 빈 결과를 패널이 0으로 보인 것이다. Mimir 3.2.2의 기본 엔진인 Mimir query engine에서도 같았다. 표본 하나짜리 시리즈 20개에 `increase()`를 걸면 결과가 비고, 표본이 둘 이상인 시리즈만 돌아온다. 누적값 만료를 늘려도 한 번만 온 조합은 여전히 샘플 하나이고, 시작 시각에 0 샘플을 넣으면 증가로는 잡히지만 `increase()`가 구간 끝까지 외삽해 8건을 10.66건으로 센다. delta를 그대로 저장하면 샘플 하나가 페이지 로드 하나의 값이라 구간 안의 샘플을 더하면 된다. 그래서 블로그의 metric 경로는 재현 환경과 다르게 구성했다. collector에서 `delta_to_cumulative`를 빼고, Mimir의 delta 저장(`-distributor.otel-native-delta-ingestion`)으로 받고, 대시보드는 `sum_over_time()`으로 센다. Mimir 3.2.2는 이 기능을 "Delta support is in an early stage of development. The ingestion and querying process is likely to change over time."이라고 설명하는 실험 기능이라, Mimir를 올릴 때는 릴리스 노트를 확인한다.

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

대가는 두 가지다. 값의 시각이 페이지 로드 시각이 아니라 collector가 받은 시각이 되고, 같은 시리즈의 값 두 개가 같은 밀리초에 들어오면 같은 시각의 표본이 되어 Mimir가 하나를 버린다. 시리즈 하나에 초당 10건이면 1.3~1.5%를 잃었다(3편, `cwv-otel-grafana/experiments/e10-concurrent-writers`). 블로그 트래픽에서는 드물 것으로 보았고, 이 손실은 뒤의 17:05에서 다시 나온다.

## 10월 17일 13:43 누구에게, 무엇이 느린가

블로그는 화면이 다섯 가지뿐이라, 화면만으로는 어디가 느린지 좁히기 어려웠다. 그래서 값의 종류가 정해진 맥락을 metric 라벨에 더했다. 언어(`page.lang`, `/en`으로 시작하면 en), 기기(`device.class`, 화면 폭 768px과 1024px으로 나눈 세 가지), 브라우저 엔진(`browser.engine`)이다. iOS의 브라우저는 이름과 상관없이 WebKit이라 엔진을 먼저 가른다. 유입 경로(검색, 소셜, AI 서비스 등), 서비스 워커를 거친 내비게이션인지, 네트워크와 기기 정보는 값의 종류가 많거나 질문이 좁아서 span 속성에만 싣는다.

LCP 요소의 selector에서는 StyleX의 원자 클래스를 지운다.

```ts
// apps/blog/src/utils/telemetry/mask.ts
    // StyleX 원자 클래스(`x1j9s4mx`)는 스타일 하나씩이라 요소를 구분하지 못하고, 스타일을 고치면 이름이 바뀐다.
    // 남겨 두면 같은 요소가 배포마다 다른 값으로 갈린다. 숫자가 많으면 아래 규칙에 걸려 `x{n}` 이 되기도 한다.
    .replace(/\.x[0-9a-z]{5,8}(?![\w-])/g, '')
```

블로그의 스타일은 StyleX가 만드는 원자 클래스라, web-vitals가 만든 selector에는 그 클래스가 줄줄이 붙는다. 그대로 두면 Tempo 대시보드의 "LCP 요소별" 패널이 같은 요소를 배포마다 다른 요소로 센다.

## 10월 17일 14:17 봇, SPA 이동, bfcache

이때 고친 것은 세 가지다. 첫째는 봇이다. PageSpeed Insights(`Chrome-Lighthouse`), headless 브라우저, 검색 엔진의 렌더러가 블로그를 열면 그 값도 수집됐다. 사람의 방문이 드문 블로그에서는 이런 값이 섞이면 분위수가 쉽게 흔들린다. 그래서 SDK를 받기 전에 user agent와 `navigator.webdriver`로 거른다. 이 판정 때문에 뒤에서 Playwright로 블로그 페이지를 직접 검증할 수 없게 됐다.

둘째는 SPA 이동이다. 블로그 안의 링크 이동은 클라이언트 이동이라, 지금까지는 첫 로드 하나의 값만 있었다. web-vitals의 `reportSoftNavs`를 켜면 Chromium 151 이상에서 이동마다 FCP, LCP, CLS, INP를 다시 잰다. TTFB는 이동에서 0으로 보고되므로 첫 로드에만 잰다. 셋째는 bfcache 복원이다. 첫 로드, SPA 이동, bfcache 복원을 각각 내비게이션 하나로 보고 trace도 따로 만든다. SPA 이동은 web-vitals가 주는 `navigationId`로, bfcache 복원은 `pageshow`의 `persisted`를 센 횟수로 가른다. 대시보드에는 첫 로드와 SPA 이동을 고르는 변수를 두었다.

## 10월 17일 16:17 리뷰가 찾은 것

여기까지 올린 뒤 수집 코드와 대시보드를 리뷰받았고, 이슈 6건이 나왔다. 6건 모두 코드와 소스로 확인됐다. 그중 블로그에서 판단이 갈린 것을 적는다.

### 일반 브라우저에서 quiet 확정은 해롭다

3편의 quiet LCP(후보가 3초 동안 바뀌지 않으면 확정)는 hidden이 오지 않는 앱 웹뷰를 위한 장치였다. 수집을 시작한 날 오후까지 Tempo에 쌓인 LCP 9건 중 7건이 quiet로 확정됐다. 글을 읽는 동안에는 3초 넘게 아무것도 누르지 않는 경우가 많아서, quiet가 web-vitals보다 먼저 확정하는 일이 흔할 것으로 본다. 그런데 일반 브라우저는 떠날 때 hidden이 오고, 그때 web-vitals가 LCP를 확정한다. quiet의 이득은 거의 없고, 3편에서 본 대로 더 큰 요소를 놓칠 위험만 남는다.

그래서 블로그는 web-vitals가 확정한 LCP만 histogram에 넣는다. quiet 타이머는 남겨 두되, 처음으로 3초 멈췄던 후보의 값을 span 속성으로만 싣는다(`cwv.lcp.quiet`가 `none`, `same`, `lower` 중 하나). `lower`는 quiet로 확정했다면 LCP를 작게 기록했을 페이지 로드다. 이 비율이 얼마나 되는지는 "닷새 동안 쌓인 값" 절에서 본다.

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

delta라서 flush가 겹쳐도 같은 값을 두 번 보내지 않는다. 블로그 페이지는 자동화 브라우저를 거르므로, 재현 환경의 데모 페이지에 이 reader를 그대로 복사해 넣고 두 reader를 Chromium에서 비교했다(`cwv-otel-grafana/experiments/e15-flush-only-reader`). 로드 1초 뒤 탭을 숨겨 첫 flush를 보내고, collector 앞의 프록시가 그 응답을 5초 늦춘다. 300ms 뒤 탭으로 돌아와 버튼을 누르고, 1.5초 뒤 다른 페이지로 떠난다. 첫 flush에는 TTFB, FCP, CLS가 실리고, 떠날 때의 두 번째 flush에 LCP와 INP가 실린다.

| 수집 코드                                    | 첫 flush의 응답 지연 | 떠날 때의 두 번째 요청 | Mimir에 도착한 LCP, INP |
| -------------------------------------------- | -------------------- | ---------------------- | ----------------------- |
| 주기 export(`PeriodicExportingMetricReader`) | 5초                  | 오지 않음              | 0건, 0건                |
| 주기 export(`PeriodicExportingMetricReader`) | 0                    | 떠난 뒤 10ms           | 1건, 1건                |
| `FlushOnlyMetricReader`                      | 5초                  | 떠난 뒤 12ms           | 1건, 1건                |
| `FlushOnlyMetricReader`                      | 0                    | 떠난 뒤 11~13ms        | 1건, 1건                |

> Playwright 1.64.1의 Chromium에서 조건마다 두 번 실행해 같았다. headless Chromium은 탭을 바꿔도 hidden이 되지 않아, 첫 hidden은 `visibilityState`를 덮어쓰고 `visibilitychange`를 만들어 냈다. 스크립트로 만든 이벤트는 `isTrusted`가 아니라서 web-vitals는 이 hidden으로 LCP를 확정하지 않고(1편), 그래서 LCP가 떠날 때의 요청에 실렸다. 실제 탭 전환에서는 LCP도 첫 flush에 실린다. 측정은 blog-experiments 커밋 [`be34dc4`](https://github.com/yceffort/blog-experiments/tree/be34dc4/cwv-otel-grafana/experiments/e15-flush-only-reader)에 있다.

대가는 hidden이 끝내 오지 않는 페이지 로드다. 그런 페이지 로드는 값이 하나도 나가지 않는다. 일반 브라우저에서는 탭을 닫거나 앱을 바꿀 때도 hidden이 오므로 드물 것으로 보았다. 비율은 다음 날 trace와 metric의 건수를 맞대어 세어 보았다(10월 18일 16:29).

### 같은 id의 갱신과 같은 밀리초의 중복은 재기만 했다

다른 두 건은 고치는 대신 재기만 했다. 하나는 같은 id로 다시 오는 CLS와 INP다. histogram은 첫 보고만 세므로, 탭에 돌아온 뒤에 나빠진 값은 들어가지 않는다. 다른 하나는 앞에서 본 같은 밀리초 중복이다. 둘 다 3편의 재현 환경처럼 gateway를 두어 측정마다 최신값을 집계하면 풀린다. 다만 상태를 들고 있는 서버를 하나 더 운영해야 하고, 시간당 몇 건인 트래픽에서는 그 정확도 차이를 데이터로 확인하기도 어렵다. 그래서 다시 온 보고는 histogram 대신 `cwv.cls.update`, `cwv.inp.update` span으로 남기고(첫 값, 등급이 바뀌었는지, 몇 번째 보고인지), Mimir가 버린 표본은 대시보드에 보이게 했다. 둘 다 "닷새 동안 쌓인 값" 절의 값을 보고 다시 판단한다.

[1편](/2026/10/web-vitals-report-timing)에서 찾은 INP 손실도 알고서 그대로 두었다. 블로그는 web-vitals 6.2.3을 쓰고 INP를 기본 설정으로 받으므로, 상호작용 뒤 1초 안에 떠난 페이지 로드의 INP는 보고되지 않는다. INP만 `reportAllChanges`로 받는 우회는 첫 보고만 세는 이 구조와 맞지 않아 넣지 않았고, 수정(#809)이 담긴 버전이 나오면 web-vitals를 올려서 해결할 생각이다. 올린 뒤에는 1편처럼 떠나는 페이지 로드를 만들어 지표별 보고 수를 다시 센다.

## 10월 17일 17:05 손실 패널이 손실을 숨기고 있었다

버린 표본을 보이려고 만든 패널을 다시 리뷰받자 3건이 나왔다. 셋 다 맞는 지적이었고, 셋 다 그 패널이 손실을 정상처럼 보이게 하는 문제였다.

collector는 1분마다 Mimir의 `cortex_discarded_samples_total`과 자기 지표(받지 못한 수, 내보내지 못한 수)를 긁어 Mimir에 넣는다. 처음 패널 쿼리는 이랬다.

```promql
max by (reason) (max_over_time(cortex_discarded_samples_total{job="mimir", user="blog"}[$__range])) or on() label_replace(vector(0), "reason", "없음", "", "")
```

첫째, `or vector(0)`는 손실 지표를 아예 긁지 못한 상태도 0으로 보여 준다. 수집 설정이 빠진 새 서버에서도, collector가 멈춘 뒤에도 패널은 "없음 0"이었다. 둘째, `max_over_time`은 조회 구간의 과거 최댓값이라 Mimir가 재시작해 카운터가 다시 시작해도 재시작 전 값을 보여 준다. 셋째는 손실과 상관없는 버그였다. 대시보드 생성 스크립트가 refId를 매기는 반복문을 마지막 패널에만 돌리고 있었는데, 손실 패널을 그 뒤에 붙이면서 대상이 바뀌어 "지표별 표본 수" 패널의 쿼리 다섯 개가 모두 refId A가 됐다. Grafana는 이 패널을 `400 query.duplicateRefId`로 거부했다. 대시보드를 화면으로 열어 보지 않고 쿼리만 확인한 탓에 놓쳤다.

고친 쿼리는 마지막 수집이 성공했고(`up == 1`) 3분 안일 때만 값을 보여 주고, 카운터의 최신값을 쓴다. 이 값은 조회 범위와 상관없이 Mimir가 시작된 뒤의 누적이라 패널 제목에도 그렇게 적었다.

```promql
(max by (reason) (cortex_discarded_samples_total{job="mimir", user="blog"}) or on() label_replace(vector(0), "reason", "없음", "", "")) and on() max((up{job="mimir"} == 1) and (time() - timestamp(up{job="mimir"}) < 180))
```

로컬에 같은 구성을 띄우고 fixture로 비교했다. 고친 쿼리는 첫 수집 전에도 "수집 안 됨"이었다.

| 상황                               | 처음 쿼리 | 고친 쿼리  |
| ---------------------------------- | --------- | ---------- |
| collector를 멈추고 4분 뒤, 6분 뒤  | 없음 0    | 수집 안 됨 |
| 버린 표본 10, Mimir 재시작, 다시 5 | 10        | 5          |

> 실제 Grafana 13.2.4의 `/api/ds/query`로 패널 쿼리를 보내 확인했고, 고친 뒤에는 Mimir 패널 30개가 모두 200이었다. 운영 서버의 Grafana에서도 같았다.

## 10월 18일 13:50 복원된 페이지의 TTFB가 첫 로드로 세어졌다

다음 날 수집 코드와 서버 구성을 다시 리뷰받았다. 처음 나온 것은 14:17에 넣은 bfcache 복원의 키였다. 복원의 TTFB는 web-vitals가 `pageshow` 리스너 안에서 바로 보고한다(1편의 `onBFCacheRestore`). 수집 코드도 window의 `pageshow`에서 복원 횟수를 세고 그 횟수를 키로 썼는데, web-vitals는 capture로 걸었고 수집 코드는 bubble로 걸었다. 현행 DOM 명세에서는 같은 타깃이라도 capture 리스너가 등록 순서와 상관없이 먼저 돌므로, 복원의 TTFB는 횟수가 늘기 전에 보고되어 다른 지표와 다른 키를 받았다. 프레임 두 개 뒤에 보고되는 FCP와 LCP는 늘어난 횟수로 갔다. 복원 하나가 trace 둘로 갈라진 것이다.

키는 web-vitals가 복원 지표에 넣어 주는 `navigationStartTime`으로 바꿨다. 1편의 코드에서 본 대로 이 값은 복원 `pageshow`의 `timeStamp`라, 같은 복원의 지표가 모두 같은 값을 들고 온다. 수집 코드는 경로만 저장한다. 복원된 문서는 SPA 이동 뒤의 URL일 수 있어서, 같은 `timeStamp`를 키로 그때의 경로를 적어 두고 지표가 올 때 찾는다.

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

그래서 복원의 TTFB는 아예 기록하지 않고(SPA 이동의 TTFB를 재지 않는 것과 같은 이유다), 변수를 첫 로드, SPA 이동, bfcache 복원, 전체의 네 가지로 나눴다. 복원이 얼마나 되는지는 복원마다 보고되는 FCP의 건수로 센다. 1편의 결론을 대시보드에서 다시 밟은 셈이다. `navigationType`이 다른 값을 한 분포에 섞으면 같은 이름의 지표가 다른 것을 잰 값끼리 섞인다.

## 10월 18일 13:50 preview 배포도 production 빌드다

두 번째는 Vercel의 preview 배포였다. 수집 코드는 `NODE_ENV`가 production일 때만 켜지는데, preview 배포도 production 빌드라 그대로 켜졌다. collector는 블로그 origin만 허용하므로 브라우저가 요청을 막고, 2편의 표대로 exporter는 CORS 거부를 네트워크 오류로 보고 약 8초 동안 5번 시도한다(처음 요청과 재시도 4번). 값이 잘못 들어가지는 않았지만, preview에서 글을 확인할 때마다 SDK를 받고 실패할 요청을 보내고 있었다. SDK를 받기 전에 origin을 비교해 거른다. 같은 자리에서 동적 import의 실패도 잡았다. 청크를 받지 못하면 `unhandledrejection`으로 새던 것을 삼켜서, 계측이 실패해도 독자에게 오류로 보이지 않게 했다.

```tsx
// apps/blog/src/components/analytics/CwvTelemetry.tsx
    // preview 배포도 production 빌드라 여기까지 온다. collector 는 블로그 origin 만 받으므로 보내지 않는다.
    if (window.location.origin !== SiteConfig.url) return
```

## 10월 18일 13:51 export마다 preflight가 붙어 있었다

서버 쪽에서는 세 가지를 고쳤다. 첫째는 CORS preflight다. exporter는 `Content-Type: application/json`으로 다른 origin에 보내므로 요청마다 preflight(OPTIONS)가 먼저 간다. collector의 CORS 응답에 `Access-Control-Max-Age`가 없으면 브라우저는 preflight 결과를 기본값인 5초만 기억한다[^3]. span은 5초 주기로 나가니 export마다 OPTIONS가 하나씩 붙고 있었던 셈이다. collector의 `cors.max_age`를 Chromium의 상한인 2시간으로 두었다. 고친 뒤 서버에서 OPTIONS를 보내 `access-control-max-age: 7200`을 확인했다.

둘째는 collector의 메모리 상한이다. `memory_limiter`는 `limit_percentage: 80`으로 두었는데, README는 이 값을 컨테이너 환경에서 호스트가 정한 메모리 제한을 기준으로 쓰라고 적는다[^4]. compose에는 collector의 메모리 제한이 없었다. 제한이 없으면 컨테이너가 쓸 수 있는 메모리는 서버 전체라 80%는 2GB의 80%가 되고, 그 선에 닿기 전에 Grafana와 Tempo와 Mimir가 먼저 밀린다. 전날 오후의 collector가 86MiB였으므로 상한을 512MiB로 두었다. `memory_limiter`가 받기를 멈추는 선은 80%에서 `spike_limit_percentage` 20%를 뺀 약 307MiB다. 넘으면 17:05의 손실 패널에 받지 못한 수로 보인다.

셋째는 컨테이너 로그다. Docker의 `json-file` 로그는 기본으로 회전하지 않는다. 1년 보존으로 둔 서버라 로그가 디스크를 채울 수 있어, 컨테이너마다 10MB 파일 3개로 돌린다.

## 10월 18일 13:54 flush 시점은 수집 코드 하나가 정한다

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

이날 리뷰 지적 중 하나는 틀렸다. 값이 없는 속성(`undefined`)이 OTLP에서 빈 값으로 나가 Tempo에 키만 남는다는 지적이었는데, sdk-trace 2.12.0의 `Span.setAttribute`는 `null`과 `undefined`를 exporter에 넘기기 전에 버린다. core의 속성 정리 함수와 변환기만 보고 Span을 보지 않은 지적이었다. 코드로 확인하고 반영하지 않았다.

## 10월 18일 16:29 대시보드를 열자 429가 났다

17:05에서 대시보드를 화면으로 열어 보지 않아 놓친 버그가 있었으므로, 이번에는 고친 대시보드를 운영 Grafana에서 열었다. 처음 열고 맨 아래로 바로 스크롤하자 패널 3개가 `too many outstanding requests`로 깨졌다. Mimir가 429를 돌려준 것이다. Mimir 로그에는 15:51:03에 10ms 사이 16건의 요청을 버렸다고 남았고, 그 전 48시간에는 한 건도 없었다. 다시 열거나 천천히 스크롤하면, 7일이나 30일 범위에서도 나지 않았다.

원인은 Mimir의 기본값이었다. 운영 Mimir의 `/config`로 확인한 값은 쿼리를 24시간 단위로 쪼개고(`split_queries_by_interval`), 쪼갠 조각을 다시 16개로 샤딩하고(`parallelize_shardable_queries`, `query_sharding_total_shards`), 테넌트당 대기 요청을 100개까지만 받는다(`max_outstanding_requests_per_tenant`). 48시간 범위면 패널 쿼리 하나가 최대 48개 하위 요청이 되고, 쿼리 다섯 개를 가진 "지표별 표본 수" 패널 하나가 240개다. 여러 querier에 일을 나누려는 설계인데, 이 Mimir는 프로세스 하나에 querier 하나(`max_concurrent: 8`)이고 결과 캐시도 꺼져 있어 나누는 만큼 큐만 길어진다.

```yaml
# infra/otel/mimir/mimir.yaml
frontend:
  split_queries_by_interval: 0
  parallelize_shardable_queries: false
```

패널 하나를 요청 하나로 둔 뒤, 서버 안에서 운영 대시보드의 Mimir 쿼리 36개를 한꺼번에 보내는 것을 48시간 범위로 한 번, 30일 범위로 두 번 돌렸다. 108개 요청이 모두 200이었고 Mimir가 버린 요청은 0건이었다. 고치기 전의 같은 실험은 돌리지 않았다. 근거는 실제로 난 429와 기본값의 계산이다.

### trace는 왔는데 metric이 없다

같은 날 넣은 패널 두 개가 처음 보여 준 값은 Tempo의 첫 로드 trace 56건, Mimir의 첫 로드 표본 40건이었다. 둘은 같은 페이지 로드를 세는 패널이다. span은 5초 주기로도 나가고 metric은 hidden의 flush로만 나가므로, 차이는 떠나는 순간의 flush가 오지 않은 페이지 로드다. 16건을 라벨과 시각으로 하나씩 맞춰 보았다. 6건은 17일 13:43 배포 전의 trace였다. Mimir의 첫 표본이 13:56이라 그 전에는 metric 경로가 없었다. 나머지 10건은 17일 저녁부터 18일 오후까지 고르게 퍼져 있었고, 대부분 데스크톱 Chromium에서 글을 연 페이지 로드였다. Mimir가 버린 표본과 collector가 받지 못한 수는 모두 0이었으므로, 남는 후보는 탭이 아직 hidden이 되지 않았거나(열어 둔 탭), 떠나는 순간의 keepalive 요청이 끝까지 가지 못한 경우다. 서버의 데이터로는 둘을 가를 수 없다. 16:17에서 미뤄 둔 비율은 이틀째 48시간 기준 50건 중 10건이었다. 닷새치 값은 "닷새 동안 쌓인 값" 절에서 다시 본다.

## 10월 22일 16:25 설정 반영을 GitHub Actions에 맡기다

지금까지는 서버 설정을 바꿀 때마다 SSH로 들어가 `git pull`과 `docker compose up -d`를 손으로 쳤다. 리뷰 반영처럼 하루에 몇 번씩 바꾸는 날에는 이 단계를 빼먹기 쉽고, 저장소와 서버가 다른 커밋에 있는 채로 글을 쓰게 된다. 그래서 `infra/otel`이 main에 올라가면 GitHub Actions가 서버에 SSH로 들어가 반영하게 했다(`.github/workflows/deploy-otel.yml`, `infra/otel/deploy.sh`).

키는 배포 전용으로 하나 새로 만들었다. 서버의 `authorized_keys`에서 이 키는 forced command에 묶여 있어, 워크플로가 어떤 명령을 보내든 `git pull` 뒤 `deploy.sh`만 실행된다. 포트 포워딩과 pty도 막았다. 키가 새어도 할 수 있는 일은 저장소의 최신 설정을 반영하는 것뿐이다. 워크플로가 SSH 명령으로 보내는 커밋 해시는 `SSH_ORIGINAL_COMMAND`로 스크립트에 전달되고, pull 뒤의 HEAD와 다르면 경고를 남긴다.

```text
# 서버의 ~/.ssh/authorized_keys
command="cd ~/blog && b=$(git rev-parse HEAD) && git pull --ff-only --quiet && exec bash infra/otel/deploy.sh $b",no-port-forwarding,no-agent-forwarding,no-X11-forwarding,no-pty ssh-ed25519 AAAA... github-actions deploy-otel
```

`deploy.sh`는 `docker compose up -d`를 돌린 뒤, pull 전후의 diff에서 설정 디렉터리가 바뀐 서비스만 다시 시작한다. bind mount한 설정 파일은 바뀌어도 compose가 컨테이너를 다시 만들지 않기 때문이다. compose.yaml이 바뀐 서비스는 `up -d`가 알아서 다시 만든다.

첫 실행은 실패했다. 22번 포트가 내 IP 하나에만 열려 있어서 GitHub 러너가 연결하지 못했다(`Connection timed out`). 러너의 IP는 실행마다 바뀌므로, 22번을 모두에게 열고 키 인증과 forced command에 기대거나, 실행마다 Lightsail API로 러너 IP를 열었다 닫거나, 서버가 주기적으로 pull하는 방식 중에서 골라야 한다. 두 번째는 AWS 키를 GitHub에 둬야 하고 실패했을 때 닫는 단계가 빠질 수 있으며, 세 번째는 푸시 즉시 배포가 아니다. sshd가 비밀번호 인증을 끈 키 전용이고 배포 키가 할 수 있는 일이 정해져 있으니 첫 번째를 골랐다. Lightsail의 `open-instance-public-ports`는 기존 규칙에 더하는 동작이라 `0.0.0.0/0`을 기존 IP 목록과 함께 지정할 수 없다고 거부했고, 22번 규칙을 닫았다가 다시 열어야 했다. 다시 실행한 워크플로는 서버를 e02c600a에서 f3c6f8ff로 pull했고, 설정 디렉터리에 바뀐 것이 없어 컨테이너를 다시 시작하지 않은 채 몇 초 만에 끝났다.

## 10월 22일 19:09 collector가 라벨을 검사하지 않았다

글을 점검하다가 12:38 절의 "받는 쪽" 설명에서 멈췄다. 누군가 임의로 요청을 보내면 값이 오염되지 않는가. 오염된다. 그리고 막을 수 있는 것과 없는 것이 갈린다. 그럴듯한 가짜 값은 막을 수 없다. 인증을 붙일 수 없는 주소이고, collector는 요청의 `service.name`과 라벨을 그대로 믿으므로 블로그 이름으로 보낸 `cwv.lcp` 값은 대시보드에 섞인다. 지표당 하루 100건 안팎이라 가짜 수십 건이면 p75가 움직인다. 막을 수 있는 것은 라벨 조합의 폭증이다. 그런데 12:38에 라벨을 몇 개로 제한한 곳은 브라우저 쪽 코드뿐이었고, 서버 쪽 상한은 Mimir의 기본값뿐이었다.

| Mimir 3.2.2 기본값 | 값      |
| ------------------ | ------- |
| 테넌트당 시리즈    | 150,000 |
| 시리즈당 라벨 수   | 30      |
| 라벨 값 길이       | 2,048자 |
| 초당 수신 샘플     | 10,000  |

직접 만든 OTLP 요청은 아무 metric 이름과 라벨 30개, 2,048자짜리 값을 실을 수 있고, Tempo는 그런 상한도 없이 256KB 본문 × 요청 수만큼 1년 보존으로 쌓인다. 그래서 브라우저 쪽 수집 코드의 스키마를 collector가 강제하게 했다. `filter` 프로세서는 `service.name`이 블로그가 아니거나, metric 이름이 다섯 지표가 아니거나, histogram이 아니거나 delta가 아니거나, 라벨 값이 허용 목록 밖이면 버린다. `transform` 프로세서는 통과한 뒤에도 모르는 라벨 키를 지우고(`keep_keys`), span 속성은 64개와 512자로 자른다.

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

같은 이미지로 로컬에서 설정을 검증하고, exporter를 debug로 바꿔 요청 12종을 보냈다. metric은 정상 요청과 "모르는 키가 섞인 요청" 둘만 살아남았고 섞인 키는 지워져 있었다. 다른 service.name, 모르는 이름, cumulative, gauge, 허용 목록 밖의 라벨 값, 라벨 누락은 모두 버려졌다. trace는 같은 trace의 두 span이 `cwv.*`면 통과했고, 2,000자짜리 속성은 512자로 잘렸다. 가짜 값 자체는 여전히 들어오지만, 라벨 조합 수는 설계값인 수천 개를 넘지 못한다.

이 변경이 자동 배포의 첫 설정 변경이었고, 그 덕에 배포 스크립트의 버그가 드러났다. 워크플로는 성공했는데 collector가 "Up 4 days" 그대로였다. 스크립트가 `cd infra/otel` 뒤에 `git diff -- infra/otel`을 쓰고 있었는데, git의 pathspec은 현재 디렉터리 기준이라 `infra/otel/infra/otel`을 찾은 셈이 되어 바뀐 파일이 없다고 본 것이다. pathspec을 `.`로 고친 커밋(19:10)이 올라가자 바뀐 파일 둘이 찍히고 collector만 다시 시작됐다. 설정을 반영하는 경로도 그 상황을 일부러 만들어 세어 봐야 한다는 17:05의 교훈이 여기서 한 번 더 맞았다.

## 닷새 동안 쌓인 값

10월 17일 13:56(Mimir의 첫 표본)부터 10월 22일 15:00까지 약 121시간 동안 쌓인 값이다. Tempo와 Mimir의 건수 비교는 같은 2시간 조각 61개(10월 17일 13:00부터 10월 22일 15:00까지)로 맞췄고, 아래의 다른 수치도 모두 같은 15:00을 끝으로 조회했다. Tempo 3.1.0의 TraceQL metrics API는 조회 구간이 2시간을 넘으면 즉시 조회와 구간 조회 모두 `end must be greater than start`로 실패했는데, Grafana 패널은 같은 구간을 조회하므로 직접 보낸 요청의 형식 문제일 수 있다. 원인은 찾지 못했고, 2시간 조각의 값이 1시간 둘로 쪼개 더한 값과 같은 것을 확인한 뒤 조각으로 합산했다.

### trace는 있는데 metric이 없는 비율은 지표마다 다르다

| 지표 | Tempo의 span | Mimir의 표본 | 차이        |
| ---- | ------------ | ------------ | ----------- |
| TTFB | 624          | 526          | 98건(15.7%) |
| FCP  | 954          | 896          | 58건(6.1%)  |
| LCP  | 902          | 880          | 22건(2.4%)  |
| CLS  | 793          | 778          | 15건(1.9%)  |
| INP  | 707          | 702          | 5건(0.7%)   |

span은 5초 주기로도 나가고 metric은 hidden의 flush로만 나간다. 그래서 차이는 지표가 보고되는 시점에 따라 갈린다. TTFB와 FCP는 로드 직후 보고되어 span은 주기 export로 이미 나갔는데, hidden이 오지 않으면 metric만 없다. LCP, CLS, INP는 web-vitals가 hidden에서 보고하므로 span과 metric이 같은 flush에 실려 함께 있거나 함께 없다. 그래서 조회 시점에 trace는 있는데 metric이 없는 첫 로드의 비율은 TTFB의 차이로 읽는다. 약 15%다(Tempo의 624건 중 6건은 13:56 전의 것이라 빼면 14.9%). 이틀째 48시간의 20%보다 낮다. 16:29에서처럼 열어 둔 탭인지 떠나는 순간의 전송 실패인지는 서버 데이터로 가르지 못한다. 서버 안에서 잃은 것은 없었다. Mimir가 버린 표본은 0, collector가 받지 못한 수와 내보내지 못한 수도 0이었다.

첫 로드 가운데 TTFB는 있는데 FCP가 없는 페이지 로드도 있다. Mimir의 첫 로드 TTFB는 524건인데, bfcache 복원을 뺀 첫 로드의 FCP는 453건이라 71건(13.5%)은 TTFB만 있다. web-vitals는 첫 그리기 전에 페이지가 숨겨지면 FCP와 LCP를 보고하지 않으므로(가시성 감시, 1편), 백그라운드 탭으로 열어 둔 글이 여기 들어간다고 본다. 이 추정은 확인하지 않았다.

### 라벨 조합 90개 중 38개는 표본이 하나였다

FCP 표본이 하나라도 있는 라벨 조합은 90개다. 그중 38개는 닷새 동안 표본이 하나뿐이었고, 가장 많은 조합이 132개였다. 조합당 하루 평균 2건이다. 13:42에서 본 대로 `delta_to_cumulative`와 `increase()`였다면 조합 38개는 결과에서 빠져 세어지지 않았을 것이다. 이틀 동안 바꾼 것 중 효과가 가장 컸던 것은 이 delta 저장이다.

### 내비게이션의 40%가 SPA 이동이었다

Tempo의 내비게이션 1,076건 중 첫 로드가 623건, SPA 이동이 433건(40%), bfcache 복원이 12건(첫 로드의 1.9%)이다(8건은 13:43 전이라 속성이 없다). 14:17에 `reportSoftNavs`를 켜지 않았다면 이 40%는 값이 없었다. FCP 기준으로 Chromium이 852건(89%), WebKit 62건, Gecko 36건이었고, SPA 이동은 Chromium 151 이상에서만 잡히므로 WebKit과 Gecko의 방문은 첫 로드 하나로만 남는다. 첫 로드 502건의 유입 경로는 직접 252건(50%), 검색 188건(37%), 블로그 안 33건, 그 밖 27건, AI 서비스 2건이었고, 212건(42%)은 서비스 워커를 거친 내비게이션이었다.

### quiet로 확정했다면 LCP를 작게 적었을 페이지 로드는 0.6%다

LCP span 902건의 `cwv.lcp.quiet`는 `none` 341건, `same` 546건, `lower` 5건이고 10건은 속성이 생기기 전 것이다. quiet 타이머가 web-vitals보다 먼저 끝난 페이지 로드가 551건(62%)이니, 글을 읽는 동안 3초 넘게 입력이 없는 경우가 흔하다는 16:17의 추정은 맞았다. 그중 quiet 값이 정식값보다 작았던 것은 5건(0.6%)이고, 5건 모두 정식값의 등급이 poor였다. 3편의 처음 구성처럼 quiet로 확정했다면 가장 느린 LCP 5건을 더 작은 값으로 기록했을 것이다. 비율이 작으므로 일반 브라우저에서는 quiet 값을 진단 속성으로만 두는 지금 구성을 유지한다.

### 탭에 돌아온 뒤 CLS 등급이 바뀐 페이지 로드는 4.4%다

같은 id로 다시 온 보고는 한 페이지 로드가 여러 번 보낼 수 있어서 span이 아니라 trace(페이지 로드) 단위로 셌다. 다시 온 보고가 있는 페이지 로드는 CLS 77건(CLS를 보고한 793건의 9.7%), INP 56건(708건의 7.9%)이고, span으로는 각각 96건과 62건이다. 그중 등급이 바뀐 페이지 로드는 CLS 35건(4.4%), INP 2건(0.3%)이다. 첫 보고만 세는 histogram은 CLS 페이지 로드 23건 중 1건의 등급을 실제보다 좋게 기록하는 셈이고, INP는 갱신이 와도 등급이 거의 바뀌지 않았다.

그러면 실제 CLS는 뒤에 나올 p75보다 나쁘다. histogram의 CLS는 "처음 숨겨질 때까지의 CLS", 즉 독자가 탭을 떠나기 전에 겪은 밀림이고, 돌아와서 더 겪은 밀림은 들어가지 않는다. 그래도 지금 구성을 유지한 이유는 두 가지다. 첫째, CLS가 약점이라는 판단은 등급이 바뀐 4.4%를 넣지 않아도 성립하고, 넣으면 더 나빠질 뿐 방향이 바뀌지 않는다. 둘째, 고치려면 어느 요소가 미는지를 알아야 하는데 그 답은 집계값이 아니라 Tempo의 span 속성(web-vitals가 주는 가장 큰 밀림의 요소)에 있고, 갱신 span에도 같은 속성이 남는다. 3편의 gateway처럼 최신값으로 교체하면 집계값은 정확해지지만, 상태를 들고 있는 서버 하나와 30분의 확정 대기, 재시작 때의 손실이 따라오고, 그 비용으로 얻는 것은 시간당 몇 건인 트래픽에서 4.4%의 등급 보정이다. 그래서 집계값은 첫 hidden까지의 CLS라는 정의를 받아들이고, 실제 값이 그보다 나쁘다는 사실은 갱신 span의 비율로 같이 읽는다.

### 블로그의 약점은 CLS다

첫 로드의 p75는 TTFB 455ms, FCP 1,191ms, LCP 1,499ms, INP 75ms, CLS 0.126이다. 등급은 LCP의 90%(445건 중 399건), INP의 96%(339건 중 326건)가 good인데 CLS는 68%(365건 중 250건)에 그쳤고, p75 0.126은 needs improvement 구간이다. LCP는 데스크톱 1,526ms와 모바일 1,473ms로 비슷했고, SPA 이동의 LCP p75는 467ms였다. 13:43에서 라벨을 더한 이유가 이 표다. 어느 화면의 어느 요소가 밀리는지는 Tempo 대시보드의 몫이고, 그 답은 다른 글에서 다룬다.

### 서버는 627MiB를 쓴다

10월 22일 오후의 메모리는 Grafana 249MiB, Tempo 192MiB, Mimir 93MiB, collector 74MiB, Caddy 19MiB로 합쳐 627MiB다. 첫날 오후의 620MiB와 거의 같고, Tempo가 158MiB에서 192MiB로 늘고 Mimir가 115MiB에서 93MiB로 줄었다. 디스크는 59GB 중 6.5GB를 쓴다. collector는 10월 18일에 다시 뜬 뒤 metric point 92,059개와 span 5,196개를 받았고, tail sampling은 trace 1,076건을 모두 통과시켰다(span 2개 미만으로 버린 trace 0건). 청구서는 월 \$12 고정이라 따로 적지 않는다.

## 바꿀 수 있는 곳의 제약

3편에서는 바꿀 수 없는 기본값이 값을 버렸다. 블로그에서는 같은 기본값을 모두 바꿀 수 있었고, 실제로 이틀 동안 대부분을 바꿨다. 누적 변환을 빼고, 시각을 받은 시각으로 바꾸고, quiet 확정을 진단으로 내리고, 주기 export를 없애고, 대시보드 쿼리를 쪼개던 Mimir의 분할과 샤딩을 껐다. 그런데 바꾼 설정마다 새 기본값을 내가 정한 셈이 됐고, 그 기본값이 맞는지 확인하는 일도 함께 내 몫이 됐다. 손실을 보이려고 만든 패널이 손실을 가린 것이 그 예다.

그래서 설정을 바꿀 때마다 쓴 확인 방법은 3편과 같았다. 보낸 수와 받은 수를 세는 것이다. 다만 대상이 하나 늘었다. 값이 사라지는 경로뿐 아니라, 사라진 것을 알려 주는 경로도 그 상황을 일부러 만들어 세어 봐야 했다. 방문이 드물어 운영 데이터로는 그런 상황이 좀처럼 생기지 않으니, 재시작이나 수집 중단 같은 상황을 직접 만들어 보는 것이 가장 확실한 방법이라고 생각한다.

닷새치 값은 이틀 동안의 판단을 대체로 뒷받침했다. 효과가 가장 컸던 것은 delta 저장이었고(표본 하나짜리 라벨 조합 38개), quiet 확정과 같은 id의 갱신은 비율이 작아 진단으로만 두는 구성을 유지한다. 남은 손실은 서버가 아니라 브라우저 쪽에 있다. 첫 로드의 15%는 조회 시점에 trace에 대응하는 metric이 없었고, 13.5%는 TTFB만 있고 FCP가 없었다. 앞의 것은 열어 둔 탭과 전송 실패를 가르지 못했고, 뒤의 것은 백그라운드 탭이라는 추정을 확인하지 못했다. 어느 쪽이든 서버 설정으로 줄일 수 있는 값은 아니다.

[^1]: [Oracle Cloud Infrastructure, Always Free Resources](https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm)의 "Reclamation of Idle Compute Instances"

[^2]: [Amazon VPC pricing](https://aws.amazon.com/vpc/pricing/)의 Public IPv4 Address 항목

[^3]: [MDN, Access-Control-Max-Age](https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/Access-Control-Max-Age). 기본값은 5초이고, Chromium은 76부터 2시간(7200초), Firefox는 24시간을 상한으로 둔다고 적는다.

[^4]: [opentelemetry-collector v0.163.0 `processor/memorylimiterprocessor/README.md`](https://github.com/open-telemetry/opentelemetry-collector/blob/v0.163.0/processor/memorylimiterprocessor/README.md)의 "For containerized environments or other similar environments that support setting memory restrictions on the Collector, `limit_percentage` should generally be used, which allows the host environment to determine the Collector's max memory allocation"
