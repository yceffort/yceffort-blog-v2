# infra/otel

블로그 페이지의 Core Web Vitals 를 OpenTelemetry 로 받아 trace 는 Tempo 에, metric 은 Mimir 에 저장하고 Grafana 대시보드 두 개로 보는 수집 서버다. [blog-experiments/cwv-otel-grafana](https://github.com/yceffort/blog-experiments/tree/main/cwv-otel-grafana) 의 lab 을 인터넷에서 받을 수 있게 바꾼 운영본이다. 버전은 lab 과 같고, metric 경로만 방문이 드문 블로그에 맞게 바꿨다(아래). 브라우저 쪽 수집 코드는 `apps/blog/src/utils/telemetry/` 에 있고, `apps/blog/src/components/analytics/CwvTelemetry.tsx` 가 production 에서만 불러온다.

## 구성

| 서비스                          | 버전    | 하는 일                                                                                                                                                                                                          |
| ------------------------------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Caddy                           | 2.11.7  | TLS 를 붙이고 `/v1/traces`, `/v1/metrics` 만 collector 로 넘긴다. 다른 경로는 404, 256KB 를 넘는 본문은 413                                                                                                      |
| OpenTelemetry Collector contrib | 0.163.1 | CORS(`ALLOWED_ORIGIN`). trace 는 tail sampling(span 2개 이상, 판정 대기 15초) 뒤 Tempo 로, metric 은 받은 시각으로 바꾼 뒤 Mimir OTLP 엔드포인트로. Mimir 와 자기 지표 중 손실 수를 1분마다 긁어 Mimir 에 넣는다 |
| Grafana Tempo                   | 3.1.0   | trace 저장. 보존 기간 `TEMPO_RETENTION`, 조회 범위 `TEMPO_QUERY_RANGE`                                                                                                                                           |
| Grafana Mimir                   | 3.2.2   | metric 저장(테넌트 `blog`). delta 를 그대로 저장한다(native delta ingestion, 실험 기능). 보존 기간 `MIMIR_RETENTION`                                                                                             |
| Grafana                         | 13.2.4  | 대시보드 `CWV 운영 (Mimir)`, `CWV 원인 분석 (Tempo)`. 로그인이 필요하고 `127.0.0.1:3000` 에만 붙는다. 쓰지 않는 플러그인을 꺼서 200MiB 안쪽으로 돈다                                                             |

밖으로 열리는 포트는 Caddy 의 80, 443 뿐이다. Tempo 3.1 은 기본값으로 TraceQL metrics 를 24시간, trace 검색을 7일까지만 조회하므로 보존 기간을 늘릴 때는 조회 범위도 같이 늘린다.

## metric 경로가 lab 과 다른 이유

lab 은 회사 구성을 따라 collector 의 `delta_to_cumulative` 로 누적값을 만들어 remote write 로 쓴다. 방문이 드문 블로그에서는 이 경로가 값을 거의 세지 못했다. 라벨 조합마다 샘플이 하나뿐이라 `increase()` 가 0 을 돌려주고, 5분 넘게 조용한 시리즈는 지워진다(blog-experiments 의 E4, E8). 그래서 delta 를 그대로 Mimir 에 저장하고 대시보드는 `sum_over_time()` 으로 센다. 샘플의 시각은 collector 가 받은 시각으로 바꾼다. 브라우저 시계가 늦으면 순서가 뒤바뀐 샘플이 오류 없이 버려지기 때문이다(E9). Mimir 의 delta 저장은 실험 기능이라 Mimir 를 올릴 때 바뀐 점이 없는지 확인해야 한다.

이 경로에도 손실이 하나 남는다. 같은 시리즈의 값 두 개가 collector 에 같은 밀리초에 들어오면 같은 시각의 표본이 되어 Mimir 가 하나를 버린다(E10, 시리즈 하나에 초당 10건일 때 1.3~1.5%). 방문이 드문 블로그에서는 드물 것으로 보지만 재지는 않았다. 버린 수는 collector 가 Mimir 의 `cortex_discarded_samples_total` 을 긁어 와서 `CWV 운영 (Mimir)` 대시보드의 손실 패널에 보인다.

## 서버에 올리기

Docker Compose v2 가 깔린 Linux 서버가 필요하다. 메모리는 2GB 이상, 방화벽은 80, 443 을 열고 22 는 내 IP 로만 연다. `OTEL_DOMAIN` 의 A 레코드는 서버 IP 를 가리키게 하고, Cloudflare 라면 프록시를 끈다(Caddy 가 인증서를 직접 받는다).

```bash
git clone --filter=blob:none --sparse https://github.com/yceffort/blog.git
cd blog && git sparse-checkout set infra/otel
cd infra/otel
cp env.example .env          # 값을 채운다
docker compose up -d
```

바꾼 설정을 반영할 때는 `git pull && docker compose up -d` 를 한다. 이미지 버전은 Dependabot(docker-compose)이 주마다 PR 을 연다. Mimir 의 delta 저장은 실험 기능이라 Mimir 판은 릴리스 노트를 확인하고 합친다.

## Grafana 보기

```bash
ssh -L 3000:127.0.0.1:3000 <서버>    # 그 뒤 http://localhost:3000, 사용자 admin
```

## 대시보드 고치기

대시보드 JSON 은 `grafana/build-dashboards.py` 가 만든다. JSON 을 직접 고치지 않고 스크립트를 고친 뒤 `python3 grafana/build-dashboards.py` 로 다시 만든다. Tempo 대시보드의 화면 변수(`home`, `post`, `tag`, `series`, `other`)는 수집 코드가 붙이는 `page.route` 값과 맞아야 한다.
