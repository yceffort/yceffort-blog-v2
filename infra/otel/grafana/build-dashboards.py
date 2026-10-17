"""grafana/dashboards 의 대시보드 JSON 두 개를 만든다. 사용: python3 grafana/build-dashboards.py

- cwv-ops.json (Mimir): 지금 얼마나 느린가, 나빠졌는가, 누구에게 느린가, 어디가 느린가, 얼마나 왔나
- cwv-why.json (Tempo): 왜 느린가(LCP 요소, 단계별 시간), 어느 글이 느린가, 유입과 캐시, 느린 페이지 로드
"""

import json
from pathlib import Path

OUT = Path(__file__).parent / "dashboards"
MIMIR = {"type": "prometheus", "uid": "mimir"}
TEMPO = {"type": "tempo", "uid": "tempo"}

# Mimir 는 브라우저가 보낸 delta 를 그대로 저장한다(native delta ingestion). 샘플 하나가 페이지 로드 하나의 값이라
# 구간 안의 샘플을 더하면(sum_over_time) 건수와 버킷이 나온다. increase() 는 쓰지 않는다.
# 다만 같은 시리즈의 값 두 개가 collector 에 같은 밀리초에 들어오면 Mimir 가 하나를 버린다(blog-experiments 의 E10).
# 버린 수는 '저장되지 못한 값' 패널에 나온다.
AGG = "sum_over_time"

# (이름, 시리즈 접두사, good 경계, poor 경계, 단위)
METRICS = [
    ("LCP", "cwv_lcp_milliseconds", 2500, 4000, "ms"),
    ("INP", "cwv_inp_milliseconds", 200, 500, "ms"),
    ("CLS", "cwv_cls", 0.1, 0.25, "none"),
    ("FCP", "cwv_fcp_milliseconds", 1800, 3000, "ms"),
    ("TTFB", "cwv_ttfb_milliseconds", 800, 1800, "ms"),
]
def selector(nav='navigation_type=~"$nav"'):
    return f'job="yceffort-blog", {nav}, page_route=~"$route", page_lang=~"$lang", device_class=~"$device", browser_engine=~"$engine"'


SEL = selector()
# 이동 변수. web-vitals 의 navigationType 값이다. SPA 이동은 Chromium 151 이상에서만 잡힌다.
NAV_OPTIONS = [
    ("첫 로드", "navigate|reload|back-forward|back-forward-cache|prerender|restore"),
    ("SPA 이동", "soft-navigation"),
    ("전체", ".*"),
]


def nav_variable():
    first_text, first_value = NAV_OPTIONS[0]
    return {
        "name": "nav",
        "label": "이동",
        "type": "custom",
        "query": ",".join(f"{text} : {value}" for text, value in NAV_OPTIONS),
        "current": {"text": first_text, "value": first_value},
        "options": [{"text": text, "value": value, "selected": i == 0} for i, (text, value) in enumerate(NAV_OPTIONS)],
    }
WIDTHS = [5, 5, 5, 5, 4]


def thresholds(good, poor):
    return {
        "mode": "absolute",
        "steps": [
            {"color": "green", "value": None},
            {"color": "orange", "value": good},
            {"color": "red", "value": poor},
        ],
    }


class Layout:
    """패널을 왼쪽에서 오른쪽으로 채우고 줄을 넘긴다."""

    def __init__(self):
        self.panels, self.x, self.y, self.row_h, self.next_id = [], 0, 0, 0, 1

    def row(self, title):
        self.newline()
        self.add({"type": "row", "title": title, "collapsed": False, "panels": []}, 24, 1)

    def newline(self):
        if self.x:
            self.x, self.y, self.row_h = 0, self.y + self.row_h, 0

    def add(self, panel, w, h):
        if self.x + w > 24:
            self.newline()
        panel["id"] = self.next_id
        panel["gridPos"] = {"x": self.x, "y": self.y, "w": w, "h": h}
        # 한 패널 안의 쿼리는 refId 가 달라야 한다. 같으면 Grafana 가 query.duplicateRefId 로 거부한다.
        for i, target in enumerate(panel.get("targets", [])):
            target["refId"] = chr(ord("A") + i)
        self.next_id += 1
        self.panels.append(panel)
        self.x += w
        self.row_h = max(self.row_h, h)


def prom(expr, legend="", instant=False):
    return {"datasource": MIMIR, "expr": expr, "legendFormat": legend, "instant": instant, "range": not instant, "refId": "A"}


def stat(title, expr, unit, th, description="", decimals=None):
    defaults = {"unit": unit, "thresholds": th, "color": {"mode": "thresholds"}}
    if decimals is not None:
        defaults["decimals"] = decimals
    return {
        "type": "stat",
        "title": title,
        "description": description,
        "datasource": MIMIR,
        "targets": [prom(expr, instant=True)],
        "fieldConfig": {"defaults": defaults, "overrides": []},
        "options": {"reduceOptions": {"calcs": ["lastNotNull"]}, "colorMode": "background", "graphMode": "none"},
    }


def timeseries(title, targets, unit, datasource, th=None, description="", stack=False):
    custom = {"lineWidth": 2, "fillOpacity": 10, "spanNulls": True}
    if stack:
        custom.update({"stacking": {"mode": "normal"}, "drawStyle": "bars", "fillOpacity": 80})
    if th:
        custom["thresholdsStyle"] = {"mode": "line+area"}
    defaults = {"unit": unit, "custom": custom}
    if th:
        defaults["thresholds"] = th
    return {
        "type": "timeseries",
        "title": title,
        "description": description,
        "datasource": datasource,
        "targets": targets,
        "fieldConfig": {"defaults": defaults, "overrides": []},
        "options": {"legend": {"displayMode": "list", "placement": "bottom"}, "tooltip": {"mode": "multi"}},
    }


def bargauge(title, expr, legend, unit, th, description=""):
    label = legend.strip("{}")
    return {
        "type": "bargauge",
        "title": title,
        "description": description,
        "datasource": MIMIR,
        "targets": [prom(expr, legend, instant=True)],
        "fieldConfig": {"defaults": {"unit": unit, "thresholds": th, "color": {"mode": "thresholds"}, "displayName": f"${{__field.labels.{label}}}"}, "overrides": []},
        "options": {"orientation": "horizontal", "displayMode": "basic", "reduceOptions": {"calcs": ["lastNotNull"]}},
    }


def p75(series, window, by="le"):
    return f"histogram_quantile(0.75, sum by ({by}) ({AGG}({series}_bucket{{{SEL}}}[{window}])))"


def good_ratio(series, good, window):
    return (
        f'sum({AGG}({series}_bucket{{{SEL},le="{good}"}}[{window}]))'
        f" / sum({AGG}({series}_count{{{SEL}}}[{window}]))"
    )


def label_variable(name, label, title):
    return {
        "name": name,
        "label": title,
        "type": "query",
        "datasource": MIMIR,
        "query": {"query": f'label_values(cwv_ttfb_milliseconds_count{{job="yceffort-blog"}}, {label})', "refId": name},
        "definition": f'label_values(cwv_ttfb_milliseconds_count{{job="yceffort-blog"}}, {label})',
        "includeAll": True,
        "allValue": ".*",
        "multi": True,
        "current": {"text": "All", "value": "$__all"},
        "refresh": 2,
    }


def ops_dashboard():
    lay = Layout()
    lay.row("지금 얼마나 느린가 (선택 구간 전체의 p75)")
    for (name, series, good, poor, unit), w in zip(METRICS, WIDTHS):
        first_hidden = "페이지가 처음 숨겨진 시점의 값이다. 다시 보였다가 숨겨지며 바뀐 값은 넣지 않는다(CWV 원인 분석 대시보드의 'CLS, INP 갱신' 패널)." if name in ("CLS", "INP") else ""
        lay.add(stat(f"{name} p75", p75(series, "$__range"), unit, thresholds(good, poor), description=first_hidden, decimals=3 if unit == "none" else None), w, 4)
    lay.row("good 비율 (good 경계 버킷 / 전체)")
    ratio_th = {"mode": "absolute", "steps": [{"color": "red", "value": None}, {"color": "orange", "value": 0.5}, {"color": "green", "value": 0.75}]}
    for (name, series, good, poor, unit), w in zip(METRICS, WIDTHS):
        lay.add(
            stat(f"{name} good 비율", good_ratio(series, good, "$__range"), "percentunit", ratio_th,
                 description=f"le=\"{good}\" 버킷이 good 경계와 정확히 같아서 보간 없이 나온다."),
            w, 4,
        )
    lay.row("나빠졌는가 (p75 추이)")
    trend = "방문이 드문 블로그라 짧은 창은 비어 있는 구간이 많다. 위쪽의 '추이 창' 으로 창 길이를 바꾼다."
    for name, series, good, poor, unit in METRICS:
        lay.add(
            timeseries(f"{name} p75 추이", [prom(p75(series, "$window"), "p75")], unit, MIMIR, thresholds(good, poor), description=trend),
            12 if name in ("LCP", "INP") else 8, 7,
        )
    lay.row("누구에게 느린가")
    for title, label in [("기기", "device_class"), ("브라우저 엔진", "browser_engine"), ("언어", "page_lang")]:
        lay.add(bargauge(f"{title}별 LCP p75", p75("cwv_lcp_milliseconds", "$__range", f"le, {label}"), f"{{{{{label}}}}}", "ms", thresholds(2500, 4000)), 8, 6)
    for title, label in [("기기", "device_class"), ("브라우저 엔진", "browser_engine"), ("언어", "page_lang")]:
        lay.add(
            bargauge(f"{title}별 INP p75", p75("cwv_inp_milliseconds", "$__range", f"le, {label}"), f"{{{{{label}}}}}", "ms", thresholds(200, 500),
                     description="엔진마다 지원하는 지표가 달라 엔진별 건수가 다를 수 있다." if label == "browser_engine" else ""),
            8, 6,
        )
    lay.row("어디가 느린가")
    lay.add(bargauge("화면별 LCP p75", p75("cwv_lcp_milliseconds", "$__range", "le, page_route"), "{{page_route}}", "ms", thresholds(2500, 4000)), 8, 6)
    lay.add(bargauge("화면별 INP p75", p75("cwv_inp_milliseconds", "$__range", "le, page_route"), "{{page_route}}", "ms", thresholds(200, 500)), 8, 6)
    lay.add(bargauge("입력 종류별 INP p75", p75("cwv_inp_milliseconds", "$__range", "le, interaction_type"), "{{interaction_type}}", "ms", thresholds(200, 500)), 8, 6)
    lay.row("얼마나 왔나")
    lay.add(
        timeseries(
            "화면별 페이지 로드",
            [prom(f"sum by (page_route) ({AGG}(cwv_ttfb_milliseconds_count{{{SEL}}}[$__interval]))", "{{page_route}}")],
            "short", MIMIR, stack=True,
            description="TTFB 는 페이지 로드마다 한 번 보고되므로 TTFB 건수를 페이지 로드 수로 쓴다. 블로그 안의 클라이언트 이동은 세지 않는다.",
        ),
        12, 7,
    )
    lay.add(
        timeseries(
            "화면별 SPA 이동",
            [prom(f"sum by (page_route) ({AGG}(cwv_fcp_milliseconds_count{{{selector('navigation_type=\"soft-navigation\"')}}}[$__interval]))", "{{page_route}}")],
            "short", MIMIR, stack=True,
            description="블로그 안에서 링크로 옮겨 간 횟수. SPA 이동 뒤 첫 contentful paint(FCP)가 보고된 수로 센다. Chromium 151 이상만 보고하고, 이동 변수와 상관없이 SPA 이동만 센다.",
        ),
        12, 7,
    )
    lay.add(
        timeseries("LCP 등급별 건수", [prom(f"sum by (rating) ({AGG}(cwv_lcp_milliseconds_count{{{SEL}}}[$__interval]))", "{{rating}}")], "short", MIMIR, stack=True),
        12, 7,
    )
    lay.add(
        timeseries(
            "지표별 표본 수",
            [prom(f"sum({AGG}({series}_count{{{SEL}}}[$__interval]))", name) for name, series, *_ in METRICS],
            "short", MIMIR,
            description="지표마다 보고 조건이 달라 건수가 다르다. INP 는 입력이 있어야, CLS 는 배치가 바뀌거나 페이지가 숨겨져야 나온다.",
        ),
        24, 7,
    )
    loss = ("위쪽 필터와 상관없다. collector 가 1분마다 Mimir 와 자기 지표를 긁어 넣는다. 버리는 일이 드물어 처음 생긴 시리즈의 첫 값을 "
            "increase() 가 세지 못하므로(blog-experiments 의 E8 과 같은 이유) 카운터의 최신값, 즉 지금 프로세스가 시작한 뒤의 누적값을 보여 준다. "
            "값은 그 대상을 3분 안에 정상으로 긁었을 때만 표시하고, 긁지 못했거나 3분 넘게 새 값이 없으면 '수집 안 됨' 이다.")

    def scraped(job):
        # 값을 믿어도 되는 상태: 마지막 수집이 성공했고(up == 1) 3분(수집 주기의 3배) 안이다.
        # 이 조건이 아니면 마지막으로 긁은 0 도 숨긴다. 조회 범위(5분) 안에 남은 옛 값이 정상처럼 보이지 않게 한다.
        return f'max((up{{job="{job}"}} == 1) and (time() - timestamp(up{{job="{job}"}}) < 180))'

    red_from_one = {"mode": "absolute", "steps": [{"color": "green", "value": None}, {"color": "red", "value": 1}]}
    discarded = bargauge(
        "Mimir 가 버린 표본 (Mimir 시작 뒤 누적)",
        f'(max by (reason) (cortex_discarded_samples_total{{job="mimir", user="blog"}}) or on() label_replace(vector(0), "reason", "없음", "", "")) and on() {scraped("mimir")}',
        "{{reason}}", "short", red_from_one,
        description="표본은 시리즈 하나의 값이다. 페이지 로드 하나의 지표 값은 버킷과 count, sum 을 합쳐 표본 11~13개다. sample_duplicate_timestamp 는 같은 시리즈의 값이 같은 밀리초에 두 번 들어와 하나를 버린 것이다. " + loss,
    )
    failed = stat(
        "collector 가 받지 못하거나 내보내지 못한 수 (collector 시작 뒤 누적)",
        f'(sum({{__name__=~"otelcol_(receiver_refused|exporter_send_failed)_(metric_points|spans)_total", job="collector"}}) or on() vector(0)) and on() {scraped("collector")}',
        "short", red_from_one,
        description="metric 데이터 포인트와 span 의 합이다. " + loss,
    )
    for panel in (discarded, failed):
        panel["fieldConfig"]["defaults"]["noValue"] = "수집 안 됨"
        lay.add(panel, 12, 6)
    return {
        "uid": "cwv-ops",
        "title": "CWV 운영 (Mimir)",
        "tags": ["cwv"],
        "timezone": "browser",
        "schemaVersion": 41,
        "time": {"from": "now-24h", "to": "now"},
        "refresh": "5m",
        "templating": {
            "list": [
                nav_variable(),
                label_variable("route", "page_route", "화면"),
                label_variable("lang", "page_lang", "언어"),
                label_variable("device", "device_class", "기기"),
                label_variable("engine", "browser_engine", "엔진"),
                {
                    "name": "window",
                    "label": "추이 창",
                    "type": "custom",
                    "query": "1h,6h,1d",
                    "current": {"text": "1h", "value": "1h"},
                    "options": [{"text": v, "value": v, "selected": v == "1h"} for v in ["1h", "6h", "1d"]],
                },
            ]
        },
        "panels": lay.panels,
    }


FILTER = 'resource.service.name="yceffort-blog" && span.cwv.navigation_type=~"$nav" && span.page.route=~"$route"'


def traceql(query, step="5m", instant=False):
    return {
        "datasource": TEMPO, "queryType": "traceql", "query": query,
        "metricsQueryType": "instant" if instant else "range", "step": step, "refId": "A",
    }


def tempo_ts(title, query, unit, description="", stack=False, th=None):
    return timeseries(title, [traceql(query)], unit, TEMPO, th=th, description=description, stack=stack)


def tempo_bar(title, query, attr, unit, th, description=""):
    """TraceQL instant 조회 결과를 막대로. 범례는 `{span.x="..."}` 꼴이라 값만 남긴다."""
    return {
        "type": "bargauge",
        "title": title,
        "description": description,
        "datasource": TEMPO,
        "targets": [traceql(query, instant=True)],
        "fieldConfig": {"defaults": {"unit": unit, "thresholds": th, "color": {"mode": "thresholds"}}, "overrides": []},
        "options": {"orientation": "horizontal", "displayMode": "basic", "reduceOptions": {"calcs": ["lastNotNull"]}, "valueMode": "color"},
        "transformations": [{"id": "renameByRegex", "options": {"regex": f'.*{attr}="([^"]*)".*', "renamePattern": "$1"}}],
    }


def why_dashboard():
    lay = Layout()
    zero = "Tempo 는 표본이 없는 구간의 분위수를 0 으로 돌려주므로 0 은 값이 아니라 빈 구간이다."
    lay.row("무엇이 LCP 인가")
    lay.add(tempo_ts("LCP 요소별 건수", f'{{{FILTER} && name="cwv.lcp"}} | count_over_time() by (span.cwv.lcp.target)', "short", stack=True), 12, 8)
    lay.add(
        tempo_ts("LCP 요소별 p75", f'{{{FILTER} && name="cwv.lcp"}} | quantile_over_time(duration, .75) by (span.cwv.lcp.target)', "s",
                 description=f"span 길이가 곧 LCP 값이라 duration 분위수가 LCP 분위수다. {zero}", th=thresholds(2.5, 4)),
        12, 8,
    )
    lay.row("어느 글이 느린가 (선택 구간 전체)")
    post = f'{{{FILTER} && span.page.route="post" && name="cwv.lcp"}}'
    lay.add(tempo_bar("글별 LCP p75 (느린 10개)", f"{post} | quantile_over_time(duration, .75) by (span.page.path) | topk(10)", "span.page.path", "s", thresholds(2.5, 4),
                      description="Tempo 의 분위수는 2 의 거듭제곱 버킷 사이를 보간한 근사값이다. 표본이 적은 글은 값이 크게 흔들린다."), 12, 9)
    lay.add(tempo_bar("글별 페이지 로드 (많은 10개)", f"{post} | count_over_time() by (span.page.path) | topk(10)", "span.page.path", "short",
                      {"mode": "absolute", "steps": [{"color": "blue", "value": None}]}), 12, 9)
    lay.row("유입과 캐시")
    lay.add(tempo_bar("유입 경로별 LCP p75", f'{{{FILTER} && name="cwv.lcp"}} | quantile_over_time(duration, .75) by (span.page.referrer_type)', "span.page.referrer_type", "s", thresholds(2.5, 4)), 8, 7)
    lay.add(tempo_bar("서비스 워커 경유별 TTFB p75", f'{{{FILTER} && name="cwv.ttfb"}} | quantile_over_time(duration, .75) by (span.page.sw_controlled)', "span.page.sw_controlled", "s", thresholds(0.8, 1.8),
                      description="yes 는 내비게이션 요청 자체가 서비스 워커를 거친 페이지 로드다(workerStart > 0)."), 8, 7)
    lay.add(tempo_bar("LCP 종류별 p75", f'{{{FILTER} && name="cwv.lcp"}} | quantile_over_time(duration, .75) by (span.cwv.lcp.kind)', "span.cwv.lcp.kind", "s", thresholds(2.5, 4),
                      description="image 는 LCP 요소에 리소스 URL 이 있는 경우(이미지, 배경 이미지), text 는 글자 블록이다."), 8, 7)
    lay.row("LCP 는 어느 단계에서 늦었나 (평균 ms)")
    note = "Tempo 데이터소스가 쿼리마다 frame refId 를 함수 이름으로 덮어써서 한 패널에 여러 쿼리를 넣으면 범례를 구분할 수 없다. 단계마다 패널을 나눴다."
    for label, attr in [("TTFB", "time_to_first_byte"), ("리소스 요청 지연", "resource_load_delay"), ("리소스 다운로드", "resource_load_duration"), ("렌더링 지연", "element_render_delay")]:
        lay.add(tempo_ts(label, f'{{{FILTER} && name="cwv.lcp"}} | avg_over_time(span.cwv.lcp.{attr})', "ms", description=note), 6, 6)
    lay.row("INP 는 어느 단계에서 늦었나 (평균 ms)")
    for label, attr in [("입력 지연", "input_delay"), ("처리 시간", "processing_duration"), ("다음 화면까지", "presentation_delay")]:
        lay.add(tempo_ts(label, f'{{{FILTER} && name="cwv.inp"}} | avg_over_time(span.cwv.inp.{attr})', "ms", description=note), 8, 6)
    lay.add(tempo_ts("INP 대상별 p75", f'{{{FILTER} && name="cwv.inp"}} | quantile_over_time(duration, .75) by (span.cwv.inp.target)', "s", th=thresholds(0.2, 0.5), description=zero), 12, 7)
    lay.add(tempo_ts("CLS 를 만든 요소", f'{{{FILTER} && name="cwv.cls" && duration > 0ms}} | count_over_time() by (span.cwv.cls.largest_shift_target)', "short", stack=True), 12, 7)
    lay.row("확정 방식과 느린 페이지 로드")
    lay.add(
        tempo_ts("quiet 3초 값과 정식 LCP", f'{{{FILTER} && name="cwv.lcp"}} | count_over_time() by (span.cwv.lcp.quiet)', "short", stack=True,
                 description="LCP 는 web-vitals 가 첫 입력이나 hidden 에서 확정한 값만 기록한다. 후보가 처음으로 3초 동안 바뀌지 않았던 때의 값을 비교용으로 남긴다. lower 는 그 값을 최종값으로 썼다면 LCP 를 작게 기록했을 페이지 로드, same 은 같은 값, none 은 3초 멈춘 적 없이 확정된 것이다."),
        8, 8,
    )
    lay.add(
        tempo_ts("CLS, INP 갱신", f'{{{FILTER} && name=~"cwv.(cls|inp).update"}} | count_over_time() by (name, span.cwv.update.rating_changed)', "short", stack=True,
                 description="같은 id 로 다시 보고된 CLS, INP. Mimir 에는 처음 숨겨진 시점의 값만 넣고 이 갱신은 넣지 않는다. rating_changed 가 true 면 등급이 바뀐 갱신이다."),
        8, 8,
    )
    lay.add(
        {
            "type": "table",
            "title": "느린 LCP (2.5초 초과)",
            "datasource": TEMPO,
            "targets": [{
                "datasource": TEMPO, "queryType": "traceql", "tableType": "spans", "limit": 50, "spss": 1, "refId": "A",
                "query": f'{{{FILTER} && name="cwv.lcp" && duration > 2.5s}} | select(span.page.path, span.cwv.lcp.target, span.cwv.lcp.kind, span.device.class, span.browser.engine, span.network.effective_type, span.cwv.lcp.time_to_first_byte, span.cwv.lcp.element_render_delay)',
            }],
            "fieldConfig": {"defaults": {}, "overrides": []},
            "transformations": [{"id": "organize", "options": {"excludeByName": {"traceService": True, "traceName": True, "service.name": True}}}],
            "options": {"showHeader": True},
        },
        16, 8,
    )
    routes = [".*", "home", "post", "tag", "series", "other"]
    return {
        "uid": "cwv-why",
        "title": "CWV 원인 분석 (Tempo)",
        "tags": ["cwv"],
        "timezone": "browser",
        "schemaVersion": 41,
        "time": {"from": "now-24h", "to": "now"},
        "refresh": "5m",
        "templating": {
            "list": [
                nav_variable(),
                {
                    "name": "route",
                    "label": "화면",
                    "type": "custom",
                    "query": ",".join(routes),
                    "current": {"text": ".*", "value": ".*"},
                    "options": [{"text": v, "value": v, "selected": v == ".*"} for v in routes],
                }
            ]
        },
        "panels": lay.panels,
    }


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for dash in (ops_dashboard(), why_dashboard()):
        (OUT / f"{dash['uid']}.json").write_text(json.dumps(dash, ensure_ascii=False, indent=2) + "\n")
        print(dash["uid"], len(dash["panels"]), "panels")
