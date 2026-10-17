"""grafana/dashboards 의 대시보드 JSON 두 개를 만든다. 사용: python3 grafana/build-dashboards.py

- cwv-ops.json (Mimir): 지금 얼마나 느린가, 나빠졌는가, 어디가 느린가
- cwv-why.json (Tempo): 왜 느린가(LCP 요소, 단계별 시간, 밀린 요소)
"""

import json
from pathlib import Path

OUT = Path(__file__).parent / "dashboards"
MIMIR = {"type": "prometheus", "uid": "mimir"}
TEMPO = {"type": "tempo", "uid": "tempo"}

# (이름, 시리즈 접두사, good 경계, poor 경계, 단위)
METRICS = [
    ("LCP", "cwv_lcp_milliseconds", 2500, 4000, "ms"),
    ("INP", "cwv_inp_milliseconds", 200, 500, "ms"),
    ("CLS", "cwv_cls", 0.1, 0.25, "none"),
    ("FCP", "cwv_fcp_milliseconds", 1800, 3000, "ms"),
    ("TTFB", "cwv_ttfb_milliseconds", 800, 1800, "ms"),
]
ROUTE = 'page_route=~"$route"'
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


def bargauge(title, expr, legend, unit, th):
    label = legend.strip("{}")
    return {
        "type": "bargauge",
        "title": title,
        "datasource": MIMIR,
        "targets": [prom(expr, legend, instant=True)],
        "fieldConfig": {"defaults": {"unit": unit, "thresholds": th, "color": {"mode": "thresholds"}, "displayName": f"${{__field.labels.{label}}}"}, "overrides": []},
        "options": {"orientation": "horizontal", "displayMode": "basic", "reduceOptions": {"calcs": ["lastNotNull"]}},
    }


def p75(series, window, by="le"):
    return f"histogram_quantile(0.75, sum by ({by}) (increase({series}_bucket{{{ROUTE}}}[{window}])))"


def good_ratio(series, good, window):
    return (
        f'sum(increase({series}_bucket{{{ROUTE},le="{good}"}}[{window}]))'
        f" / sum(increase({series}_count{{{ROUTE}}}[{window}]))"
    )


def ops_dashboard():
    lay = Layout()
    lay.row("지금 얼마나 느린가 (선택 구간 전체의 p75)")
    for (name, series, good, poor, unit), w in zip(METRICS, WIDTHS):
        lay.add(stat(f"{name} p75", p75(series, "$__range"), unit, thresholds(good, poor), decimals=3 if unit == "none" else None), w, 4)
    lay.row("good 비율 (good 경계 버킷 / 전체)")
    ratio_th = {"mode": "absolute", "steps": [{"color": "red", "value": None}, {"color": "orange", "value": 0.5}, {"color": "green", "value": 0.75}]}
    for (name, series, good, poor, unit), w in zip(METRICS, WIDTHS):
        lay.add(
            stat(f"{name} good 비율", good_ratio(series, good, "$__range"), "percentunit", ratio_th,
                 description=f"le=\"{good}\" 버킷이 good 경계와 정확히 같아서 보간 없이 나온다."),
            w, 4,
        )
    lay.row("나빠졌는가 (10분 창 p75 추이)")
    for name, series, good, poor, unit in METRICS:
        lay.add(timeseries(f"{name} p75 추이", [prom(p75(series, "10m"), "p75")], unit, MIMIR, thresholds(good, poor)), 12 if name in ("LCP", "INP") else 8, 7)
    lay.row("어디가 느린가")
    lay.add(bargauge("화면별 LCP p75", p75("cwv_lcp_milliseconds", "$__range", "le, page_route"), "{{page_route}}", "ms", thresholds(2500, 4000)), 8, 6)
    lay.add(bargauge("화면별 INP p75", p75("cwv_inp_milliseconds", "$__range", "le, page_route"), "{{page_route}}", "ms", thresholds(200, 500)), 8, 6)
    lay.add(bargauge("입력 종류별 INP p75", p75("cwv_inp_milliseconds", "$__range", "le, interaction_type"), "{{interaction_type}}", "ms", thresholds(200, 500)), 8, 6)
    lay.row("등급 분포와 표본 수")
    lay.add(
        timeseries("LCP 등급별 건수", [prom(f"sum by (rating) (increase(cwv_lcp_milliseconds_count{{{ROUTE}}}[$__interval]))", "{{rating}}")], "short", MIMIR, stack=True),
        12, 7,
    )
    lay.add(
        timeseries(
            "지표별 표본 수(하한)",
            [prom(f"sum(increase({series}_count{{{ROUTE}}}[$__interval]))", name) for name, series, *_ in METRICS],
            "short", MIMIR,
            description="collector 의 delta_to_cumulative 는 5분(만료 검사는 1분마다) 넘게 샘플이 없는 시리즈를 지운다. 다시 1부터 쌓인 값을 Prometheus 는 증가 0 으로 본다. 드문 라벨 조합일수록 이 값은 실제보다 작다. 페이지 로드 수로 읽지 않는다.",
        ),
        12, 7,
    )
    for i, t in enumerate(lay.panels[-1]["targets"]):
        t["refId"] = chr(ord("A") + i)
    return {
        "uid": "cwv-ops",
        "title": "CWV 운영 (Mimir)",
        "tags": ["cwv"],
        "timezone": "browser",
        "schemaVersion": 41,
        "time": {"from": "now-1h", "to": "now"},
        "refresh": "1m",
        "templating": {
            "list": [
                {
                    "name": "route",
                    "label": "화면",
                    "type": "query",
                    "datasource": MIMIR,
                    "query": {"query": "label_values(cwv_lcp_milliseconds_count, page_route)", "refId": "route"},
                    "definition": "label_values(cwv_lcp_milliseconds_count, page_route)",
                    "includeAll": True,
                    "allValue": ".*",
                    "multi": True,
                    "current": {"text": "All", "value": "$__all"},
                    "refresh": 2,
                }
            ]
        },
        "panels": lay.panels,
    }


FILTER = 'resource.service.name="yceffort-blog" && span.page.route=~"$route"'


def traceql(query, step="5m"):
    return {"datasource": TEMPO, "queryType": "traceql", "query": query, "metricsQueryType": "range", "step": step, "refId": "A"}


def tempo_ts(title, query, unit, description="", stack=False, th=None):
    return timeseries(title, [traceql(query)], unit, TEMPO, th=th, description=description, stack=stack)


def why_dashboard():
    lay = Layout()
    lay.row("무엇이 LCP 인가")
    lay.add(tempo_ts("LCP 요소별 건수", f'{{{FILTER} && name="cwv.lcp"}} | count_over_time() by (span.cwv.lcp.target)', "short", stack=True), 12, 8)
    lay.add(
        tempo_ts("LCP 요소별 p75", f'{{{FILTER} && name="cwv.lcp"}} | quantile_over_time(duration, .75) by (span.cwv.lcp.target)', "s",
                 description="span 길이가 곧 LCP 값이라 duration 분위수가 LCP 분위수다. Tempo 는 표본이 없는 구간의 분위수를 0 으로 돌려주므로 0 은 값이 아니라 빈 구간이다.", th=thresholds(2.5, 4)),
        12, 8,
    )
    lay.row("LCP 는 어느 단계에서 늦었나 (평균 ms)")
    note = "Tempo 데이터소스가 쿼리마다 frame refId 를 함수 이름으로 덮어써서 한 패널에 여러 쿼리를 넣으면 범례를 구분할 수 없다. 단계마다 패널을 나눴다."
    for label, attr in [("TTFB", "time_to_first_byte"), ("리소스 요청 지연", "resource_load_delay"), ("리소스 다운로드", "resource_load_duration"), ("렌더링 지연", "element_render_delay")]:
        lay.add(tempo_ts(label, f'{{{FILTER} && name="cwv.lcp"}} | avg_over_time(span.cwv.lcp.{attr})', "ms", description=note), 6, 6)
    lay.row("INP 는 어느 단계에서 늦었나 (평균 ms)")
    for label, attr in [("입력 지연", "input_delay"), ("처리 시간", "processing_duration"), ("다음 화면까지", "presentation_delay")]:
        lay.add(tempo_ts(label, f'{{{FILTER} && name="cwv.inp"}} | avg_over_time(span.cwv.inp.{attr})', "ms", description=note), 8, 6)
    lay.add(tempo_ts("INP 대상별 p75", f'{{{FILTER} && name="cwv.inp"}} | quantile_over_time(duration, .75) by (span.cwv.inp.target)', "s", th=thresholds(0.2, 0.5),
                     description="표본이 없는 구간은 0 으로 그려진다."), 12, 7)
    lay.add(tempo_ts("CLS 를 만든 요소", f'{{{FILTER} && name="cwv.cls" && duration > 0ms}} | count_over_time() by (span.cwv.cls.largest_shift_target)', "short", stack=True), 12, 7)
    lay.row("확정 방식과 느린 페이지 로드")
    lay.add(
        tempo_ts("LCP 확정 방식", f'{{{FILTER} && name="cwv.lcp"}} | count_over_time() by (span.cwv.lcp.finalized_by)', "short", stack=True,
                 description="web-vitals 는 첫 입력이나 hidden 에서 확정한다. quiet 는 후보가 3초 동안 바뀌지 않아 이 코드가 확정한 것이다."),
        8, 8,
    )
    lay.add(
        {
            "type": "table",
            "title": "느린 LCP (2.5초 초과)",
            "datasource": TEMPO,
            "targets": [{
                "datasource": TEMPO, "queryType": "traceql", "tableType": "spans", "limit": 50, "spss": 1, "refId": "A",
                "query": f'{{{FILTER} && name="cwv.lcp" && duration > 2.5s}} | select(span.cwv.lcp.target, span.cwv.lcp.element_render_delay, span.cwv.lcp.time_to_first_byte)',
            }],
            "fieldConfig": {"defaults": {}, "overrides": []},
            "transformations": [{"id": "organize", "options": {"excludeByName": {"traceService": True, "traceName": True, "service.name": True}}}],
            "options": {"showHeader": True},
        },
        16, 8,
    )
    return {
        "uid": "cwv-why",
        "title": "CWV 원인 분석 (Tempo)",
        "tags": ["cwv"],
        "timezone": "browser",
        "schemaVersion": 41,
        "time": {"from": "now-1h", "to": "now"},
        "refresh": "1m",
        "templating": {
            "list": [
                {
                    "name": "route",
                    "label": "화면",
                    "type": "custom",
                    "query": ".*,home,post,tag,series,other",
                    "current": {"text": ".*", "value": ".*"},
                    "options": [{"text": v, "value": v, "selected": v == ".*"} for v in [".*", "home", "post", "tag", "series", "other"]],
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
