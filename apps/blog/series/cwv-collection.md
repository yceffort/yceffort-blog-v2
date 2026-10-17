---
name: 'Core Web Vitals 직접 수집하기'
title: '<em>Core Web Vitals</em> 직접 수집하기'
description: 'web-vitals로 잰 CWV를 OpenTelemetry로 trace와 metric에 나눠 모으면서, 값이 브라우저에서 Grafana 패널까지 가는 동안 오류 없이 사라진 지점을 재현 환경에서 다시 재고 기록한 시리즈'
art:
  undraw: browser-stats
  hue: blue
  tone: light
---

CWV를 직접 수집할 때 어려운 것은 측정보다 측정한 값이 대시보드까지 가는 길이었다. web-vitals가 계산한 값은 브라우저 안에서, 페이지를 떠나는 순간에, collector에서, 저장소에서 조회할 때 사라졌고, 그 어느 지점에서도 오류는 나지 않았다. 이 시리즈는 그 지점들을 하나씩 찾아 공개 재현 환경에서 다시 잰 기록이다.

같은 질문을 두 환경에서 풀었다. 하나는 앱 웹뷰에서 열리는 화면의 값을 여러 팀이 같이 쓰는 사내 관측 인프라로 보내는 경우로, 떠날 때 신호가 오지 않는 웹뷰와 설정을 바꿀 수 없는 collector가 제약이었다. 다른 하나는 같은 수집을 이 블로그에 올리고 수집 서버까지 직접 운영하는 경우로, 모든 설정을 바꿀 수 있는 대신 방문이 드물다는 점이 문제가 됐다.

측정 코드와 실험은 [blog-experiments/cwv-otel-grafana](https://github.com/yceffort/blog-experiments/tree/main/cwv-otel-grafana)에 있고, 각 편의 수치를 어느 버전에서 쟀는지는 글과 실험별 README에 적었다.
