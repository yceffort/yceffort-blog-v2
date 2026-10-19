---
name: '코딩 에이전트 모델 교체 실험'
title: '코딩 에이전트의 <em>모델 교체</em>를 판단하기'
description: 'SWE-bench의 문항과 채점 방식을 살펴보고, Claude Code에서 모델을 비교할 실험을 설계한 뒤 해결률과 비용을 바탕으로 교체 여부를 판단한다.'
art:
  undraw: cli-coding-agent
  hue: violet
  tone: light
---

Claude Code에서 쓰는 모델을 오픈웨이트 모델로 바꿔도 될지 알아보기 위한 기록이다. 1편에서는 SWE-bench의 문항과 채점 방식을 분석하고 비교 원칙을 정리한다. 2편에서는 실제 실행 환경과 실험 조건을 구성하고, 3편에서는 해결률과 비용, 실패 사례를 바탕으로 모델 교체 여부를 판단할 예정이다.
