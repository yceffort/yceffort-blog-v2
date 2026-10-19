---
name: 'number-flow 개선기'
title: '<em>number-flow</em> 개선기'
description: 'number-flow를 포크해 구형 브라우저를 지원하고, 모던 브라우저의 애니메이션 성능을 개선한 과정을 기록한다.'
art:
  undraw: going-upwards
  hue: cyan
  tone: light
---

[number-flow](https://github.com/barvian/number-flow)의 숫자 애니메이션을 구형 브라우저에서도 보여주려고 [포크](https://github.com/yceffort/number-flow)를 만들었다. 첫 글에서는 원본의 움직임을 유지하는 rAF 폴백 엔진을 다뤘다. 두 번째 글에서는 모던 브라우저에서도 남아 있던 스타일 재계산을 줄이기 위해 애니메이션을 합성 스레드에서 실행할 수 있는 형태로 바꾼 과정을 다룬다.
