---
title: 'nodejs의 buffer 이해하기'
tags:
  - javascript
  - nodejs
published: true
date: 2021-10-15 23:48:00
description: 'nodejs도 본격적으로 해보고 싶네여'
---

Nodejs에서 buffer는 raw 2진 데이터를 저장할 수 있는 특수한 유형의 객체다. 버퍼는 일반적으로 컴포터에 할당된 메모리 청크 - 일반적으로 RAM -을 나타낸다. 일단 버퍼크기를 설정하게 되면, 이후에는 변경할 수 없다.

버퍼는 바이트를 저장하는 단위라고 볼 수 있다. 그리고 바이트는 8비트 순서로 이루어져있다. 비트는 컴퓨터의 가장 기본적인 저장 단위이며, 0 또는 1로 이루어져 있다.

Nodejs는 버퍼클래스를 전역 스코프에 expose 하므로