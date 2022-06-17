---
title: 'JSON.stringify 만들어보기'
tags:
  - javascript
  - typescript
published: true
date: 2022-06-17 12:19:04
description: '갑자기 무슨 뻘짓'
---

## Table of Contents

## JSON이 지원하는 타입

JSON 무려 [공식 홈페이지](https://www.json.org/json-en.html)가 존재하는데, 여기서 어떤 데이터 타입을 지원하는지 나와있다. JSON은 우리가 매일 쓰고 또 그다지 어렵지 않기 때문에 그렇게 복잡하게 생각해본적이 없는데, 공식 문서의 그래프를 보면 살짝 어지러워진다. 이래저래 읽는게 귀찮고 복잡하므로, 타입스크립트로 간단하게 요약해보자면 다음과 같다.

```typescript
type JSONType =
  | null
  | boolean
  | number
  | string
  | JSONType[]
  | { [key: string]: JSONType }
```
