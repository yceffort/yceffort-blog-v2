---
title: '타입스크립트 컴파일러는 어떻게 동작하는가?'
tags:
  - typescript
  - javascript
published: true
date: 2022-05-15 08:32:39
description: "그리고 이를 위협하는 swc..."
---

## Table of Contents

## Introduction

jQuery와 angular, react의 등장으로 프론트엔드 생태계에 많은 변화가 있었다고 한다면, 타입스크립트도 그에 못지 않은 영향력을 끼쳤다고 볼 수 있다. 타입스크립트의 등장 전후로 프론트엔드 개발, 특히 협업하는 데 있어서 큰 도움을 얻을 수 있었다. 

그런데 우리는 타입스크립트는 어떻게 동작할까? `tsc`라는 명령어 뒤에는 어떤 일이 벌어지고 있을까? 리처드 파인만이 말했던 것처럼, 스스로 만들어 보는 수준까지는 아니더라도, 타입스크립트 컴파일러가 동작하는 방식에 대해서 하나하나씩 뜯어보고, 직접 코드도 살펴보면서 이해해보고자 한다.

## 참고한 내용

주로 참고한 내용은 tsconf 2021에 있었던 키노트다. 

- [typescript repo](https://github.com/microsoft/TypeScript)
- [typescript-compiler-notes](https://github.com/microsoft/TypeScript-Compiler-Notes)
- [How the TypeScript Compiler Compiles - understanding the compiler internal](https://www.youtube.com/watch?v=X8k_4tZ16qU)
- [tsconf-slide-show](https://keybase.pub/orta/talks/tsconf-2021/)

## 대략적인 흐름

타입스크립트 컴파일러가 동작하는 방식, 즉 `tsc` 명령어를 눌렀을 때 일어나는 작업은 크게 아래와 같이 나눠볼 수 있다.

1. tsconfig 읽기: 타입스크립트 프로젝트라면, root에 `tsconifg.json`을 읽는 작업부터 시작할 것이다.
2. preprocess: 파일의 root 부터 시작해서 imports로 연결된 가능한 모든 파일을 찾는다.
3. tokenize & parse: `.ts`로 작성된 파일을 신택스 트리로 변경한다. 
4. binder: 3번에서 변경한 신택스 트리를 기준으로, 해당 트리에 있는 symbol (`const` 등) 을 identifier로 변경한다.
5. 타입체크: binder와 신택스 트리를 기준으로 타입을 체크한다.
6. transform: 신택스트리를 1번에서 읽었던 옵션에 맞게 변경한다.
7. emit: 신택스 트리를 `.js` `.d.ts`파일 등으로 변경한다.

- 3번까지의 과정이 소스코드를 읽어 데이터로 만드는 과정
- 4, 5가 타입체킹 과정
- 6, 7 을 파일을 만드는 과정이라 볼 수 있다.

## 소스코드를 데이터로 만들기

1번과 2번 과정을 제외하고, 가장 먼저 해야할 일은 코드를 신택스트리로 변경하는 일이다.

`index.ts`

```typescript
const message: string = "Hello, world!"
welcome(message)

function welcome(str: string) {
  console.log(str)
}
```

위와 같은 파일이 있다고 가정해보자. 일반적으로 자바스크립트 코드는 `;`, 줄바꿈, 내지는 `{}` 등으로 나눠서 이해할 수 있다. 여기에서는 세가지 구문으로 나눠 볼 수 있다.

- `const message: string = "Hello, world!"` 변수를 선언하는 구문
- `welcome(message)` 함수를 호출하는 구문
- `function ...{...}` 함수를 정의하는 구문

타입스크립트는 일단 이렇게 3가지 구문으로 나누어서 시작할 것이다. 

```typescript
const message: string = "Hello, world!"
```

위 코드를 또 자세히 보면, 각각을 다음과 같은 chunk 로 나눌 수 있다.

- `const`
- `message`
- `:`
- `string`
- `=`
- `"Hello, world!"`


