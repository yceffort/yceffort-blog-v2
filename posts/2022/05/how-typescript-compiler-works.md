---
title: '타입스크립트 컴파일러는 어떻게 동작하는가?'
tags:
  - typescript
  - javascript
published: true
date: 2022-05-15 08:32:39
description: '그리고 이를 위협하는 swc...'
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
const message: string = 'Hello, world!'
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
const message: string = 'Hello, world!'
```

위 코드를 또 자세히 보면, 각각을 다음과 같은 chunk 로 나눌 수 있다.

- `const`
- `message`
- `:`
- `string`
- `=`
- `"Hello, world!"`

이런식으로 일반적인 코드 문자열을 데이터로 만드는 과정이 바로 신택스 트리를 생성하는 과정이라 볼 수 있다. 그리고 이렇게 만들어진 트리가 [abstract syntax tree, 즉 추상구문트리](https://ko.wikipedia.org/wiki/%EC%B6%94%EC%83%81_%EA%B5%AC%EB%AC%B8_%ED%8A%B8%EB%A6%AC)라 불리우는 것이다.

그리고 이 신택스 트리를 만들기 위해서 필요한 것이 `scanner`와 `parser`다.

### scanner

[https://github.com/microsoft/TypeScript/blob/main/src/compiler/scanner.ts](https://github.com/microsoft/TypeScript/blob/main/src/compiler/scanner.ts): 이 코드를 잘 살펴보면, 코드 문자열을 읽기 위한 사전작업, 예를 들어 예약어 (`abstract`, `case` 등)를 읽어들이거나 `{}`와 같은 토큰을 분석하기 위한 작업들이 준비되어 있는 것을 볼 수 있다. (이 스캐너는 무려 26,000줄의 단일파일로 구성되어 있는데, 이제 앞으로 살펴볼 파일들 대비 귀여운(?)편에 속한다.) 이 스캐너의 역할은 일반적인 코드 문자열을 토큰으로 변환하는 것이다. 위의 토큰은 아래와 같이 변환된다.

- `Const Keyword`
- `WhitespaceTrivia`
- `Identifier`
- `ColonToken`
- `WhitespaceTrivia`
- `StringKeyword`
- `WhitespaceTrivia`
- `EqualToken`
- `WhitespaceTrivia`
- `StringLiteral`

[tsplayground 에서 확인해보기](https://www.typescriptlang.org/pt/play?#code/MYewdgzgLgBAtgUwhAhgcwQLhtATgSzDRgF4YAiACQQBsaQAaGAdxFxoBMBCcgWACggA)

> 우측 사이드바에 scanner가 뜨지 않는다면 plugins에서 scanner

> 참고로 실제 타입스크립트에서 동작하는 것과 약간의 차이가 있다.

이과정은 굉장히 선형적으로 단순하게 이루어진다. 즉 파일을 처음부터 주욱 읽어 가면서, 특정 키워드내지는 예약어가 있는지, `identifier`가 있는지, 등을 순차적으로 확인한다.

스캐너는 이 과정에서 코드 문자열의 정합성도 검사한다. 예를 들어 다음과 같은 것들이 있다.

```ts
let noEnd = " // Unterminated string literal.(1002)
let num = 2__3  // Multiple consecutive numeric separators are not permitted.(6189)
const 🤔 = 'hello' // Invalid character.(1127)
let x1 =  1} // Declaration or statement expected.(1128)
```

### parser

[https://github.com/microsoft/TypeScript/blob/main/src/compiler/parser.ts](https://github.com/microsoft/TypeScript/blob/main/src/compiler/parser.ts) `parser`도 비교적 적은 양의 코드인 9,000줄로 구성되어 있다. 이 파서의 역할은, 스캐너가 읽어들인 token을 기준으로 트리를 만드는 것이다.

앞서 언급했던 토큰들은, parser에 의해 아래와 같은 트리로 만들어 진다.

![ts-ast](./images/ts-ast.png)

> https://ts-ast-viewer.com/#code/MYewdgzgLgBAtgUwhAhgcwQLhtATgSzDRgF4YByACQQBsaQAaGAdxFxoBMBCcgKF6A

```
AST
SourceFile
    pos: 0
    end: 43
    flags: 0
    modifierFlagsCache: 0
    transformFlags: 2229249
    kind: 303 (SyntaxKind.SourceFile)
    statements: [
    FirstStatement
    ]
    endOfFileToken: EndOfFileToken
    fileName: /input.tsx
    text: const message: string = 'Hello, world!'
    languageVersion: 4
    languageVariant: 1
    scriptKind: 4
    isDeclarationFile: false
    hasNoDefaultLib: false
    externalModuleIndicator: undefined
    bindDiagnostics:
    bindSuggestionDiagnostics: undefined
    pragmas: [object Map]
    checkJsDirective: undefined
    referencedFiles:
    typeReferenceDirectives:
    libReferenceDirectives:
    amdDependencies:
    commentDirectives: undefined
    nodeCount: 8
    identifierCount: 1
    identifiers: [object Map]
    parseDiagnostics:
    path: /input.tsx
    resolvedPath: /input.tsx
    originalFileName: /input.tsx
    impliedNodeFormat: undefined
    imports:
    moduleAugmentations:
    ambientModuleNames:
    resolvedModules: undefined
    locals: [object Map]
    endFlowNode: [object Object]
    symbolCount: 1
    classifiableNames: [object Set]
    id: 58041
```

> 위와 같은 내용은 typescript playground > settings > AST Viewer를 누르면 확인해볼 수 있다.

내용을 잘 살펴보면, 앞서 scanner 가 만들었던 토큰을 기준으로 다음과 같은 ast 트리를 만들어 낸 것을 알 수 있다.

- `VariableStatement`: `const`를 시작으로 한 변수 선언 구문을 의미한다.
- `VariableDeclarationList`: 여기에서 선언된 변수 배열을 나타낸다.

> 왜 배열이냐하면, `let a, b, c = 3` 와 같이 여러변수를 한구문에서 선언할 수있기 때문이다.

- `VariableDeclaration`: `message` 선언부를 의미한다.
- `Identifier`: `message`
- `StringKeyword`: `string` 타입 선언부
- `StringLiteral`: `Hello, world!'`

이러한 과정을 거쳐, parser는 scanner가 만들어준 token을 기준으로 신택스 트리를 만들게 된다.

parser에서는, 다음과 같은 내용을 분석하여 에러가 있는지 살펴보고 있다면 에러를 던진다.

```ts
#var = 123 // The left-hand side of an assignment expression must be a variable or a property access.(2364)
const decimal = 4.1n // A bigint literal must be an integer.(1353)
var extends = 123 // 'extends' is not allowed as a variable declaration name.(1389)
var x = { class C4 {} } // ':' expected.(1005)
```

parser가 분석하는 내용은 일반적으로 자바스크립트 구문이 올바른 위치에 있는지 여부를 확인한다고 보면 된다.

## 타입 검사

앞선 과정은 자바스크립트 컴파일러에도 존재하는 과정이었다면, 타입스크립트만의 특별한 과정인 타입검사가 다음으로 존재한다.
