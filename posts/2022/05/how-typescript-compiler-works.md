---
title: '타입스크립트 컴파일러는 어떻게 동작하는가?'
tags:
  - typescript
  - javascript
published: true
date: 2022-05-15 08:32:39
description: 'clone 받아서 읽어보세여 재밌어여 (안재밌음)'
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

### binder

[https://github.com/microsoft/TypeScript/blob/main/src/compiler/binder.ts](https://github.com/microsoft/TypeScript/blob/main/src/compiler/binder.ts)

바인더는 전체 파일(전체 신택스 트리)를 읽어서 타입 검사에 필요한 데이터를 수집하는 과정이라고 볼 수 있다. 전체를 읽어 드린다는 말에서 느낌이 오는 것 처럼, 이 과정은 꽤나 무거운 작업으로 볼 수 있다. 이 과정을 통해서 메타데이터를 수집하고, 타입분석에 필요한 계층 구조등을 만든다.

```typescript
const message: string = 'Hello, world!'
welcome(message)

function welcome(str: string) {
  console.log(str)
}
```

위 파일을 다시 살펴보면 크게 global scope와 function scope 두가지로 나눠져 있는 것을 볼 수 있다.

- global scope
  - `message`
  - `welcome`
- function scope (welcome)
  - `str`

바인더는 이를 순회하면서 어디에, 그리고 어떤 `identifier`가 있는지 확인한다. 자세한 과정을 아래를 통해서 확인해보자.

- `message`가 그 첫번째 `identifier`로, 0번째 식별자로 설정해두고, `const`이기 때문에 `BlockScopedVariable`로 기억해둔다.
- 그 다음엔 `welcome`을 찾을 수 있다. 그러나 아직 이 식별자는 선언되지 않았으므로, 지나간다.
- 인수로 선언되어 있는 `message`도 현재는 그 쓰임새를 알 수 없으므로 지나간다.
- `welcome`을 드디어 찾았다. 이제 `welcome`을 만날 때 마다 무엇을 실행해야하는지 알 수 있게 되었다. 그리고 `welcome`을 `Function`으로 기억해둦다.
  - `welcome`은 함수 스코프로, `parent`인 global scope를 등록한다.
  - `str`은 함수의 인수로, `BlockScopeVariable` 로 등록한다.

이렇게 등록해둔 내용, 이른바 symbol은 향후 스코프에서 이 `identifier`를 만날때 무엇인지 판단할 때 쓸 수 있는 테이블에 등록한다.

![symbol](./images/symbol.png)

또 한가지 binder에서 알아두어야 할 것은 `flw nodes`라는 개념이다.

```ts
// string, number
function log(x: string | number) {
  // string number
  if (typeof x === 'string') {
    // string
    // (1)
    return x
  } else {
    // number
    return x + 1
  }
  // 사실 여기는 unreachable
  // string number
  return x
}
```

위 코드를 보면, `x`라고 하는 변수의 타입이 각각 무엇이 될 수 있는지 머릿속에 흐름을 그려볼 수 있을 것이다. 타입스크립트에서 이러한 기능이 가능한 것은, 기본적으로 타입스크립트는 이러한 타입의 흐름을 추적하고 있기 때문인데, 이에 추가로 타입스크립트는 앞서 언급했던 스코프 내에서의 변수의 타입 변화도 추적한다.

`typeof x === 'string'`과 같은 구문을 flow condition, 그리고 이러한 플로우를 추적하는 스코프를 `flow container`라고 한다. `flow condition`을 기점으로 두개의 `flow container`가 두개 생긴것을 알 수 있다. 그리고 이러한 컨테이너 내부에서 해당 노드 (변수, identifier)가 어떤 타입인지 기억한다. 이를 바탕으로 코드가 내부에서 어떤 흐름으로 작동하는지 판단할 수 있게 된다.

이러한 flow는 타입스크립트가 해당 변수가 어떤 타입인지 추적할 수 있게 도와주는데, 이러한 추적은 밑에서 위로 올라가는 방식으로 진행된다. `(1)`에서 시작한다고 가정해보자. 해당 위치에서, 타입스크립트는 `x`의 타입이 무엇인지 flow node를 통해 물어보게되고, 가장 먼저 만나는 `flow condition`을 통해 `x`는 `string`임을 알게 된다. 이처럼, 해당 변수의 타입을 알기 위해서 `flow container` 내부에서 `flow condition`을 만나는지, 혹은 `flow container`의 시작지점에서 어떻게 선언되어있는지를 확인하는 bottom-to-top 방식으로 확인한다.

바인더도 여타 다른 과정과 마찬가지로 코드를 검사하는 과정을 거친다. binder에서 확인할 수 있는 것들은 다음과 같다.

```ts
const a = 123
delete a // 'delete' cannot be called on an identifier in strict mode.(1102)

const abc = 123
const abc = 123 // Cannot redeclare block-scoped variable 'abc'.(2451)

yield // Identifier expected. 'yield' is a reserved word in strict mode.

class A {} // duplicate identifier 'A;
type A {}
```

이처럼 바인더는 전체를 읽어 드리는 과정에서 전체적인 context를 이해하였으므로, 이러한 전체 신택스 트리를 기준으로 잡아낼 수 있는 문제점을 지적할 수 있다. 예를 들어 `strict mode`에 대한 검사나, 스코프 내에서 중복된 `identifier` 등을 잡아낼 수 있다.

### checker

[https://github.com/microsoft/TypeScript/blob/main/src/compiler/checker.ts](https://github.com/microsoft/TypeScript/blob/main/src/compiler/checker.ts) 는 이름에서도 알 수 있는 것 처럼 실제 타입을 체크하는 파일이다. 타입스크립트의 꽃이라고 볼 수 있으며, github의 file을 보면 알 수 있지만 2.67mb의 위용을 자랑한다. 대략 42000줄의 코드가 포함되어 있으며, 여기에 우리가 상상할 수 있는 흥미로운 것들이 많이 존재한다. (왜 `unknown`이 `any`보다 나은지, 타입스크립트의 구조적 타이핑은 무엇인지 등등..)

> 이렇게 하나의 파일에 크게 모두 담아 둔 이유는, 파일을 나눠서 관리하는 것 보다 하나의 파일에서 관리하는 것이 속도 측면에서 훨씬 좋기 때문이다. 특히 `checker`의 경우, 기존에는 100개가 넘는 import 가 존재하였는데, 이것이 속도에 있어 많은 걸림돌이 되었다고 한다.

> 참고자료
>
> - https://github.com/microsoft/TypeScript/issues/27891#issuecomment-530535972
> - https://twitter.com/orta/status/1178805954869125125
> - https://twitter.com/SeaRyanC/status/1178848975656345601

여기에 있는 내용을 모두 다 다루기 위해서는, 코드의 길이만큼의 설명이 필요하기 때문에 개괄적인 내용에 대해서만 다루고자한다.

`checker` 라는 이름에서 알수 있듯이, 대다수의 타입스크립트 validation이 여기에서 이루어진다..

![ts-diagnostics](./images/ts-diagnostics.png)

> https://orta.keybase.pub/talks/tsconf-2021/long-tsconf-2021.pdf?dl=1

여기에서 중점적으로 다루고자 하는 것은, 어떻게 체크를 하는지, 그리고 어떻게 타입을 비교하는지, 그리고 추론 시스템은 어떻게 구성되어 있는 지 등 총 3가지에 대해 이야기 해보고자 한다.

```ts
const message: string = 'Hello, world'
```

거의 대부분의 신택스 트리에는, 이에 맞는 checker 함수가 있다고 보면 된다. 타입스크립트는 이 신택스 트리를 순회하면서 대부분의 객체들을 체크 하게 된다.

- `VariableStatement`
  - `VariableDeclarationList`
  - `VariableDeclaration`
    - `Identifier`
    - `StringKeyword`
    - `StringLiteral`

```
checker.checkSourceElementWorker
checker.checkVariableStatement
checker.checkGrammarVariableDeclarationList
checker.checkVariableDeclaration
checker.checkVariableLikeDeclaration
checker.checkTypeAssignableToAndOptionallyElaborate
checker.isTypeRelatedTo
```

이렇듯 신택스 트리를 순차적으로 순회하면서, 체크 가능한 모든 것들을 체크하면서 validation을 진행하게 된다.

앞서, `string = "Hello, world"` 구문은, 다음의 과정을 거치게 된다. (`checker.isTypeRelatedTo`)

```ts
function isSimpleTypeRelatedTo(
  source: Type,
  target: Type,
  relation: ESMap<string, RelationComparisonResult>,
  errorReporter?: ErrorReporter,
) {
  const s = source.flags
  const t = target.flags
  if (
    t & TypeFlags.AnyOrUnknown ||
    s & TypeFlags.Never ||
    source === wildcardType
  )
    return true
  if (t & TypeFlags.Never) return false
  if (s & TypeFlags.StringLike && t & TypeFlags.String) return true
  if (
    s & TypeFlags.StringLiteral &&
    s & TypeFlags.EnumLiteral &&
    t & TypeFlags.StringLiteral &&
    !(t & TypeFlags.EnumLiteral) &&
    (source as StringLiteralType).value === (target as StringLiteralType).value
  )
    return true
  if (s & TypeFlags.NumberLike && t & TypeFlags.Number) return true
  if (
    s & TypeFlags.NumberLiteral &&
    s & TypeFlags.EnumLiteral &&
    t & TypeFlags.NumberLiteral &&
    !(t & TypeFlags.EnumLiteral) &&
    (source as NumberLiteralType).value === (target as NumberLiteralType).value
  )
    return true
  if (s & TypeFlags.BigIntLike && t & TypeFlags.BigInt) return true
  if (s & TypeFlags.BooleanLike && t & TypeFlags.Boolean) return true
  if (s & TypeFlags.ESSymbolLike && t & TypeFlags.ESSymbol) return true
  if (
    s & TypeFlags.Enum &&
    t & TypeFlags.Enum &&
    isEnumTypeRelatedTo(source.symbol, target.symbol, errorReporter)
  )
    return true
  if (s & TypeFlags.EnumLiteral && t & TypeFlags.EnumLiteral) {
    if (
      s & TypeFlags.Union &&
      t & TypeFlags.Union &&
      isEnumTypeRelatedTo(source.symbol, target.symbol, errorReporter)
    )
      return true
    if (
      s & TypeFlags.Literal &&
      t & TypeFlags.Literal &&
      (source as LiteralType).value === (target as LiteralType).value &&
      isEnumTypeRelatedTo(
        getParentOfSymbol(source.symbol)!,
        getParentOfSymbol(target.symbol)!,
        errorReporter,
      )
    )
      return true
  }
  // In non-strictNullChecks mode, `undefined` and `null` are assignable to anything except `never`.
  // Since unions and intersections may reduce to `never`, we exclude them here.
  if (
    s & TypeFlags.Undefined &&
    ((!strictNullChecks && !(t & TypeFlags.UnionOrIntersection)) ||
      t & (TypeFlags.Undefined | TypeFlags.Void))
  )
    return true
  if (
    s & TypeFlags.Null &&
    ((!strictNullChecks && !(t & TypeFlags.UnionOrIntersection)) ||
      t & TypeFlags.Null)
  )
    return true
  if (s & TypeFlags.Object && t & TypeFlags.NonPrimitive) return true
  if (relation === assignableRelation || relation === comparableRelation) {
    if (s & TypeFlags.Any) return true
    // Type number or any numeric literal type is assignable to any numeric enum type or any
    // numeric enum literal type. This rule exists for backwards compatibility reasons because
    // bit-flag enum types sometimes look like literal enum types with numeric literal values.
    if (
      s & (TypeFlags.Number | TypeFlags.NumberLiteral) &&
      !(s & TypeFlags.EnumLiteral) &&
      (t & TypeFlags.Enum ||
        (relation === assignableRelation &&
          t & TypeFlags.NumberLiteral &&
          t & TypeFlags.EnumLiteral))
    )
      return true
  }
  return false
}
```

> https://raw.githubusercontent.com/microsoft/TypeScript/main/src/compiler/checker.ts

먼저 타입 `string`과 `string literal`즉, 값의 `string`을 비교하여 확인하게 되는데, 이 둘이 일치하면 `true`를 리턴하게 된다. 코드의 길이는 길지만, 이 경우에는 비교가 비교적 간단하여 함수 전체를 실행하지는 않을 것이다.

만약 아래와 같은 코드는 어떻게 될까?

```ts
{ hello: number} = {hello: "world"}
```

타입스크립트는 구조적 타이핑 (structural typing)을 기반으로 하고 있으므로 먼저 외적인 구조 부터 비교를 시작하여 안으로 파고 들게 된다. 이 경우 둘다 `object`의 형태를 띄고 있기 때문에 이 시점의 비교에서는 `true`가 리턴될 것이다. 그리고 그 다음 내부의 필드를 비교하는데, 두개 모두 `hello`를 가지고 있으므로 여기서도 `true`가 될 것이다. 그 다음 필드의 값을 비교하게 되는데, `number` (위 코드 기준 `TypeFlags.NumberLike`), 그리고 `string literal`인 `"world"`가 들어가 있으므로 결국에는 `false`가 리턴될 것이다.

이와 거의 유사한 방식으로 `type generic`도 비교하게 된다.

```ts
Promise<string> = Promise<{ hello: string }>
```

> 물론, [제네릭의 공변성과 반공변성](<https://en.wikipedia.org/wiki/Covariance_and_contravariance_(computer_science)>)에 대해 다루기 시작하면 복잡해진다.

`Promise`, `Generic` (`<string>`, `{hello: string}`)

그 다음으로는 타입 추론에 대해서 알아보자. 코드의 타입 추론도 마찬가지로 `checker`의 역할 중 하나다.

```typescript
const message: string = 'Hello, world!'
```

처럼 쓸 수도 있지만, 대부분의 경우에는

```typescript
const message = 'Hello, world!'
```

를 더 선호할 것이다.

위 와 같은 경우에는, 아래와 같은 신택스 트리가 생성된다.

- `VariableStatement`
  - `VariableDeclarationList`
    - `VariableDeclaration`
      - `StringKeyword` (`name`)
      - 타입이 없다! 🤔
      - `StringLiteral` (`initializer`)

이 경우에는 간단하게 `initializer`의 타입을 비어 있는 타입쪽으로 이동 시키면 된다.

```ts
declare function setup<T>(config: { initial(): T }): T
```

위와 같은 타입 파라미터 추론의 경우에는 조금 복잡해진다. 여기에서 선언된 `T`는 실제 함수가 사용되기 전까지 무슨 타입이 올지 알 수 없다. 여기서 `checker`가 얻을 수 있는 정보는 다음과 같다.

- Generic Function
- Generic Arg T
- Return Value T
- parameter는 객체이며 `initial`이 키이고, 값은 `T`

그리고 함수의 사용이 다음과 같을 때, 위와 마찬가지 프로세스로 비교하게 된다.

```ts
const abc = setup({
  initial() {
    return 'abc'
  },
})
```

```ts
// 객체를 시작으로 밖에서부터 비교
// T를 만날때까지 안으로 파고 든다.
// T는 string으로 추론할 수 있다.
{ initial(): T } = { initial(): string }
```

`T`를 만나서 `string`이라는 것이 확인 되는 순간, `cheker`는 해당 함수`setup`으로 새로운 인스턴스를 만들게 된다. 이 인스턴스에는 `T`가 `string`이라는 정보가 담기게 되고, 모든 `T`를 `string`으로 변경한다. 그리고 그 다음에 다시 `checker`는 비교를 시작하게 된다.

```ts
const abc = setup({initial() { return "abc" }})
{initial(): string} = {initial():"abc"}
```

마지막으로 알아볼 것은 `contextual typing`, 즉 문맥상의 타이핑이다. 문맥상의 타이핑이란, 타입의 결정이 코드의 위치 (문맥) 을 기준으로 일어난다는 것을 의미한다.

```ts
type Adder = {
  inc(n: number): number
}

const adder: Adder = {
  inc(n) {
    return n + 1
  },
}
```

여기에서도 마찬가지로, parameter 에서 시작하여 신택스 트리를 분석 하여 비교한다. 가장먼저 `n`의 경우에는 타입이 현재 존재하지 않는다. 때문에 `Adder` 타입으로 돌아가 타입 비교를 시작하게 된다. 이 과정은 앞서 이야기한 타입 비교 과정과 동일하다. `Adder`는 객체타입으로 비교 확인이 완료되고, 그 이후에 `inc`라는 `attribute`가 있는지 확인하고, 일치하였으므로, 내부의 `n`을 찾아 `number`라는 타입을 얻을 수 있게 된다.

즉 타입을 알아내기 위해 파라미터에서 시작을 했다가, 파라미터에서 타입을 알아내지 못했다면 이에 해당하는 타입으로 돌아가 `n`에 해당하는 타입을 가져오게 된다.

만약 `adder`에 `n`이 `number`로 선언되어 있다 하더라도, `Adder`의 객체 타입과 비교해야 하기 때문에 동일한 과정을 또 거쳤을 것이다.

결론적으로, 타입스크립트는 이에 타입을 알아야하는 무언가가 있다면 이에 매칭하는 타입을 찾을 때 까지 신택스 트리를 거슬러 올라갈 것이다.

이렇듯 `checker`에는 타입을 체크하기 위핸 다양한 방법과 아이디어가 담겨 있다. 이를 근본적으로 이해하는 가장 좋은 방법은, github에서는 큰 파일이라 볼 수 없으므로, 직접 로컬에서 클론해서 `checker.ts`의 구조를 파악해보는 것이다. 그리고 여기에서 흥미로운 부분이 있다면, 직접 `console.log`를 넣어 디버깅 해보거나, 혹은 `debugger`를 넣는 방법 등으로 일련의 동작을 파악해본다면 도움이 될 것이다.

## 파일 생성하기

`checker`까지 거쳤다면, 이제 `.js`, 자바스크립트 파일을 만들 시간이다. 각종 도구들로 만들고 분석한 신택스 트리로, 어떻게 자바스크립트 파일을 만드는지 살펴보자.

[https://github.com/microsoft/TypeScript/blob/main/src/compiler/emitter.ts](https://github.com/microsoft/TypeScript/blob/main/src/compiler/emitter.ts)

`emitter`의 역할은 신택스 트리를 읽어서 파일로 리턴하는 것이다. `emitter`의 다음 네가지로 크게 분류할 수 있다.

- `.js` `.map` `.d.ts`를 만들어냄
- 신택스 트리를 text로 변환
- 루프 내부 `_i` `_i2` 와 같은 임시 변수를 추적
- 신택스 트리를 신택스 트리로 변환 (aka `transformer`)

신택스 트리를 신택스 트리로 변환한다는 것은 무엇일까? 아래 예제를 보면 명확히 이해할 수 있다.

```ts
const message: string = 'Hello, world'
```

**타입스크립트의 신택스 트리**

- `VariableStatement`
  - `VariableDeclarationList`
    - `VariableDeclaration`
      - `identifier` (name)
      - `StringKeyword` (type)
      - `StringLiteral` (initializer)

_자바스크립트의 신택스 트리_

- `VariableStatement`
  - `VariableDeclarationList`
    - `VariableDeclaration`
      - `identifier` (name)
      -
      - `StringLiteral` (initializer)

타입스크립트는 자바스크립트의 신택스트리와 다르게 `type` 이라는 것이 존재하므로, 이를 제거하는 과정을 거치게 된다.

이외에도 타입스크립트 transformer는 설정에 따라 다양한 일들을 하는데, 이는 모두 신택스트리를 기준으로 이루어진다.

- ts syntax 제거
- 클래스 필드 (타입스크립트에만 있는)
- ESNext transform
- ES2020 ~ ES2015 transform
- ES2015 Generator
- Module Transformer
- ES Transformer
- Done!

이러한 과정도, `ts playground`에서 `plugin`을 추가하여 확인해 볼 수 있다.

![ts-transformer](./images/ts-transformer.png)

위 코드는 최신문법이 없기 때문에 굉장히 간단하지만, 최신 코드 (`ES2020` 등) 를 사용해보면 `transforming` 하는 모습을 직접 볼 수 있을 것이다.

여기에 덧붙여, `transformer`이 어떤 코드를 실제로 `transform`을 해야하는지 알기 위해, `transformer`는 `treefacts`를 사용하여 이 코드가 어떤 문법으로 이루어져있는지 확인한다. 즉, 각 파트가 나중에 어떤 식으로 변경되어야 하는지 이해하는 과정을 거친다.

![ts-treefacts](./images/ts-treefacts.png)

첫번째 코드 `const`의 경우 ES2015의 문법이 들어가 있기 때문에, `AssetES2015`라는 플래그를 기록해둔다. 그리고 이후 `transformer`에서 `ES2015` 과정을 거치게 되면 해당 플래그 부분을 변환하게 될 것이다.

`welcome(msg)`는 es3에서도 실행될 수 있는
무난한 자바스크립트 코드(?) 이므로 별다른 체크를 해두지 않는다.

마지막은 특별한 부분은 없지만, typescript 향 코드 이므로 (타입이 있으므로) 타입스크립트 코드라는 체크만 해둔다.

`treefacts`는 이처럼 각 `transformer`가 거쳐야 할 일들을 체크해 두며, 각 `transform`과정을 거칠 때 마다 하나씩 제거해서 우리가 원하는 js 파일로 만들어준다.

이와 반대로, `d.ts`의 경우에는 타입만을 남겨두길 원하기 때문에, 앞서 언급했던 트리에서 `initializer`를 제거하고 타입만 남겨 둘 것이다.

그러나 반대로 자바스크립트를 기준으로 `.d.ts`를 만드는 경우도 있을 것이다. (레거시 js 라이브러리를 사용하기 위해서 custom `d.ts`를 추가하거나 `@types/***`를 만드는 경우) 이 경우에는 앞서 이야기 했던 과정이 반대로 이루어진다고 생각하면 된다. 자바스크립트 신택스 트리를 만든 다음, checker를 통해서 필요한 타입을 추론하고, 이를 `DTS Transformer`를 거쳐 `d.ts` 신택스를 만들게 된다. 한가지 더 번거로운 것은, 당연하게도 자바스크립트는 타입이 없기 때문에 가능한 타입에 대해서 모든 것들을 체크해서 확인한다는 것이다.

## 마무리

타입스크립트 컴파일러는 앞서 살펴보았듯 여러개의 파일, 그리고 각 파일 마다 수천 수만 라인의 코드로 이루어져 있고, 이를 몇 백줄의 글로 요약하기란 불가능에 가깝다. 이글은 어디까지나 코드나 강의자료를 참고 하면서 요약해둔 내용일 뿐이다.

실제 타입스크립트 컴파일러 내부에서는 여기에서 언급한 내용 이외에도 수많은 동작과 또 우리가 알지 못한 다양한 최적화 기법이 적용되어 있다. 이를 확인해 보기 위해 직접 저장소를 방문해 코드를 보는 것을 추천한다.

## 회고

- 신기 혹은 당연한 일이지만, 모든 타입스크립트 컴파일러들은 타입스크립트로 작성되어 있다.
- 코드를 보고 나니 SWC가 왜 만들어진지 알 수 있었다. 신택스 트리르 만들고, 분석하고, 파일을 순회해서 분석하는 일은 상당히 복잡하고 많은 시간이 소요되는 일이다. 자바스크립트 개발자로서 자바스크립트를 디스하고 싶지 않지만, 확실히 이런 일은 '안' 자바스크립트가 어울릴 수 있겠다는 생각이 든다.
  - 굳이 wasm때문이 아니더라도, 자바스크립트 생태계에 rust가 조금씩 들어오는 건 어쩌면 필연적인 일이었을 수도?
- 타입스크립트를 자주 쓰지만, 컴파일러가 어떻게 동작하는 지를 탐구해볼 생각을 많이 안해본 것 같다. 이번 기회에 많이 배우게 되었다.
