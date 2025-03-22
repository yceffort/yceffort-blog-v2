---
title: '타입스크립트 성능을 위한 팁'
tags:
  - typescript
published: true
date: 2021-03-22 23:14:07
description: '아무튼 스터디 중임'
---

타입스크립트 공식 레포에 있는 위키에는, 타입스크립트 성능을 위한 몇 가지 팁을 기재해 놓은 위키가 있다.

https://github.com/microsoft/TypeScript/wiki/Performance

그 위키에 대한 내용을 한글로 번역해보면서 간단히 요약도 하고, 또 이해가 안되는 부분은 조금씩 설명을 달아두려고 한다. 물론, 원문을 보는 것이 제일 좋다.

## Table of Contents

## 컴파일하기 쉬운 코드를 작성하기

### 타입간 결합이 필요하다면 `type`대신 `interface` 사용하기

객체에 사용하는 `type`과 `interface`는 매우 유사하게 사용되고 있다.

```typescript
interface Foo {
  prop: string
}

type Bar = {prop: string}
```

그러나 타입간 결합이 필요할 때는, `interface`를 확장하는 것이 성능상으로 유리하다. `interface`는 단순히 객체에 대한 모양을 표현하는 것이기 때문에, 여러개가 올 경우 단순히 합쳐버리면 된다. 그러나 `type`은 객체 뿐 만 아니라 단순히 원시타입도 올 수 있기 때문에 재귀적으로 속성을 머지해야 하고, 때때로 `never`가 나오곤 한다. (아래 참고)

```typescript
type type2 = {a: 1} & {b: 2} // 잘 머지됨
type type3 = {a: 1; b: 2} & {b: 3} // resolved to `never`

const t2: type2 = {a: 1, b: 2} // good
const t3: type3 = {a: 1, b: 3} // Type 'number' is not assignable to type 'never'.(2322)
const t3: type3 = {a: 1, b: 2} // Type 'number' is not assignable to type 'never'.(2322)
```

> 인터페이스를 쓴다면 이럴일이 없다.

따라서 여러개의 객체 타입을 합성해야 한다면, `interface`의 `extends`를 사용하는 것이 좋다.

### 타입 어노테이션 사용하기

타입 어노테이션, 특시 리턴 타입을 지정하는 것은 컴파일러에 많은 도움을 준다. 당연하게도, 직접 리턴타입을 지정해준다면 타입스크립트 컴파일러가 함수의 타입을 추론하는 것 보다 훨씬더 성능적으로 이점을 얻을 수 있고, 이는 declaration 파일을 읽고 쓰는데 많은 시간을 절약해준다. (incremental builds) 물론 타입 추론은 매우 편리한 기능이기 때문에, 다 이걸 처리할 필요는 없지만, 코드에서 약간의 병목현상이 생긴다면 고려해볼만 하다.

```typescript
import {bar, barType} from 'bar'
function foo() {
  return bar
}
```

이거보단, 아래 코드가 낫다.

```typescript
import {bar, barType} from 'bar'
function foo(): barType {
  return bar
}
```

### Union 보다는 Base type을 만들어두자

타입 union은 훌륭한 기능이다. 이는 값에 대한 다양한 타입의 가능성을 열어준다.

```typescript
interface WeekdaySchedule {
  day: 'Monday' | 'Tuesday' | 'Wednesday' | 'Thursday' | 'Friday'
  wake: Time
  startWork: Time
  endWork: Time
  sleep: Time
}

interface WeekendSchedule {
  day: 'Saturday' | 'Sunday'
  wake: Time
  familyMeal: Time
  sleep: Time
}

declare function printSchedule(schedule: WeekdaySchedule | WeekendSchedule)
```

그러나 이러한 타입 유니온은 비용이 발생한다. `printSchedule`에 인수가 넘어갈 때마다, 각 인수들을 union에 있는 타입들과 대조하기 시작한다. 물론 단순히 타입이 두개 뿐이라면 (성능적인 차이는) 무시할만하다. 그러나 이 숫자가 많아진다면, 컴파일 속도에 문제가 될 수 있다. 예를 들어, union에서 중복을 제거하기 위해 각각의 요소를 쌍으로 비교해야 하며, 이는 2차적으로 드는 비용이다. 이러한 종류의 검사는 union이 커질 수록 더욱 많이 발생할 수 있으며, 이 규모를 줄여야 한다. 이를 위해 union 보다는 하위 유형을 사용하는 것이 좋다.

```typescript
interface Schedule {
  day:
    | 'Monday'
    | 'Tuesday'
    | 'Wednesday'
    | 'Thursday'
    | 'Friday'
    | 'Saturday'
    | 'Sunday'
  wake: Time
  sleep: Time
}

interface WeekdaySchedule extends Schedule {
  day: 'Monday' | 'Tuesday' | 'Wednesday' | 'Thursday' | 'Friday'
  startWork: Time
  endWork: Time
}

interface WeekendSchedule extends Schedule {
  day: 'Saturday' | 'Sunday'
  familyMeal: Time
}

declare function printSchedule(schedule: Schedule)
```

더욱 현실적인 예 중하나라노는 built-in DOM 엘리먼트의 타입을 만드는 경우다. 이 경우에는 `HtmlElement`를 기본 엘리먼트로 두고 `DivElement` `ImgElement` 등을 만드는 것이, `DivElement | ImageElement ...` 보다 좋다.

## 프로젝트 레퍼런스 사용하기

타입스크립트를 사용하여 커대한 크기의 코드를 작성할 때, 코드베이스를 여러개의 독립적인 프로젝트로 구성하는 것이 도움이 된다. 이렇게 할 경우, 각 프로젝트에는 다른 프로젝트에 종속된 자체 `tsconfig.json`이 있을 수 있다. 이렇게 하면, 단일 컴파일에 너무 많은 파일이 로드 되지 않도록 도움을 줄 수 있으며, 코드 베이스 배치 전략을 쉽게 구성할 수 있다.

[코드베이스를 여러개의 프로젝트 단위로 나누는 기초적인 방법이 있다.](https://www.typescriptlang.org/docs/handbook/project-references.html) 예를 들어, 프로젝트에 클라이언트와 서버가 동시에 있고, 이 사이에 공유하는 모듈이 있다고 가정해두자.

```bash
              ------------
              |          |
              |  Shared  |
              ^----------^
             /            \
            /              \
------------                ------------
|          |                |          |
|  Client  |                |  Server  |
-----^------                ------^-----
```

테스트는 아래와 같이 분리될 수 있다.

```bash
              ------------
              |          |
              |  Shared  |
              ^-----^----^
             /      |     \
            /       |      \
------------  ------------  ------------
|          |  |  Shared  |  |          |
|  Client  |  |  Tests   |  |  Server  |
-----^------  ------------  ------^-----
     |                            |
     |                            |
------------                ------------
|  Client  |                |  Server  |
|  Tests   |                |  Tests   |
------------                ------------
```

한 가지 자주 묻는 질문 중에 하나는, '도대체 얼마나 프로젝트가 커야 하는가' 에 대한 것이다. 이는 마치 '함수/클래스는 어디까지 커져도 되나요' 와 같은 질문과 비슷한데, 결국은 경험에 의지할 수 밖에 없다. 한가지 익숙한 방식은 `js`와 `ts`를 폴더 단위로 나누는 것이다. 또한 같은 폴더에 있기에 충분히 비슷한 내용의 코드라면, 프로젝트도 같은 단위에 있는 것이 좋다. 그리고 너무 크거나 작은 프로젝트는 지양하는 것이 좋다. 만약 한가지 프로젝트가 다른 것들을 합친것보다 더 크다면, 일종의 경고 싸인으로 보는 것이 좋다. 비슷하게, 오버헤드 증가를 막기 위해서 단일 파일을 내지한 수십개의 프로젝트는 지양하는 것이 좋다.

참고: https://www.typescriptlang.org/docs/handbook/project-references.html

## `tsconfig.json`이나 `jsconfig.json` 설정하기

타입스크립트 유저는 `tsconfig.json`로 컴파일 환경을 설정할 수 있다. [`jsconfig.json`은 마찬가지로 자바스크립트 유저의 개발 환경을 설정하는데 도움을 줄 수 있다.](https://code.visualstudio.com/docs/languages/jsconfig)

### 파일 명시하기

항상 설정파일이 한번에 너무 많은 파일을 포함하지 않도록 조심해야 한다.

`tsconfig.json`을 사용한다면, 프로젝트의 파일을 특정하는 방법이 두가지가 있다.

- `files`
- `include` `exclude`

두 개의 차이라면, `files`는 소스 파일의 path를 명시해야 하고, `include` `exclude`는 파일의 globbing pattern을 사용한다는 것이다.

`files`을 지정한다면, 파일을 직접 빠르게 로드할 수 있다는 장점이 있지만, 최상위 진입점이 별도로 존재하지 않고 프로젝트에 많은 파일이 있는 경우 조금 번거로워 질 수 있다. 또한 `tsconfig.json`에 새파일을 추가하는 것을 까먹는 일도 종종 발생할 수 있으므로, 조금 번거로울 수 있다.

`include`/`exclude`는 위 처럼 파일을 특정해야 하는 번거로움을 없앨 수 있지만, 역시 이에 따른 비용이 발생한다. 파일을 찾기 위해서 디렉토리를 계속해서 순회해야 한다는 것이다. 만약 폴더가 엄청 많을 경우, 컴파일 속도가 느려질 수 있다. 이에 덧붙여, 때때로 컴파일 과정에서 불필요한 `.d.ts`와 테스트 파일이 추가되버리는 경우, 컴파일 속도와 메모리 오버헤드가 발생할 수 있다. 마지막으로, `exclude`에는 `node_modules`와 같은 몇가지 이유 있는 기본값들이 존재하고는 있지만, 잘 관리하지 못할 경우 아주 무거운 폴더가 포함될 위험성 또한 존재한다.

최상의 개발 경험을 위해서, 아래와 같이 설정할 것을 추천한다.

- 프로젝트의 input 폴더 만을 명시해 둘 것
- 다른 프로젝트의 소스파일을 같은 폴더에 짬뽕해서 보관해두지 말것
- 테스트를 다른 원본 파일과 같은 폴더에 두는 경우, 쉽게 제외 시킬 수 있도록 고유한 이름을 지정할 것 (`*.test.ts`와 같이)
- 소스 디렉토리에서 `node_modules`와 같은 대규모 빌드 아티팩트와 dependency 폴더를 피할 것

> `exclude`가 비어있다고 하더라도, `node_modules`는 기본값으로 제외된다

```json
{
  "compilerOptions": {
    // ...
  },
  "include": ["src"],
  "exclude": ["**/node_modules", "**/.*/"]
}
```

### `@types`

기본값으로, 타입스크립트는 개발자가 import 했던 안했던 간에 `node_modules`에 있는 `@types` 패키지를 자동으로 포함시킨다. 이 말인 즉슨, `node.js` `jasmine` `mocha` 와 같이 import 하지 않은 패키지라 할지라도, 단순히 글로벌 환경에서 로드 되어 사용될 수 있다는 것을 의미한다.

이는 때때로 컴파일과 코드 에디팅 하는 시간을 지연시킬 수 있으며, 심지어 이것들의 선언이 서로 충돌이 나서 다음과 같은 문제가 날 수도 있다.

```bash
Duplicate identifier 'IteratorResult'.
Duplicate identifier 'it'.
Duplicate identifier 'define'.
Duplicate identifier 'require'.
```

따라서, 글로벌 패키지가 필요하지 않은 상황이라면, `type` 옵션을 비워 둠으로써 이러한 문제를 해결할 수 있다.

```json
// src/tsconfig.json
{
  "compilerOptions": {
    // ...

    // Don't automatically include anything.
    // Only include `@types` packages that we need to import.
    "types": []
  },
  "files": ["foo.ts"]
}
```

만약 몇가지 패키지가 글로벌로 필요하다면, 아래와 같이 추가할 수 있다.

```json
// tests/tsconfig.json
{
  "compilerOptions": {
    // ...

    // Only include `@types/node` and `@types/mocha`.
    "types": ["node", "mocha"]
  },
  "files": ["foo.test.ts"]
}
```

### 점진적 프로젝트 빌드 옵션 사용하기

`--incremental` 옵션은 타입스크립트가 마지막 컴파일 정보를 `.tsbuildinfo`에 저장해두도록 한다. 이 파일은 `--watch` 가 작동하는 방식과 비슷하게, 마지막 컴파일 이후 다시 체크 혹은 내보내야 하는 (emit) 가장 작은 파일 집합을 파악하는데 사용된다.

이러한 점진적 컴파일은, 프로젝트 설정에 `composite`를 설정해둘 때 기본으로 사용하는데, 이를 사용해서 선택한 프로젝트에 대한 동일한 속도 향상을 가져 올 수 있다.

### `.d.ts` 체크 생략

기본값으로, 타입스크립트는 프로젝트 내에 있는 `.d.ts` 파일을 모두 체크하여 일관성을 유지하고 이슈를 찾는다. 그러나, 이는 일반적으로 불필요한 작업이다. ~~대부분의 경우, `.d.ts`는 잘 작동하는 파일일 가능성이 크다. 타입스크립트는 `.d.ts`의 체크를 끄는 `skipDefaultLibCheck` 옵션을 제공한다.~~ (이는 deprecated되었다. 그냥 `skipLibCheck`을 쓰면 된다.)

이 옵션은 빌드를 빠르게하는 목적으로만 사용하는 것이 좋다.

### 빠른 분산 검사

dog list는 animal list 일까? 다시말해, `List<Dog>`은 `List<Animals>`에 할당 가능한가? 이를 확인할 수 있는 가장 정확한 방법은, 각 타입의 구조를 멤버 대 멤버로 하나씩 검사하는 것이다. 그러나 이는 매우 느릴 수 있다. 그러나 만약 우리가 `List<T>`에 대해서만 할 수 있다면, `Dog`이 `Animal`에 할당 가능한지만 확인하면 될 것이다. (`List<T>`의 각 멤버를 일일이 검사할 필요 없이) 컴파일러가 `strictFunctionTypes` 플래그를 활성화 시켜 잠재적으로 성능을 향상시킬 수 있다.

#### 🤔 뭔개소리야,,,

```typescript
interface Animal {}
interface Dog extends Animal {}
interface JayG extends Dog {}
```

타입 시스템에는, 타입 가변성이라는 개념이 존재한다. (Type Variance) 이는 타입과 서브타입의 관계를 서술한 것을 의미한다. 여기에는 네가지가 존재한다.

- Covariance: `A`가 `B`의 서브타입일 경우, `T<A>` 도 `T<B>`의 서브타입인 경우

  ```typescript
  function hello(d: Dog) {}
  hello(animal) //error
  hello(dog) // ok
  hello(jayg) //ok
  ```

- Contravariance: `A`가 `B`의 서브타입일 경우, `T<B>`가 `T<A>`의 서브타입인 경우

  ```typescript
  function hello(d: Dog) {}
  hello(animal) //ok
  hello(dog) // ok
  hello(jayg) // error
  ```

- Invariance: 다른 타입을 허용하지 않음
- Bivariance: 아무 타입이나 다 허용

위 설명에서, 타입스크립트는 기본적으로 Covariance 하다는 것을 의미한다.

```typescript
interface Animal {
  name: string
}
interface Dog extends Animal {
  kind: string
}
interface JayG extends Dog {
  age: number
}

const a: Animal = {name: 'hi'}
const d: Dog = {name: 'hi', kind: 'mix'}
const j: JayG = {name: 'hi', kind: 'mix', age: 34}

const animals: Animal[] = new Array(5)
const dogs: Dog[] = new Array(5)
const jaygs: JayG[] = new Array(5)

animals[0] = a // ok
animals[1] = d // ok
animals[2] = j // ok

dogs[0] = a // error  Property 'kind' is missing in type 'Animal' but required in type 'Dog'.
```

그러나 메서드 인수에서는 Contravariance 하다.

```typescript
let helloAnimal: (x: Animal) => void = () => console.log('animal')
let helloDog: (x: Dog) => void = () => console.log('dog')
let helloJayG: (x: JayG) => void = () => console.log('jayg')

helloAnimal = helloDog // Error with --strictFunctionTypes Type '(x: Dog) => void' is not assignable to type '(x: Animal) => void'.
helloDog = helloAnimal // ok
helloDog = helloJayG // Error with --strictFunctionTypes Type '(x: JayG) => void' is not assignable to type '(x: Dog) => void'.
helloJayG = helloDog // ok
```

메소드 인수는 이처럼, `Contravariance`한 특징으르 가지고 있는데, 기존에는 메소드 인수가 `Bivariance`, 즉 아무타입이나 다 허용 했다. 그러나 이 옵션 `strictFunctionTypes`이 등장하면서 이러한 문제를 막아주기 시작했다.

다시말해, `strictFunctionTypes` 옵션을 통해서 변수의 가변성을 엄격하게 체크할 수 있으므로, 이것을 통해서 다양한 경우의 수를 고려하지 않아도 되기 때문에 빌드가 빨라질 수 있다는 것을 의미한다.

## 다른 빌드 툴 설정하기

타입스크립트 컴파일러는 종종 다른 빌드 툴과 함께 실행되는데, 이는 특히 웹 애플리케이션 제작시에 번들러가 포함되는 상황이 종종 발생한다. 여기에서는 모든 빌드 툴을 다 다루지는 않지만, 기본적인 접근방식은 비슷하다고 보면 된다. 본격적으로 아래 섹션을 읽기 전에, 다음과 같은 아티클을 보는 것도 좋다.

- https://github.com/TypeStrong/ts-loader#faster-builds
- https://github.com/s-panferov/awesome-typescript-loader/blob/master/README.md#performance-issues

### 동시성 타입 체크

타입 체크는 일반적으로 다른 파일에 있는 정보를 필요로 하는데, 이 때문에 코드변환, 생성 등의 과정에서 상대적으로 더 많은 비용이 발생할 수 있다. 타입 체크는 더욱이 시간이 더 걸릴 수 있기 때문에, 이는 내부 개발 루프에 영향을 미칠 수 있다. 즉 다시 말해, 코드 편집, 컴파일, 실행 주기가 길어져 번거로울 수 있다.

이러한 이유로, 일부 빌드 툴은 타입체크를 다른 프로세스와 분리 시켜서 수행할 수 있다. 이는 타입스크립트가 빌드 툴 내부의 에러를 보고하기전에, 잘못된 코드가 실행될 수도 있다는 것을 의미하지만, 편집기에서 오류가 먼저 나타나는 경우가 더많고 작업코드를 실행하는 동안 차단하지 않는다.

- https://github.com/TypeStrong/fork-ts-checker-webpack-plugin
- https://github.com/s-panferov/awesome-typescript-loader

## 이슈 분석하기

뭔가 잘못되고 있다는 걸 느낄때, 아래의 방법을 통해서 힌트를 얻을 수 있다.

### 에디터 플러그인 비활성화

에디터는 설치된 플러그인에 따라 영향을 받을 수 있다. 플러그인, 특히 자바스크립트와 타입스크립트와 연관된 플러그인을 비활성화 해서 성능과 반응성에 영향이 있는지 확인해볼 필요가 있다.

### `extendDiagnostics`

`--extendedDiagnostics`를 활성화 하면, 아래와 같은 정보를 컴파일러로 부터 얻을 수 있다.

```bash
Files:                         6
Lines:                     24906
Nodes:                    112200
Identifiers:               41097
Symbols:                   27972
Types:                      8298
Memory used:              77984K
Assignability cache size:  33123
Identity cache size:           2
Subtype cache size:            0
I/O Read time:             0.01s
Parse time:                0.44s
Program time:              0.45s
Bind time:                 0.21s
Check time:                1.07s
transformTime time:        0.01s
commentTime time:          0.00s
I/O Write time:            0.00s
printTime time:            0.01s
Emit time:                 0.01s
Total time:                1.75s
```

> `Total Time`은 위 정보를 모두 합한 시간이 아니다. (일부 누락된 시간 등이 존재)

- `files`: 프로그램에 포함된 파일
- `I/O Read Time`: 파일 시스템에 접근하면서 읽는데 소요된 시간 `include`를 순회하면서 걸리는 시간 포함
- `Parse time`: 프로그램을 스캔하고 파싱하는데 걸리는 시간
- `Bind time`: 단일 파일에 다양한 정보를 구축하는데 소요된 시간
- `Check time`: 타입 체크에 소요된 시간
- `transformTime time`: 타입스크립트 AST를 구식 런타임형태로 재작성하는데 걸리는 시간
- `commentTime` 결과 파일에 코멘트를 계산하는데 소요된 시간
- `I/O Write time`: 디스크에 파일을 쓰고 업데이트 하는데 소요된 시간
- `printTime time`: 출력 파일내 문자열 을 계산하여 디스크로 내보내는데 걸리는 시간

- `printTime`가 너무 높다면, `emitDeclarationOnly` 옵션을 고려해보자
- `Program Time` `I/O Read time`이 높다면, `include`/`exclude`가 적절하게 설정되어 있는지 확인해보자.

### `showConfig`

`tsc` 실행시에는 컴파일이 어떤 설정을 가지고 실행되는지 알 수 없으며, 특히 `tsconfig.json`이 다른 설정파일로 확장될 수 있다는 점을 고려할 때 더욱 헷갈릴 수 있다. 아래 옵션을 통해 실제로 어떤 설정을 가지고 컴파일 되는지 확인하라 필요가 있다.

```bash
tsc --showConfig

# or to select a specific config file...

tsc --showConfig -p tsconfig.json
```

### `traceResolution`

`traceResolution`는 특정 파일이 왜 컴파일에 포함되어 있는지 추적해준다.

```bash
tsc --traceResolution > resolution.txt
```

만약 특정 존재하지 않아야 하는 파일이 보인다면, `include` `exclude` 옵션을 살펴보거나, `types` `typeRoots` `path` 등을 확인할 필요가 있다.

### `tsc` 만 단독으로 실행해보기

대부분의 시간을, 써드 파티 툴인 Gulp, Rollup, Webpack 등과 함께 실행하기 때문에 성능이 느려보일 수 있다. `tsc --extendedDiagnostics` 를 사용하여 타입스크립트와 툴간의 주요 불일치를 찾아 낸다면, 잘못된 설정 또는 비효율적인 부분을 짚어낼 수 있다.

이를 통해 염두해야 할 점은

- `tsc` 단독 실행과 타입스크립트와 연동한 다양한 빌드 툴 사이에 빌드 시간 차이가 현격하게 나는지
- 빌드 툴이 진단을 제공하는 경우, 타입스크립트의 결과와 차이가 있는지
- 빌드 툴에 원인이 될 수 있는 자체 옵션이 있는지
- 빌드 툴에 원인이 될 수 있는 타입스크립트 구성이 있는지 (`ts-loader` 와 같이)

등이 있다.

### Dependencies 업그레이드

typescript 의 버전과 `@types`의 패키지 버전을 업그레이드 해보자.

### 성능 추적

위의 옵션으로도 왜 타입스크립트가 느려졌는지 이해하기 어려울 때, 타입스크립트 4.1 버전 이상에서 제공하는 `--generateTrace`를 사용하여 컴파일러가 시간을 소비하는 작업을 파악해보자. 이 옵션은 엣지 또는 크롬에서 분석할 수 있는 출력 파일을 제공한다.

```bash
tsc -p ./some/project/src/tsconfig.json --generateTrace tracing_output_folder
```

1. `about://tracing`
2. `load` 클릭
3. 아웃풋 폴더 내의 `trace.*.json` 열기

자세한 내용은 [여기](https://github.com/microsoft/TypeScript/wiki/Performance-Tracing)에서 확인할 수 있다.

## (개인적인) 결론

곧 엄청나게 큰 프로젝트에 타입스크립트를 도입을 앞두고 있어서, 잃어버린 타입스크립트에 대한 기억을 되찾고자 다시한번 공부해보았다. 설정 파일을 만드는 것은 잠깐이지만, 그것을 기반으로 성을 쌓는건 엄청나게 긴 시간이 든다. 기반을 잘못 다지게 되면 성을 아무리 쌓는들 무슨 소용이 있으랴. 🤪 부디 모두가 행복하게 타입스크립트를 well-form으로 적용할 수 있도록 기반을 다지고, 중간 중간 성능 이슈도 점검하면서 잘 만들어 갔으면 조헧다.
