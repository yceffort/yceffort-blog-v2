---
title: 'Nodejs 모듈 (CommonJS, ECMAScript) 과 패키지, 그리고 Semver'
tags:
  - javascript
  - nodejs
published: true
date: 2021-07-05 21:41:20
description: '어우 피곤해'
---

## Node.js의 모듈

아주 간단히 이야기 하자면, Node.js의 모듈이란 필요로하는 자바스크립트 파일을 의미한다. Node.js 런타임은 현재 두가지 유형의 모듈을 지원한다. 첫번째는 `CommonJS` 모듈이며, 이는 Node.js가 가장 오랫동안 지원해온 모듈 시스템이다. 이는 `*.js`나 `*.cjs`로 끝난다. 두 번째로는 요즘 자주 사용되는, ECMAScript 방식이다. 이 파일 확장자는 `*.js`나 `*.mjs`로 끝난다.

`CommonJS` 모듈은 일반적으로 웹 브라우저에서 로드되는 자바스크립트 파일과는 약간 다르다. CommonJS 자바스크립트 파일에는 다른 공통 파일을 참조하기 위해 사용할 수 있는 `require()`가 존재하고, 다른 공통파일에서 참조할 수 있도록 하는 `exports`가 존재한다. Webpack이나 Browserify와 같은 도구들은 브라우저 환경에서 CommonJS 파일을 사용할 수 있게 해준다.

CommonJS 모듈이 Nodejs 내부에서 참조되면, 파일의 내용을 즉시실행함수로 감싸 버린다. 이 방법을 통해 `exports`와 `require` 기능을 사용할 수 있게 된다. 이 래퍼는 대략 아래와 같은 모양을 띈다.

```javascript
;(function (exports, require, module, __filename, __dirname) {
  // original content here
})
```

https://www.freecodecamp.org/news/node-module-exports-explained-with-javascript-export-function-examples/

`require('module')` 함수가 알아서 필요한 모듈을 찾으려면 몇가지 단계를 통과해야 한다. 이러한 프로세스를 module resolution algorithm 이라고 한다. 이는 대략 아래와 같은 과정을 거친다.

- `module`이 `http`와 같은 nodejs 내장 모듈이라면 그것을 로드한다.
- `module`이 `/` `./` `../`로 시작하면 파일이나 디렉토리를 로드한다.
- 디렉토리라면, `package.json` 파일의 `main`필드를 보고, 그것을 로드한다.
- 디렉토리인데, `package.json`이 없다면 `index.js`를 로드한다.
- 파일인데, 확장자까지 정확이 있다면 그 파일을 로드하고, 확장자가 없다면, `.js` `.json` `.node`를 로드한다.
- `./node_modules`를 살펴본다.
- `./node_modules`디렉토리를 찾기 위해 각 상위 디렉토리를 살펴본다.

위를 간단히 표로 요약해보자.

| require                   | Module Path                                                      |
| ------------------------- | ---------------------------------------------------------------- |
| `require('path')`         | built-in _path_ module                                           |
| `require('./my-mode.js')` | `/srv/my-mod.js`                                                 |
| `require('redis')`        | `/srv/node_modules/redis/`, `/node_modules/redis/`               |
| `require('foo.js')`       | `/srv/node_modules/foo.js/`, `/node_modules/foo.js`              |
| `require('./foo')`        | `/srv/foo.js` `/srv/foo.json` `/srv/foo.nde` `/srv/foo/index.js` |

한가지 팁을 주자면, 명시적으로 파일이 필요한 경우라면 확장자까지 제공하는 것이 좋다. 확장자를 생략하면, `require`가 모호해지고, 만약 같은 파일명의 `.json` 이나 `.js`가 추가된다면 코드가 깨져버릴 수도 있다.

파일이 로드되면 `require` cache에 추가된다. key/value 쌍으로 저장되는데, 여기서 키는 확인된 모듈 파일의 이름에 대한 절대경로이고, 값은 해당 모듈의 export 객체다. 따라서 단일 인스턴스를 여러번 exports 하더라도 동일한 싱글톤 객체를 참조할 수 있게 된다.

## SemVer: 시멘틱 버저닝

`SemVer`란 packages를 릴리즈하는데 있어 일종의 규칙이라 볼 수 있다. npm을 비롯한 여러 플랫폼에서 사용되고 있다. `SemVer`의 버전 문자열은 `1.2.3`과 같이 마침표로 구분된 세개의 숫자로 이루어져 있다. 첫번째는 메이저, 두번째는 마이너, 세번째는 패치 버전이다.

각 버전은 다른 의미를 가지고 있다. 일반적으로 브레이킹 체인지가 있을 경우 (= 패키지의 작동 방식이 바뀌는 경우) 메이저 버전이 변경된다. 새로운 기능이 추가 되는 경우 마이너 버전이 증가한다. 마지막으로 버그 수정의 경우에는 패치버전이 변경된다. 숫자가 커지는 경우 오른쪽 숫자가 0으로 바뀔 수 있다. `9.0.0`이 메이저 버전업을 거치게 되면 `10.0.0`이 될 수 있는 것이다.

만약 `0.1.2` 와 같이 선행버전이 0으로 시작하는 경우, 가장 중요한 숫자는 그 다음 숫자가 된다. 즉, `0`으로 시작하는 패키지는 아직 안정적인 프로젝트는 아님을 의미한다.

![semver](https://thomashunter.name/media/2021/packages-modules/semver-ranges.png)

이 SemVer의 철학을 고수하는 것이 바로 npm 커뮤니티를 하나로 묶는 것이다. 호환성에 대한 일종의 이러한 범용적인 가정 덕분에, 애플리케이션은 특정 패키지 버전 대신 패키지의 '범위'에 자유롭게 의존활 수 있다. `package.json`에는 키와 값으로, 즉 키는 패키지 이름, 값은 패키진의 버전 범위 (또는 특정 버전)을 나타낸다.

```json
"dependencies": {
  "fastify": "^3.11.1",
  "ioredis": "~4.22.0",
  "pg": "8.5.1"
}
```

`fastify`는 `^`기호를 사용하여 버전 범위를 나타낸다. 이는 지정된 버전이 호환되는 모든 버전을 허용한다. (`3.11.1` `3.11.9` `3.19.3`은 가능하지만, `3.11.0`과 같이 이전 버전, 혹은 `4.0.1`과 같은 더 높은 메이저 버전은 허용하지 않는다.) 일반적으로 npm으로 새패키지를 설치할때 기본적으로 사용된다.

`ioredis`는 `~`를 사용한다. 이는 버그 수정 (패치 버전 업데이트)만 가능하고, 마이너버전 업데이트도 허용하지 않는다는 것을 의미한다. 이는 패키지와의 강력한 연결이 요구될 떄 사용할 수 있다.

`pg`는 어떠한 기호도 사용하고 있지 않는다. 이 특정 패키지만 사용하라 수 있는데 이를 패키지 버전 고정이라고도 한다.

## npm package와 `node_modules` 디렉토리

npm 패키지는 node.js 모듈및 json 파일, `README.md` 등을 포함하는 아카이브다. 공용 패키지는 `npmjs.com` 레지스트리 등에 업데이트 할 수 있으며, private 패키지는 private registry 또는 회사 소유의 레지스트리에 업로드 할 수 있다. Node.js 자체는 npm 패키지가 무엇인지 인식하지 않고, `node_modules` 디렉토리에 있는 디렉토리와 파일만 인식한다. 이러한 패키지를 추출하여 올바른 위치에 콘텐츠를 배치하는 것이 npm CLI의 몫이다. 

Node.js 자체는 다른 플랫폼에서 제공하는 많은 기능이 없기 때문에, npm 패키지는 node.js 애플리케이션에 매우 중요하다고 볼 수 있다. 이는 npm 패키지 생태계가 성장할 수 있도록 장려된 의도적인 설계 철학이다. 

따라서 거의 모든 node.js 애플리케이션에 dependencies, 종속성이 있다. dependencies란 애플리케이션이 의존하고 있는 npm 패키지다. 이러한 dependency는 직접적인 의존성일 수도 있고, dependency가 의존하는 또다른 하위 dependency일 수도 있다. 이는 dependency의 계층 구조를 만들어 낸다.

아래 구조를 예를 들어보자.

```bash
node_modules/
  foo/ (1.0.0)
  bar/ (2.0.0)
    node_modules/
      foo/ (1.0.0)
```

여기서 한가지 발견할 수 있는 문제는 순환 의존성이다. `foo` 패키지가 만약에 `bar`에 의존하게 되면 무한히 순환하게 되는 중첩된 폴더구조가 생겨버린다. 또, 그렇지 않더라도, `foo` 모듈이 두번 설치되어 공간을 낭비하게 된다. npm은 이를 위해 패키지를 설치 할 때 패키지를 트리 위에 올려 중복을 제거한다. 

```bash
node_modules/
  foo/ (1.0.0)
  bar/ (2.0.0)
```

`bar` 패키지는 이제 `foo` 패키지를 자신의 `node_modules` 대신, 상위 폴더로 접근하여 자신과 동등한 위치에 있는 `foo`를 사용할 것이다.

더 복잡한 예를 살펴보자. 예를 들어, 서로다른 패키지는 각각 서로다른 버전의 패키지에 의존하고 있을 수 있다. npm은 각각의 패키지를 만족하는 최적의 패키지 버전을 찾아내서, 호이스팅 시켜 디스크 사용량을 줄이게 된다.

그러나 호이스팅이 불가능한 경우도 있다. 아래와 같이 다른 버전을 사용하는 경우, 각각 다른 버전의 패키지가 설치되어 버릴 수도 있다.

```bash
node_modules/
  foo/ (1.0.0)
  bar/ (2.0.0)
    node_modules/
      foo/ (2.0.0)
```

`package-lock.json` (구 `npm-shrinkwrap.json`)는, 패키지의 직접적인 의존성,그리고 일시적인 의존성도 차단하기 위해 만들어졌다. 이 파일이 없다면, 새패키지 버전이 나올떄마다 디스크에 설치파는 패키지의 버전을 매번 확인해야 할 것이다.

## npm install vs npm ci

의존성을 설치하는 npm cli 명령어는 `install`과 `ci`가 있다. 한줄로 요약하자면, `package-lock.json`을 오염시키지 않기 위한 환경 (빌드, ci, 배포 등)에서는 `ci`를, 그외의 개발 과정중에서는 `install`을 사용하는 것이 좋다.

https://docs.npmjs.com/cli/v7/commands/npm-ci

`npm ci`는

- `package-lock.json` `npm-shrinkwrap.json`이 있을 경우 (패키지 버전을 다시 확인하지 않고 두 파일에 기재된 그대로 설치)
- `node_modules`가 없는 경우

에 매우 빠르게 동작한다.

따라서 ci를 사용하기 위해서는,

- `package-lock.json` `npm-shrinkwrap.json`가 반드시 존재해야 한다.(없다면 `npm install`)
- `package-lock.json`의 종속성이 `package.json`과 일치하지 않는다면, 업데이트 되는 것이 아니고 에러가 난다. (이를 해결하려면 `npm instlal`)
- `npm ci`는 전체 종속성을 설치할 때만 사용 `npm ci react`는 불가능
- `node_modules`가 존재한다면 `npm ci`는 해당 폴더를 삭제
- 절대로 `package.json`이나 `package-lock.json`을 수정하지 않는다.