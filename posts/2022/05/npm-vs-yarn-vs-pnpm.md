---
title: 'npm, yarn, pnpm 비교해보기'
tags:
  - javascript
  - npm
  - yarn
  - pnpm
published: true
date: 2022-05-20 22:26:01
description: '그리고 승자는 🤔'
---

## Table of Contents

## Introduction

npm 에서 시작한 node package management의 역사는, 이제 3가지 옵션이 주어져 있다. yarn 1.0 (이제 yarn classic 이라고 부르겠다) 과 yarn 2.0 (yarn berry) 두 가지 버전도 사뭇 다른 점이 많다는 것을 감안한다면, 이제 크게 4가지 선택지가 존재 한다고 볼 수 있다.

그리고 위 3가지 패키지 관리자들은 아래와 같은 기본적인 기능 (node 모듈을 설치하고, 관리하는 등)을 제공하고 있다.

- metadata 작성 및 관리
- 모든 dependencies 일괄 설치 또는 업데이트
- dependencies 추가, 업데이트, 삭제
- 스크립트 실행
- 패키지 퍼블리쉬
- 보안 검사

따라서 설치 속도나 디스크 사용량, 또는 기존 워크 플로우 등과 어떻게 매칭 시킬지와 같은 기능 외적인 요구 사항에 따라 패키지 관리자를 선택하는 시대가 도래했다고 볼 수 있다.

겉으로는 기능적으로 비슷해보이고 무엇을 선택하든 별 차이는 없어보이지만, 패키지 관리자들의 내부 동작은 매우 다르다. npm 과 yarn의 경우 flat 한 node_modules 폴더에 dependencies 를 설치했다. 그러나 이러한 전략은 비판에서 자유롭지 못하다. (어떤 문제인지는 뒤에서 설명하도록 한다.)

그래서 등장한 pnpm은 이러한 dependencies를 중첩된 node_modules 폴더에 효율적으로 저장하기 시작했고, yarn berry는 plug and play (pnp) 모드를 도입하여 이러한 문제를 해결하기 시작했다.

이 세가지 패키지 관리자는 각각 어떤 특징과 역사를 가지고 있으며, 무엇을 선택해야할까?

- [npm](https://www.npmjs.com/)
- [yarn classic](https://classic.yarnpkg.com/lang/en/)
- [yarn berry](https://github.com/yarnpkg/berry)
- [pnpm](https://pnpm.io/ko/)

> pnpm 홈페이지에 있는 yarn과 npm을 쓰레기통에 쳐박은 이미지가 매우 인상적이다. 🤔 vue가 react를 쓰레기통에 쳐박는 이미지를 달아놨다면...

## 자바스크립트 패키지 관리자의 역사

모두가 잘 알고 있는 것 처럼 최초의 패키지 매니저는 2010년 1월에 나온 npm 이다. npm 은 패키지 매니저가 어떤 동작을 해야하는 지에 대한 핵심적인 개념을 잡았다고 볼 수 있다.

10여년이 넘는 시간 동안 npm 이 존재했는데, yarn, pnpm 등이 등장하게 된 것일까?

- `node_modules` 효율화를 위한 다른 구조 (nested vs flat, node_modules, vs pnp mode)
- 보안에 영향을 미치는 호이스팅 지원
- 성능에 영향을 미칠 수 있는 `lock`파일 형식
- 디스크 효율성에 영향을 미치는 패키지를 디스크에 저장하는 방식
- 대규모 모노레포의 유지 보수성 과 속도에 영향을 미치는 workspace라 알려진 멀티 패키지 관리 및 지원
- 새로운 도구와 명령어 관리에 대한 관리
  - 이와 관련된 다양하고 확장가능한 플러그인과 커뮤니티 툴
- 다양한 기능 구현 가능성과 유연함

npm 이 최초로 등장하 이래로 이러한 니즈가 어떻게 나타났는지, yarn classic은 그 이후 등장해서 어떻게 해결햏ㅅ는지, pnpm이 이러한 개념을 어떻게 확장했는지, yarn berry가 전통적인 개념과 프로테스에 의해 설정된 틀을 깨기 위해 어떠한 노력을 했는지 간략한 역사를 파악해보자.

### 선구자 npm

본격적으로 시작하기에 앞서 재밌는 사실을 이야기 해보자면, `npm`은 `node package manager`의 약자가 아니다. npm의 전신은 사실 `pm`이라 불리는 bash 유틸리티인데, 이는 `pkgmakeinst`의 약자다. 그리고 이의 node 버전이 `npm`인 것이다.

> [https://github.com/npm/cli#is-npm-an-acronym-for-node-package-manager](https://github.com/npm/cli#is-npm-an-acronym-for-node-package-manager)

npm 이전에는 프로젝트의 dependencies를 수동으로 다운로드하고 관리하였기 때문에 엄청난 혁명을 가져왔다고 볼 수 있다. 이와 더불어 메타데이터를 가지고 있는 `package.json`와 같은 개념, dependencies를 `node_modules`라 불리는 폴더에 설치한다는 개념, 커스텀 스크립트, public & private 패키지 레지스트리와 같은 개념들 모두 npm에 의해 도입되었다.

### 많은 혁명을 가져온 yarn classic

[2016년의 블로그 글](https://engineering.fb.com/2016/10/11/web/yarn-a-new-package-manager-for-javascript/)에서, 페이스북은 구글과 몇몇 다른 개발자들과 함께 npm이 가지고 있던 일관성, 보안, 성능 문제 등을 해결하기 위한 새로운 패키지 매니저를 만들기 위한 시도를 진행 중이라고 발표 했다. 그리고 이듬해 `Yet Another Resource Negotiator`의 약자인 yarn을 발표했다.

yarn은 대부분의 개념과 프로세스에 npm을 기반으로 설계했지만, 이외에 패키지 관리자 환경에 큰 영향을 미쳤다. npm과 대조적으로, yarn은 초기버전의 npm의 주요 문제점 중 하나였던 설치 프로세스의 속도를 높이기 위해 작업을 병렬화 하였다.

yarn은 dx(개발자 경험), 보안 및 성능에 대한 기준을 높였으며, 다음과 같은 개념을 패키지 매니저에 도입하였다.

- native 모노레포 지원
- cache-aware 설치
- 오프라인 캐싱
- lock files

yarn classic은 2020년 부터 유지보수 모드로 전환되었다. 그리고 1.x 버전은 모두 레거시로 간주하고 yarn classic으로 이름이 바뀌었다. 현재는 yarn berry에서 개발과 개선이 이루어지고 있다.

### pnpm 빠르고 휴올적인 디스크 관리

pnpm은 2017년에 만들어졌으며, npm 의 drop-in replacement(설정을 바꿀 필요 없이 바로 사용가능하며, 속도와 안정성 등 다양한 기능 향상이 이루어지는 대체품) 으로, npm만 있다면 바로 사용할 수 있다.

pnpm 제작자들이 생각한 npm 과 yarn의 가장 큰 문제는 프로젝트 간에 사용되는 dependencies의 중복 저장이다. yarn classic이 물론 npm 보다 빠르지만, 두 매니저 모두 node_modules 내부에 flat하게 패키지를 설치하여 (=동일한 디렉토리에 flat하게 저장) 관리했다.

pnpm은 이러한 호이스트 방식 대신, 다른 dependencies를 해결하는 전략인 [content-addressable storage](https://pnpm.io/next/symlinked-node-modules-structure)를 사용했다. 이 방법을 사용하면, home 폴더의 글로벌 저장소 (`~/.pnpm-store`)에 패키지를 저장하는 중첩된 node_modules 폴더가 생성된다. 따라서 모든 버전의 dependencies은 해당 폴더에 물리적으로 한번만 저장되므로, single source of truth를 구성하고, 상당한 디스크 공간을 절약할 수 있다.

이는 node_modules의 레이아웃을 통해 이루어지고, `symlinks`를 사용하여 dependencies의 중첩된 구조를 생성한다. 여기서 폴더 내부의 모든 패키지 파일은 저장소에 대한 하드 링크로 구성되어 있다.

![pnpm](https://d33wubrfki0l68.cloudfront.net/64b2f62af3b1c3dc4314df0ec517d9661d03b934/aca71/assets/images/node-modules-structure-8ab301ddaed3b7530858b233f5b3be57.jpg)

> [https://pnpm.io/blog/2021/12/29/yearly-update](https://pnpm.io/blog/2021/12/29/yearly-update)

### yarn berry, plug n play

yarn berry 는 2020년 1월에 출시되었으며 yarn classic의 업그레이드 버전이다. yarn 팀은 본질적으로 새로운 코드 베이스와 새로운 원칙을 가진 완전히 새로운 패키지 매니저라는 것을 분명하게 하기 위해 `yarn berry`라고 부르기 시작했다.

yarn berry에서 눈여겨 봐야 할 것은 [plug n play](https://yarnpkg.com/features/pnp/)로, [node_modules를 fix 위한 전략](https://yarnpkg.com/features/pnp#fixing-node_modules)이다. node_modules를 생성하는 대신, `.pnp.cjs`라 불리는 의존성 lookup 파일이 생성되는데, 이는 중첩된 폴더 구조 대신 단일 파일 이기 때문에 더 효율적으로 처리할 수 있다. 또한 모든 패키지는 `.yarn/cache` 폴더 내부에 zip 파일로 저장되므로, node_modules 폴더보다 더 디스크 공간을 적게 차지한다.

이 모든 변화는, 릴리즈 이후에 많은 논란을 일으켰다. pnp의 breaking change는 [메인테이너들로 하여금 기존에 존재하는 패키지를 업데이트 하게 끔 만들었다.](https://blog.hao.dev/state-of-yarn-2-berry-in-2021) 새로운 pnp 방식은 default로 설정되었고, node_modules로 돌아가는 것 또한 간단하지 않았다. 이 때문에 [많은 유명한 개발자들이 yarn berry를 opt-in으로 만들지 않은 것에 대해 비판하기 시작했다.](https://www.youtube.com/watch?v=bPae4Z8BFt8)

yarn berry 팀은 이후 릴리즈에서 많은 문제를 해결하고자 노력했다. PnP의 비호환성을 해결하기 위해 default 작동 모드를 쉽게 바꾸기 위한 몇가지 방법을 제안했다. [node_modules plugin](https://github.com/yarnpkg/berry/tree/master/packages/plugin-nm)의 도움으로, 기본적인 node_modules로 돌아가는 데 한 줄의 코드만으로 가능해졌다.

[호환성 표](https://yarnpkg.com/features/pnp#compatibility-table)에서 볼 수 있듯이, 많은 대형 프로젝트 들이 점차 yarn berry를 지원하는 방향으로 가기 시작했다.

앞선 3가지 패키지 매니저 중에서 가장 최근에 나왔지만, 패키니 매니저 환경에 많은 영향을 미쳤다. 2020년말, pnpm도 plug n play 방식을 지원하기 시작했다.

## 패키지 매니저 설치하기

패키지 매니저를 사용하기 위해서는, 개발자의 로컬 혹은 CI/CD 시스템에 설치해야 한다.

### npm

nodejs 내부에 npm이 내장되어 있으므로, 추가적으로 작업을 할 필요가 없다. [nvm](https://github.com/nvm-sh/nvm)이나 [volta](https://volta.sh/)를 사용하면, node와 npm 버전을 관리하는데 매우 유용하게 쓸 수 있다.

### yarn classic

`npm i -g yarn`으로 설치하면 된다.

### yarn berry

[yarn classic에서 yarn berry로 넘어가는 방법](https://yarnpkg.com/getting-started/migration)으로 추천할만한 것은 다음과 같다.

- yarn 1.x 등 최신버전으로 업데이트
- `yarn set version berry`

[사실 추천하는 방법](https://yarnpkg.com/getting-started/install#install-corepack)은 Corepack을 사용하는 것이다.

[Corepack](https://nodejs.org/api/corepack.html)은 yarn berry 개발자에 의해 만들어진 도구로, [package manager manager](https://github.com/nodejs/TSC/issues/904) (;;;) 라는 이름으로 처음 제안되었고, node lts v16에 머지되었다.

Corepack의 도움으로 node는 yarn classic, yarn berry, pnpm의 바이너리를 shim으로 가지고 있기 때문에 npm의 대체 패키지 매니저를 별도로 설치할 필요는 없다. 이 shim을 활용하면, yarn과 pnpm 명령어를 명시적으로 설피할 필요 없이, 실행할 수 잇다.

Corepack은 nodejs@16.9.0 부터 사전 설치되며, 이전 버전에서는 `npm install -g corepack`으로 설치할 수 있다.

Corepack을 사용하기 위해서는, 먼저 활성화를 해야 한다.

```
$ corepack enable
$ corepack prepare yarn@3.1.1 --activate
```

### pnpm

pnpm 도 마찬가지 두 가지 방법으로 설치 할 수 있다.

- `$ npm i -g pnpm`
- `$ corepack prepare pnpm@6.24.2 --activate`

## 프로젝트 구조

프로젝트의 구조를 살펴보면, 각 패키지 매니저의 주요 특성을 한눈에 살펴볼 수 있다. 특정 패키지 매니저를 구성하는데 사용하는 파일과, 설치단계에서 생성되는 파일을 쉽게 알아볼 수 있다.

기본적으로, 모든 패키지 매니저는 모든 중요한 메타 정보를 `package.json`에 저장한다. 또한 루트 레벨에 설정파일을 사용하여 프라이빗 레지스트리나 dependency resolution 방법을 설정할 수 있다. 그리고 이 단계에서 dependencies를 파일 구조 (node_modules)에 저장하고 lock 파일이 생성된다.

> 이 글에서는 workspaces에 대해서는 다루지 않는다.

### npm

`$npm install` 또는 `$npm i` 명령어를 실행하면, `package-lock.json`이 생성되고 `node_modules` 폴더도 생성된다. 이 외에도 `.npmrc` 설정 파일도 생성될 수 있다.

```
.
├── node_modules/
├── .npmrc
├── package-lock.json
└── package.json
```

### yarn classic

`$yarn`을 실행하면, `yarn.lock`과 `node_modules` 폴더가 생성된다. 마찬가지로 [`.yarnrc` 파일](https://classic.yarnpkg.com/en/docs/yarnrc)도 옵셔널로 생성할 수 있다. 이에 더해 `.npmrc` 파일이 있으면 이를 이용할 수도 있다. 그리고 캐시 폴더인 `.yarn/cache/`와 현재 yarn classic의 버전을 저장하는 `.yarn/releases/`도 생성될 수 있다. 이처럼 설정에 따라서 다양하게 변경될 수 있다.

```.
├── .yarn/
│   ├── cache/
│   └── releases/
│       └── yarn-1.22.17.cjs
├── node_modules/
├── .yarnrc
├── package.json
└── yarn.lock
```

### yarn berry와 `node_modules`

install mode에 관계 없이, yarn berry 프로젝트에서는 다른 패키지 관리자보다 더 많은 파일 보다 폴더를 처리해야 한다. 일부는 선택사항이고, 그리고 일부는 필수 사항이다.

yarn berry는 더이상 `.npmrc` 다 `.yarnrc`를 사용하지 않는다. 대신 [`yarnrc.yml` 설정 파일](https://yarnpkg.com/configuration/yarnrc)을 필요로 한다. 전통적인 `node_modules`를 생성하는 워크플로우가 존재하는 경우, [nodeLinker config](https://yarnpkg.com/configuration/yarnrc#nodeLinker) 파일을 아래와 같은 형태로 제공해야 한다.

```
# .yarnrc.yml
nodeLinker: node-modules # or pnpm
```

`$ yarn`을 실행하면, 모든 의존성을 `node_modules`에 설치한다. `yarn.lock` 파일이 생성되는데, 이 파일은 기존 `yarn classic`과 호환되지는 않는다. 또한 오프라인 모드에서 설치를 위해 `.yarn/cache` 폴더도 생성된다. `releases` 폴더는 프로젝트에서 사용하는 yarn berry의 버전을 저장하기 위해 옵셔널로 생성된다.

```
.
├── .yarn/
│   ├── cache/
│   └── releases/
│       └── yarn-3.1.1.cjs
├── node_modules/
├── .yarnrc.yml
├── package.json
└── yarn.lock
```

### yarn berry with pnp

PnP 모드에는 [strict](https://yarnpkg.com/features/pnp)와 [loose](https://yarnpkg.com/features/pnp#pnp-loose-mode) 모드가 있는데, 일단은 모드에 상관없이 `yarn`을 실행하면 `.yarn/cache`와 `.yarn/unplugged`, `.pnp.cjs` `yarn.lock` 파일이 생성된다. strict 모드는 기본 값이고, loose는 아래 처럼 옵셔널로 설정해두어야 한다.

```
# .yarnrc.yml
nodeLinker: pnp
pnpMode: loose
```

PnP 프로젝트에서, `.yarn/` 폴더 내부에는 `release/`외에도 [ide 지원](https://yarnpkg.com/getting-started/editor-sdks)을 위한 `sdk/` 폴더를 포함할 가능성이 높다. [이외에도 사용례에 따라서, 다양한 폴더들이 생성될 수 있다.](https://yarnpkg.com/getting-started/qa#which-files-should-be-gitignored)

```
.
├── .yarn/
│   ├── cache/
│   ├── releases/
│   │   └── yarn-3.1.1.cjs
│   ├── sdk/
│   └── unplugged/
├── .pnp.cjs
├── .pnp.loader.mjs
├── .yarnrc.yml
├── package.json
└── yarn.lock
```

### pnpm

`pnpm`도 다른 패키지 매니저와 마찬가지로 `package.json` 이 필요하다. `$ pnpm i`를 실행하면, `node_modules` 가 생성되는 것 까지는 다른 패키지 관리자와 동일하지만, 앞서 언급한 `content-addressable storage approach`라는 특성 때문에 이후의 구조가 완전히 다르다.

pnpm은 자체 lock 파일인 `pnp-lock.yml`을 생성한다. 그리고 마찬가지로 `.npmrc`로 설정을 추가할 수도 있다.

## Lock 파일과 dependency 저장

앞서 언급한 것 처럼, 모든 패키지 매니저는 각자 다른 형태의 lock 파일이 존재한다.

일단 lock 파일의 정의를 먼저 살펴보면, lock 파일이란 매 설치시 결정적이고 (= 항상 같은 버전을 설치하고) 예측가능한 특성을 보장하기 위하여, 각 버전의 정확한 의존성 버전을 저장하고 있는 파일을 의미한다. `package.json`은 정확한 버전이 기재되어 있는 것이 아니고, `>= 1.2.5`와 같은 형식의 [버전 범위 aka 시멘틱 버저닝](https://docs.npmjs.com/about-semantic-versioning)이 존재하기 때문에, lock파일이 없다면 매 설치마다 설치하는 버전이 달라질 수 있다.

lock 파일은 또한 체크섬이 존재하는데, 이에 대해서는 보안 관련 섹션에서 다룬다.
