---
title: 패키지 가이드 3탄) 패키지 제작을 위한 가이드
marp: true
paginate: true
theme: default
tags:
  - npm
  - bundler
  - web-performance
date: 2025-03-28
description: '🤔'
published: true
art:
  undraw: guidelines
---

# 패키지 가이드 3탄) 패키지 제작을 위한 가이드

<!-- _class: invert deck-cover -->

@yceffort

---

## 가능한 파일 구조와 빌드 구조는 유지하는 것이 좋다

<style scoped>section { font-size: 22px; }</style>

- 일반적인 개발자라면 파일 구조를 부수효과를 최소화 하는 조건으로 격리해둘 것임
- 이 구조를 빌드 결과물에 그대로 반영한다면, 파일별로 부수효과를 격리할 수 있고, 트리쉐이킹을 원활하게 할 수 있으며, 나아가 `sideEffects` 필드를 적절하게 활용하는 것도 가능
- 다만 다음과 같은 예외도 존재
  - 배럴파일로 내보내진 CommonJS 의 경우: 배럴파일에서 이미 `require`가 발생하였고, 그 순간 부터 정적 분석이 불가능해지므로 파일구조를 쪼개는 것이 의미가 없어짐. 단, `date-fns/something`과 같이 하위 디렉토리로 나누어져 있다면 파일별로 쪼개는 것이 의미가 있음
  - 브라우저에서 직접 불러와서 사용하는 경우: (<https://unpkg.com/react@18/umd/react.production.min.js>)
    - 파일이 쪼개져있다면 그만큼 요청 수가 증가하여 오버헤드가 발생
    - CDN 은 정적 자원이므로 파일 하나로 만들어두면 한번의 캐싱만으로 사용 가능
    - 모든 자원을 하나로 합쳐두면, 중복된 토큰, 변수명, 코드 패턴 등을 효과적으로 압축할 수 있음

---

## 코드 난독화 (압축)은 선택적으로, 필수는 아님

<style scoped>section { font-size: 22px; }</style>

- `Terser`, `ESBuild` 등을 사용하면 간단한 설정만으로도 코드를 90% 이상 압축할 수 있어 다운로드 속도가 빨라지고 코드를 최적화 할 수 있음
- constant inlining 과 같은 고급 기능은 서비스 번들러인 웹팩에서는 잘되지 않고, 롤업 등 고오급 번들러만 가능하므로 어느 정도 라이브러리가 미리 압축해주는 것도 좋은 전략이 될 수 있음
- 그러나 난독화 및 압축이 꼭 좋은 것 만은 아님
  - 일반적인 압축은 서비스 번들러에서 이미 대부분 수행하므로, 굳이 미리 할 필요가 없음
  - 코드가 난독화 되어 있으면 서비스에서 발생한 패키지 문제를 디버깅하기 매우 까다로움
- 압축이 필요한 경우
  - 보안 민감도가 높은 패키지 (물론 이건 어디까지나 예방차원일 뿐, 100% 방지는 절대안됨)
  - 브라우저에서 직접 불러와서 사용되는 패키지 (CDN)
  - 사용자가 굳이 코드를 따라서 읽지 않아도 되거나 읽을 필요가 없는 패키지 (like 리액트)

---

## 프레임워크는 절대 패키지에 포함시키면 안된다

- 리액트, 뷰와 같은 프레임워크 패키지, 혹은 그에 준하는 패키지들은 절대 패키지에 포함시키면 안됨
- 이 밖에도 서비스에 '단 하나' 만 설치되기 원하는 패키지의 경우에도 마찬가지
- `peerDependencies`를 사용하여 해당 패키지가 프레임워크에 의존하고 있다는 것을 선언하는 것 만으로도 충분

---

## 모던 브라우저를 타겟으로 빌드

- 지원해야할 폴리필이 많고 트랜스파일이 많아진다는 것은 그만큼 패키지의 성능을 키우는 셈
- 사용자 중 매우 '일부' 가 사용하는 낮은 브라우저를 위해서 '대다수' 의 사용자에게 성능 저하를 주는 것은 바람직하지 않음
- 꼭 필요하다면 모던 브라우저 용, 리거시 브라우저 용 으로 나눠서 제공하는 것이 바람직
- `browserslist` 및 `tsconfig`의 `target` 필드를 사용하여 가능한 최신 브라우저를 타겟으로 빌드하고, 패키지를 설치하는 사용자들로 하여금 이 패키지가 어떤 브라우저를 타겟으로 만들어 졌는지 명시적으로 알 수 있도록 하는 것이 좋음
  - <https://ko.legacy.reactjs.org/docs/javascript-environment-requirements.html>

---

## CDN 으로 제공하는 것을 고려한다면

<style scoped>section { font-size: 18px; }</style>

- 대표적인 CDN인 [unpkg](https://unpkg.com/) 와 [jsdelivr](https://www.jsdelivr.com/) 를 사용하여 패키지를 제공하는 것을 고려해보는 것도 좋음
- 두 CDN 에서 패키지를 제공하고 싶다면, `package.json`에 다음과 같이 선언해주면 됨

  ```json
  {
    "unpkg": "./dist/index.umd.js",
    "jsdelivr": "./dist/index.umd.js"
  }
  ```

  ```js
  ;(function (c, x) {
    if (typeof exports === 'object' && typeof module !== 'undefined') {
      // ✅ CommonJS 환경 (예: Node.js)
      x(exports)
    } else if (typeof define === 'function' && define.amd) {
      // ✅ AMD 환경 (예: RequireJS)
      define(['exports'], x)
    } else {
      // ✅ 전역 환경 (예: <script> 태그로 직접 로딩하는 브라우저)
      c = c || self
      x((c.React = {}))
    }
  })(this, function (c) {
    // 여기부터 실제 React 구현
  })
  ```

---

## CSS 파일을 분리하기

- bootstrap, tailwindcss 와 같은 css 라이브러리는 모든 기능을 가지고 있는 하나의 CSS 번들을 제공하는 것이 더 쉬울 수도 있음
- 그러나 CSS 파일이 하나로 합쳐져 있다면, 사용자가 사용하지 않는 CSS 도 함께 제공되서 사잌즈가 매우 커질 수 있음
- 그래서 일반적인 경우에는 다음과 같이 원하는 스타일만 불러올 수 있도록 함
  - [bootstrap](https://getbootstrap.com/docs/5.2/customize/optimize/)
  - [tawilwindcss](https://v3.tailwindcss.com/docs/optimizing-for-production)
- 만약 CSS 가 패키지의 일부 정도 수준이라면, 각 컴포넌트마다 CSS 를 따로 분리해서 별도로 제공하고, 컴포넌트를 사용할 때 마다 해당 스타일을 불러오도록 설정하는 것이 이상적
