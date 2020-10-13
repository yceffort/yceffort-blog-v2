---
title: eslint-config-yceffort, 나만의 eslint-config 만들기
tags:
  - javascript
  - typescript
published: true
date: 2020-09-15 11:07:10
description: '나만의 일관된 javascript code를 위하여 만들어보았습니다.'
category: npm
template: post
---

전 회사에서 자체적으로 만든 `eslint-config-***`를 쓰고 있었는데, private 레파지토리에 있어서 내 public 레파지토리에 적용해서 쓰는데에 어려움이 있었다. 1년간 쓰면서 자체적으로 정한 규칙도 맘에 들었고, 만들어 주신 분께서 꽤나 많은 공을 쏟아 주셔서 정말 잘 쓸 수 있었다. 그래서 이와 거의 흡사한 룰을 가진 나만의 `eslint-config-yceffort` 를 만들어서 써보기로 했다. 룰은 물론 거의 비슷하지만, 갖다 배낄 수는 없는 노릇이고 - 이미 퇴사해서 코드는 없으므로 기억나는 룰을 최대한 비슷하게 맞춰보았다.

## 1. eslint-config-\*\*\* 만드는 법

만드는 방법은 https://tech.kakao.com/2019/12/05/make-better-use-of-eslint/ 여기에 잘나와 있어서 따로 자세히 포스팅 하지 않으려고 한다. 분명히 예전에 다닐 때는 저런게 없었던 것 같은데 🤔 어느 틈엔가 만들어 쓰고 있었나보다.

## 2. github npm registry를 쓰고 싶었지만...

github의 리치한 대부분의 기능, 단순 소스 관리 부터 workflows 에 이르기 까지 모든 기능들을 쓰는데 심취하면서, 이 package registry 까지 github에서 사용해보고 싶었다. https://github.com/features/packages

결론부터 말하자면 그러지 못했다.

https://github.com/yceffort/eslint-config-yceffort/packages

패키지를 올리는 것은 꽤나 단순하지만, 사용하는 입장에서 `.npmrc`에 `registry`를 아래 처럼 별도로 등록해줘야 하는 허들이 있었다. https://docs.github.com/en/packages/using-github-packages-with-your-projects-ecosystem/configuring-npm-for-use-with-github-packages

```
registry=https://npm.pkg.github.comOWNER
@OWNER:registry=npm.pkg.github.com
@OWNER:registry=npm.pkg.github.com
```

어차피 나 밖에 쓸일이 없으므로 별다른 허들이 되지 않겠지만서도 (...) 매번 만드는 나의 레파지토리에 한단계라도 허들을 낮추고자 그냥 npm registry를 쓰기로 했다.

https://www.npmjs.com/package/eslint-config-yceffort

## 3. 버전 관리의 중요성

최초의 버전은 0.01 이었는데, 몇가지를 `README.md`에 잘못써서 그것만 따로 커밋 푸쉬했더니, github의 README와 npm의 READEME가 다른 사태가 발생했다.

- https://github.com/yceffort/eslint-config-yceffort
- https://www.npmjs.com/package/eslint-config-yceffort

허허~~

## 4. prettier의 일부 기능을 끄고 싶은데..

`mathjax`를 사용하기 위해서 `$$..$$` 문법을 쓰고 있는게 있었다. 근데 이걸 뭔가 계속 escape 처리를 해서.. 뭔가 수정할 방법이 있는 것 같은데 귀찮아서 다음으로 미뤘다.

## 5. 결론

https://www.npmjs.com/package/eslint-config-yceffort

많은 이용 부탁드립니다.
