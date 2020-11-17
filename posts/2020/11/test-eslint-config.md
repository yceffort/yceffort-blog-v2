---
title: 'eslint-config 를 위한 테스트 코드를 작성하기 (CI)'
tags:
  - javascript
published: true
date: 2020-11-03 18:03:54
description: 'eslint config 테스트 코드 작성'
---

[eslint-config-yceffort](https://github.com/yceffort/eslint-config-yceffort)를 사용하면서 개인적으로 굉장히 만족도가 높아졌다. 하지만 한가지 아쉬운 것은 테스트 코드가 없다는 것과, 배포를 할 때 별도의 절차 없이 내 로컬에서 그때 그때 수동으로 하고 있다는 것이었다. 여기에도 CI CD절차가 있다면 좋다고 생각했다.

## 방법1

간단한 방법으로는, 올바른 코드를 작성한다음에, 해당 코드를 `test`시에 `eslint`를 돌리는 방법이다.

`./tests/.eslintrc`

```json
{
  "extends": ["../index"],
  "rules": {
    // your custom rules..
  }
}
```

`camelcase.test.js`

```javascript
const hello_world = 'hello_world'
console.log(hello_world)
```

나의 eslintrc 옵션에서는 `camelcase`가 off로 되어 있고, 정상적으로 off가 되어 있다면 test 시에 eslint 에러가 나지 않을 것이다.

하지만 이 방법은 아쉽게도, lint가 정상적으로 작동하는지에 대해서만 알 수 있을 뿐, eslint가 실패시 원하는 에러가 뜨는지, 경고가 떴을 경우에는 해당 경고가 뜨는지 까지는 알 수 없다.

## 방법2

확실한 방법은 [eslint-nodejs-api](https://eslint.org/docs/developer-guide/nodejs-api)를 사용하는 것이다. nodejs api 레벨에서 테스트를 해보고, 그 결과를 가지고 좀더 정확히 분석하는 방법이다.

```javascript
const { ESLint } = require('eslint')
const config = require('../../../index')
const {
  rules: { curly },
} = require('../../../rules/style')

const RULE_ID = 'curly'

const eslint = new ESLint({
  // 특정 룰만 가져온 이유는, 다른 룰로 인해서 에러가 나는 것을 방지하기 위해서다.
  // 즉 순수하게 테스트 하고 싶은 룰에 대해서만 룰을 집어 넣었다.
  baseConfig: { ...config, rules: { curly } },
})

describe('eslint-config-yceffort curly', function () {
  it('right curly', async function () {
    const [result] = await eslint.lintFiles([`${__dirname}/curly.right.js`])

    const errors = result.messages.some((message) => message.ruleId === RULE_ID)

    // 올바른 케이스이기 때문에 true로 비교 하고 싶어서 이렇게 했다.
    expect(!errors).toBe(true)
  })

  it('wrong curly', async function () {
    const [result] = await eslint.lintFiles([`${__dirname}/curly.wrong.js`])

    const errors = result.messages.some((message) => message.ruleId === RULE_ID)

    // 잘못된 케이스이기 때문에 false로 비교 하고 싶어서 이렇게 했다.
    expect(!errors).toBe(false)
  })
})
```

`curly.wrong.js`

```javascript
if (true) {
  if (true) console.log('hello')
}
```

`curly.right.js`

```javascript
if (true) {
  for (let i of [1, 2, 3, 4, 5]) {
    console.log(i)
  }
}
```

github workflow 결과: https://github.com/yceffort/eslint-config-yceffort/runs/1345142937?check_suite_focus=true

## 결론

방법1은 작성하기 간단한 반면, 아주 정확하게 거를 수가 없다는 단점이 있고, 방법2는 작성하기엔 빡세지만 원하는 만큼 다양한 케이스에 대해서 테스트 코드를 작성할 수 있다는 장점이 있다.

방법2로 우리 조직의 `eslint-config`도 관리해보고 싶었지만, test case 작성이 너무 어렵다는 이유로 방법1을 선택했다. 🤔 (서운하지 않습니다.) 아무도 안쓰는 내 `eslint-config-yceffort`는 저렇게 관리해봐야겠다.