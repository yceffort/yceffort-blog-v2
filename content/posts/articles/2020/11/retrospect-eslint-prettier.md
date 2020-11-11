---
title: 'eslint, prettier, editorconfig 로 코드 컨벤션을 맞춘 후기'
tags:
  - javascript  
  - eslint
published: true
date: 2020-11-11 22:06:31
description: '예민이가 된 기분'
---

`eslint-config-***` 시리즈를 만들면서 몇 가지 배운 것, 그리고 잊지 않기 위해 기억해야 할 것을 요약해 둔다.

https://github.com/yceffort/eslint-config-yceffort

## eslint 와 prettier의 충돌

`eslint`와 `prettier`를 적요한 사람들이 가장 많이 겪는 문제 중 하나는, vs code 등 에디터에서 `eslint`를 적용했는데, 룰이 서로 충돌을 한다는 문제였다.

대표적인 예가 `indent`룰 인데, 놀랍게도 `eslint`와 `prettier`모두 indent 룰이 존재한다.

- `eslint`: https://eslint.org/docs/rules/indent
- `prettier`: https://prettier.io/docs/en/options.html#tab-width

예를 들어, 나는 2칸을 규칙으로 지정해서 하고 싶어서, 아래와 같이 두개 다 적용했다고 가정해보자.

```javascript
rules: {
  indent: ['error', 2],
  'prettier/prettier': ['error', { tabWidth: 2 }],
},
```

![indent1](./images/indent1.png)

![indent2](./images/indent2.png)

뭔가 둘이 똑같은 일을 하는 `indent`와 `tabWidth`의 동작이 미묘하게 다른 것을 알 수 있다.

https://github.com/eslint/eslint/issues/10930

> @jlchereau ESLint's indent rule and Prettier's indentation styles do not match - they're completely separate implementations and are two different approaches to solving the same problem ("how do we enforce consistent indentation in a project").

> When using Prettier, you shouldn't be using ESLint's indent rule at all. In your current configuration, the prettier config disables indent and then you're turning it back on using "indent": ["error", 4, { "SwitchCase": 1 }], in the rules property of your config. I don't use eslint-plugin-prettier, but I believe you should be letting that rule handle any warnings about indentation.

이러한 `indent`룰 이외에도 충돌하는 룰이 몇가지 있다.

이런 경우에는 `prettier`가 할 수 있는 일은 `prettier`에게 모두 맡기고, 관련 `eslint`룰은 모두 끄는 것이 좋다.


https://github.com/prettier/eslint-config-prettier

> Turns off all rules that are unnecessary or might conflict with Prettier. ;
> This lets you use your favorite shareable config without letting its stylistic choices get in the way when using Prettier.

## 특정 라이브러리 import 방지

`lodash`는 다양한 기능을 제공하는 좋은 라이브러리이지만, 여기저기서 언급한 것처럼 - 오래된 설계 방식으로 인해 잘못 import 를 하면 트리쉐이킹이 되지 않는다.

```javascript
// 설정이 잘 되어있어도 lodash 모든 것들을 가져온다.
import { sortBy } from "lodash";

// sortBy 경로에서 가져온다.
import sortBy from "lodash-es/sortBy";
```

https://yceffort.kr/2020/07/how-commonjs-is-making-your-bundles-larger

https://yceffort.kr/2020/07/how-commonjs-is-making-your-bundles-larger

이것을 [no-restricted-imports](https://eslint.org/docs/rules/no-restricted-imports)로 막을 수 있다.

```javascript
"no-restricted-imports": [
    "error", 
    {
    "name": "lodash",
    "message": "lodash has been prohibited due to bundle size. use lodash-es instead."
    }
]
```

![no-restricted-imports](./images/no-restricted-imports.png)

만약 기존에 이미 `no-restricted-imports` 룰이 있다면, 별개의 `eslintrc`룰을 만들어서 `eslint`를 한번 더 돌리면 된다. 뭔가 이 룰을 고도화 하자는 issue도 있었는데, `eslint`팀에서 복잡성을 이유로 거절했었다.

## husky, lint-staged 

`prettier`와 `lint`에 대한 점검은 보통 CI 단계에서 하는 경우가 많은데, 코드를 커밋하기 전에도 할 수 있다.

- `husky`: https://github.com/typicode/husky
- `lint-staged`: https://github.com/okonet/lint-staged

두 라이브러리를 devDependencies에 설치해두고, `package.json`에 아래와 같이 추가해주면 된다.

```json
{
   "husky": {
    "hooks": {
      "pre-commit": "lint-staged -q"
    }
  },
  "lint-staged": {
    "**/*.{js}": [
      "eslint",
      "git add"
    ]
  },
}
```

코드의 양이 많아질수록, git add 하는 과정이 버벅일 수 있으므로, 이는 적당히 고려해보기만 하면 될 것 같다.

## 취향을 타는 룰은 변경보다는 현상 유지가 낫다.

나는 대표적인 80width 2indent 파인데, 이는 굉장히 개인적인 취향의 영역이다. ([물론 prettier는 합리적인 선택이라고 주장하지만 서도](https://prettier.io/docs/en/options.html#print-width)) 이렇게 취향을 타는, 그리고 전체적인 코드 베이스를 수정해야 하는 룰은 그대로 두는게 낫다. 

고치는게 어렵다거나, 수정이 많아진다거나 하는 이유가 아니고, 전반적인 코드의 history를 오염시킬 수 있기 때문이다. 온갖 코드에 히스토리가 바로 보이지 않고, 직전 히스토리에 `변경된 eslint 룰 적용`이라는 커밋 메시지만 잔뜩 남아있으면 git blame 하기가 여간 쉽지 않을 것이다.

## prettier 1.0과 2.0 사이에 default 값의 변화가 있다.

이건 내가 ~~멍청해서~~ 잘 못 찾고 삽질하던 영역인데, 1.0에서 2.0으로 업그레이드 하면서 기본값에서 몇가지 변화가 있었다. (내가 헤매던건 `trailing commas`) 

https://prettier.io/docs/en/options.html

1.0에서 버전업 할 때 이를 잘 살펴볼 필요가 있다.

## react/exhaustive-deps는 죄가 없다.

리액트 코드들을 많이 살펴보면 이 `react/exhaustive-deps`룰을 warning처리 해놓거나, 혹은 eslint-disable을 도배해놓은 걸 볼 때가 많다. 

경험적으로 봤을 떄, 그리고 많은 아티클을 참고 했을 때 이룰은 `error` 로 두고, 필요할 떄만 `eslint-disable-line` 을 해두는 것이 좋다.

이 논쟁에 대한 많은 글들이 있기 때문에,, 더이상 설명하는 것은 생략..

- https://reactjs.org/docs/hooks-faq.html#is-it-safe-to-omit-functions-from-the-list-of-dependencies
- https://github.com/facebook/react/issues/14920
- https://yceffort.kr/2020/10/think-about-useEffect

## camelcase

나는 camelCase와 PascalCase를 사용하는 것을 좋아한다. 그리고 대부분의 자바스크립트 라이브러리들이 둘 중 하나으로 작성되어 있다. 그러나 가끔 라이브러리들을 보면 snake_case로 작성되어 있는 경우가 있는데, 이 camelcase룰이 잘 되어 있어서 쓰는데 많은 도움이 되었다. destructuring시에 무시하거나, 혹은 특정 string에 대해서는 예외로 설정해 둘 수 있다.

https://eslint.org/docs/rules/camelcase

## .editorconfig도 쓰자.

[.editorconfig](https://editorconfig-specification.readthedocs.io/en/latest/)는 다양한 편집기와 IDE에서 작엏바는 여러 개발자들을 일관된 코딩스타일로 묶어주는데 도움을 준다. 따라서 이것도 프로젝트 코딩 컨벤션과 맞춰서 작성해서 제공하는게 좋다. 

`.editorconfig`는 편집기 그 자체를 코딩 컨벤션에 맞춰서 자동으로 구성되게 해주기 때문에, `editorconfig` `prettier` `eslint`이 모든 것을 사용하는 것이 모두에게 일관된 코딩 스타일을 제공하는데 효과적이다.

아래는 내 전용 `.editorconfig`다.

https://github.com/yceffort/eslint-config-yceffort/blob/master/.editorconfig

```yaml
# http://editorconfig.org
root = true

[*.{js,ts,tsx}]
charset = utf-8
end_of_line = lf
indent_style = space
indent_size = 2
insert_final_newline = true
max_line_length = 80
trim_trailing_whitespace = true

[*.{json,yml,yaml}]
charset = utf-8
end_of_line = lf
indent_style = space
indent_size = 2
insert_final_newline = true
trim_trailing_whitespace = true
```

https://stackoverflow.com/questions/48363647/editorconfig-vs-eslint-vs-prettier-is-it-worthwhile-to-use-them-all

## 오타도 주의 하자.

eslint와는 조금 다른 얘기지만, 오타도 점검내지는 고쳐줬으면 좋겟다.

우리는 영어권 민족이 아니기 때문에, 영문 오타를 낼 가능성이 많다. 물론 영어권 사람들도 그렇겠지만. 그래서 대규모 프로젝트를 진행하다보면, 많은 사람들이 실수로 저질러 놓은 영어 오타들을 많이 보게 된다.

이를 점검할 수 있는 방법엔 여러가지가 있다.

- https://github.com/aotaduy/eslint-plugin-spellcheck
- https://github.com/TypoCI/spellcheck-action
- https://github.com/marketplace/actions/check-spelling

다양한 방법이 있지만, 개인적으로는 [vscode의 code spell checker](https://marketplace.visualstudio.com/items?itemName=streetsidesoftware.code-spell-checker)를 사용해서 내 코드 - 내지는 내 담당 PR만이라도 보고 있다.

왜냐하면, 아무래도 오타를 죄다 `error` 내지는 `warning`으로 띄운다면 `naver`나 `daum`같은 것도 죄다 오타로 걸리고, 그래서 이 예외적인 표현을 예외 목록에 추가하고, 그러다보면 예외목록에 갖가지 단어가 추가되는 등의 번거로움이 있을 것 같아서 적극적인 적용에는 망설이고 있다.

변수나 함수명의 귀여운 오타들이 코드의 성능이나 생산성에 영향을 미치는 것은 아니라서 모두 고칩시다! 라고 주장하기엔 무리가 있지만, 아무래도 거슬리는 건 어쩔 수가 없다. 나라도 조심하자.

## `warn`은 결국 `off`나 `error`로 가야 한다.

`warn`으로 설정하는 이유의 대부분은 

> 문제가 있는건 알지만 일단 나중에 한꺼번에 고치자
> 문제인건 알지만 나중에 다시 논의 해보자
> 문제인건 알지만 에러까지는 아니고, 알아서 조심하자

정도로 나눠 볼 수 있을 것 같다. 그러나 이 `warn`을 장기간 방치해두면 그 경고가 쌓이게 되어, 정작 코드 컨벤션을 맞추기 위한 `eslint`의 의미를 퇴색시키고, 다른 경고들도 볼 수 없게 된다. 

개인적인 생각으로는, `warn`으로 할 바엔 `off`로 하든, 혹은 `error`로 두고 예외적인 부분만 disable 처리하고 코멘트를 달아두는게 나은 것 같다. `warn`은 어디까지나 임시일뿐, 결국엔 `eslint`의 이점을 누릴 수 없게 만든다.

## AST 에 대한 이해

`eslint`를 만들면서 병적으로 사소한 룰을 추가하고 빼보고 카나리 배포해보고 적용해보고 fix해보고 commit 해보고 충돌나고 다시 고치고를 반복하면서 배운 것 이외에, 한가지 배운 것이 있다면 AST(Abstract Syntax Tree) 에 대한 이해였다.

eslint 는 아래와 같은 순서로 동작한다.

1. javascript 코드를 읽는다
2. parser로 AST를 만든다.
  - 여기서 parser는 `Espree` `babel-eslint`등이 존재한다. 
  - AST는 소스코드의 구조를 트리 형태로 표현한 것이다. 
3. linter와 rule을 활용하여 AST를 검사한다.
4. 검사결과에 따라서 에러를 뱉거나 고친다.

```javascript
var hello = 'hello'
console.log(hello)
```

이 parser를 거치면 아래와 같은 AST 형태로 바뀌게 된다.

```json
{
  "type": "Program",
  "start": 0,
  "end": 38,
  "body": [
    {
      "type": "VariableDeclaration",
      "start": 0,
      "end": 19,
      "declarations": [
        {
          "type": "VariableDeclarator",
          "start": 4,
          "end": 19,
          "id": {
            "type": "Identifier",
            "start": 4,
            "end": 9,
            "name": "hello"
          },
          "init": {
            "type": "Literal",
            "start": 12,
            "end": 19,
            "value": "hello",
            "raw": "'hello'"
          }
        }
      ],
      "kind": "var"
    },
    {
      "type": "ExpressionStatement",
      "start": 20,
      "end": 38,
      "expression": {
        "type": "CallExpression",
        "start": 20,
        "end": 38,
        "callee": {
          "type": "MemberExpression",
          "start": 20,
          "end": 31,
          "object": {
            "type": "Identifier",
            "start": 20,
            "end": 27,
            "name": "console"
          },
          "property": {
            "type": "Identifier",
            "start": 28,
            "end": 31,
            "name": "log"
          },
          "computed": false
        },
        "arguments": [
          {
            "type": "Identifier",
            "start": 32,
            "end": 37,
            "name": "hello"
          }
        ]
      }
    }
  ],
  "sourceType": "module"
}
```

이러한 구조를 이해해야 `eslint`가 어떻게 동작하는지 알수 있으며, 나아가 단순히 룰을 조합하는 `eslint-config`뿐만 아니라 내가 직접 custom rule을 만들 수도 있다.

