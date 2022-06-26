---
title: '나만의 eslint 룰 만들어보기'
tags:
  - javascript
  - eslint
published: true
date: 2022-06-26 20:52:14
description: 'rust로 eslint를 만들어도 재밌겠네용'
---

## Table of Contents

## Introduction

react@17 이 업데이트 되면서 더이상 `jsx, tsx` 파일에 `import React`를 할 필요가 없어졌다. [참고](https://ko.reactjs.org/blog/2020/09/22/introducing-the-new-jsx-transform.html) 이를 사용함으로써 여러가지 이점이 있지만, 무엇보다 번들 사이즈가 줄어든 다는 장점이 가장 크다. (아주 작은 정도지만)

그러나 기존 react@16 기반의 코드에서 저 `import React from 'react'` 코드를 모두 제거하기란 쉽지 않다. `import React from 'react'`를 모두 찾고 검색해서 지우는 방법도 있겠지만, 저 사이에 무엇이라도 껴 있다면, (`import React, { MouseEvent } from 'react'` 와 같이) 이 방법도 소용이 없다. 그래서 어떻게 해결할까 고민하던 중, `eslint`가 있으니 이를 활용하면 쉽게 해결할 수 있지 않을까 하는 아이디어가 떠올랐다.

## `no-restricted-imports`를 쓰는 방법

아마도 대부분의 프로젝트에서는 eslint를 사용 중일 것이다. 그래서 eslint에 있는 기본 룰인 [no-restricted-imports](https://eslint.org/docs/latest/rules/no-restricted-imports)를 사용해서 해결해보자.

```javascript
module.exports = {
  rules: {
    'react/react-in-jsx-scope': ['off'],
    'no-restricted-imports': [
      'error',
      {
        paths: [
          {
            name: 'react',
            importNames: ['default'],
            message: "import React from 'react' makes bundle size larger.",
          },
        ],
      },
    ],
  },
}
```

`react` 라는 import가 있고, 이 importNames이 기본값 (`React`)일 경우 에러메시지를 띄우는 방법이다. 이방법을 활용하면 같은 원리로 [트리쉐이킹이 안되는 `lodash`](/2021/08/javascript-tree-shaking#%EB%AC%B4%EC%97%87%EC%9D%84-%ED%95%B4%EC%95%BC%ED%95%A0%EC%A7%80-%EA%B0%90%EC%9D%B4-%EC%98%A4%EC%A7%80-%EC%95%8A%EC%9D%84-%EB%95%8C) import 하는 것을 막을 수 있다.

> 하지만 아쉽게도 이 방법은 자동으로 fix 까지 해주지 않는다. 물론 자동으로 import를 해서 fix 할 수도 있겠지만, 그것보다는 개발자가 직접 수정하는 것이 더 안전할 것이다.

## eslint 룰 만들기?

이 방법으로 문제를 해결하긴 했지만, 갑자기 궁금했졌다. 내가 직접 관련된 문제를 해결할 수 있는 rules을 만들어 볼 순 없을까? 🧐

### eslint 동작 방식 이해

eslint 의 동작방식을 이해하기 위해서 알아야 하는 단 한가지는 바로 [AST](/2021/05/ast-for-javascript)다. 이 글을 요약해서 설명하자면, AST는 우리가 작성한 코드를 기반으로 트리 구조의 데이터 스트럭쳐를 만들어 낸다. 즉, eslint 는 코드를 AST를 활용해서 트리구조를 만든 다음, 여기에서 지적하고 싶은 코드를 만들어서 룰로 저장하는 것이다.

### 간단한 예제

먼저, 한 글자 짜리 변수를 막는 룰을 만든다고 가정해보자. https://astexplorer.net/ 에서 변수 선언문 트리를 만들면, 아래와 같은 결과를 얻을 수 있다.

```javascript
const hello = 'world'
```

그럼 아래와 같은 트리를 확인할 수 있다.

```json
{
  "type": "Program",
  "start": 0,
  "end": 21,
  "loc": {
    "start": {
      "line": 1,
      "column": 0
    },
    "end": {
      "line": 1,
      "column": 21
    }
  },
  "range": [0, 21],
  "errors": [],
  "comments": [],
  "sourceType": "module",
  "body": [
    {
      "type": "VariableDeclaration",
      "start": 0,
      "end": 21,
      "loc": {
        "start": {
          "line": 1,
          "column": 0
        },
        "end": {
          "line": 1,
          "column": 21
        }
      },
      "range": [0, 21],
      "declarations": [
        {
          "type": "VariableDeclarator",
          "start": 6,
          "end": 21,
          "loc": {
            "start": {
              "line": 1,
              "column": 6
            },
            "end": {
              "line": 1,
              "column": 21
            }
          },
          "range": [6, 21],
          "id": {
            "type": "Identifier",
            "start": 6,
            "end": 11,
            "loc": {
              "start": {
                "line": 1,
                "column": 6
              },
              "end": {
                "line": 1,
                "column": 11
              },
              "identifierName": "hello"
            },
            "range": [6, 11],
            "name": "hello",
            "_babelType": "Identifier"
          },
          "init": {
            "type": "Literal",
            "start": 14,
            "end": 21,
            "loc": {
              "start": {
                "line": 1,
                "column": 14
              },
              "end": {
                "line": 1,
                "column": 21
              }
            },
            "range": [14, 21],
            "value": "world",
            "raw": "\"world\"",
            "_babelType": "Literal"
          },
          "_babelType": "VariableDeclarator"
        }
      ],
      "kind": "const",
      "_babelType": "VariableDeclaration"
    }
  ]
}
```

그리고 룰을 작성하기에 앞서, 먼저 룰이 포함되어 있는 프로젝트를 하나 만들어야 한다. (`npm init`) 그리고 중요한 것은, `eslint-plugin-`으로 시작해야 한다.

그리고 `index.js`를 만들고 다음과 같이 파일을 만들어보자.

```javascript
module.exports = {
  rules: {
    // 룰 이름을 선언한다.
    'variable-length': (context) => ({
      // 변수 선언하는 부분은 VariableDeclarator 이다.
      VariableDeclarator: (node) => {
        // 변수명은 여기에 있다. (위 json 참고)
        if (node.id.name.length < 2) {
          context.report(
            node,
            `Variable names should be longer than 1 character`,
          )
        }
      },
    }),
  },
}
```

그리고 해당 룰을 적용해보자

```bash
/workspaces/eslint-plugin-import-yceffort/test/index.js
  3:7   warning  Variable names should be longer than 1 character VariableDeclarator  yceffort-rules/var-length
```

와...!!!

이번에는 옵션을 주어서, 특정 한글자 짜리 변수는 허용하도록 해보자. 예를 들어서 `_`와 같이.

```javascript
module.exports = {
    'var-length': (context) => ({
      VariableDeclarator: (node) => {
        const { options } = context
        const allowedList = options.find((opt) => 'allowed' in opt)
        const allowed = allowedList.allowed || []

        if (node.id.name.length < 2 && !allowed.includes(node.id.name)) {
          context.report(
            node,
            `Variable names should be longer than 1 character ${node.type}`,
          )
        }
      },
    }),
  },
}
```

```javascript
const rootRule = require('../.eslintrc.js')

module.exports = {
  ...rootRule,
  plugins: ['yceffort-rules'],
  rules: {
    'yceffort-rules/var-length': ['warn', { allowed: ['_'] }],
  },
}
```

### import React 만들어보기

원래 글의 목적이었던, `import React from "react"`나 `import React from "lodash"`와 같은 default import 를 막는 rule을 만들어보자.

먼저 AST Explorer로 트리구조를 살펴보자.

```javascript
import React, { MouseEvent } from 'react'
import lodash from 'lodash'
```

```json
{
  "type": "Program",
  "start": 0,
  "end": 69,
  "loc": {
    "start": {
      "line": 1,
      "column": 0
    },
    "end": {
      "line": 2,
      "column": 27
    }
  },
  "comments": [],
  "range": [0, 69],
  "sourceType": "module",
  "body": [
    {
      "type": "ImportDeclaration",
      "start": 0,
      "end": 41,
      "loc": {
        "start": {
          "line": 1,
          "column": 0
        },
        "end": {
          "line": 1,
          "column": 41
        }
      },
      "specifiers": [
        {
          "type": "ImportDefaultSpecifier",
          "start": 7,
          "end": 12,
          "loc": {
            "start": {
              "line": 1,
              "column": 7
            },
            "end": {
              "line": 1,
              "column": 12
            }
          },
          "local": {
            "type": "Identifier",
            "start": 7,
            "end": 12,
            "loc": {
              "start": {
                "line": 1,
                "column": 7
              },
              "end": {
                "line": 1,
                "column": 12
              },
              "identifierName": "React"
            },
            "name": "React",
            "range": [7, 12],
            "_babelType": "Identifier"
          },
          "range": [7, 12],
          "_babelType": "ImportDefaultSpecifier"
        },
        {
          "type": "ImportSpecifier",
          "start": 16,
          "end": 26,
          "loc": {
            "start": {
              "line": 1,
              "column": 16
            },
            "end": {
              "line": 1,
              "column": 26
            }
          },
          "imported": {
            "type": "Identifier",
            "start": 16,
            "end": 26,
            "loc": {
              "start": {
                "line": 1,
                "column": 16
              },
              "end": {
                "line": 1,
                "column": 26
              },
              "identifierName": "MouseEvent"
            },
            "name": "MouseEvent",
            "range": [16, 26],
            "_babelType": "Identifier"
          },
          "importKind": null,
          "local": {
            "type": "Identifier",
            "start": 16,
            "end": 26,
            "loc": {
              "start": {
                "line": 1,
                "column": 16
              },
              "end": {
                "line": 1,
                "column": 26
              },
              "identifierName": "MouseEvent"
            },
            "name": "MouseEvent",
            "range": [16, 26],
            "_babelType": "Identifier"
          },
          "range": [16, 26],
          "_babelType": "ImportSpecifier"
        }
      ],
      "importKind": "value",
      "source": {
        "type": "Literal",
        "start": 34,
        "end": 41,
        "loc": {
          "start": {
            "line": 1,
            "column": 34
          },
          "end": {
            "line": 1,
            "column": 41
          }
        },
        "extra": {
          "rawValue": "react",
          "raw": "'react'"
        },
        "value": "react",
        "range": [34, 41],
        "_babelType": "StringLiteral",
        "raw": "'react'"
      },
      "range": [0, 41],
      "_babelType": "ImportDeclaration"
    },
    {
      "type": "ImportDeclaration",
      "start": 42,
      "end": 69,
      "loc": {
        "start": {
          "line": 2,
          "column": 0
        },
        "end": {
          "line": 2,
          "column": 27
        }
      },
      "specifiers": [
        {
          "type": "ImportDefaultSpecifier",
          "start": 49,
          "end": 55,
          "loc": {
            "start": {
              "line": 2,
              "column": 7
            },
            "end": {
              "line": 2,
              "column": 13
            }
          },
          "local": {
            "type": "Identifier",
            "start": 49,
            "end": 55,
            "loc": {
              "start": {
                "line": 2,
                "column": 7
              },
              "end": {
                "line": 2,
                "column": 13
              },
              "identifierName": "lodash"
            },
            "name": "lodash",
            "range": [49, 55],
            "_babelType": "Identifier"
          },
          "range": [49, 55],
          "_babelType": "ImportDefaultSpecifier"
        }
      ],
      "importKind": "value",
      "source": {
        "type": "Literal",
        "start": 61,
        "end": 69,
        "loc": {
          "start": {
            "line": 2,
            "column": 19
          },
          "end": {
            "line": 2,
            "column": 27
          }
        },
        "extra": {
          "rawValue": "lodash",
          "raw": "'lodash'"
        },
        "value": "lodash",
        "range": [61, 69],
        "_babelType": "StringLiteral",
        "raw": "'lodash'"
      },
      "range": [42, 69],
      "_babelType": "ImportDeclaration"
    }
  ]
}
```

위 AST 트리를 기반으로 룰을 만들어보자.

```javascript
module.exports = {
  rules: {
    'default-import': (context) => ({
      ImportDeclaration: (node) => {
        const found = node.specifiers.find(
          (i) => i.type === 'ImportDefaultSpecifier',
        )

        if (found) {
          const { options } = context
          const option = options.find((opt) => 'path' in opt)
          const paths = option.path || []

          if (paths.includes(node.source.value)) {
            context.report(node, `import ${node.source.value}는 하면 안되잉`)
          }
        }
      },
    }),
}
```

```bash
@yceffort ➜ /workspaces/eslint-plugin-import-yceffort/test (main ✗) $ npm run lint

> test@1.0.0 lint
> eslint '**/*.{js,ts}'

/workspaces/eslint-plugin-import-yceffort/test/index.js
  1:1  warning  import react는 하면 안되잉                                                 yceffort-rules/default-import
  2:1  warning  import lodash는 하면 안되잉                                                yceffort-rules/default-import
```

## 참고

[eslint-plugin-yceffort-rules](https://github.com/yceffort/eslint-plugin-yceffort-rules)

> 물론 그냥 연습만 해보느라 여러가지로 썩 좋지 못한 코드니, 실제로 사용할 때는 적절하게 리팩토링을 해서 써보자.
