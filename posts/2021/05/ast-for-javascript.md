---
title: '자바스크립트 개발자를 위한 AST 이해하기'
tags:
  - javascript
published: true
date: 2021-05-10 09:40:39
description: '이걸 이제 하네 완전 게을러짐'
---

요즘 자바스크립트 프로젝트를 하다보면, `devDependencies`에 정말 많은 의존성이 있음을 알 수 있다. 자바스크립트 트랜스파일링, 코드 최소화, CSS pre-processor, eslint, prettier 등등등. 이러한 기능들은 실제 프로덕션 코드로 올라가는 것은 아니지만, 개발 과정에서 중요한 것들을 담당한다. 그리고 이러한 툴들은 AST processing 을 기반으로 작동한다.

## AST 란 무엇인가?

> 컴퓨터 과학에서 추상 구문 트리(abstract syntax tree, AST), 또는 간단히 구문 트리(syntax tree)는 프로그래밍 언어로 작성된 소스 코드의 추상 구문 구조의 트리이다. 이 트리의 각 노드는 소스 코드에서 발생되는 구조를 나타낸다. 구문이 추상적이라는 의미는 실제 구문에서 나타나는 모든 세세한 정보를 나타내지는 않는다는 것을 의미한다. 예를 들어, 그룹핑을 위한 괄호는 암시적으로 트리 구조를 가지며, 분리된 노드로 표현되지는 않는다. 마찬가지로, if-condition-then 표현식과 같은 구문 구조는 3개의 가지에 1개의 노드가 달린 구조로 표기된다.

https://ko.wikipedia.org/wiki/%EC%B6%94%EC%83%81_%EA%B5%AC%EB%AC%B8_%ED%8A%B8%EB%A6%AC

(뭔 소리 인지 모르겠으니) 예제를 살펴보자.

> 모든 예제는 https://astexplorer.net/ 여기에서 가져왔다

```javascript
function square(n){
  return n * n
}
```

를 AST로 변환하면

```json
{
  "type": "Program",
  "start": 0,
  "end": 36,
  "loc": {
    "start": {
      "line": 1,
      "column": 0
    },
    "end": {
      "line": 3,
      "column": 1
    }
  },
  "range": [
    0,
    36
  ],
  "errors": [],
  "comments": [],
  "sourceType": "module",
  "body": [
    {
      "type": "FunctionDeclaration",
      "start": 0,
      "end": 36,
      "loc": {
        "start": {
          "line": 1,
          "column": 0
        },
        "end": {
          "line": 3,
          "column": 1
        }
      },
      "range": [
        0,
        36
      ],
      "id": {
        "type": "Identifier",
        "start": 9,
        "end": 15,
        "loc": {
          "start": {
            "line": 1,
            "column": 9
          },
          "end": {
            "line": 1,
            "column": 15
          },
          "identifierName": "square"
        },
        "range": [
          9,
          15
        ],
        "name": "square",
        "_babelType": "Identifier"
      },
      "generator": false,
      "async": false,
      "expression": false,
      "params": [
        {
          "type": "Identifier",
          "start": 16,
          "end": 17,
          "loc": {
            "start": {
              "line": 1,
              "column": 16
            },
            "end": {
              "line": 1,
              "column": 17
            },
            "identifierName": "n"
          },
          "range": [
            16,
            17
          ],
          "name": "n",
          "_babelType": "Identifier"
        }
      ],
      "body": {
        "type": "BlockStatement",
        "start": 18,
        "end": 36,
        "loc": {
          "start": {
            "line": 1,
            "column": 18
          },
          "end": {
            "line": 3,
            "column": 1
          }
        },
        "range": [
          18,
          36
        ],
        "body": [
          {
            "type": "ReturnStatement",
            "start": 22,
            "end": 34,
            "loc": {
              "start": {
                "line": 2,
                "column": 2
              },
              "end": {
                "line": 2,
                "column": 14
              }
            },
            "range": [
              22,
              34
            ],
            "argument": {
              "type": "BinaryExpression",
              "start": 29,
              "end": 34,
              "loc": {
                "start": {
                  "line": 2,
                  "column": 9
                },
                "end": {
                  "line": 2,
                  "column": 14
                }
              },
              "range": [
                29,
                34
              ],
              "left": {
                "type": "Identifier",
                "start": 29,
                "end": 30,
                "loc": {
                  "start": {
                    "line": 2,
                    "column": 9
                  },
                  "end": {
                    "line": 2,
                    "column": 10
                  },
                  "identifierName": "n"
                },
                "range": [
                  29,
                  30
                ],
                "name": "n",
                "_babelType": "Identifier"
              },
              "operator": "*",
              "right": {
                "type": "Identifier",
                "start": 33,
                "end": 34,
                "loc": {
                  "start": {
                    "line": 2,
                    "column": 13
                  },
                  "end": {
                    "line": 2,
                    "column": 14
                  },
                  "identifierName": "n"
                },
                "range": [
                  33,
                  34
                ],
                "name": "n",
                "_babelType": "Identifier"
              },
              "_babelType": "BinaryExpression"
            },
            "_babelType": "ReturnStatement"
          }
        ],
        "_babelType": "BlockStatement"
      },
      "_babelType": "FunctionDeclaration"
    }
  ]
}
```

이렇게 나온다. (`babel-eslint9`)

코드 텍스트에서, 트리 구조의 데이터 스트럭쳐를 만들어 낸다. 코드에 있는 아이템이 각 트리에 있는 노드와 매치된다.

그런데 어떻게 코드에서 AST를 가져오는 것일까? 이러한 작업은 컴파일러가 하는데, 일반적인 컴파일러가 어떻게 처리하는지 알아봐야 한다. 그전에, 여기에서는 고수즌의 언어코드를 컴퓨터가 이해하는 바이트 코드로 까지 변환하는 모든 단계를 다루지는 않으려고 한다. 우리는 렉시컬, 그리고 신택스 분석에만 관심이 있다. 그리고 이 두 단계는 코드에서 AST를 생성하는데 중요한 역할을 한다.

첫번째 단계, 렉시컬 분석 (Lexical analyzer aka scanner)는 코드의 문자들을 읽어서 정해진 룰에 따라서 이들을 토큰으로 만들어 합친다. 또한 여기에서 공백, 주석 등을 지우고 마지막으로는 이 전체 코드를 토큰들로 나눈다. 이 렉시컬 분석기가 소스 코드를 읽을 때, 코드를 글자단위로 읽는다. 그 과정에서 공백, operator symbol, special symbol 를 만나면 해당 단어가 끝난 것으로 간주한다.

두번째 단계, 신택스 분석 (Syntax analyzer aka parser) 에서는 위에서 결과로 나온 토큰 목록을 트리 구조로 만들며, 구조적 혹은 언어적으로 문제가 있을 경우 에러를 내뱉는다. 트리를 만드는 과정에서, 일부 파서들은 불필요한 토큰 (중복된 괄호라든지)를 생략한다. 그리고 그 결과로 `Abstract Syntax Tree`를 만든다. 이는 코드와 100% 일치하지는 않지만, 코드를 다루기에는 충분하다. 

다음은 AST를 배우고 싶을 때 참고 할만한 레포 목록이다.

- 컴파일러에 대해서 배우고 싶다면, [The-super-tiny-compiler](https://github.com/jamiebuilds/the-super-tiny-compiler)를 보는 것을 추천한다. 자바스크립트 로 쓰여진 가장 간단한 컴파일러 예제를 구현해두었다.
- [LangSandbox](https://github.com/ftomassetti/LangSandbox) 또한 재밌는 프로젝트다. 여기에서는 어떻게 프로그래밍 언어를 만들어야 하는지 보여준다.,
- [AST Explorer](https://astexplorer.net/)
- [@babel/parser](https://github.com/babel/babel/tree/master/packages/babel-parser) 구 babylon

이제 AST가 코드에서 어떻게 만들어졌는지 알아보았으니, 실제 유즈케이스를 확인해보자.

가장 첫번째로 이야기할 사용사례는 당연히, 트랜스파일링이다.

https://babeljs.io/ 바벨은 다들 아시다시피, 자바스크립트 컴파일러다. 바벨은 크게 3단계로 이루어진다. (parsing, transforming, generation) 바벨에 자바스크립트 코드를 넘기면, AST를 활용해서 코드를 새롭게 변환해서 만든다.

```javascript
import * as parser from "@babel/parser";
import generate from "@babel/generator";
// string code
const pureCode = `
  const welcome = 'hello world'
`;

// ast
const ast = parser.parse(pureCode);

// generate
const finalCode = generate(ast);
```

```json
{
   "type":"File",
   "start":0,
   "end":33,
   "loc":{
      "start":{
         "line":1,
         "column":0
      },
      "end":{
         "line":3,
         "column":0
      }
   },
   "errors":[
      
   ],
   "program":{
      "type":"Program",
      "start":0,
      "end":33,
      "loc":{
         "start":{
            "line":1,
            "column":0
         },
         "end":{
            "line":3,
            "column":0
         }
      },
      "sourceType":"script",
      "interpreter":null,
      "body":[
         {
            "type":"VariableDeclaration",
            "start":3,
            "end":32,
            "loc":{
               "start":{
                  "line":2,
                  "column":2
               },
               "end":{
                  "line":2,
                  "column":31
               }
            },
            "declarations":[
               {
                  "type":"VariableDeclarator",
                  "start":9,
                  "end":32,
                  "loc":{
                     "start":{
                        "line":2,
                        "column":8
                     },
                     "end":{
                        "line":2,
                        "column":31
                     }
                  },
                  "id":{
                     "type":"Identifier",
                     "start":9,
                     "end":16,
                     "loc":{
                        "start":{
                           "line":2,
                           "column":8
                        },
                        "end":{
                           "line":2,
                           "column":15
                        },
                        "identifierName":"welcome"
                     },
                     "name":"welcome"
                  },
                  "init":{
                     "type":"StringLiteral",
                     "start":19,
                     "end":32,
                     "loc":{
                        "start":{
                           "line":2,
                           "column":18
                        },
                        "end":{
                           "line":2,
                           "column":31
                        }
                     },
                     "extra":{
                        "rawValue":"hello world",
                        "raw":"'hello world'"
                     },
                     "value":"hello world"
                  }
               }
            ],
            "kind":"const"
         }
      ],
      "directives":[
         
      ]
   },
   "comments":[
      
   ]
}
```

babel과 관련된 자세한 내용은 https://github.com/jamiebuilds/babel-handbook 에서 공부해볼 수 있다.

다음으로 알아볼 유즈 케이스는, 코드를 자동으로 리팩토링 해주는 [JSCodeShift](https://github.com/facebook/jscodeshift)다. 예를 들어, 아래와 같은 코드 리팩토링이 있다고 가정해보자.

```javascript
load().then(function (response) {
  return response.data
})
```

```javascript
load().then(response => response.data)
```

단순한 찾아 바꾸기가 아니기 때문에, 일반적인 코드 에디터에서는 이러한 리팩토링이 불가능할 것이다. 그리고 이것을 가능하게 하는 것이 `jscodeshift`다.

`jscodeshift`에 대해서 들어봤다면, `codemods`에 대해서도 들어봤을 것이다. `Jscodeshift`는 `codemods`를 실행시키는 툴킷이다. `codemods`에서 실제 AST 를 활용한 변환이 일어난다. 따라서, 기본적인 아이디어는 babel과 하위 플러그인과 유사하다.

- https://github.com/reactjs/react-codemod
- https://github.com/facebook/jscodeshift


다음으로 알아보는 유즈케이스는 `Prettier`다. prettier는 자동으로 줄바꿈도 해주고, 공백도 제거해주며, 괄호도 다듬어 주는 도구다. 따라서 prettier는 기존의 코드를 받아서, 이를 다듬은 코드로 리턴한다는 것을 알 수 있다. 이러한 과정 또한 마찬가지로, AST를 거친다.

아이디어는 기본적으로 똑같다. 코드를 받아서 AST를 만든다. 그리고, 여기에서 prettier가 활약한다. AST는 `intermediate representation`나 `Doc` 형태로 변환된다. AST 노드는 여기에서 포맷팅의 관점에서 어떻게 서로 연관되어 있는지 정보도 추가되어서 변환된다. 만약, 리스트가 길고 한줄에 맞지 않으면 각 파라미터를 별도의 줄로 분할한다. 그다음, `printer`라고 하는 알고리즘이 IR을 거쳐서 전체 그림을 바탕으로 코드를 포맷팅할 방법을 결정하게 된다.

> prettier에 대해 더 자세히 알고 싶다면 https://homepages.inf.ed.ac.uk/wadler/papers/prettier/prettier.pdf 이 논문을 참고하면 좋다.

마지막으로 알아볼 유즈케이스느 [js2flowchart](https://github.com/Bogdan-Lyashenko/js-code-to-svg-flowchart) 다. 이 라이브러리는 코드를 기준으로 플로우차트를 그려주는 라이브러리다. 플로우차트로 코드를 설명/문서화 할 수 있으며, 시각적인 이해를 바탕으로 다른 코드를 학습할 수 있고, 나아가 유효한 JS 문법을 기준으로 플로우차트를 간단하게 만들어 볼 수도 있다.

```javascript
 /**
 * Binary search
 * @param {Array} list
 * @param {Number} element
 * @returns {number}
 */
function indexSearch(list, element) {
    let currentIndex,
        currentElement,
        minIndex = 0,
        maxIndex = list.length - 1;

    while (minIndex <= maxIndex) {
        currentIndex = Math.floor((maxIndex + maxIndex) / 2);
        currentElement = list[currentIndex];

        if (currentElement === element) {
            return currentIndex;
        }

        if (currentElement < element) {
            minIndex = currentIndex + 1;
        }

        if (currentElement > element) {
            maxIndex = currentIndex - 1;
        }
    }

    return -1;
}
```

![flowchart](./images/flowchart.svg)

https://bogdan-lyashenko.github.io/js-code-to-svg-flowchart/docs/live-editor/index.html


이 라이브러리는 어떻게 동작할까? 먼저 코드를 AST로 변환한다음, AST를 순회하며 Flowtree라고 하는 또다른 트리를 만든다. 여기에서는 중요하지 않은 토큰들을 생략하는 등의 과정을 거친다. 그 다음에, Flowtree를 순회하며 ShapesTree라는 것을 만든다. 여기에 각각 노드들은 각 노드의 타입, 위치, 트리사이의 관계등을 나타내고 있다. 마지막으로, 이를 기준으로 SVG 파일을 만들어 낸다.

