---
title: 리액트 텍스트 하이라이트 만들기
date: 2019-10-15 07:51:36
published: true
tags:
  - react
  - typescript
  - algorithm
  - javascript
description: '리액트에서 텍스트 강조하는 방법'
slug: /2019/10/15/react-text-highlight/
template: post
---

## 요구사항

한 엘리먼트안에서 특정한 키워드를 다른 색싱으로 바꿔서 출력하는 것이다.

아래 예시를 살펴보자

### before

```jsx
<Text>카카오 페이지 카카오 스토리 카카오톡</Text>
```

### after

```jsx
<Text>
  <Text color="blue">카카오 </Text>페이지
  <Text color="blue">카카오 </Text>스토리
  <Text color="blue">카카오</Text>톡
</Text>
```

## 의식의 흐름

특정 키워드가 포함되어 있는지, 그리고 그것을 따로 뽑아 낼 수 있는 가장 간단한 방법은 무엇일까? 바로 [split](https://developer.mozilla.org/ko/docs/Web/JavaScript/Reference/Global_Objects/String/split) 일 것이다.

```javascript
const splitResult = '카카오 페이지, 카카오 스토리, 카카오톡'
splitResult.split('카카오') // ["", " 페이지, ", " 스토리, ", "톡"]
```

그러나 여기서 두 가지 몰랐던 사실을 알게 된다.

1. 첫 문자에 seperator 가 동일하게 나올 경우, 앞에 ""가 무조건 나온다.
2. text === seperator 면 결과는 빈 문자열 두개다.

```javascript
const splitResult = '카카오'
splitResult.split('카카오') //  ["", ""]
```

> 문자열에서 separator가 등장하면 해당 부분은 삭제되고 남은 문자열이 배열로 반환됩니다. separator가 등장하지 않거나 생략되었을 경우 배열은 원본 문자열을 유일한 원소로 가집니다. separator가 빈 문자열일 경우, str은 문자열의 모든 문자를 원소로 가지는 배열로 변환됩니다. separator가 원본 문자열의 처음이나 끝에 등장할 경우 반환되는 배열도 빈 문자열로 시작하거나 끝납니다. 그러므로 원본 문자열에 separator 하나만이 포함되어 있을 경우 빈 문자열 두 개를 원소로 가지는 배열이 반환됩니다.

평소에 잘 몰랐던 split의 심오한 철학이 많이 있으니 가서 확인해보는 것도 좋을 듯 하다.

암튼 첫 번째 결과물은 이렇다.

```tsx
const [initial, ...rest] = text.split(highlight)
return (
  <Text>
    {rest.reduce(
      (partialResult, current) => [
        ...partialResult,
        <Text
          key={highlight + current}
          color={highlightColor}
          inlineBlock
          size={fontSize}
          whiteSpace="pre"
        >
          {highlight}
        </Text>,
        current,
      ],
      [initial],
    )}
  </Text>
)
```

[reduce](https://developer.mozilla.org/ko/docs/Web/JavaScript/Reference/Global_Objects/Array/Reduce) 를 활용해서, 처리했다.

근데 어차피, map으로 돌면서 하는게 더 간단하지 않을까 하는 아이디어가 나왔다.

## 결과

```tsx
const initial = text.split(highlight)
return (
  <Text>
    {initial.map((normal, i) =>
      i > 0 ? (
        <>
          <Text
            key={highlight + i.toString()}
            color={highlightColor}
            inlineBlock
            size={fontSize}
            whiteSpace="pre"
          >
            {highlight}
          </Text>
          {normal}
        </>
      ) : (
        <>{normal}</>
      ),
    )}
  </Text>
)
```

`i > 0` 을 처리한 이유는, 어차피 첫번째 엘리먼트는 무조건 하이라이트가 안되는 텍스트가 오기 때문이다! 첫단어가 일치하는 단어라면 ""가 올 것이고, 일치 하지 않는 단어라면 그 단어 그대로 올라오기 때문에 첫번째 단어는 별도처리를 하지 않아도 된다.

그리고 두번째 엘리먼트 부터 해당 text가 있어서 쪼개진 단어가 올것이기 때문에, 앞에 하이라이트 텍스트를 붙여주고, 그 다음 평범한 단어를 붙여주면 된다.
만약 여러 단어에 하이라이팅이 필요하다면, 리스트 사이사이에 검색한 내용을 넣어주면 된다.
