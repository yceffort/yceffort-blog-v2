---
title: 두 String의 유사도를 측정해보자 - Levenshtein distance
date: 2018-05-31 08:11:51
tags:
  - algorithm
published: true
description: 두 개의 String이 있을때, 그 두개를 비교하는 작업은 어떻게 할 수 있을까? str.equalsOf(str2) 이런
  것이 아니라, 두 단어의 비슷한 정도를 말하는 것이다. 예를 들어보자.  사용자가 Toast라고 말을 했다. 그러면 구글 Voice는
  Toast라는 사용자 사운드에 가장 비슷한 단어 몇가지를 추천해준다.  ![구글보이스로 toast를 ...
category: algorithm
slug: /2018/05/31/Levenshtein-distance/
template: post
---
두 개의 String이 있을때, 그 두개를 비교하는 작업은 어떻게 할 수 있을까?
str.equalsOf(str2) 이런 것이 아니라, 두 단어의 비슷한 정도를 말하는 것이다.

예를 들어보자.

사용자가 Toast라고 말을 했다. 그러면 구글 Voice는 Toast라는 사용자 사운드에 가장 비슷한 단어 몇가지를 추천해준다.

![구글보이스로 toast를 외쳐보았다](../images/google_voice_test.png)

나의 저질 발음으로 인해, 안타깝게도 toast를 인식하지 못하고 저렇게 다섯 개의 후보를 주고 말았다.
그렇다면 나는, 내가 가지고 있는 DB의 데이터 중에서 가장 post와 유사한 단어를 찾아서 돌려줘야 한다.

그렇다면, 내가 가지고 있는 DB의 단어와 구글이 return한 단어는 어떻게 비교할 수 있을까?
정확히 말하면, 두개의 String의 유사도는 어떻게 판단할 수 있을까?

- [**Levenshtein Distance**](http://en.wikipedia.org/wiki/Levenshtein_distance) : The minimum number of single-character edits required to change one word into the other. Strings do not have to be the same length  
  – 한 글자 글자의 차이(삽입, 삭제, 대체) 를 거리로 계산한다.
- [**Hamming Distance**](http://en.wikipedia.org/wiki/Hamming_distance) : The number of characters that are different in two equal length strings.  
  – 길이가 같은 두 단어에서 몇개를 대체하면 같아지는지 계산한다. 근데 이미 길이가 같아야 한다는 전재가 있으므로 글러먹음.
- [**Smith–Waterman**](http://en.wikipedia.org/wiki/Smith-Waterman_algorithm) : A family of algorithms for computing variable sub-sequence similarities.  
  – 배열들의 가능한 모든 길이로 쪼개서 비교하는 방식
- [**Sørensen–Dice Coefficient**](http://en.wikipedia.org/wiki/Dice%27s_coefficient) : A similarity algorithm that computes difference coefficients of adjacent character pairs.  
  – 배열들의 쌍을 묶어보면서 비교하는 방식

정도가 있다.  그중에서 가장 접근하고 이해하기 쉬운 Levenshtein distance 에 대해 알아보고자 한다.

두 배열을 비교하기위해서는, 두 배열이 같아지는 과정이 얼마나 필요한지 (거리가 어떻게 되는지) 구하는 과정을 거치면 된다.

그 과정은 딱 3개다. 새로운걸 삽입(insertion), 기존의 원소를 삭제(deletion), 기존의 원소를 다른 것으로 대체(substitution)
예를 들어보자.

ghost > toast

1. g를 t로 대체한다 (subsitution) thost > toast
2. h를 o로 대체한다 (subsitution) toost > toast
3. o를 a로 대체한다 (subsitution) toast = toast!

이런 과정을 거치면, ghost와 toast의 거리는 3이 되는 것이다.

이제 느낌을 보면 알겠지만, 두 단어의 거리는 둘 중에 가장 긴 단어의 거리가 최대다. (zzz > effoooooooooooort를 비교한다고 생각해보자)

그렇기 때문에, 두단어의 유사도는

return (longerLength - getDistance(longer, shorter)) / (double) longerLength;

이렇게 나올 것이다.

그렇다면 이것을 알고리즘으로 구현하기 위해선 어떻게 해야할까?

이것을 알고리즘으로 구성하기 위한 단계를 다시 한번 생각해보자.

임의로 이렇게 한다고 치자.

두 단어 = s1, s2

길이 = s1.length, s2.length

1. 두 단어 중에 0인게 있다면, 당연한 말이겠지만, 다른 단어의 길이를 리턴한다.
2. 두개 배열의 For문을 같이 돈다.
3. 만약 s1[n] == s2[m] 이라면 변경에 필요한 거리는 0이 된다.
4. 그렇지 않다면, 대체, 삽입, 수정 중에서 가장 최소의 비용이 되는 방법을 고른다.
5. 이렇게 해서 쌓은 코스트를 배열에 저장한다.
6. 맨마지막 값을 리턴한다.

```java
public static int getDistance(String s1, String s2) {
  int longStrLen = s1.length() + 1;
  int shortStrLen = s2.length() + 1; // 긴 단어 만큼 크기가 나올 것이므로, 가장 긴단어 에 맞춰 Cost를 계산
  int[] cost = new int[longStrLen];
  int[] newcost = new int[longStrLen]; // 초기 비용을 가장 긴 배열에 맞춰서 초기화 시킨다.
  for (int i = 0; i < longStrLen; i++) { cost[i] = i; } // 짧은 배열을 한바퀴 돈다.
  for (int j = 1; j < shortStrLen; j++) {
    // 초기 Cost는 1, 2, 3, 4...
    newcost[0] = j; // 긴 배열을 한바퀴 돈다.
    for (int i = 1; i < longStrLen; i++) {
      // 원소가 같으면 0, 아니면 1
      int match = 0;
      if (s1.charAt(i - 1) != s2.charAt(j - 1)) { match = 1; }
      // 대체, 삽입, 삭제의 비용을 계산한다.
      int replace = cost[i - 1] + match;
      int insert = cost[i] + 1;
      int delete = newcost[i - 1] + 1;
      // 가장 작은 값을 비용에 넣는다.
      newcost[i] = Math.min(Math.min(insert, delete), replace);
    } // 기존 코스트 & 새 코스트 스위칭 int[] temp = cost; cost =
    newcost; newcost = temp;
  }
  // 가장 마지막값 리턴
  return cost[longStrLen - 1];
}
```

처음에는 무식하게 재귀함수로 계속해서 호출하는 방법을 썼는데, 이 방법이 더 빠르다고 한다.

암튼 이런 방식으로 한다면, 비교가 엄청빠르다.

한건의 단어를 대상으로 350000건의 사전 단어를 비교하는데, 속도는 재보지 않았지만 꽤 눈깜짝 할 새에 비교가 된다.

안드로이드 (갤럭시 s3 기준) 으로도 무식하게 35000줄짜리 text 파일을 일일이 비교해서 가져오는 데도 꽤 빠른 속도로 비교된다.

사실 이게 String 비교에도 쓰이지만, 유전체의 염기서열 등등의 다양한 배열을 비교하는데 더 많이 쓰인 다고 한다.
