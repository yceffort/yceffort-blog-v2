---
title: 발음 기반으로 String의 유사도를 비교해 보자.
date: 2018-05-31 09:01:45
published: true
tags:
  - programming
  - java
description: 앞선 [포스팅](https://yceffort.github.io/notes/Levenshtein-distance) 을
  통해서 두 String을 문자열 기반으로 분석하였을때. 유사도를 어떻게 측정하는지 알아보았다. 그러나 음성인식으로 String을 비교 했을
  때,  다음과 같은 문제에 직면하였다.  “Eggs”를 말했을때, Android Voice API는...
category: programming
slug: /2018/05/31/compare-string-with-voice/
template: post
---
앞선 [포스팅](https://yceffort.github.io/notes/Levenshtein-distance) 을 통해서 두 String을 문자열 기반으로 분석하였을때. 유사도를 어떻게 측정하는지 알아보았다.

그러나 음성인식으로 String을 비교 했을 때,  다음과 같은 문제에 직면하였다.

“Eggs”를 말했을때, Android Voice API는 “X” 를 반환하는 것이 아닌가?

String의 물리적인 비교를 해보았을때, 두 비교값은 앞선 나의 방식으로 한다면 당연히 0 일 것이다.

그렇다면, 발음 기반으로 비교를 하고 싶으면 어떻게 할 수 있을까?

[아파치 코덱 라이브러리](http://commons.apache.org/proper/commons-codec/download_codec.cgi) 가 그 문제의 해결책을 줄 수 있다.

그 중에서도 사용할 것은 [SoundEx](http://commons.apache.org/proper/commons-codec/apidocs/org/apache/commons/codec/language/Soundex.html) 다.

사용법은 다음과 같다.

일단 위의 링크에서 관련 라이브러리를 받아서 Import를 한다.

```java
 public static double getVoiceSimilarity(String s1, String s2)
{
  Soundex soundex = new Soundex();
  double sim = 0;
    try {
      sim = soundex.difference(s1, s2);
    } catch (Exception e) {
      sim = 0;
    }
  return sim;
}
```

대략 원리는 이렇다.

The correct value can be found as follows:

1. Retain the first letter of the name and drop all other occurrences of a, e, i, o, u, y, h, w.
2. Replace consonants with digits as follows (after the first letter): - b, f, p, v → 1

- c, g, j, k, q, s, x, z → 2
- d, t → 3
- l → 4
- m, n → 5
- r → 6

3. If two or more letters with the same number are adjacent in the original name (before step 1), only retain the first letter; also two letters with the same number separated by ‘h’ or ‘w’ are coded as a single number, whereas such letters separated by a vowel are coded twice. This rule also applies to the first letter.
4. If you have too few letters in your word that you can’t assign three numbers, append with zeros until there are three numbers. If you have more than 3 letters, just retain the first 3 numbers.

영어는 귀찮으니까 delphi라는 단어로 예를 들어보자.

1.  맨 앞글자를 제외하고 나머지 단어는 냄겨둔다.  
    결과: d?????
2.  ‘A’, E’, ‘I’, ‘O’, ‘U’, ‘H’, ‘W’, ‘Y’는 숫자 0으로 바꾼다.  
    결과: d0??00
3.  나머지단어는 아래처럼 바꾼다.  
    b, f, p, v → 1  
    c, g, j, k, q, s, x, z → 2  
    d, t → 3  
    l → 4  
    m, n → 5  
    r → 6  
    결과: d04100
4.  연속해서 붙어있는 글자들은 하나를 남기고 지운다.  
    결과: d0410
5.  0을 다 지운다.  
    결과: d41
6.  4자리를 만드는 대신, 오른쪽 빈 공간을 0으로 채운다.  
    최종결과: d410

```java
System.out.println(soundex(“dellfhai”));
System.out.println(soundex(“delphi”));
```

를 하게 되면, 동일하게 d410이 나온다. 싱기방기.

영어의 알파벳별 단어발음을 적극적으로 활용 한 것이다.

보면 알겠지만,  하나의 단어에 대해서 무조건 4글자의 규칙적인 발음표(?) 를 만들어서 리턴한다.

어떤 단어든 최초 앞글자 + lenght 3의 String을 리턴하며, 이런 변환 과정에 있어서 비슷한 자음 (b, p 를 같은 것으로 본다던지) 을 같은것으로 가정하여 판단하는 것이다.

```java
public static void main(String[] args) {
  System.out.println(soundex("eggs"));
  System.out.println(soundex("X"));
  System.out.println(soundex("Soundex"));
  System.out.println(soundex("Saundthax"));
}
```

결과

```
E200
X000
S532
S532
```

꽤비슷한 단어로 추정하는 것을 볼수 있다.  
eggs, x 는 스펠링 비교를 했을 때 절대로 같거나 비슷한 단어로 볼수 없지만,

이알고리즘을 사용할 경우 비슷한단어라고 추정이 가능하다.
