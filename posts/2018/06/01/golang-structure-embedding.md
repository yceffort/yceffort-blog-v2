---
title: GoLang) 구조체와 임베딩
date: 2018-06-01 07:57:05
published: true
tags:
  - golang
description: 'Golang에는 클래스가 없는 대신, 아래와 같은 구조체가 존재한다.'
category: golang
slug: /2018/06/01/golang-structure-embedding/
template: post
---

Golang에는 클래스가 없는 대신, 아래와 같은 구조체가 존재한다.

```go
package main import "fmt"

type work struct {
 mission string
 time int
 boss string
 salary int
}

func main() {
 programming := work{"잡일", 5, "김악덕", 100}
 fmt.Println(programming)
 fmt.Println(programming.time)
 programming.time = 100
 fmt.Println(programming.time)
}
```

```
$ go run GoStructure.go {잡일 5 김악덕 100} 5 100
```

위에 struct 를 mission, boss String 이런식으로도 쓸 수 있다.

메소드는 어떻게 구현할 수 있을까?

```go
func (total *work) getTotal() int {
  return total.time * total.salary
}

func main() {
  fmt.Println(programming.getTotal())
}
```

이런식으로 할 수 있다.

특별한 건 여기서, work 는 구조체를 total이라는 변수명을 사용해서 (리시버 변수) 접근하고 있고, work에 \*을 붙였다는 것아다.

\*은 모두가 예상한 바로 그것이다. 포인터.

\*이 지정되지 않으면(주소값을 할당해주지 않으면) 저 메소드는 구조체 내부의 값에 영향을 미치지 않게 된다.

암튼 쓰는건 여타 언어와 크게 다를 바는 없다.

다만 클래스가 없기 때문에 상속이라는 개념도 없다.

그렇다면 GoLang에서 상속을 흉내내려면 어떻게 해야할까?

바로 그것은 임베딩이다.

```go
package main import "fmt"
type work struct {
  mission string
  time int
  boss string
  salary int
}

func (total *work) getTotal() int {
  return total.time * total.salary
}

func (total *hardWork) getTotal() int {
  return total.w.time * total.w.salary - 100000
}

type hardWork struct {
  w work
  level int
  yaguen bool
}


func main() {
  var spring hardWork
  spring.w.boss = "김악마"
  spring.w.time = 100
  spring.w.mission = "의미없이 스프링 문서를 보고 있을 것"
  spring.w.salary = -100
  spring.level = 10000
  spring.yaguen = true
  fmt.Println(spring)
  fmt.Println(spring.w.getTotal())
  fmt.Println(spring.getTotal())
}
```

대충 보면 느낌이 올 것이다. hardwork는 work를 임베딩 하고 있으며, 변수명을 w 로 work를 가지고 있다. 변수명은 생략가능하다.

그리고 명칭이 같은 메소드를 오버라이딩하고 있는데, 이 두개에 대한 동작이 다른 것을 볼 수 있다.
