---
title: '개발자가 알아야 하는 4가지 nodejs 디자인 패턴'
tags:
  - javascript
  - nodejs
published: true
date: 2021-01-11 23:33:24
description: '옛날 스타일의 가능한 객체 지향 프로그래밍'
---

디자인 패턴에는 세가지 유형이 있다.

- Creational: 객체 인스턴스 생성
- Structural: 객체 설계 방식
- Behavioural: 객체가 상호 작용하는 방식

## Singleton

클래스의 단일 인스턴스만을 원할 때 이 패턴을 사용한다. 즉, 여러개의 인스턴스를 생성하는 것이 아니라 하나만 생성하는 것이다. 인스턴스가 없다면 새 인스턴스를 생성한다. 인스턴스가 있는 경우에는, 해당 인스턴스를 사용한다.

```javascript
class DatabaseConnection {
  constructor() {
    this.databaseConnection = 'dummytext'
  }

  getNewDBConnection() {
    return this.databaseConnection
  }
}

class Singleton {
  constructor() {
    throw new Error('Use the getInstance() method on the Singleton object!')
  }

  getInstance() {
    if (!Singleton.instance) {
      Singleton.instance = new DatabaseConnection()
    }

    return Singleton.instance
  }
}

module.exports = Singleton
```

위에서 보이는 것 처럼, 싱클턴을 구축할 수 있는 많은 예제가 있다. 이 외에 이 설계 패턴을 구현하는 더 짧은 방법이 있다.

```javascript
class DatabaseConnection {
  constructor() {
    this.databaseConnection = 'dummytext'
  }

  getNewDBConnection() {
    return this.databaseConnection
  }
}

module.exports = new DatabaseConnection()
```

이것이 작동할 수 있는 이유는 module caching system 이다. [module caching system이란, 모듈이 처음 로딩 된 이후에 캐싱이 되는 것을 의미한다.](https://nodejs.org/api/modules.html#modules_caching) 즉, 위의 예제에서는, 새롭게 exported된 인스턴스는 캐싱이 되며, 이것이 재 사용될 때마다 이 캐쉬댄 내용을 불러온다는 뜻이다.

따라서,Nodejs에서 싱글턴을 구현하는 방법은 위 처럼 두가지로 볼 수 있다.

### 요약

- 싱클턴 방식은 단 하나의 클래스 인스턴스가 필요할 때 유용하다.
- Nodejs에서는, module caching system을 활용해서 export한 모듈을 바로 쓸 수 있다.

## 팩토리

팩토리 디자인 패턴은, 객체를 생성하는데 사용되는 인터페이스 또는 추상 클래스를 정의 하는 것이다. 이렇게 생성된 인터페이스 및 추상클래스를 사용하여 다른 객체를 초기화 한다. 아래의 예를 살펴보자.

```javascript
import Motorvehicle from './Motorvehicle'
import Aircraft from './Aircraf'
import Railvehicle from './Railvehicle'

const VehicleFactory = (type, make, model, year) => {
  if (type === car) {
    return new Motorvehicle('car', make, model, year)
  } else if (type === airplane) {
    return new Aircraft('airplane', make, model, year)
  } else if (type === helicopter) {
    return new Aircraft('helicopter', make, model, year)
  } else {
    return new Railvehicle('train', make, model, year)
  }
}

module.exports = VehicleFactory
```

이렇게 각 클래스 인스턴스를 별개로 만드는 대신에, `VehicleFactory`를 활용해서 타입을 명시하는 방법을 택할 수 있다. 위 예제를 활용해서, `car` 인스턴스를 만들려면 아래처럼 실행하면 된다.

```javascript
// 첫번째 매개변수에서 타입을 지정하고, 나머지는 그대로 변수를 넘긴다.
const audiAllRoad = VehicleFactory('car', 'Audi', 'A6 Allroad', '2020')
```

팩토리 디자인 패턴을 사용하면 객체의 구조가 객체 그 자체 사이를 디커플링 시킬 수 있다는 장점이 있다. 기존 코드를 손상시키지 않더라도 새 객체를 응용프로그램에 사용할 수 있다. 마지막으로, 인스턴스 생성과 관련된 모든 코드가 한 곳에 있으므로 코드를 더 잘 꾸밀 수 있다.

### 요약

- 팩토리 디자인 패턴은 객체 생성을 위한 인터페이스 및 추상 클래스를 제공한다.
- 동일한 인터페이스 및 추상 클래스를 사용하여 다른 객체를 만들 수 있다.
- 코드의 구조를 개선하고 유지관리가 더 쉬워 진다.

## 빌더

빌더 디자인 패턴 또한 마찬가지로 객체 구조와 객체를 분리할 수 있다. 따라서 복잡한 객체를 생성하는 코드를 단순화 한다. 단순한 객체를 만들 때는 과한 기능일 수 있지만, 복잡한 객체를 만들 때는 단순화 하는데 도움을 준다.

```javascript
class Car {
  constructor(make, model, year, isForSale = true, isInStock = false) {
    this.make = make
    this.model = model
    this.year = year
    this.isForSale = isForSale
    this.isInStock = isInStock
  }

  toString() {
    return console.log(JSON.stringify(this))
  }
}

class CarBuilder {
  constructor(make, model, year) {
    this.make = make
    this.model = model
    this.year = year
  }

  notForSale() {
    this.isForSale = false

    return this
  }

  addInStock() {
    this.isInStock = true

    return this
  }

  build() {
    return new Car(
      this.make,
      this.model,
      this.year,
      this.isForSale,
      this.isInStock,
    )
  }
}

module.exports = CarBuilder
```

위 패턴을 사용하면 `Car` 대신에 `CarBuilder`를 사용하여 객체를 만들 수 있다.

```javascript
const CarBuilder = require('./CarBuilder')

const bmw = new CarBuilder('bmw', 'x6', 2020).addInStock().build()
const audi = new CarBuilder('audi', 'a8', 2021).notForSale().build()
const mercedes = new CarBuilder('mercedes-benz', 'c-class', 2019).build()
```

만약에 이런 빌더 패턴 없이 복잡한 객체를 만들게 되면 에러를 발생할 가능성이 커진다.

```javascript
const bmw = new CarBuilder('bmw', 'x6', 2020, true, true)
```

뒤 이어 있는 `true`가 각각 무엇을 의미하는지 알아야 하기 때문에 객체 생성이 복잡해 지고, 에러를 만들어낼 가능성이 커진다. 따라서 빌더 디자인 패턴은 복잡한 객체 생성과 사용을 분리하는데 도움을 준다.

## 프로토타입

자바스크립트는 프로토타입 기반 언어이기 때문에, 프로토타입으로 상속이 구현되어 있다. 이 말인 즉슨, 모든 객체는 어떤 객체를 상속하고 있다는 뜻이다.

따라서 이른바 예제 객체 라고 불리우는 프로토타입 객체의 값을 복제 하여 새로운 객체를 만든다. 이는 프로토 타입이 새 객체의 일종의 청사진 역할을 하는 것이다. 이 설계 패턴을 활용하면 객체에 정의된 함수가 참조에 의해 생성된다는 이점을 얻을 수 있다. 즉, 모든 객체가 해당 기능의 복사본을 보유하는 것이 아니라 동일한 기능을 가르키게 된다. 간단히 말해, 프로토타입 기능은 프로토타입에 상속된 모든 객체에 사용할 수 있다.

```javascript
const atv = {
  make: 'Honda',
  model: 'Rincon 650',
  year: 2018,
  mud: () => {
    console.log('Mudding')
  },
}

const secondATV = Object.create(atv)
```

프로토타입에서 새로운 객체를 생성하기 위해서는, `Object.create()`를 활용하면 된다. 두번째 객체인 `secondATV`는 첫번째 객체인 `atv`와 같은 값을 가지게 된다. `mud()`를 호출해보면 같은 값을 찍는 것을 알 수 있다.

프로토타입 디자인 패턴을 활용하는 다른 방법은 클래스 안에 프로토타입을 명시하는 것이다.

```javascript
const atvPrototype = {
  mud: () => {
    console.log('Mudding')
  },
}

function Atv(make, model, year) {
  function constructor(make, model, year) {
    this.make = make
    this.model = model
    this.year = year
  }

  constructor.prototype = atvPrototype

  let instance = new constructor(make, model, year)
  return instance
}

const atv1 = Atv()
const atv2 = Atv('Honda', 'Rincon 650', '2018')
```

마찬가지로 두 인스턴스 모두 `atv` 객체에 정의된 항목에 액세스 할 수 있다.

결론적으로, 프로토타입 설계 패턴은 객체가 동일한 기능 또는 속성을 공유하기를 원할 때 유용하다.

### 요약

- 자바스크립트는 프로토타입 기반 언어다.
- 프로토타입 기반 상속을 사용한다.
- 각 객체는 다른 객체로 부터 상속된다.
- 새 객체는 프로토타입이라는 일종의 청사진에 따라 생성된다.
- 프로토타입에 정의된 함수는 모든 새 클래스에서 상속된다.
- 새 클래스는 개별 복사본을 갖는 대신 동일한 기능을 가리킨다.
-
