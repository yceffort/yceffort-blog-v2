---
title: '좋은 자바스크립트 테스트 코드를 짜는 방법'
tags:
  - javascript
  - nodejs
published: true
date: 2021-10-10 11:21:24
description: ''
---

## Table of Contents

## tl;dr

![BASIC Principles](https://miro.medium.com/max/1400/1*D_CFjHViMGu6HcidoSlR9Q.png)

### Black-Box

테스트는 내부가 아닌, 외부 결과물을 테스트 해야 한다.

- REST API 또는 Public API를 통해 테스트 하기
- mock을 최소화
- 테스트 더블 (실제 객체를 사용하기 어렵거나 모호할 떄 대신해 줄 수 있는 테스트 객체)을 사용하여 컴포넌트를 테스트 할 것

### Annotative

각 테스트는 예측가능하고 선언적인 구조로 이루어져있어야 한다.

- 테스트명은 3개의 반복적인 구조를 가지고 있어야 한다.
- AAA Pattern을 사용할 것 (Arrange, Act, Assert)
- Assertion은 선언적인 스타일에서 확인이 이루어져야 한다.
- 테스트는 7 문장 이상으로 이루어져 있으면 안된다.

### Single Door

각 테스트 검사는 하나의 액션과 하나의 응답으로 이루어져야 한다.

- 각 테스트는 하나의 애플리케이션 액션, 즉 하나의 함수 호출을 테스트 해야 한다.
- 각 테스트 결과물은 호출에 따른 하나의 결과물만 확인 해야 한다.
- 결과물이 성공적이라는 것을 증명하기 위해 최소한의 assertion을 사용해야 한다.

### Independent

테스트는 어떠한 상태도 공유하지 않는 독립된 공간이어야 한다.

- 테스트는 긴 문제를 위한 7개의 문장으로 이루어져 있어야 한다.
- 다른 테스트와 객체나 데이터를 공유해서는 안된다.
- 테스트는 다른 테스트의 결과에 영향을 미쳐서는 안된다. 테스트의 실행순서도 고려되어서는 안된다.
- 코드를 공유해야하는 부득이한 상황에서는 아주 작은 helper만 사용해야 한다.

### Copy

테스트를 작성한 의도를 이해하기 위해 필요한 모든 것을 포함해야 한다.

- 테스트 결과물은 쉽게 추론 가능해야 한다.
- 테스트에 필요한 의미있는 정보를 테스트 외부에 두어선 안된다.
- 필요한 경우에만 코드를 복사한다.
- 테스트에 집중해라. 중요하지 않은 정보는 외부 헬퍼나 훅을 이용해라
- 팩토리 기법을 사용하여 긴 구조를 만들어라. - 의미있는 정보만 넘겨서 오버라이딩 해라.

## BASIC

BASIC 이라는 약자를 바탕으로, 한글자씩 살펴보자. 각 글자는 테스트 코드를 만들 때 고려해야하는 원칙을 나타낸다. 그리고, 이 BASIC 원칙을 적용하여 길고 번거로운 테스트 코드가 아닌 간결하고 아름다운 테스트 코드를 짜는 방법을 알아볼 것이다.

### Black-ㅠox

테스트 코드는 단지 테스트 중인 컴포넌트가 객체가 무엇을 만들고 호출하는 사람이 무엇을 받게 될 것인지에 대해서만 관심을 가진다. 즉, API 또는 코드 객체 (유닛) 에 대해서만 신경 쓸 것이다. 만약 버그가 밖에서 일어나느 게 아니라면, 사용자에게도 별로 중요하지 않을 것이다. 따라서 우리에게도 이는 주요 관심사가 아니다. 작동 방식을 테스트 하지말고, 작업 그 자체에만 집중해야 한다. 어떤 함수를 호출하는지는 중요하지 않고, 결과에만 집중하면 된다.예를 들어, 유닛테스트의 응답, 관찰할 수 있는 상태값, 외부 코드에 대한 호출 등을 예를 들 수 있다. 외부로 노출 되는 것에만 초점을 맞춤으로써, 세부적인 코드의 양을 줄이면 테스트 코드 작성자는 UX에 영향을 미칠 수 있는 중요한 사항에 우선순위를 정하게 될 것이다. 이는 본질적으로 테스트 코드의 길이와 복잡성을 낮출 수 있다.

### Annotative

테스트 코드는 선언적인 언어로 예측 가능한 구조를 가져야 한다. 코드라기 보다는 일종의 주석 처럼 느껴져야 한다. 테스트 코드와 다르게 프로덕션 코드를 훑어 보는 것은 명확한 시작과 끝을 알 수 없는 일종의 여행이다. 프로덕션 코드를 이해하기 위해서는 애플리케이션에 대한 깊이 있는 이해가 필요하다. 반면에 HTML 코드는 어떤가? 태그가 시작하면, 우리는 태그의 끝을 찾게 되고 쉽게 다음에 올 코드를 예상할 수 있다. HTML 코드를 읽는 것은 프로덕션 코드를 읽는 것보다 확실히 쉽다. HTML 코드를 짜듯이, 테스트 코드도 선언적이고 구조적으로 만들어야 한다.

어떻게하면 테스트 코드를 선언적이고 구조적으로 만들수 있을까? `AAA Pattern`을 준수하고, 선언적 assertion, 최대 7 구문 내에서 아래 6개의 절차를 끝내야 한다. 참고: https://github.com/goldbergyoni/javascript-testing-best-practices

```javascript
// 1. Unit
describe('돈을 송금한다.', () => {
  // 2. 시나리오, 3. 기대 값
  test('유효한 송금이 완료되면, 올바른 응답이 와야 한다.', () => {
    // 4. arrange
    const moneyTransferRequest = {
      sender: 'yceffort@gmail.com',
      amount: 15000,
      receiver: 'root@yceffort.kr',
    }
    const transferService = new TransferService()

    // 5. act
    const result = transferService.transfer(moneyTransferRequest)

    // 6. assert
    expect(result.status).toBe('approved')
  })
})
```

### Single Door

모든 테스트 코드는 단 한가지에만 집중해야 한다. 일반적으로 애플리케이션에서 하나의 작업을 실행하고, 이 작업에 대한 응답으로 발생한 결과 하나 만을 확인해야 한다. 여기서 말하는 '작업'이란 함수 호출, 버튼 클릭, Rest API 호출, 큐에 있는 메시지, 스케쥴 된 작업 등 기타 여러 시스템 이벤트가 될 수 있다. 이 작업을 수행한 후에는 최대 세 가지 결과가 일어날 수 있다. 응답이 오거나, 상태 값이 변하거나 (DB, 인메모리) 또는 써드파티 서비스 호출. 통합테스트 (integration test)내에는 아래 6 종류의 결과가 도출 될 수 있다.

#### 백엔드 테스트 체크리스트

##### API 응답

아래 내용을 점검

- http status
- http body 내 데이터
- http body 구조
- (openapi 등의) 문서 준수 여부

##### State (DB)

API를 기반으로 아래 내용을 점검

- 데이터 저장/수정
- 실패시 데이터가 저장/수정되지 않음
- 관계 없는 데이터가 수정되지 않음
- 마이그레이션이 성공됨

##### Observability(관측성)와 에러

아래 내용을 점검

- 구체적인 에러가 로깅 되었는지
- 에러가 모니터링을 위해 전송되었는지
- 에러이름, 코드가 정확한지
- HTTP 응답 코드
- 프로세스가 예기치 않게 종료되었는지

##### Integrations (통합)

stub, mock, nock 등으로 아래 요소를 점검

- 외부 서비스가 200 이외의 응답이 오는지
- 외부 서비스가 응답하지 않는지
- 외부 서비스 호출을 양식에 맞게 하는지
- 외부 서비스 응답이 느리게 오는지

##### 보안

- 인증이 충분하지 않으면 401이 오는지
- 사용자 A가 사용자 B의 데이터를 볼수 없어야 함
- 주어진 권한 이상으로 작업을 할 수 없어야 함
- 인증이 만료되면 401이 오는지

##### 메시지 큐

- 메시지가 큐로 전송되는지
- 받은 메시지가 ACK/NACK 인지
- 중복 메시지를 잘 처리하는지
- 오염된(잘못된) 메시지를 수신하지 않는지
- 재시도가 실패로 이어지는지

[checklist](https://pbs.twimg.com/media/FAm8QTnXIAQRzCq?format=jpg&name=large)

> https://pbs.twimg.com/media/FAm8QTnXIAQRzCq?format=jpg&name=large

이러한 잠재적인 결과는 assertion을 통해 테스트 되어야 한다. 이로 인해 얻을 수 있는 것은 무엇인가? 각 테스트를 유닛 단위로 좁히면 테스트 시간이 단축되고 근본적인 원인을 분석하는데 도움을 준다. 장애가 발생할 경우 무엇이 작동하지 않아서 장애가 일어나는지 명확하게 분석할 수 있게 된다. 적당히 좋은 규칙이면서도, 너무 엄격하지도 않다. 하나의 테스트에서 2~3개 정도 테스트 하는 것은 괜찮다. (너무 많으면 좋지 않다) 숫자가 아닌 목표에 집중하라. 간단하고 간결한 테스트를 작성해야 한다.

### Independent

테스트는 7~10줄 사이의 문제이며, 다른 어떤 코드와도 겹치지 않는 독립적인 공간이다. 독립적이고, 짧으며, 선언적인 테스트 코드를 유지하는 것이 중요하다. 그러나 테스트 코드를 일부 글로벌 객체와 연동하는 것은 주의를 기울여야 한다. 갑자기 테스트가 많은 부수효과 (side effect)에 노출되며 복잡성이 급격히 커진다. 커플링은 실행 순서, UI 상태, 불문명한 mock 등 어떤 상황에서도 일어날 수 있다. 이 모든 것들을 피해야 한다. 대신, 자체 DB 등을 활용하여 각 테스트 에서 종속성을 분리해야 하고, 코드 그 자체 만으로도 설명할 수 있는 환경을 유지해야 한다.

### Copy only what is necessary

코드 이해에 필요한 모든 세부 사항을 테스트에 포함시켜야 한다. 그 이상일 필요는 없다. 딱 필요한 것만. 중복이 너무 많으면 유지 보수 하기 어려워지는 테스트가 만들어질 수 있다. 반면에 중요한 세부 정보를 외부에 노출하면 많은 파일에 걸쳐 이것이 무슨 의미 인지 검색해야 한다. 예를 들어 100여줄의 JSON 파일을 테스트 하는 코드가 있다고 가정해보자. 모든 테스트에서 이를 붙여 넣는 것은 매우 지겨운 일이다. 그렇다고 외부에서 `transferFactory.getJSON()`으로 압축해버리면 무엇을 하는지 모호 해진다. 데이터가 없으면 테스트 결과, 왜 이 결과가 400이 되어야 하는지를 이해 하는 것이 모호해진다. 고전적인 x-unit 패턴은 이를 'mystery guest'라고 이름 지었다. 눈에 보이지 않는 무언가가 테스트 결과에 영향을 미친다면, 우리는 정확히 알 수가 없다. 오직 필요한 내용만 테스트 코드에 존재해야 한다.

외부에서는 반복 가능한 긴 부분을 따로 추출하고, 테스트에서 어떤 세부사항이 중요한지 자세히 언급함으로써 더 좋은 테스트 코드를 짤 수 있다. 위의 예제를 활용한다면, `transferFactory.getJSON({sender: undefined})` 와 같이 함수의 인수를 활용하는 방식이 있을 수 있다. 이 테스트에서는 `sender`가 없다면 테스트가 오류를 내뱉거나 기타 다른 결과를 내야한다는 것을 적절히 유추할 수 있다.

## 실제 예제로 살펴보기

위 5가지 기본 원리를 적용하여 복잡도가 높고 얽혀있는 테스트를 보다 간단한 테스트로 변환해 보자.

아래는 우리가 테스트할 애플리케이션이다. 송금 서비스가 제공되고 있다. 이 코드는 송금 요청을 확인하고, 몇가지 로직을 적용하며, 이후에 은행 http 서비스에 송금을 요청하고, 마지막으로 이 결과를 DB에 기록한다.

```javascript
class TransferService {
  async transfer({ id, sender, receiver, transferAmount, bankName }) {
    // 유효성 검사
    if (!sender || !receiver || !transferAmount || !bankName) {
      throw new Error('Some mandatory property was not provided')
    }

    // 보낼 수 있는지 잔고 확인
    if (
      this.options.creditPolicy === 'zero' &&
      sender.credit < transferAmount
    ) {
      this.numberOfDeclined++ // 거절 횟수 증가
      return { id, status: 'declined', date }
    }

    // 돈을 보내고
    await this.bankProviderService.transfer(
      sender,
      receiver,
      transferAmount,
      bankName,
    )
    // 우리 DB에 쌓는다
    await this.repository.save({
      id,
      sender,
      receiver,
      transferAmount,
      bankName,
    })

    return { id, status: 'approved', date: new Date() }
  }

  getTransfers(senderName) {
    // Query the DB
  }
}
```

### 구린 테스트 예제

먼저 좋지 않은 테스트 예제를 아래와 같이 나열해보았다. ❌ 표시는 무언가 개선할 여지가 있다는 뜻이다.

```javascript
// ❌
test('Should fail', () => {
  const transferRequest = testHelpers.factorMoneyTransfer({}) // ❌
  serviceUnderTest.options.creditPolicy = 'zero' // ❌
  transferRequest.howMuch = 110 // ❌

  // sinon 라이브러리를 사용하여 db 저장 함수를 흉내냄
  const databaseRepositoryMock = sinon.stub(dbRepository, 'save') //❌
  const transferResponse = serviceUnderTest.transfer(transferRequest)
  expect(transferResponse.currency).toBe('dollar') // ❌
  expect(transferResponse.id).not.toBeNull() // ❌
  expect(transferResponse.date.getDay()).toBe(new Date().getDay()) // ❌
  expect(serviceUnderTest.numberOfDeclined).toBe(1) // ❌
  expect(databaseRepositoryMock.calledOnce).toBe(false) // ❌

  // 사용자 송금 이력 가져오기
  const allUserTransfers = serviceUnderTest.getTransfers(
    transferRequest.sender.name,
  )
  expect(allUserTransfers).not.toBeNull() // ❌ Overlapping
  expect(allUserTransfers).toBeType('array') // ❌ Overlapping

  // 거부된 송금이 사용자 기록에 남는지 확인 ❌
  let transferFound = false
  allUserTransfers.forEach((transferToCheck) => {
    if (transferToCheck.id === transferRequest.id) {
      transferFound = true
    }
  })
  expect(transferFound).toBe(false)

  // 이메일이 전송되었는지 확인 ❌
  if (
    transferRequest.options.sendMailOnDecline &&
    transferResponse.status === 'declined'
  ) {
    const wasMailSent = testHelpers.verifyIfMailWasSentToTransfer(
      transferResponse.id,
    )
    expect(wasMailSent).toBe(true)
  }
})
```

### BASIC 원칙으로 좋은 테스트 만들어보기

```javascript
test(‘Should fail’, () => {
```

```javascript
describe(‘transferMoney’)// The operation under test
{
  test(‘When the user has not enough credit, then decline the   request’, () =>
}
```

Pattern: Annotative

- 테스트는 구조화되고 잘 꾸며진 형식으로 테스트의 의도를 명확히 알수 있어야 한다.

---

```javascript
const transferRequest = testHelpers.factorMoneyTransfer({})
```

```javascript
const transferRequest = testHelpers.factorMoneyTransfer({
  credit: 50,
  transferAmount: 100,
})
```

Pattern: Copy only what is necessary

이 라인은 JSON을 단순히 추상화 하고 있기 때문에, 어떤 데이터인지 알수 있는 방법이 없다. 따라서 읽는 사람은 이 데이터에 무언가 문제가 있다고 추정하는 수밖에 없다. 테스트는 보는 사람이 테스트를 벗어나지 않고도 명확하게 이해할 수 있도록 잘 선언되어 있어야 한다.

항상 테스트의 성공 또는 실패에 대한 중요한 세부 정보를 테스트 내에서 가지고 있어야 한다. 나머지는 모두 밖에 있어야 한다. 실제로 다이나믹 팩토리 기법은 특정 부분을 오버라이드 하는 동시에 기본 값을 제공하는데 큰 도움이 될 수 있다.

---

```javascript
serviceUnderTest.options.creditPolicy = 'zero'
```

모든 테스트에 대한 각 서비스를 별도로 생성해서, 글로벌 서비스가 아닌 자신만의 인스턴스를 사용해야 한다.

Pattern: Independent

이 라인은 테스트의 복잡성을 크게 높였다. 모든 테스트가 글로벌 객체를 공유하고, 해당 속성을 수정할 수 있으면 다른 파일에 있는 다른 테스트에서 실패가 발생할 수 있다. 위의 테스트는 시스템의 상태를 잘 알지 못한채로 수정을 가할 수 있다. 이제 한번의 짧은 테스트에서 실패를 확인하는 대신, 같은 파일에 있는 다른 테스트 사례를 훑어보고 어디가 잘못되었는지 확인할 수 있어야 한다.

---

```javascript
transferRequest.howMuch = 110 //
```

```javascript
const amountMoreThanTheUserCredit = 110
transferRequest.howMuch = amountMoreThanTheUserCredit
// 또는
transferRequest = { credit: 50, howMuch: 100 }
```

Pattern: Copy only what is necessary

110은 무엇을 의미하는가? 이는 읽는 사람으로 하여금 무엇인가 추정할 수 없게 한다. 단지 그냥 숫자일 뿐이다. 높은 숫자인가? 낮은 숫자인가? 이 숫자의 의미는 무엇인가? 작성자는 유저의 크레딧 보다 높은 값, 즉 테스트 결과와 깊은 관계가 있는 중요한 정보를 선택하였지만 읽는 사람으로 하여금 이 의미를 전달하지 못했다. 이를 해결하기 위해서는 적당한 네이밍이 있는 변수에 값을 넣어서 사용해야 한다.

---

```javascript
const databaseRepositoryMock = sinon.stub(dbRepository, ‘save’);
// save 함수를 mock 으로 만들어, 이 함수가 불리지 않는지 테스트 한다.
```

```javascript
// Assert - 송금이 이루어지지 않았는지 확인
const senderTransfersHistory = transferServiceUnderTest.getTransfers(
  transferRequest.sender.name,
)
expect(senderTransfersHistory).not.toContain(transferRequest)
```

Pattern: Black box

작성자의 의도는 무엇인가? 거절된 이체가 실제로 DB에 까지 저장되지 않았는지를 확인하기 위해서다. 이를 위해 DB 엑세스 함수를 만들고, 호출되지 않는지를 확인한다. 그러나 이는 불필요한 테스트다. 가능하면 프로덕션에서 발생할 수 있는 시나리오 대로 사용자의 흐름을 테스트하는 것이 좋다.

---

```javascript
expect(transferResponse.currency).toBe(‘dollar’);
 expect(transferResponse.id).not.toBeNull();
 expect(transferResponse.date.getDay()).toBe(new Date().getDay());
```

이런 테스트 코드는 삭제하는 것이 좋다.

Pattern: Single door

이 테스트를 만든 개발자는 전체 송금 흐름에서 모든 버그를 잡기를 원하는 것 같다. 그러나 이는 잘못되었다. 테스트는 짧고 한가지에 집중해서 이루어 져야 한다. 단일 테스트에서 많은 결과를 가지고 긴 흐름을 가져 갈경우 테스트 전체의 가독성을 떨어뜨린다. 테스트가 실패했을 때, 무시해도 될 세부적인 내용일지, 혹은 시스템 전체가 다운되서 그런건지 알수가 없다. 근본적인 원인을 찾기가 어려워 진다는 것이다. 테스트 코드를 만들때에는, 몇가지 아주 세부적인 사항은 희생하는 것이 좋다.

---

```javascript
expect(serviceUnderTest.numberOfDeclined).toBe(1)
```

삭제하는 것이 좋다. 우리는 구현이 어떻게 되었는지는 관심갖지 않아도 된다. 결과물이 괜찮다면, 구현도 괜찮은 것으로 간주한다.

Pattern: black box

코드 내부적으로 `numberOfDeclined`를 사용하여 오류를 저장하고 있는 것으로 보인다. 아마도 이는 오류 보고를 하는데 사용뙤고 있을 것이다. 본질적으로, 구현 세부사항은 단순히 결과 보다 훨씬 많다. 모든 기능, 필드, 상호작용을 검사하면 함수당 수십개 또는 수백개의 테스트가 나올 수도 있고 결과적으로 테스트 파일도 엄청나게 길어질 것이다. 공개적으로 사용 가능한 결과만 확인할 때 테스트 코드의 크기는 줄어들고, 문제를 확인하는게 훨씬 쉬워 진다. 내부적으로 뭔가 잘못되었찌만, 외부에 반영되고 있지 않은 버그는 사용자에게 영향을 주지 않는다. 우리에게 중요한 것은 구현 세부사항이 아닌 결과물이다.

---

```javascript
// 사용자 송금 이력 가져오기
const allUserTransfers = serviceUnderTest.getTransfers(transferRequest.sender.name);
expect(allUserTransfers).not.toBeNull(); // ❌ Overlapping
expect(allUserTransfers).toBeType(‘array’); // ❌ Overlapping
```

```javascript
expect(senderTransfersHistory).not.toContain(transferRequest)
```

Pattern: Single-door

테스트 코드 작성자는 응답 배열이 유효한지 확인하고자 한다. 훌륭하지만, 중복이 일어나고 있다. array에 transfer가 없다면, 다른 모든 테스트는 암묵적으로 증명되는 거나 다름 없다. assertion은 많을 필요가 없다. 코드의 논점을 흐리기 때문이다. 테스트는 적게 노력하고, 코드를 읽는 사람을 배려하며, 중요한 부분을 강조해야 한다. 적은 코드를 사용하여 동일한 신뢰도를 달성할 수 있다면, 그렇게 하는 것이 좋다.

---

```javascript
// 거부된 송금이 사용자 기록에 남는지 확인 ❌
let transferFound = false
allUserTransfers.forEach((transferToCheck) => {
  if (transferToCheck.id === transferRequest.id) {
    transferFound = true
  }
})
```

```javascript
expect(senderTransfersHistory).not.toContain(transferRequest)
```

Pattern: Annotative

이 테스트에서는 거절된 송금이 시스템에 저장되지 않는지, 그리고 검색 가능한지 확인하려고 한다. 따라서 거절된 송금이 시스템에 저장되지 않는지를 확인하기 위해 모든 사용자의 송금을 순환하면서 확인한다. 필수적인 코드이지만, 너무 복잡하다. 테스트에 루프, 조건문, 상속, 트라이 캐치 등 모든 프로그래밍 요소가 들어가 있을 경우 복잡성이 커질 수 있다. 선언적인 코드로 테스트를 단순하게 유지하는 것이 중요하다. 위 변경된 스타일은 세부 구현 사항을 이해할 필요 없이 즉시 읽고 이해 하면 된다. 선언적인 코드를 고수해야 한다.

### 결과물

```javascript
test('크레딧이 없을 경우, 거절된 송금은 송금자의 송금 히스토리에 남아서는 안된다.', () => {
  // Arrange
  const transferRequest = testHelpers.factorMoneyTransfer({
    sender: { credit: 50 },
    transferAmount: 100,
  })
  const transferServiceUnderTest = new TransferService({
    creditPolicy: 'NoCredit',
  })

  // Act
  transferServiceUnderTest.transfer(transferRequest)

  // Assert
  const senderTransfersHistory = transferServiceUnderTest.getTransfers(
    transferRequest.sender.name,
  )
  expect(senderTransfersHistory).not.toContain(transferRequest)
})
```

## 결론

아름다운 테스트란 최소한의 노력으로 적절한 자신감을 얻는 것이다. TDD의 아버지인 Kent Beck도 이렇게 얘기 했다.

> I get paid for code that works, not for tests, so my philosophy is to test as little as possible to reach a given level of confidence
>
> 우리는 테스트가 아닌 프로덕션 코드로 돈을 벌기 때문에, 주어진 신뢰에 도달하기 위해 가능한 한 적게 테스트 하는 것이 제 철학입니다.

테스트 커버리지를 넓히는 것 만큼, 테스트에 쓰이는 노력을 줄이는 것도 중요하다.
