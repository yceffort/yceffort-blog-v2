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

## BASIC

BASIC 이라는 약자를 바탕으로, 한글자씩 살펴보자. 각 글자는 테스트 코드를 만들 때 고려해야하는 원칙을 나타낸다. 그리고, 이 BASIC 원칙을 적용하여 길고 번거로운 테스트 코드가 아닌 간결하고 아름다운 테스트 코드를 짜는 방법을 알아볼 것이다.

### Black-box

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
    const moneyTransferRequest = { sender: 'yceffort@gmail.com', amount: 15000, receiver: 'root@yceffort.kr' }
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

코드 이해에 필요한 모든 세부 사항을 테스트에 포함시켜야 한다. 그 이상일 필요는 없다. 딱 필요한 것만. 중복이 너무 많으면 유지 보수 하기 어려워지는 테스트가 만들어질 수 있다. 반면에 중요한 세부 정보를 외부에 노출하면 많은 파일에 걸쳐 이것이 무슨 의미 인지 검색해야 한다. 예를 들어 100여줄의 JSON 파일을 테스트 하는 코드가 있다고 가정해보자. 모든 테스트에서 이를 붙여 넣는 것은 매우 지겨운 일이다.  그렇다고 외부에서 `transferFactory.getJSON()`으로 압축해버리면 무엇을 하는지 모호 해진다. 데이터가 없으면 테스트 결과, 왜 이 결과가 400이 되어야 하는지를 이해 하는 것이 모호해진다. 고전적인 x-unit 패턴은 이를 'mystery guest'라고 이름 지었다. 눈에 보이지 않는 무언가가 테스트 결과에 영향을 미친다면, 우리는 정확히 알 수가 없다. 오직 필요한 내용만 테스트 코드에 존재해야 한다.

외부에서는 반복 가능한 긴 부분을 따로 추출하고, 테스트에서 어떤 세부사항이 중요한지 자세히 언급함으로써 더 좋은 테스트 코드를 짤 수 있다. 위의 예제를 활용한다면, `transferFactory.getJSON({sender: undefined})` 와 같이 함수의 인수를 활용하는 방식이 있을 수 있다. 이 테스트에서는 `sender`가 없다면 테스트가 오류를 내뱉거나 기타 다른 결과를 내야한다는 것을 적절히 유추할 수 있다.

## 실제 예제로 살펴보기

위 5가지 기본 원리를 적용하여 복잡도가 높고 얽혀있는 테스트를 보다 간단한 테스트로 변환해 보자.

아래는 우리가 테스트할 애플리케이션이다. 송금 서비스가 제공되고 있다. 이 코드는 송금 요청을 확인하고, 몇가지 로직을 적용하며, 이후에 은행 http 서비스에 송금을 요청하고, 마지막으로 이 결과를 DB에 기록한다. 

```javascript
class TransferService {
async transfer({ id, sender, receiver, transferAmount, bankName }) {
    // Validation
    if (!sender || !receiver || !transferAmount || !bankName) {
      throw new Error("Some mandatory property was not provided");
    }
  
    // Handle insufficient credit
    if (this.options.creditPolicy === "zero" && sender.credit < transferAmount) {
      this.numberOfDeclined++; //incrementing interal metric
      return {id, status: "declined",date};
    }

    // All good, let's transfer using the bank 3rd party service + save in DB
    await this.bankProviderService.transfer(sender, receiver, transferAmount, bankName);
    await this.repository.save({id,sender,receiver,transferAmount,bankName});

    return {id,status: "approved", date: new Date()};
  }  
  
  getTransfers(senderName){
    // Query the DB
  }
}
```