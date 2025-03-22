---
title: 'Nodejs에서 로깅하기'
tags:
  - javascript
  - nodejs
published: true
date: 2021-02-26 19:56:41
description: '어쩌다 보니 nodejs도 하고 있🤣'
---

\*\* 주의: 제가 하고 있는 프로젝트의 방향성과는 다를 수 있습니다

로깅은 서버사이드에서 중요한 처리 중 하나다. 서버에서 어떤 일들이 일어나고 있는지 알 수 있고, 의도치 않은 동작이나 버그가 발생했을 경우 재빠르게 원인을 찾을 수 있다. 본문에서는 nodejs에서 로깅을 남기는 몇가지 좋은 사례를 알아본다.

## Table of Contents

## 0. 시작하기전에

한가지 알아둬야 할 것은, 모든 정보를 로깅으로 남겨서는 안된다는 것이다. 로깅이 성능과 데이터 용량에 영향을 미치는 것도 있지만, 그것보다도 더 중요한 것은 주민등록번호, 카드번호, 암호와 같은 민감한 정보는 절대로 남겨서는 안된다.

## 1. console.log로 시작하기

자바스크립트를 배운 순간부터 지금까지 (...) 가장 많이 사용하 있는 `console.log`다. 개인적으로도 급한 프로젝트를 휙휙 처리하다보면 로깅을 `console.log`로 남기곤 했다 (죄송합니다). 조금 더 나아가서 `console.error` `console.group` 등을 사용하는 경우도 있다. `console.log`는 코드 자체가 없어보이는 것은 둘째치고 [자바스크립트의 성능에 안좋은 영향을 미친다.](https://stackoverflow.com/a/11426318) 따라서 본격적으로 로깅을 원한다면, 진짜 로깅에 사용되는 라이브러리를 사용하는 것이 좋다.

## 2. 로그 라이브러리 사용하기

node 진영에서 가장 많이 사용되는 로깅 라이브러리는 크게 다음과 같다.

- [winston](https://github.com/winstonjs/winston): 로그를 별도의 데이터베이스 등에 저장하고 싶을 때. 내가 거친 프로젝트가 대부분 이걸 썼던 것 같다.
- [bunyan](https://github.com/trentm/node-bunyan): CLI가 기가막히다
- [log4js](https://github.com/log4js-node/log4js-node): 로그 스트림, aggregator 등 지원

```javascript
const winston = require('winston')
const config = require('./config')

const enumerateErrorFormat = winston.format((info) => {
  if (info instanceof Error) {
    Object.assign(info, {message: info.stack})
  }
  return info
})

const logger = winston.createLogger({
  level: config.env === 'development' ? 'debug' : 'info',
  format: winston.format.combine(
    enumerateErrorFormat(),
    config.env === 'development'
      ? winston.format.colorize()
      : winston.format.uncolorize(),
    winston.format.splat(),
    winston.format.printf(({level, message}) => `${level}: ${message}`),
  ),
  transports: [
    new winston.transports.Console({
      stderrLevels: ['error'],
    }),
  ],
})

module.exports = logger
```

로깅 라이브러리는 일반적인 `console.log`을 사용하는 것보다 여러 측면에서 좋다. 성능에도 더 좋고, 기능도 다양하고, 쉽게 알록달록하게 만들수도 있다.(?)

## 3. Morgan으로 node내 http 요청 로깅하기

또 다른 좋은 습관 중 하나는 nodejs 애플리케이션 내 http 요청을 로깅하는 것이다. 이를 위한 좋은 라이브러리가 바로 [morgan](https://github.com/expressjs/morgan) 이다. 이 도구는 서버 로그를 가져와서 체계화 시켜 읽기 쉽게 만들어 준다.

```javascript
const morgan = require('morgan')
app.use(morgan('dev'))
```

이미 정의된 문자열 포맷을 사용하려면

```javascript
morgan('tiny')
```

를 쓰면 된다.

### winston + morgan

```javascript
const morgan = require('morgan')
const config = require('./config')
const logger = require('./logger')

morgan.token('message', (req, res) => res.locals.errorMessage || '')

const getIpFormat = () => (config.env === 'production' ? ':remote-addr - ' : '')
const successResponseFormat = `${getIpFormat()}:method :url :status - :response-time ms`
const errorResponseFormat = `${getIpFormat()}:method :url :status - :response-time ms - message: :message`

const successHandler = morgan(successResponseFormat, {
  skip: (req, res) => res.statusCode >= 400,
  stream: {write: (message) => logger.info(message.trim())},
})

const errorHandler = morgan(errorResponseFormat, {
  skip: (req, res) => res.statusCode < 400,
  stream: {write: (message) => logger.error(message.trim())},
})

module.exports = {
  successHandler,
  errorHandler,
}
```

위 예제에서 보이는 것처럼, 두 라이브러리를 함께 쓰기 위해서는 단순히 winston에 morgan에서 나온 결과물을 넘겨주면 된다.

## 4. 로그레벨 정의하기

로그의 이벤트를 구분하기 위해서 로그 수준을 체계적으로 정리하는 것이 중요하다. 이를 잘 정리해두면, 필요한 정보만 빼다가 쉽게 알아낼 수 있다. 로그 수준에는 여러개가 있지만, 다음과 같이 나누는 것이 일반적이다.

- error
- warning
- info
- debug

## 5. 로그 관리 시스템 사용하기

애플리케이션의 크기에 따라서 별도로 로그를 관리하는 시스템을 가져다 쓰는 것이 좋을 수도 있다. (물론 그냥 cat grep 할 수도 있겠지만) 로그 관리 시스템을 사용하면, 실시간으로 로그를 추적하고 분석할 수 있으므로 코드를 개선하는데 용이하다. 주로 사용하는 프로그램은 다음과 같다.

- [Sentry](https://sentry.io/welcome/) 사실 이거밖에 안써봄
- [Loggly](https://www.loggly.com/)
- [McAfee Enterprise Log Search](https://www.mcafee.com/enterprise/ko-kr/products/enterprise-log-search.html)
- [Graylog](https://www.graylog.org/)
- [Splunk](https://www.splunk.com/)
- [Logmatic](https://logmatic.com/)
- [Logstash](https://www.elastic.co/kr/logstash) 생각해보니 엘라스틱 서치 쓰느라 이것도 써본듯

## 6. 상태 모니터링 도구

상태 모니터링 도구는 서버 성능을 추적하고, 애플리케이션 충돌 또는 다운타임의 원인을 식별해 낼 수 있는 좋은 방법이다. 대부분의 도구는 오류 스택 추적과, 성능 모니터링 기능을 제공한다. Nodejs에서 유명한 도구는 다음과 같다.

- [PM2](https://pm2.keymetrics.io/): 가장 유명한 도구
- [Sematext](https://sematext.com/)
- [Appmetrics](https://www.app-metrics.io/)
- [ClinicJS](https://clinicjs.org/)
- [AppSignal](https://appsignal.com/)
- [Express Status Monitor](https://github.com/RafalWilinski/express-status-monitor)

## 7. 결론

로그도 확인 할 필요 없이 24/365로 잘돌아가는 서비스가 있다면 좋겠지만, 사실 그런 인프라는 존재 하지 않는다. 운영환경을 모니터링하고, 오류를 줄이기 위해서 로깅은 개발자들에게 필수다.
