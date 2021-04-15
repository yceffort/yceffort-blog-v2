---
title: Nodejs에서의 암/복호화
tags:
  - TIL
  - javascript
  - nodejs
published: true
date: 2020-06-09 09:56:46
description:
  '### Nodejs에서의 암호화와 복호화 만약 같은 텍스트로 암호화를 동일하게 시도했을 때, 암호화된 결과가 동일하게
  나온다면 이 암호화는 굉장히 약한 암호화라 볼 수 있다. 강력한 암호화는 매번 암호화를 시도할 때마다 (설령 같은 텍스트라 할지라도) 다른
  결과가 나와야 한다.  물론, 어쨌든 암호화 되어 있다는 사실 만으로도 만족할 수도 있다. 그러나 ...'
category: TIL
slug: /2020/06/encryption-decryption-nodejs/
template: post
---

### Nodejs에서의 암호화와 복호화

만약 같은 텍스트로 암호화를 동일하게 시도했을 때, 암호화된 결과가 동일하게 나온다면 이 암호화는 굉장히 약한 암호화라 볼 수 있다. 강력한 암호화는 매번 암호화를 시도할 때마다 (설령 같은 텍스트라 할지라도) 다른 결과가 나와야 한다.

물론, 어쨌든 암호화 되어 있다는 사실 만으로도 만족할 수도 있다. 그러나 공격자가 암호화된 데이터에 접근했을 때 유사한 패턴을 찾게 된다면 그것으로 패턴을 분석할 수 있게 된다. 공격자가 입력한 텍스트가 동일하게 암호화 되었고, 그 암호화된 텍스트가 DB에서 발견 된다면, 그 텍스트는 추론할 수 있게 되며 더이상 암호화는 유효하지 않게 된다. 따라서 암호화된 출력이 항상 다르게 하기 위해서는, 임의성 (randomness)을 추가해야 한다.

암호화가 매번 다른 결과를 만들어 주는 것이 [Initialize Vector (IV) 초기화 벡터](https://en.wikipedia.org/wiki/Initialization_vector)이다. 암호화 알고리즘에 IV를 추가하여 위에서 언급한 임의성을 얹을 수 있고, 이 임의성은 암호화가 매번 다른 결과를 만들수 있도록 도와 준다. 강력한 암호화를 위해서는, 매번 암호화를 시도할 때마다 서로다른 랜덤값을 IV로 제공해야 한다. 이는 암호 해싱의 [salt](<https://en.wikipedia.org/wiki/Salt_(cryptography)>)와 유사하다.

단순함을 유지하면서, 암호화된 데이터에 대한 단일 데이터베이스 필드와 값을 사용하기 위해, 암호화전에 IV를 준비하여 암호화된 결과에 맞게 준비한다. 그런 다음, 암호를 해독하기전에, IV를 읽고 이를 키와 함께 사용하면 된다.

```javascript
'use strict'

const crypto = require('crypto')

const ENCRYPTION_KEY =
  process.env.ENCRYPTION_KEY || 'abcdefghijklmnop'.repeat(2) // Must be 256 bits (32 characters)
const IV_LENGTH = 16 // For AES, this is always 16

function encrypt(text) {
  const iv = crypto.randomBytes(IV_LENGTH)
  const cipher = crypto.createCipheriv(
    'aes-256-cbc',
    Buffer.from(ENCRYPTION_KEY),
    iv,
  )
  const encrypted = cipher.update(text)

  return (
    iv.toString('hex') +
    ':' +
    Buffer.concat([encrypted, cipher.final()]).toString('hex')
  )
}

function decrypt(text) {
  const textParts = text.split(':')
  const iv = Buffer.from(textParts.shift(), 'hex')
  const encryptedText = Buffer.from(textParts.join(':'), 'hex')
  const decipher = crypto.createDecipheriv(
    'aes-256-cbc',
    Buffer.from(ENCRYPTION_KEY),
    iv,
  )
  const decrypted = decipher.update(encryptedText)

  return Buffer.concat([decrypted, decipher.final()]).toString()
}

const text = 'hello my name is yceffort'
const encryptResult = encrypt(text)
console.log('encrypt result:', encryptResult)

const decryptResult = decrypt(encryptResult)
console.log('decrypt result:', decryptResult)
```

```
encrypt result: bad1fdcaa253c63bc3c8a5aadb7d8913:97a0a7ab35c84d3e52b56afeb717ad888669c67132cc97f941f7969ec52a1732
decrypt result: hello my name is yceffort
```

[여기](https://gist.github.com/vlucas/2bd40f62d20c1d49237a109d491974eb)의 코드를 참고했다.

IV에 대해서 설명하려면 한도 끝도 없고 내 두뇌 용량도 초과하므로, 자세한 설명은 생략하지만 서도 - 암호화를 할 때 꼭꼭꼭꼮 IV를 쓰도록 하자. [이미 nodejs의 crypto 라이브러리에서는 iv 를 쓰지 않는 코드는 deprecated 되었다.](https://nodejs.org/api/crypto.html#crypto_crypto_createcipher_algorithm_password_options)
