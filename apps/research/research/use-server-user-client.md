---
title: "'use client'는 무슨일을 하는걸까?"
marp: true
paginate: true
theme: default
tags:
  - react
  - nextjs
date: 2025-05-15
description: '🤔'
published: true
art:
  undraw: web-app
---

<!-- _class: deck-cover -->

# `'use client'` & `'use server'` 제대로 이해하기 🚀

[블로그 "What Does 'use client' Do?" (overreacted.io) 요약](https://overreacted.io/what-does-use-client-do/)

---

## 🤔 기존 클라이언트-서버 통신, 뭐가 문제였을까?

- **API 라우트 정의:** 서버에 `/api/user` 같은 엔드포인트 생성
- **`Workspace` 호출:** 클라이언트에서 문자열 기반 URL로 `Workspace` 사용

  ```javascript
  // Client
  fetch('/api/user', {/* ... */})
  ```

- **분리된 프로그램:** 클라이언트와 서버는 완전히 별개의 세상
- **문자열 기반 타입:** 함수 호출이 아닌, 문자열 (URL)에 의존
- **번거로운 데이터 전달:** 서버에서 클라이언트로 초기 데이터 전달 시 `<script>` 태그에 JSON 주입 등

---

## ✨ 새로운 패러다임: "하나의 프로그램"

`'use client'`와 `'use server'` 지시어는...

- 클라이언트와 서버를 **하나의 프로그램**처럼 모델링
- 네트워크 경계를 **모듈 시스템** 안으로
- **타입 안전성**과 **정적 분석**의 이점

> 저자의 주장: 이 지시어들은 구조화 프로그래밍, 일급 함수, `async/await` 급의 혁신! (본인피셜)

---

## `🚀 'use server'` : 서버 함수를 클라이언트에

서버 모듈 상단에 `'use server';`

```javascript
// /actions/updateUsername.js (Server)
'use server'

export async function updateUsername(userId, newName) {
  // ... DB 업데이트 로직 ...
  return {success: true, newName}
}
```

이렇게 써둔다면...?

---

## 클라이언트에서 그냥 `import` 해서 사용

```javascript
// MyComponent.jsx (Client)
import {updateUsername} from './actions/updateUsername'

async function handleClick() {
  const result = await updateUsername('user123', '새이름')
  console.log(result.newName) // '새이름'
}
```

- **동작 원리:** 클라이언트의 `import`는 실제론 서버 함수를 호출하는 **타입화된 `Workspace` (RPC) 프록시**를 생성.
- **장점:** 코드 간결화, 타입 안전성, "모든 참조 찾기" 등 개발 도구 활용 극대화

---

- 타입화된 `Workspace` (RPC) 프록시: 클라이언트 측에 존재하는 대리 함수로서, 호출 시 타입 시스템의 보호를 받으며 마치 로컬 함수처럼 원격 서버의 특정 함수를 네트워크 요청(마치 Workspace처럼)을 통해 실행시키는 중간 매개체
- 한국어로 된 사이트에서 물건의 스펙(타입)을 보고 주문(함수 호출) 하지만, 실제로는 대행 사이트가 해외 판매처(서버)에 요청(RPC, 네트워크 통신) 을 해서 물건을 가져다 주는느낌
- RPC (Remote Procedure Call - 원격 프로시저 호출): 별도의 원격 제어를 위한 코딩 없이 다른 주소 공간에서 함수나 프로시저를 실행할 수 있게하는 프로세스 간 통신 기술

---

## `🎨 'use client'` : 클라이언트 컴포넌트를 서버에

### 클라이언트 컴포넌트 모듈 상단에 `'use client';`

```javascript
// /components/LikeButton.jsx (Client)
'use client'

export function LikeButton({initialLikes}) {
  const [likes, setLikes] = useState(initialLikes)
  // ... 클릭 핸들러 등 ...
  return <button onClick={handleClick}>{likes} Likes</button>
}
```

---

<style scoped>section { font-size: 22px; }</style>

### 서버(예: RSC)에서 `import` 해서 JSX처럼 사용

```javascript
// app/page.jsx (Server Component)
import { LikeButton } from './components/LikeButton';

export default function Page() {
  const initialLikes = await getLikesFromDB();
  return (
    <div>
      <h1>My Post</h1>
      <LikeButton initialLikes={initialLikes} />
    </div>
  );
}
```

- **동작 원리:** 서버의 `import`는 실제 컴포넌트 코드가 아닌 **"클라이언트 참조"** (예: `/components/LikeButton.jsx#LikeButton`)를 가져옴. 이는 최종적으로 클라이언트에 `<script>` 태그와 직렬화된 props로 전달됨.
- **장점:** 서버 렌더링(초기 HTML) + 클라이언트 인터랙션의 자연스러운 결합, 타입 안전성.

---

## 🚪 두 개의 세계, 두 개의 문

`'use client'`와 `'use server'`는 코드를 "표시"하는 것이 아니라, 두 환경 사이에 **"문"을 여는 것!**

- `'use server'`: **클라이언트 → 서버**로의 문 (서버 함수 호출)
  - 프론트엔드는 서버 함수를 HTTP 호출하는 `async` 함수로 인식.
- `'use client'`: **서버 → 클라이언트**로의 문 (클라이언트 코드/컴포넌트 전송)
  - 백엔드는 클라이언트 코드를 `<script>` 태그로 변환될 수 있는 참조로 인식.
- 'use client' === `<script />`
- 'use server' === `fetch()`

---

## 🌟 핵심 비전 및 이점

- **단일 프로그램 모델:** 클라이언트/서버 앱을 두 환경에 걸친 하나의 프로그램으로.
- **모듈 시스템 내 네트워크 추상화:** `import`를 통해 네트워크 경계를 자연스럽게.
- **리액트를 넘어서:** 자바스크립트, 나아가 다른 언어에도 적용 가능한 패턴.
- **개발 경험 향상:** 타입 안전성, 도구 지원, 모듈화 용이.
- **"두 세계를 엮다":** 양쪽 환경의 로직을 가진 재사용 가능한 추상화 구성.

---

## 💡 결론

- `'use client'`와 `'use server'`는 단순히 리액트 서버 컴포넌트의 일부가 아니라, 클라이언트-서버 아키텍처의 **근본적인 변화**를 가져오는 개념
- 이들은 클라이언트와 서버 간의 **상호작용 방식을 근본적으로 바꾸는** 강력한 컨셉
- 네트워크의 복잡성을 추상화하고, 개발자가 **하나의 통합된 애플리케이션**을 구축하는 데 집중할 수 있도록 돕는다.

> **이것이 모듈 시스템 레벨의 RPC + 클라이언트에 코드 보내기의 혁신?!**

---

## `'use client'` 및 `'use server'` 비판적 시각

`'use client'`와 `'use server'` 지시어 및 그 철학은 클라이언트-서버 개발의 새로운 가능성을 보여주지만, 몇 가지 비판적인 관점과 잠재적인 문제점도 고려해볼 필요가 있다.

---

## 1. **추상화의 이면 (Leaky Abstraction) 및 복잡성 증가:**

- **숨겨진 복잡성:** `import` 한 줄로 네트워크 통신을 처리하는 것은 간결해 보이지만, 그 이면에는 RPC, 직렬화, 역직렬화, 오류 처리, 청크 로딩 등 복잡한 메커니즘이 숨어있음. 문제가 발생했을 때, 이 추상화 계층 때문에 디버깅이 어려워짐. 개발자는 여전히 네트워크의 본질을 이해해야한다.
- **새로운 학습 곡선:** 문법은 단순하지만, 이 모델이 실제로 어떻게 동작하는지, 어떤 경우에 사용하고 어떤 경우에 사용하지 말아야 하는지, 상태는 어떻게 동기화되는지 등 새로운 패러다임에 대한 깊은 이해가 필요. 이는 또 다른 학습 비용.

---

## 2. **강한 결합도 (Tight Coupling):**

- **"단일 프로그램"의 함정:** 클라이언트와 서버를 "하나의 프로그램"으로 간주하는 것은 양날의 검. 이는 두 환경 간의 결합도를 높여, 한쪽의 변경이 다른 쪽에 예기치 않은 영향을 미칠 가능성을 키움.
- **독립적인 진화의 어려움:** 서버 API를 여러 종류의 클라이언트(웹, 모바일 앱, 외부 서비스 등)에 제공해야 하는 경우, 특정 프레임워크의 `import` 시스템에 강하게 결합된 방식은 유연성을 저해할 수 있음. 전통적인 REST나 GraphQL API가 제공하는 명확한 경계와 느슨한 결합이 더 유리할 수 있음.

---

## 3. **도구 및 프레임워크 의존성 (일종의 마법):**

- **빌드 시스템 의존:** `'use client'`나 `'use server'`는 특정 빌드 도구(예: Next.js의 웹팩/터보팩 설정)의 "마법"에 크게 의존. 이 도구 없이는 단순한 주석에 불과하며, 해당 생태계를 벗어나기 어렵게 만드는 일종의 벤더 종속성을 야기할 수 있음.
- **표준의 부재:** 아직 웹 표준이 아니므로, 특정 프레임워크나 라이브러리에 국한된 기능일 수 있음. 이는 장기적인 유지보수나 기술 스택 변경 시 위험 요소가 될 수 있음.

---

## 4. **디버깅 및 오류 추적의 어려움:**

- **모호한 오류 지점:** 클라이언트에서 서버 함수 호출 시 발생하는 오류가 네트워크 문제인지, 직렬화 문제인지, 서버 로직의 문제인지, 아니면 프레임워크 내부의 문제인지 파악하기 어려울 수 있음.

---

## 아래 코드에서 에러가 난다면 뭐가 문젤까?

```js
import {someServerFunction} from './serverActions'

try {
  const result = await someServerFunction(data)
  // 성공 로직
} catch (error) {
  // 오류 처리
  console.error('서버 함수 호출 오류:', error)
}
```

- 네트워크? 직렬화? 그냥 서버액션이 잘못됨? 프레임워크 내부 이슈? 인증?
- 파악해야할 지점이 너무 많음

---

<style scoped>section { font-size: 22px; }</style>

## 그래도 이건 뭐가 문제인지 알기는 쉽지

```js
// 클라이언트 측 코드
try {
  const response = await fetch('/api/someEndpoint', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(data),
  })

  if (!response.ok) {
    // HTTP 상태 코드가 200-299 범위가 아닐 때
    // (예: 404 Not Found, 500 Internal Server Error 등)
    // 여기서 오류는 "서버 엔드포인트" 자체와의 통신에서 발생했음을 알 수 있음
    throw new Error(`HTTP error! status: ${response.status}`)
  }
  const result = await response.json()
  // 성공 로직
} catch (error) {
  // 오류 처리
  console.error('API 호출 오류:', error)
  // 만약 fetch 자체가 실패했다면 (예: 네트워크 연결 없음)
  // error 객체는 보통 TypeError를 가지고 있으며, 이걸로 파악가능
}
```

---

<style scoped>section { font-size: 22px; }</style>

## 5. **직렬화의 한계 및 오버헤드:**

- **전송 가능한 데이터 제약:** 클라이언트와 서버 간에 전달되는 모든 데이터는 직렬화 가능해야 함. 이는 기존에도 마찬가지였지만, 추상화로 인해 개발자가 이 제약을 간과하기 쉬워지며, 함수나 복잡한 객체 (Date, Map, Set 등) 전달 시 예기치 않은 문제를 겪을 수 있음.
- **성능 고려:** 자동으로 처리되는 직렬화/역직렬화 과정이 빈번하거나 데이터 크기가 클 경우 성능에 영향을 줄 수 있음.
- 엥 근데 rest api 도 직렬화 해야 하는거 아님?
  - REST API (명시적): 개발자가 Workspace를 사용하고 JSON.stringify()로 요청 본문을 만들거나, response.json()으로 응답을 파싱하는 등 직렬화/역직렬화 과정을 비교적 명시적으로 인지하고 코드를 작성하는 경우가 대부분. 어떤 데이터가 어떻게 변환되는지, 어떤 형식으로 주고받는지에 대해 더 직접적으로 관여.
  - 'use server' (암묵적/추상화): 프레임워크가 직렬화/역직렬화 과정을 내부적으로 "마법처럼" 처리. 개발자는 마치 로컬 함수를 호출하듯 코드를 작성할 수 있어 편리하지만, 이 과정이 숨겨져 있기 때문에 어떤 값이 직렬화 가능한지, 어떤 값이 문제를 일으킬 수 있는지에 대한 고민을 간과하기 쉬울 수 있음. 이 과정이 추상화 되어 있어서 개발자가 간과하고 실수하기 쉬움

---

## 6. **보안 고려사항 증가:**

- `'use server'`를 통해 서버 함수를 클라이언트에서 쉽게 호출할 수 있게 되면, 의도치 않게 내부 로직이나 민감한 작업을 수행하는 함수가 노출될 위험이 있음.
- 물론 이는 개발자의 주의가 필요한 부분이지만, 추상화로 인해 경계가 모호해지면서 실수의 가능성이 커질 수 있음.
- 입력값 검증 및 권한 부여 로직이 더욱 중요.

---

## 결론의 결론

- `'use client'`와 `'use server'`는 특정 사용 사례(특히 풀스택 프레임워크 내에서의 긴밀한 클라이언트-서버 통합)에서 개발 경험을 크게 향상시킬 잠재력을 가지고 있음.
- 하지만 이것이 모든 문제를 해결하는 "은총알"은 아니며, 도입 시 위에서 언급된 복잡성, 결합도, 의존성, 디버깅 문제 등을 충분히 고려해야 한다.
- 프로젝트의 특성, 팀의 경험, 장기적인 유지보수 계획 등을 종합적으로 검토하여 신중하게 접근할 필요가 있음.
- "마법"처럼 보이는 편리함 뒤에는 항상 그에 상응하는 비용이나 제약이 따를 수 있음을 인지해야 함.
