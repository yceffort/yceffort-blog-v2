---
title: 'nextjs에서 api에러 핸들링하기'
tags:
  - javascript
  - typescript
  - nextjs
published: true
date: 2021-10-22 18:30:22
description: '결국 여기까지 와버렸'
---

nextjs로 동작하는 일반적인 애플리케이션을 상상하자면, 아래와 같은 요소를 가정하고 개발할 수 있을 것이다.

- api 호출이 있으며, 경우에 따라서 api 호출 과정에서 인증 등의 에러가 발생함
  - 위 인증 에러가 발생할 경우, 원래 가려던 페이지가 아닌 특정 페이지로 이동시켜야 함
- api 호출은 서버사이드, 클라이언트 사이드에서 모두 일어날 수 있으며 두 경우에 모두 위 처리를 해야함

## 1. 에러 정의 

먼저 api 호출시 발생할 수 있는 에러에 대해 정의해야 한다. 가장 일반적인 에러는 인증 에러가 있을 것이다. api 호출시 정상적인 응답 (200) 이 아닌, 에러 응답이 왔을 때 에러를 throw 하는 코드를 짜보자.

### api.ts

```typescript
import axios, { AxiosRequestConfig, AxiosResponse } from 'axios';
import { AuthError, ForbiddenError } from './error';

export interface RequestConfig extends AxiosRequestConfig {
  suppressStatusCode?: number[];
}

function AxiosAuthInterceptor<T>(response: AxiosResponse<T>): AxiosResponse {
  const status = response.status;

  if (status === 403) {
    throw new ForbiddenError(status);
  }

  if (status === 401) {
    throw new AuthError(status);
  }

  return response;
}

export default async function withAxios(requestConfig: RequestConfig) {
  const instance = axios.create();

  instance.interceptors.response.use((response) =>
    AxiosAuthInterceptor(response)
  );

  const response = await instance.request({
    ...requestConfig,
    baseURL: `${process.browser ? process.env.HOST_URL : ''}/api`,
    validateStatus: (status) =>
      [...(requestConfig.suppressStatusCode || [])].includes(status) ||
      status < 400,
  });

  return response
}
```

### error.ts

