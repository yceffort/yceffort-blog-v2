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

### error.ts

```typescript
export function isInstanceOfAPIError(object: unknown): object is ApiError {
  return object instanceof ApiError && 'code' in ApiError
}

export class ApiError extends Error {
  name: string;

  message: string;

  constructor(readonly code: number) {
    super();
  }
}

export class ForbiddenError extends ApiError {
  name = 'ForbiddenError';

  message = '인증처리에 실패했습니다.';
}

export class AuthError extends ApiError {
  name = 'AuthError';

  message = '인증되지 않은 사용자입니다.';
}
```

일단 자바스크립트의 기본 Error Class를 확장해서 우리가 사용할 커스텀 에러를 만들었다.

### api.ts

```typescript
import axios, { AxiosRequestConfig, AxiosResponse } from 'axios';
import { AuthError, ForbiddenError } from './error';

// axios는 400 이상의 status 가 오면 다 에러를 리턴한다.
// 이를 커스텀 할 수 있도록 하여 개발자가 정의한 에러일 때만 에러를 던질 수 있도록 인수를 받는다.
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



