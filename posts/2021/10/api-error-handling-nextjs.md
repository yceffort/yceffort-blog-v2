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
  return (
    object instanceof ApiError &&
    ('redirectUrl' in object || 'notFound' in object)
  )
}

export class ApiError extends Error {
  redirectUrl: string = ''

  notFound: boolean = false
}

export class NotFoundError extends ApiError {
  name = 'NotFoundError'

  message = '찾을 수 없습니다.'

  notFound = true
}

export class ForbiddenError extends ApiError {
  name = 'ForbiddenError'

  message = '인증처리에 실패했습니다.'

  redirectUrl = '/error'
}

export class AuthError extends ApiError {
  name = 'AuthError'

  message = '인증되지 않은 사용자입니다.'

  redirectUrl = '/auth'
}
```

일단 자바스크립트의 기본 Error Class를 확장해서 우리가 사용할 커스텀 에러를 만들었다.

### api.ts

```typescript
import axios, { AxiosRequestConfig, AxiosResponse } from 'axios'
import { AuthError, ForbiddenError } from './error'

// axios는 400 이상의 status 가 오면 다 에러를 리턴한다.
// 이를 커스텀 할 수 있도록 하여 개발자가 정의한 에러일 때만 에러를 던질 수 있도록 인수를 받는다.
export interface RequestConfig extends AxiosRequestConfig {
  suppressStatusCode?: number[]
}

// axios에 넣을 interceptor.응답에 따라 각각 다른 처리를 한다.
// 굳이 axios가 아니더라도 다른 처리를 할 수 있음.
function AxiosAuthInterceptor<T>(response: AxiosResponse<T>): AxiosResponse {
  const status = response.status

  if (status === 404) {
    throw new NotFoundError()
  }

  if (status === 403) {
    throw new ForbiddenError()
  }

  if (status === 401) {
    throw new AuthError()
  }

  return response
}

export default async function withAxios(requestConfig: RequestConfig) {
  const instance = axios.create()

  instance.interceptors.response.use((response) =>
    AxiosAuthInterceptor(response),
  )

  const response = await instance.request({
    ...requestConfig,
    baseURL: `${!process.browser ? HOST_URL : ''}/api`,
    validateStatus: (status) =>
      [...(requestConfig.suppressStatusCode || [])].includes(status) ||
      status < 500,
  })

  return response
}
```

이제 api는 준비되었으니, 에러를 핸들링할 준비를 해보자.

## 2. 에러 핸들링

`getServerSideProps`는 서버에서 별도로 실행되는 영역이므로, 여기에서 그냥 throw error가 발생하면 nextjs의 에러페이지에 도착해버릴 것이다. 따라서 이를 적절하게 처리해줄 필요가 있다.

### withServerSideProps

```typescript
import { GetServerSideProps, GetServerSidePropsContext } from 'next'
import { ApiError, isInstanceOfAPIError } from './error'

export default function withGetServerSideProps(
  getServerSideProps: GetServerSideProps,
): GetServerSideProps {
  return async (context: GetServerSidePropsContext) => {
    try {
      // getServerSideProps를 평소대로 실행
      // await 를 꼭 붙여서 try catch에서 에러가 잡히도록
      return await getServerSideProps(context)
    } catch (error) {
      // apiError라면
      if (isInstanceOfAPIError(error)) {
        const { redirectUrl, notFound } = error
        // 404로 보내거나
        if (notFound) {
          return {
            notFound: true,
          }
        }
        // 원하는 페이지로 보낸다.
        // https://nextjs.org/docs/basic-features/data-fetching#getserversideprops-server-side-rendering 참고
        return {
          redirect: {
            destination: redirectUrl,
            permanent: false,
          },
        }
      }

      console.error('unhandled error', error)

      throw error
    }
  }
}
```

에러를 처리할 higher order component를 만들었으니, 이제는 getServerSideProps를 이 컴포넌트로 감싸주기만 하면 된다.

```typescript
import Head from 'next/head'
import { GetServerSideProps } from 'next'
import styles from '../styles/Home.module.css'
import withGetServerSideProps from '../withServerSideProps'

export default function Home() {
  return (
    <div>
      <h1>결과</h1>
    </div>
  )
}

export const getServerSideProps: GetServerSideProps = withGetServerSideProps(
  async (ctx) => {
    const { status = 200 } = ctx.req?.query
    const response = await fetch(`/api/hello?status=${status}`)

    const result = await response.json()

    return {
      props: {
        result,
      },
    }
  },
)
```

이제 `getServerSideProps`를 사용할 때 `withGetServerSideProps`로 감싸준다면, api에서 에러가 나도 적절하게 redirect 처리를 해줄 것이다.

### 클라이언트

이제 똑같이 클라이언트에서도 처리가 필요하다. 여기에서는 ErrorBoundary를 사용할 것이다.

```typescript
import Router from 'next/router'
import { isInstanceOfAPIError } from './error'
import Error from './pages/error'
import Page404 from './pages/404'

type ErrorBoundaryProps = React.PropsWithChildren<{}>

interface ErrorBoundaryState {
  error: Error | null
}

const errorBoundaryState: ErrorBoundaryState = {
  error: null,
}

export default class ErrorBoundary extends React.Component<
  ErrorBoundaryProps,
  ErrorBoundaryState
> {
  constructor(props: ErrorBoundaryProps) {
    super(props)
    this.state = errorBoundaryState
  }

  static getDerivedStateFromError(error: Error) {
    console.error(error)
    return { error }
  }

  private resetState = () => {
    if (this.state.error) {
      this.setState(errorBoundaryState)
    }
  }

  private setError = (error: Error) => {
    console.error(error)

    this.setState({ error })
  }

  // 전역 에러 중 캐치하지 못한 에러
  private handleError = (event: ErrorEvent) => {
    this.setError(event.error)
    event.preventDefault?.()
  }

  // promise 중 캐치하지 못한 rejection
  private handleRejectedPromise = (event: PromiseRejectionEvent) => {
    event?.promise?.catch?.(this.setError)
    event.preventDefault?.()
  }

  componentDidMount() {
    window.addEventListener('error', this.handleError)
    window.addEventListener('unhandledrejection', this.handleRejectedPromise)

    Router.events.on('routeChangeStart', this.resetState)
  }

  componentWillUnmount() {
    window.removeEventListener('error', this.handleError)
    window.removeEventListener('unhandledrejection', this.handleRejectedPromise)

    Router.events.off('routeChangeStart', this.resetState)
  }

  render() {
    const { error } = this.state

    if (isInstanceOfAPIError(error)) {
      const { redirectUrl, notFound } = error

      if (notFound) {
        return <Page404 />
      }

      if (redirectUrl) {
        window.location.href = redirectUrl
      }

      return <Error />
    }

    console.log('unhandled client error')

    return this.props.children
  }
}
```

이제 클라이언트와 서버사이드 모두에서 우리가 공통으로 정의한 에러에 대해 처리를 할 수 있게되었다.
