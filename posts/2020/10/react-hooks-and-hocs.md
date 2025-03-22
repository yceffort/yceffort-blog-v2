---
title: '리액트의 Hooks과 HOC, HOC의 사용이 복잡해지는 경우'
tags:
  - react
  - javascript
published: true
date: 2020-10-19 21:19:43
description: 'HOC는 좋지만, hooks을 사용하는 습관을 기르자.'
---

요즘 대부분의 리액트 코드는 함수형 컴포넌트와 리액트 hooks의 조합으로 개발된다. 그러나 여전히 [higher-order components(이하 HOC)](https://ko.reactjs.org/docs/higher-order-components.html)는 클래스형, 그리고 함수형 모두에 적용할 수 있다. 따라서 HOC는 레거시와 모던한 리액트 컴포넌트 사이에서 재사용 가능성을 높이며 쓸 수 있는 훌륭한 다리 역할을 하고 있다.

그러나 때때로 HOC의 사용은 자제해야하며, 몇몇 문제들은 hooks만으로도 해결할 수 있다.

## HOC와 HOOKS: Prop에서 오는 혼동

조건부 렌더링 기능을 적용하기 위해 HOC를 사용한다고 가정해보자.

```javascript
import * as React from 'react'

const withError = (Component) => (props) => {
  if (props.error) {
    return <div>Something went wrong ...</div>
  }

  return <Component {...props} />
}

export default withError
```

에러가 없는 경우, HOC가 어떻게 모든 props를 넘기는지 보자. 에러가 없는 경우에 정상적으로 작동할 것이지만, 다음 컴포넌트에 전달해야 하는 props 들이 굉장히 많으며, 이 props를 모두 신경쓰기는 어렵다.

```javascript
import * as React from 'react'

const withError =
  (Component) =>
  ({error, ...rest}) => {
    if (error) {
      return <div>Something went wrong ...</div>
    }

    return <Component {...rest} />
  }

export default withError
```

위 코드 또한 `error` prop을 제거하고도 정상적으로 작동한다. 그러나 위 버전 또한 HOC를 사용하는 경우 오는 Props의 혼동을 피할 수는 없다. 전개 연산자를 사용하여 HOC에 props를 넘겨주었지만, 이 props들이 어디에 필요한지 명확히 하기가 굉장히 어렵다.

이는 HOC의 첫번째 약점이다. HOC가 어떤 컴포넌드들과 합성되어 있는지 빠르게 알 수 없기 때문에, 어떤 컴포넌에 어떤 것을 넘겨야 할지 예측이 어렵다. 예를 들어, loading 인디케이터가 있는 HOC를 하나더 만들어 보자.

```javascript
import * as React from 'react'

const withLoading =
  (Component) =>
  ({isLoading, ...rest}) => {
    if (isLoading) {
      return <div>Loading ...</div>
    }

    return <Component {...rest} />
  }

export default withLoading
```

이제 두개의 HOC를 사용해야 한다면, 아래와 같이 써야할 것이다.

```javascript
const DataTableWithFeedback = compose(
  withError,
  withLoading,
)(DataTable);

const App = () => {
  ...

  return (
    <DataTableWithFeedback
      columns={columns}
      data={data}
      error={error}
      isLoading={isLoading}
    />
  );
};
```

HOC에 대해 세세히 알고 있지 않다면, 어떤 props가 어떤 HOC에 넘어가는지 알수 없다. 한단계 더 나아가 이제 데이터를 가져오는 HOC까지 있다고 가정해보자.

```javascript
const DataTableWithFeedback = compose(
 withFetch,
 withError,
 withLoading,
)(DataTable);

const App = () => {
 ...

 const url = 'https://api.mydomain/mydata';

 return (
   <DataTableWithFeedback
     url={url}
     columns={columns}
   />
 );
};
```

갑자기 `withFetch`를 사용하면서 `error`와 `isLoading`이 필요 없게 되었다. 정확히는, 아래와 같이 props가 흘러가게 된다.

```bash

App     withFetch   withError   withLoading   DataTable

        data->      data->      data->        data
url->   error->     error
        isLoading-> isLoading-> isLoading
```

`withFetch`는 내부적으로 `error`와 `isLoading`을 처리하고, 이를 위해 `withLoading`과 `withError`를 필요로 하게 된다. 이에 대한 이해가 부족할 경우 버그를 만들어낼 가능성이 존재한다.

결과적으로, HOC를 통해서 넘어가는 props는 블랙박스로 남게 되어 이를 이해하기 위해서는 꽤나 많은 주의를 기울여야 한다. HOC에 대한 자세한 이해가 없이는, HOC 들 사이에서 무슨일이 일어나고 있는지 알기 어렵다.

반면에 리액트 hook 에서는 어떻게 처리하는지 살펴보자.

```javascript
const App = () => {
  const url = 'https://api.mydomain/mydata'
  const {data, isLoading, error} = useFetch(url)

  if (error) {
    return <div>Something went wrong ...</div>
  }

  if (isLoading) {
    return <div>Loading ...</div>
  }

  return <DataTable columns={columns} data={data} />
}
```

리액트 hooks을 사용하면, 블랙박스 안으로 들어가는 `url`과 이를 통해 나오게 되는 `data`, `isLoading`, `error`를 모두 볼수 있다. `useFetch`가 어떻게 구현되어 있는지는 몰라도, 이 함수를 통해 들어가는 input과 output을 명확하게 볼 수 있다. `useFetch`가 다른 HOC와 같이 블랙박스처럼 취급 된다하더라도, 복잡했던 HOC와는 다르게 단 한줄로 모든 것을 표현하고 있다. 그러나 HOC를 합성해서 사용하는 경우 input과 output이 명확하지 않았다.

## HOC와 HOOKS: 이름 간의 충돌

만약 컴포넌트에서 두개의 prop이 사용된다면, 후자가 전자를 엎어버리게 된다. (그전에 에러가 나겠지만)

```javascript
<Headline text="Hello World" text="Hello React" />
```

만약 HOC에서 다음과 같은 일이 벌어진다면 어떻게 될까?

```javascript
const UserWithData = compose(
  withFetch,
  withFetch,
  withError,
  withLoading,
)(User);

const App = () => {
  ...

  const userId = '1';

  return (
    <UserWithData
      url={`https://api.mydomain/user/${userId}`}
      url={`https://api.mydomain/user/${userId}/profile`}
    />
  );
};
```

위 예제는 두 번의 fetch를 위해서 HOC를 합성한 예제다. 그러나 앞서 언급한 것처럼, 동일한 prop이 존재한다면 후자만 유효하게 된다. 이를 위해서, `withFetch`의 `url`을 배열로 만들었다고 가정해보자. 그렇다고해서 문제가 해결되는 것이 아니다.

- 모든 요청이 끝났을 때 `loading` 인디케이터가 사라져야 하는가?
- 하나만 실패해도 에러 페이지를 보여줘야 하는가?
- 만약 한 요청이 다른 요청에 의존하면 어떻게 되는가?

이 문제를 HOOKS로 풀어보도록 하자.

```javascript
const App = () => {
  const userId = '1';

  const {
    data: userData,
    isLoading: userIsLoading,
    error: userError
  } = useFetch(`https://api.mydomain/user/${userId}`);

  const {
    data: userProfileData,
    isLoading: userProfileIsLoading,
    error: userProfileError
  } = useFetch(`https://api.mydomain/user/${userId}/profile`);

  if (userError || userProfileError) {
    return <div>Something went wrong ...</div>;
  }

  if (userIsLoading) {
    return <div>User is loading ...</div>;
  }

  const userProfile = userProfileIsLoading
    ? <div>User profile is loading ...</div>
    : <UserProfile userProfile={userProfileData} />;

  return (
    <User
      user={userData}>
      userProfile={userProfile}
    />
  );
};
```

앞선 HOC 예제에 비해 복잡도도 줄어들고, 조건별로 처리할 수 있는 여지도 많아졌다.

HOC를 사용할 때는, 내부적으로 같은 prop 명을 사용하고 있는 컴포넌트가 있는지 조심해야 한다.

## HOC와 HOOKS: 의존성

HOC는 강력하지만, 때로는 너무 강력할 때가 있다. HOC는 부모로 부터 props를 받거나, 혹은 컴포넌트 내부에서 처리하는 방법으로 변수를 받을 수 있다. 아래 예제를 살펴보자.

```javascript
const withLoading =
  ({loadingText}) =>
  (Component) =>
  ({isLoading, ...rest}) => {
    if (isLoading) {
      return <div>{loadingText ? loadingText : 'Loading ...'}</div>
    }

    return <Component {...rest} />
  }

const withError =
  ({errorText}) =>
  (Component) =>
  ({error, ...rest}) => {
    if (error) {
      return <div>{errorText ? errorText : 'Something went wrong ...'}</div>
    }

    return <Component {...rest} />
  }
```

```javascript
const DataTableWithFeedback = compose(
  withError({ errorText: 'The data did not load' }),
  withLoading({ loadingText: 'The data is loading ...' }),
)(DataTable);

const App = () => {
  ...

  return (
    <DataTableWithFeedback
      columns={columns}
      data={data}
      error={error}
      isLoading={isLoading}
    />
  );
};
```

`errorText`와 `loadingText`를 각각 받아서, 에러와 로딩 문구를 커스터마이징 할 수 있도록 처리했다. 그러나 이제 props를 받는 곳이 한군데 가 더 늘어나서 혼란이 가중되었다. 만약 여기에서, `userId` 까지 필요하다면 어떻게 될까?

```javascript
const UserWithData = compose(
  withFetch(props => `https://api.mydomain/user/${props.userId}`),
  withFetch(props => `https://api.mydomain/user/${props.userId}/profile`),
)(User);

const App = () => {
  ...

  const userId = '1';

  return (
    <UserWithData
      userId={userId}
      columns={columns}
    />
  );
};
```

만약 여기서 더 나아가 두번째 요청이 첫번째 요청에 의존적이라고 한다면 어떻게 될끼?

```javascript
const UserProfileWithData = compose(
  withFetch(props => `https://api.mydomain/users/${props.userId}`),
  withFetch(props => `https://api.mydomain/profile/${props.profileId}`),
)(UserProfile);

const App = () => {
  ...

  const userId = '1';

  return (
    <UserProfileWithData
      columns={columns}
      userId={userId}
    />
  );
};
```

HOC간에 강한 연결을 만들어 문제를 해결했지만, HOC간에 강결합을 통해서 문제를 해결하는 것은 굉장히 어렵고 혼란이 가중된다.

그러나 Hooks을 쓰면 더 쉽게 해결할 수 있다.

```javascript
const App = () => {
  const userId = '1';

  const {
    data: userData,
    isLoading: userIsLoading,
    error: userError
  } = useFetch(`https://api.mydomain/user/${userId}`);

  const profileId = userData?.profileId;

  const {
    data: userProfileData,
    isLoading: userProfileIsLoading,
    error: userProfileError
  } = useFetch(`https://api.mydomain/user/${profileId}/profile`);

  if (userError || userProfileError) {
    return <div>Something went wrong ...</div>;
  }

  if (userIsLoading || userProfileIsLoading) {
    return <div>Is loading ...</div>;
  }

  return (
    <User
      user={userData}>
      userProfile={userProfileData}
    />
  );
};
```

Hook은 오직 함수형 컴포넌트에서만 직접적으로 쓰이기 때문에, 데이터를 넘기기 유용하다. 또한 블랙박스가 존재하지 않고, custom hook간에 데이터를 넘기는 것 또한 명확하게 볼 수 있다. 의존성이 있는 경우에는, hook을 쓰는 것이 더 코드에 많은 이점을 가져올 수 있다.
