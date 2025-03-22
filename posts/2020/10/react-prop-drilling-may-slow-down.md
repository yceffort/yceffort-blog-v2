---
title: 'Prop drilling 해결을 위해 context를 사용하기 전에 구조를 생각해보자.'
tags:
  - react
  - javascript
published: true
date: 2020-10-19 23:59:32
description: '처음부터 구조를 잘 생각해 둔다면 성능상 에 이점을 가져갈 수 있다.'
---

리액트로 웹페이지를 만들때, 최대한 작은 단위로 쪼개서 이를 개별 컴포넌트로 만들고 조립하는 방식을 택한다. 그런 방식을 택하다 보면, 십중 팔구 아래와 같은 구조로 구성하게 된다.

```javascript
function App() {
  return (
    <div>
      <MainNav />
      <Homepage />
    </div>
  )
}
function MainNav() {
  return (
    <div>
      <GitHubLogo />
      <SiteSearch />
      <NavLinks />
      <NotificationBell />
      <CreateDropdown />
      <ProfileDropdown />
    </div>
  )
}
function Homepage() {
  return (
    <div>
      <LeftNav />
      <CenterContent />
      <RightContent />
    </div>
  )
}
function LeftNav() {
  return (
    <div>
      <DashboardDropdown />
      <Repositories />
      <Teams />
    </div>
  )
}
function CenterContent() {
  return (
    <div>
      <RecentActivity />
      <AllActivity />
    </div>
  )
}
function RightContent() {
  return (
    <div>
      <Notices />
      <ExploreRepos />
    </div>
  )
}
```

위 구조는 실제 깃헙 홈페이지를 React로 구성한다고 했을 때의 모습이다. 매우 일반적인 구조고, 작동하는데 문제는 없지만, 아래와 같은 구조를 사용해본다면 어떨까?

```javascript
function App() {
  return (
    <div>
      <MainNav>
        <GitHubLogo />
        <SiteSearch />
        <NavLinks />
        <NotificationBell />
        <CreateDropdown />
        <ProfileDropdown />
      </MainNav>
      <Homepage
        leftNav={
          <LeftNav>
            <DashboardDropdown />
            <Repositories />
            <Teams />
          </LeftNav>
        }
        centerContent={
          <CenterContent>
            <RecentActivity />
            <AllActivity />
          </CenterContent>
        }
        rightContent={
          <RightContent>
            <Notices />
            <ExploreRepos />
          </RightContent>
        }
      />
    </div>
  )
}
function MainNav({children}) {
  return <div>{children}</div>
}
function Homepage({leftNav, centerContent, rightContent}) {
  return (
    <div>
      {leftNav}
      {centerContent}
      {rightContent}
    </div>
  )
}
function LeftNav({children}) {
  return <div>{children}</div>
}
function CenterContent({children}) {
  return <div>{children}</div>
}
function RightContent({children}) {
  return <div>{children}</div>
}
```

이렇게 코드를 바꾼 구조적인 아이디어는, 대부분의 구성요소가 단순히 레이아웃만 담당한다는 것이다. 이들은 상태를 스스로 관리하기 위해 무언가 일을 하는 코드는 적고 (물론 자체적인 상태관리가 필요할 수도 있지만) 단순히 상태를 받아서 이를 나타나야 하는 페이지에 표시하는 역할만 한다.

만약 첫번째 예제와 같은 구조를 가지고 있다면, `prop-drilling`이 발생하게 된다. `<App/>` 에서 `<HomePage />`로 다시, `<CenterContent />`에서 `<AllActivity />`로 가는 등, prop을 넘기고 넘기는 과정이 반복된다. 이런 구조를 보게 되면 사람들은 아마도 React API인 `context` 를 사용하여 해결하려 할 것이다. 하지만 그러기 전에, 좀더 간단하게 생각해볼 필요가 있다. 자체적인 상태관리가 필요없고, 단순히 레이아웃인 컴포넌트라면 `prop`을 직접적으로 넘기면 된다. `<App />` 에서 `<AllActivity/>`로.

너무 많은 사람들이 `prop-drilling`에서 `context`로 섣불리 넘어가버리려고 한다. 더 많은 컴포넌트를 염두에 두고, 구성요소를 구조화 한다면 더 유지 보수가 쉬워지고, 성능및 관리 문제가 줄어들 것이다.

출처: https://epicreact.dev/one-react-mistake-thats-slowing-you-down
