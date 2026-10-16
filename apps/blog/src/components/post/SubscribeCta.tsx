'use client'

import type {ReactNode} from 'react'
import {useSyncExternalStore} from 'react'

import {usePushSubscription} from '@/hooks/usePushSubscription'

// iOS는 홈 화면에 추가한 앱에서만 PushManager가 있어서, 브라우저 탭에서는 추가하는 방법을 안내한다.
// iPadOS는 데스크톱 UA를 쓰므로 터치 지원 여부로 구분한다.
function isIOSBrowserTab() {
  const isIOS =
    /iP(hone|ad|od)/.test(navigator.userAgent) ||
    (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1)
  return isIOS && !('PushManager' in window)
}

const subscribe = () => () => {}

function SubscribeBox({
  title,
  children,
  actions,
}: {
  title: string
  children?: ReactNode
  actions: ReactNode
}) {
  return (
    <aside className="post-subscribe" data-nav="subscribe">
      <span className="post-subscribe-icon" aria-hidden="true">
        <svg
          width="20"
          height="20"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.8"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9" />
          <path d="M10.3 21a1.94 1.94 0 0 0 3.4 0" />
        </svg>
      </span>
      <div className="post-subscribe-body">
        <p className="post-subscribe-title">{title}</p>
        {children}
      </div>
      <div className="post-subscribe-actions">{actions}</div>
    </aside>
  )
}

// 노출과 RSS 클릭은 data-nav로 InternalNavTracker가, 구독은 push_subscribe(source: post_end)가 잡는다.
export default function SubscribeCta({lang = 'ko'}: {lang?: 'ko' | 'en'}) {
  const isKo = lang === 'ko'
  const {pushState, togglePush} = usePushSubscription('post_end')
  const needsHomeScreen = useSyncExternalStore(
    subscribe,
    isIOSBrowserTab,
    () => false,
  )
  const canPush =
    pushState !== 'unsupported' && Notification.permission !== 'denied'
  const rssLink = (
    <a
      href={isKo ? '/feed.xml' : '/en/feed.xml'}
      className="post-subscribe-link"
    >
      {isKo ? 'RSS 구독 →' : 'Subscribe via RSS →'}
    </a>
  )

  if (pushState === 'on') {
    return (
      <SubscribeBox
        title={isKo ? '새 글 알림이 켜져 있습니다' : 'New post alerts are on'}
        actions={rssLink}
      >
        <p>
          {isKo
            ? '새 글이 나오면 알림으로 알려 드립니다. 끄려면 상단 톱니바퀴 설정을 열어 주세요.'
            : "You'll get a notification when a new post is published. To turn it off, open settings (the gear icon) at the top."}
        </p>
      </SubscribeBox>
    )
  }

  if (canPush) {
    return (
      <SubscribeBox
        title={
          isKo
            ? '새 글이 나오면 알려 드릴까요?'
            : 'Want to know when a new post is published?'
        }
        actions={
          <>
            <button
              type="button"
              className="post-subscribe-button"
              disabled={pushState === 'busy'}
              onClick={() => void togglePush()}
            >
              {isKo ? '알림 받기' : 'Get notified'}
            </button>
            {rssLink}
          </>
        }
      >
        <p>
          {isKo
            ? '새 글을 발행하면 브라우저 알림을 보내 드립니다. 언제든 끌 수 있습니다.'
            : "I'll send a browser notification when a new post goes up. You can turn it off anytime."}
        </p>
      </SubscribeBox>
    )
  }

  if (needsHomeScreen) {
    return (
      <SubscribeBox
        title={
          isKo
            ? 'iPhone, iPad에서 새 글 알림 받기'
            : 'Get new post alerts on iPhone or iPad'
        }
        actions={rssLink}
      >
        <ol className="post-subscribe-steps">
          {isKo ? (
            <>
              <li>공유 버튼을 누르고 &apos;홈 화면에 추가&apos;를 고릅니다.</li>
              <li>홈 화면에 생긴 yceffort 앱을 엽니다.</li>
              <li>상단 톱니바퀴 설정에서 new post alerts를 켭니다.</li>
            </>
          ) : (
            <>
              <li>Tap Share and choose &apos;Add to Home Screen&apos;.</li>
              <li>Open the yceffort app from your Home Screen.</li>
              <li>
                Open settings (the gear icon) and turn on new post alerts.
              </li>
            </>
          )}
        </ol>
      </SubscribeBox>
    )
  }

  return (
    <SubscribeBox
      title={isKo ? '새 글을 놓치고 싶지 않으시다면' : 'Enjoyed this post?'}
      actions={rssLink}
    >
      <p>
        {isKo ? 'RSS로 구독해 주세요.' : 'Subscribe via RSS to get new posts.'}
      </p>
    </SubscribeBox>
  )
}
