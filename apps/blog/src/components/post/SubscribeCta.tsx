'use client'

import {usePushSubscription} from '@/hooks/usePushSubscription'

// 노출과 RSS 클릭은 data-nav로 InternalNavTracker가, 구독은 push_subscribe(source: post_end)가 잡는다.
export default function SubscribeCta({lang = 'ko'}: {lang?: 'ko' | 'en'}) {
  const isKo = lang === 'ko'
  const {pushState, togglePush} = usePushSubscription('post_end')
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

  if (!canPush) {
    return (
      <aside className="post-subscribe" data-nav="subscribe">
        <p>
          {isKo
            ? '새 글을 놓치고 싶지 않으시다면 RSS로 구독해 주세요.'
            : 'Enjoyed this post? Subscribe via RSS to get new posts.'}
        </p>
        {rssLink}
      </aside>
    )
  }

  if (pushState === 'on') {
    return (
      <aside className="post-subscribe" data-nav="subscribe">
        <p>
          {isKo
            ? '새 글이 나오면 알림으로 알려 드립니다.'
            : "You'll get a notification when a new post is published."}
        </p>
        {rssLink}
      </aside>
    )
  }

  return (
    <aside className="post-subscribe" data-nav="subscribe">
      <p>
        {isKo
          ? '새 글이 나오면 알려 드릴까요?'
          : 'Want to know when a new post is published?'}
      </p>
      <div className="post-subscribe-actions">
        <button
          type="button"
          className="post-subscribe-button"
          disabled={pushState === 'busy'}
          onClick={() => void togglePush()}
        >
          {isKo ? '알림 받기' : 'Get notified'}
        </button>
        {rssLink}
      </div>
    </aside>
  )
}
