'use client'

import {useEffect, useState} from 'react'

export type PushState = 'unsupported' | 'off' | 'on' | 'busy'
export type PushSource = 'tweaks' | 'post_end'

// Tweaks 패널과 글 끝 안내가 각자 상태를 들고 있으므로, 한쪽에서 바꾸면 다른 쪽이 다시 읽게 한다.
const CHANGE_EVENT = 'push-subscription-change'

function urlBase64ToUint8Array(base64: string) {
  const padding = '='.repeat((4 - (base64.length % 4)) % 4)
  const raw = atob((base64 + padding).replaceAll('-', '+').replaceAll('_', '/'))
  return Uint8Array.from(raw, (char) => char.charCodeAt(0))
}

function trackPush(eventName: string, source: PushSource) {
  if (typeof window.gtag === 'function') {
    window.gtag('event', eventName, {placement: source})
  }
}

export function usePushSubscription(source: PushSource) {
  const [pushState, setPushState] = useState<PushState>('unsupported')

  useEffect(() => {
    let cancelled = false
    if (!('serviceWorker' in navigator) || !('PushManager' in window)) {
      return undefined
    }
    // 첫 방문에는 마운트 시점에 SW가 아직 등록 전이라 getRegistration()이 비어 있다.
    // 검색으로 처음 들어온 독자가 대부분 이 경우라 ready를 기다린다.
    // production에서만 SW가 등록되므로 dev에서는 ready가 끝나지 않아 unsupported로 남는다.
    const syncPushState = async () => {
      const registration = await navigator.serviceWorker.ready
      const subscription = await registration.pushManager.getSubscription()
      if (!cancelled) {
        setPushState(subscription ? 'on' : 'off')
      }
    }
    void syncPushState()
    window.addEventListener(CHANGE_EVENT, syncPushState)
    return () => {
      cancelled = true
      window.removeEventListener(CHANGE_EVENT, syncPushState)
    }
  }, [])

  const togglePush = async () => {
    if (pushState === 'unsupported' || pushState === 'busy') {
      return
    }
    const registration = await navigator.serviceWorker.getRegistration()
    if (!registration) {
      return
    }
    setPushState('busy')
    try {
      if (pushState === 'on') {
        const subscription = await registration.pushManager.getSubscription()
        if (subscription) {
          await fetch('/api/push/subscribe', {
            method: 'DELETE',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              endpoint: subscription.endpoint,
            }),
          })
          await subscription.unsubscribe()
        }
        trackPush('push_unsubscribe', source)
        setPushState('off')
        return
      }
      const permission = await Notification.requestPermission()
      if (permission !== 'granted') {
        trackPush('push_permission_denied', source)
        setPushState('off')
        return
      }
      const subscription = await registration.pushManager.subscribe({
        userVisibleOnly: true,
        applicationServerKey: urlBase64ToUint8Array(
          process.env.NEXT_PUBLIC_VAPID_PUBLIC_KEY || '',
        ),
      })
      const response = await fetch('/api/push/subscribe', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(subscription.toJSON()),
      })
      if (!response.ok) {
        await subscription.unsubscribe()
        trackPush('push_subscribe_failed', source)
        setPushState('off')
        return
      }
      trackPush('push_subscribe', source)
      setPushState('on')
    } catch {
      trackPush(
        pushState === 'on'
          ? 'push_unsubscribe_failed'
          : 'push_subscribe_failed',
        source,
      )
      setPushState('off')
    } finally {
      window.dispatchEvent(new Event(CHANGE_EVENT))
    }
  }

  return {pushState, togglePush}
}
