'use client'

import {useSyncExternalStore} from 'react'

const QUERY = '(display-mode: standalone)'

function subscribe(onChange: () => void) {
  const media = window.matchMedia(QUERY)
  media.addEventListener('change', onChange)
  return () => media.removeEventListener('change', onChange)
}

function getSnapshot() {
  return (
    window.matchMedia(QUERY).matches ||
    (navigator as Navigator & {standalone?: boolean}).standalone === true
  )
}

function getServerSnapshot() {
  return false
}

export function useIsStandalone() {
  return useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot)
}
