import type {SavedSize} from './cache'

export type SavedMeta = {
  count: 0 | 5 | 10 | 20
  images: boolean
  locale: 'ko' | 'en'
  auto: string[]
  excluded: string[]
  lastSync: number | null
  announced: boolean
  sizes: Record<string, SavedSize>
}

const META_KEY = 'offline-saved-meta'

const defaultMeta = (): SavedMeta => ({
  count: 5,
  images: true,
  locale: 'ko',
  auto: [],
  excluded: [],
  lastSync: null,
  announced: false,
  sizes: {},
})

export function readMeta(storage: Storage | null): SavedMeta {
  try {
    const raw = storage?.getItem(META_KEY)
    return raw ? {...defaultMeta(), ...JSON.parse(raw)} : defaultMeta()
  } catch {
    return defaultMeta()
  }
}

export function writeMeta(storage: Storage | null, meta: SavedMeta) {
  try {
    storage?.setItem(META_KEY, JSON.stringify(meta))
  } catch {
    // 저장 공간이 꽉 찼거나 막혀 있으면 이번 기록은 남기지 않는다
  }
}
