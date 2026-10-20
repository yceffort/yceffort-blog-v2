import {vi, type Mock} from 'vitest'

import type {OfflineDeps} from '../cache'

const keyOf = (request: RequestInfo | URL) => new Request(request).url

function createFakeCache(): Cache {
  const entries = new Map<string, Response>()

  return {
    match: async (request: RequestInfo | URL) =>
      entries.get(keyOf(request))?.clone(),
    put: async (request: RequestInfo | URL, response: Response) => {
      const key = keyOf(request)
      entries.delete(key)
      entries.set(key, response.clone())
    },
    delete: async (request: RequestInfo | URL) =>
      entries.delete(keyOf(request)),
    keys: async () => [...entries.keys()].map((url) => new Request(url)),
  } as unknown as Cache
}

export function createFakeCaches(): CacheStorage {
  const caches = new Map<string, Cache>()

  return {
    open: async (name: string) => {
      let cache = caches.get(name)
      if (!cache) {
        cache = createFakeCache()
        caches.set(name, cache)
      }
      return cache
    },
    delete: async (name: string) => caches.delete(name),
  } as unknown as CacheStorage
}

export function createFakeStorage(): Storage {
  const items = new Map<string, string>()

  return {
    get length() {
      return items.size
    },
    clear: () => items.clear(),
    getItem: (key: string) => items.get(key) ?? null,
    key: (index: number) => [...items.keys()][index] ?? null,
    removeItem: (key: string) => {
      items.delete(key)
    },
    setItem: (key: string, value: string) => {
      items.set(key, value)
    },
  }
}

export function createDeps(
  route: (url: URL) => Response,
): OfflineDeps & {fetch: Mock<typeof fetch>} {
  return {
    caches: createFakeCaches(),
    fetch: vi.fn<typeof fetch>(async (input) =>
      route(new URL(input instanceof Request ? input.url : input)),
    ),
    storage: createFakeStorage(),
    origin: 'https://yceffort.kr',
    saveData: false,
    now: () => 1_000,
  }
}
