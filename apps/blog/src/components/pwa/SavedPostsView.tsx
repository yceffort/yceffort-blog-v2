'use client'

import * as stylex from '@stylexjs/stylex'
import Link from 'next/link'
import {useEffect, useId, useState, useSyncExternalStore} from 'react'

import {useIsStandalone} from '@/hooks/useIsStandalone'
import {listSavedPosts, type OfflineDeps} from '@/utils/offline/cache'
import {readMeta, type SavedMeta} from '@/utils/offline/meta'
import {formatBytes} from '@/utils/offline/size'
import {
  browserDeps,
  clearAllSaved,
  deleteSaved,
  syncRecent,
  updateSettings,
} from '@/utils/offline/sync'

const COUNTS = [0, 5, 10, 20] as const

const COPY = {
  ko: {
    title: '저장한 글',
    summary: (n: number, size: string | null, time: string | null) =>
      `${n}편 저장됨${size ? `, ${size}` : ''}${time ? `, 마지막 갱신 ${time}` : ''}`,
    external: (n: number) => ` + 외부 이미지 ${n}장`,
    estimate: (avg: string, twenty: string) =>
      `한 편당 평균 ${avg}, 20편이면 약 ${twenty}`,
    autoSave: '최근 글 자동 저장',
    counts: ['끄기', '5편', '10편', '20편'],
    images: '본문 이미지도 저장',
    auto: '자동',
    read: '읽음',
    delete: '삭제',
    deleteAll: '전체 삭제',
    confirm: '저장한 글을 모두 지울까요?',
    empty: '저장한 글이 없습니다',
    hint: '앱으로 설치하면 글을 오프라인에 저장해 둘 수 있습니다.',
  },
  en: {
    title: 'Saved posts',
    summary: (n: number, size: string | null, time: string | null) =>
      `${n} saved${size ? `, ${size}` : ''}${time ? `, last updated ${time}` : ''}`,
    external: (n: number) => ` + ${n} external images`,
    estimate: (avg: string, twenty: string) =>
      `About ${avg} per post, roughly ${twenty} for 20`,
    autoSave: 'Auto-save recent posts',
    counts: ['Off', '5', '10', '20'],
    images: 'Also save images in posts',
    auto: 'Auto',
    read: 'Read',
    delete: 'Delete',
    deleteAll: 'Delete all',
    confirm: 'Delete all saved posts?',
    empty: 'No saved posts yet',
    hint: 'Install this site as an app to save posts for offline reading.',
  },
}

const sx = stylex.create({
  root: {
    '@layer utilities': {
      maxWidth: 'var(--container-3xl)',
      marginInline: 'auto',
    },
  },
  h1: {
    '@layer utilities': {
      marginBottom: 'calc(var(--spacing) * 2)',
      fontSize: 'var(--text-2xl)',
      lineHeight: 'var(--blog-leading, var(--text-2xl--line-height))',
      fontWeight: 'var(--font-weight-bold)',
      color: {
        default: 'oklch(21% 0.006 285.885)',
        ':is(.dark *)': 'oklch(96.7% 0.001 286.375)',
      },
    },
  },
  note: {
    '@layer utilities': {
      marginBottom: 'calc(var(--spacing) * 6)',
      fontSize: 'var(--text-sm)',
      lineHeight: 'var(--blog-leading, var(--text-sm--line-height))',
      color: {
        default: 'oklch(44.2% 0.017 285.786)',
        ':is(.dark *)': 'oklch(70.5% 0.015 286.067)',
      },
    },
  },
  settings: {
    '@layer utilities': {
      display: 'flex',
      flexDirection: 'column',
      gap: 'calc(var(--spacing) * 3)',
      marginBottom: 'calc(var(--spacing) * 6)',
      padding: 'calc(var(--spacing) * 4)',
      borderWidth: '1px',
      borderStyle: 'solid',
      borderColor: {
        default: 'oklch(92% 0.004 286.32)',
        ':is(.dark *)': 'oklch(27.4% 0.006 286.033)',
      },
      borderRadius: 'var(--radius-lg)',
    },
  },
  settingLabel: {
    '@layer utilities': {
      fontSize: 'var(--text-sm)',
      lineHeight: 'var(--blog-leading, var(--text-sm--line-height))',
      fontWeight: 'var(--font-weight-medium)',
      color: {
        default: 'oklch(21% 0.006 285.885)',
        ':is(.dark *)': 'oklch(96.7% 0.001 286.375)',
      },
    },
  },
  segmented: {
    '@layer utilities': {
      display: 'flex',
      minWidth: '0',
      gap: 'calc(var(--spacing) * 1)',
      padding: 'calc(var(--spacing) * 1)',
      borderWidth: '1px',
      borderStyle: 'solid',
      borderColor: {
        default: 'oklch(92% 0.004 286.32)',
        ':is(.dark *)': 'oklch(27.4% 0.006 286.033)',
      },
      borderRadius: 'var(--radius-lg)',
    },
  },
  segment: {
    '@layer utilities': {
      flex: '1',
      minWidth: '0',
      paddingBlock: 'calc(var(--spacing) * 1.5)',
      borderRadius: 'var(--radius-md)',
      fontSize: 'var(--text-sm)',
      lineHeight: 'var(--blog-leading, var(--text-sm--line-height))',
      fontWeight: 'var(--font-weight-medium)',
      color: {
        default: 'oklch(44.2% 0.017 285.786)',
        ':is(.dark *)': 'oklch(70.5% 0.015 286.067)',
      },
      backgroundColor: {
        default: null,
        '@media (hover: hover)': {
          default: null,
          ':hover': 'oklch(96.7% 0.001 286.375)',
          ':is(.dark *):hover': 'oklch(27.4% 0.006 286.033)',
        },
      },
      opacity: {
        default: null,
        ':disabled': '0.6',
      },
    },
  },
  segmentSelected: {
    '@layer utilities': {
      color: {
        default: 'var(--color-white)',
        ':is(.dark *)': 'oklch(21% 0.006 285.885)',
      },
      backgroundColor: {
        default: 'oklch(21% 0.006 285.885)',
        ':is(.dark *)': 'oklch(96.7% 0.001 286.375)',
      },
    },
  },
  toggle: {
    '@layer utilities': {
      display: 'flex',
      alignItems: 'center',
      gap: 'calc(var(--spacing) * 2)',
      fontSize: 'var(--text-sm)',
      lineHeight: 'var(--blog-leading, var(--text-sm--line-height))',
      color: {
        default: 'oklch(37% 0.013 285.805)',
        ':is(.dark *)': 'oklch(87.1% 0.006 286.286)',
      },
    },
  },
  checkbox: {
    '@layer utilities': {
      width: 'calc(var(--spacing) * 4)',
      height: 'calc(var(--spacing) * 4)',
      accentColor: {
        default: 'oklch(21% 0.006 285.885)',
        ':is(.dark *)': 'oklch(96.7% 0.001 286.375)',
      },
    },
  },
  row: {
    '@layer utilities': {
      display: 'flex',
      alignItems: 'center',
      gap: 'calc(var(--spacing) * 3)',
      paddingBlock: 'calc(var(--spacing) * 3)',
      borderBottomWidth: {
        default: '1px',
        ':last-child': '0',
      },
      borderBottomStyle: 'solid',
      borderBottomColor: {
        default: 'oklch(92% 0.004 286.32)',
        ':is(.dark *)': 'oklch(27.4% 0.006 286.033)',
      },
    },
  },
  title: {
    '@layer utilities': {
      flex: '1',
      minWidth: '0',
      overflowWrap: 'anywhere',
      fontSize: 'var(--text-base)',
      lineHeight: 'var(--leading-snug)',
      color: {
        default: 'oklch(21% 0.006 285.885)',
        ':is(.dark *)': 'oklch(96.7% 0.001 286.375)',
      },
      textDecorationLine: {
        default: null,
        '@media (hover: hover)': {
          default: null,
          ':hover': 'underline',
        },
      },
    },
  },
  tag: {
    '@layer utilities': {
      flexShrink: '0',
      borderRadius: 'calc(infinity * 1px)',
      paddingInline: 'calc(var(--spacing) * 2)',
      paddingBlock: 'calc(var(--spacing) * 0.5)',
      fontSize: 'var(--text-xs)',
      lineHeight: 'var(--blog-leading, var(--text-xs--line-height))',
      fontWeight: 'var(--font-weight-medium)',
      color: {
        default: 'oklch(44.2% 0.017 285.786)',
        ':is(.dark *)': 'oklch(70.5% 0.015 286.067)',
      },
      backgroundColor: {
        default: 'oklch(96.7% 0.001 286.375)',
        ':is(.dark *)': 'oklch(27.4% 0.006 286.033)',
      },
    },
  },
  size: {
    '@layer utilities': {
      flexShrink: '0',
      whiteSpace: 'nowrap',
      fontSize: 'var(--text-xs)',
      lineHeight: 'var(--blog-leading, var(--text-xs--line-height))',
      color: {
        default: 'oklch(44.2% 0.017 285.786)',
        ':is(.dark *)': 'oklch(70.5% 0.015 286.067)',
      },
    },
  },
  remove: {
    '@layer utilities': {
      flexShrink: '0',
      display: 'inline-flex',
      alignItems: 'center',
      justifyContent: 'center',
      width: 'calc(var(--spacing) * 8)',
      height: 'calc(var(--spacing) * 8)',
      borderRadius: 'var(--radius-md)',
      color: {
        default: 'oklch(55.2% 0.016 285.938)',
        ':is(.dark *)': 'oklch(70.5% 0.015 286.067)',
      },
      backgroundColor: {
        default: null,
        '@media (hover: hover)': {
          default: null,
          ':hover': 'oklch(96.7% 0.001 286.375)',
          ':is(.dark *):hover': 'oklch(27.4% 0.006 286.033)',
        },
      },
      opacity: {
        default: null,
        ':disabled': '0.4',
      },
    },
  },
  empty: {
    '@layer utilities': {
      paddingBlock: 'calc(var(--spacing) * 10)',
      textAlign: 'center',
      fontSize: 'var(--text-sm)',
      lineHeight: 'var(--blog-leading, var(--text-sm--line-height))',
      color: {
        default: 'oklch(55.2% 0.016 285.938)',
        ':is(.dark *)': 'oklch(70.5% 0.015 286.067)',
      },
    },
  },
  deleteAll: {
    '@layer utilities': {
      marginTop: 'calc(var(--spacing) * 6)',
      fontSize: 'var(--text-sm)',
      lineHeight: 'var(--blog-leading, var(--text-sm--line-height))',
      fontWeight: 'var(--font-weight-medium)',
      color: {
        default: 'oklch(57.7% 0.245 27.325)',
        ':is(.dark *)': 'oklch(70.4% 0.191 22.216)',
      },
      textDecorationLine: {
        default: null,
        '@media (hover: hover)': {
          default: null,
          ':hover': 'underline',
        },
      },
      opacity: {
        default: null,
        ':disabled': '0.5',
      },
    },
  },
})

type Saved = {
  posts: Array<{path: string; title: string}>
  meta: SavedMeta
}

async function loadSaved(deps: OfflineDeps): Promise<Saved> {
  const posts = await listSavedPosts(deps)
  return {posts, meta: readMeta(deps.storage)}
}

const subscribe = () => () => {}

export function SavedPostsView({locale}: {locale: 'ko' | 'en'}) {
  const copy = COPY[locale]
  const standalone = useIsStandalone()
  const labelId = useId()
  // 서버 HTML에 안내 문구가 들어가면 앱에서도 하이드레이션 전까지 보이므로 본문은 마운트 뒤에만 그린다
  const mounted = useSyncExternalStore(
    subscribe,
    () => true,
    () => false,
  )
  const [saved, setSaved] = useState<Saved | null>(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    if (standalone) {
      void loadSaved(browserDeps()).then(setSaved)
    }
  }, [standalone])

  const run = async (action: (deps: OfflineDeps) => Promise<unknown>) => {
    setBusy(true)
    try {
      const deps = browserDeps()
      await action(deps)
      setSaved(await loadSaved(deps))
    } finally {
      setBusy(false)
    }
  }

  const changeSettings = (
    patch: Partial<Pick<SavedMeta, 'count' | 'images'>>,
  ) =>
    run(async (deps) => {
      const next = updateSettings({...patch, locale}, deps)
      setSaved((current) => current && {...current, meta: next})
      await syncRecent(deps)
    })

  const deleteAll = () => {
    if (window.confirm(copy.confirm)) {
      void run(clearAllSaved)
    }
  }

  const meta = saved?.meta
  const posts = saved?.posts
  const lastSync =
    !meta || meta.lastSync === null
      ? null
      : new Date(meta.lastSync).toLocaleString(
          locale === 'en' ? 'en' : 'ko-KR',
          {dateStyle: 'short', timeStyle: 'short'},
        )

  const sized = (posts ?? []).flatMap((post) => {
    const size = meta?.sizes[post.path]
    return size ? [size] : []
  })
  const totalBytes = sized.reduce((sum, size) => sum + size.bytes, 0)
  const totalExternal = sized.reduce((sum, size) => sum + size.external, 0)
  const externalNote = (n: number) => (n > 0 ? copy.external(n) : '')
  const totalSize =
    sized.length > 0
      ? formatBytes(totalBytes) + externalNote(totalExternal)
      : null
  const average = sized.length > 0 ? totalBytes / sized.length : 0

  return (
    <div className={stylex.props(sx.root).className}>
      <h1 className={stylex.props(sx.h1).className}>{copy.title}</h1>
      {mounted && !standalone && (
        <p className={stylex.props(sx.note).className}>{copy.hint}</p>
      )}
      {mounted && standalone && (
        <>
          {posts && (
            <p className={stylex.props(sx.note).className}>
              {copy.summary(posts.length, totalSize, lastSync)}
            </p>
          )}
          <section className={stylex.props(sx.settings).className}>
            <p id={labelId} className={stylex.props(sx.settingLabel).className}>
              {copy.autoSave}
            </p>
            <fieldset
              aria-labelledby={labelId}
              disabled={!meta || busy}
              className={stylex.props(sx.segmented).className}
            >
              {COUNTS.map((count, i) => {
                const selected = meta?.count === count
                return (
                  <button
                    key={count}
                    type="button"
                    aria-pressed={selected}
                    onClick={() => void changeSettings({count})}
                    className={
                      stylex.props(sx.segment, selected && sx.segmentSelected)
                        .className
                    }
                  >
                    {copy.counts[i]}
                  </button>
                )
              })}
            </fieldset>
            <label className={stylex.props(sx.toggle).className}>
              <input
                type="checkbox"
                checked={meta?.images ?? false}
                disabled={!meta || busy}
                onChange={(event) =>
                  void changeSettings({images: event.target.checked})
                }
                className={stylex.props(sx.checkbox).className}
              />
              {copy.images}
            </label>
          </section>
          {sized.length > 0 && (
            <p className={stylex.props(sx.note).className}>
              {copy.estimate(formatBytes(average), formatBytes(average * 20))}
            </p>
          )}
          {posts?.length === 0 && (
            <p className={stylex.props(sx.empty).className}>{copy.empty}</p>
          )}
          {posts && posts.length > 0 && (
            <>
              <ul>
                {posts.map((post) => (
                  <li
                    key={post.path}
                    className={stylex.props(sx.row).className}
                  >
                    <Link
                      href={post.path}
                      // 목록이 화면에 들어올 때마다 글마다 RSC를 미리 받지 않는다
                      prefetch={false}
                      className={stylex.props(sx.title).className}
                    >
                      {post.title}
                    </Link>
                    {meta?.sizes[post.path] && (
                      <span className={stylex.props(sx.size).className}>
                        {formatBytes(meta.sizes[post.path].bytes) +
                          externalNote(meta.sizes[post.path].external)}
                      </span>
                    )}
                    <span className={stylex.props(sx.tag).className}>
                      {meta?.auto.includes(post.path) ? copy.auto : copy.read}
                    </span>
                    <button
                      type="button"
                      aria-label={copy.delete}
                      disabled={busy}
                      onClick={() =>
                        void run((deps) => deleteSaved(post.path, deps))
                      }
                      className={stylex.props(sx.remove).className}
                    >
                      <svg
                        width="16"
                        height="16"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        aria-hidden="true"
                      >
                        <path d="M18 6L6 18M6 6l12 12" />
                      </svg>
                    </button>
                  </li>
                ))}
              </ul>
              <button
                type="button"
                disabled={busy}
                onClick={deleteAll}
                className={stylex.props(sx.deleteAll).className}
              >
                {copy.deleteAll}
              </button>
            </>
          )}
        </>
      )}
    </div>
  )
}
