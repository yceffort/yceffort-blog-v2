import {stripTitleEmphasis} from '@yceffort/shared/utils'
import Image from 'next/image'
import Link from 'next/link'

import * as recentStyles from '@/components/home/recent.styles'
import {EmphasizedTitle} from '@/components/post/EmphasizedTitle'
import * as listStyles from '@/components/post/list.styles'
import PostDate from '@/components/post/PostDate'
import type {Post} from '@/type'
const ROW_CLASS = {
  default: recentStyles.rec_row,
  series: recentStyles.series_row,
  thumb: recentStyles.thumb_row,
} as const
export default function RecentRow({
  post,
  index,
  variant = 'default',
  pathPrefix = '',
}: {
  post: Post
  index: number
  variant?: keyof typeof ROW_CLASS
  pathPrefix?: string
}) {
  const {
    fields: {slug},
    frontMatter: {
      date,
      updated,
      title: rawTitle,
      description,
      tags,
      thumbnail,
      published,
    },
    readingTime,
  } = post
  const plainTitle = stripTitleEmphasis(rawTitle)
  return (
    <div className={`rec-row ${ROW_CLASS[variant]}`}>
      <Link
        href={`${pathPrefix}/${slug}`}
        aria-label={plainTitle}
        prefetch={false}
        className={recentStyles.element_a}
      />
      {!published && variant !== 'thumb' && (
        <span className={listStyles.draft_badge}>Draft</span>
      )}
      <div className={`rn ${recentStyles.rn}`}>
        {String(index + 1).padStart(2, '0')}
      </div>
      {variant === 'thumb' && (
        <div className={recentStyles.rthumb}>
          {thumbnail && (
            <Image
              src={thumbnail}
              alt=""
              fill
              sizes="(min-width: 901px) 128px, 88px"
              className={recentStyles.rthumb_img}
            />
          )}
          {!published && <span className={listStyles.draft_badge}>Draft</span>}
        </div>
      )}
      <div>
        <h4 className={recentStyles.element_h4}>
          <EmphasizedTitle title={rawTitle} />
        </h4>
        <div className={`rtags ${recentStyles.rtags}`}>
          {tags.slice(0, 3).map((t) => (
            <span key={t}>#{t}</span>
          ))}
        </div>
      </div>
      <div className={`rd ${recentStyles.rd}`}>{description}</div>
      <div className={`rmeta ${recentStyles.rmeta}`}>
        <b className={recentStyles.element_b}>
          <PostDate value={date} />
        </b>
        {updated && (
          <>
            {pathPrefix ? 'Updated' : '수정'} <PostDate value={updated} />
            <br />
          </>
        )}
        {pathPrefix ? `${readingTime} min · read` : `${readingTime}분 · read`}
      </div>
      <div className={`rarrow ${recentStyles.rarrow}`} aria-hidden="true">
        <svg
          width="16"
          height="16"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
        >
          <path d="M5 12h14M13 5l7 7-7 7" />
        </svg>
      </div>
    </div>
  )
}
