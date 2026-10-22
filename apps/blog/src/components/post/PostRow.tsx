import * as stylex from '@stylexjs/stylex'
import {stripTitleEmphasis} from '@yceffort/shared/utils'
import Image from 'next/image'
import Link from 'next/link'
import {ViewTransition} from 'react'

import {EmphasizedTitle} from '@/components/post/EmphasizedTitle'
import * as listStyles from '@/components/post/list.styles'
import PostDate from '@/components/post/PostDate'
import type {Post} from '@/type'
const sx = stylex.create({
  article: {
    '@layer utilities': {
      position: 'relative',
    },
  },
})
export default function PostRow({
  post,
  pathPrefix = '',
}: {
  post: Post
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
      series,
      seriesOrder,
      published,
    },
    readingTime,
  } = post
  const plainTitle = stripTitleEmphasis(rawTitle)
  const transitionName = `post-${slug.replace(/\//g, '-')}`
  return (
    <article
      className={`post-row ${listStyles.post_row} ${stylex.props(sx.article).className}`}
    >
      <Link
        href={`${pathPrefix}/${slug}`}
        aria-label={plainTitle}
        className={`post-row-link ${listStyles.post_row_link}`}
        prefetch={false}
      />
      {!published && <span className={listStyles.draft_badge}>Draft</span>}
      {thumbnail ? (
        <ViewTransition name={`${transitionName}-thumbnail`}>
          <div className={`post-row-thumb ${listStyles.post_row_thumb}`}>
            <Image
              src={thumbnail}
              alt=""
              fill
              sizes="(min-width: 768px) 120px, 84px"
              className={listStyles.post_row_thumb_img}
            />
          </div>
        </ViewTransition>
      ) : (
        <div
          className={`post-row-thumb post-row-thumb-empty ${listStyles.post_row_thumb_empty_box}`}
          aria-hidden="true"
        >
          <svg
            viewBox="0 0 48 32"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
            className={listStyles.post_row_thumb_empty_svg}
          >
            <path
              d="M8 10h22M8 16h32M8 22h18"
              stroke="currentColor"
              strokeWidth="1.6"
              strokeLinecap="round"
              opacity="0.55"
            />
          </svg>
        </div>
      )}

      <div className={`post-row-body ${listStyles.post_row_body}`}>
        <div className={`post-row-head ${listStyles.post_row_head}`}>
          {series && (
            <span className={`series ${listStyles.post_row_series}`}>
              ◆ {series}
              {seriesOrder != null &&
                ` · ${pathPrefix ? `Part ${seriesOrder}` : `${seriesOrder}편`}`}
            </span>
          )}
          <ViewTransition name={`${transitionName}-tags`}>
            <div className={`post-row-tags ${listStyles.post_row_tags}`}>
              {tags.slice(0, 3).map((tag) => (
                <span key={tag} className={`tag-chip ${listStyles.tag_chip}`}>
                  #{tag}
                </span>
              ))}
            </div>
          </ViewTransition>
        </div>

        <ViewTransition name={transitionName}>
          <h3 className={`post-row-title ${listStyles.post_row_title}`}>
            <EmphasizedTitle title={rawTitle} />
          </h3>
        </ViewTransition>

        {description && (
          <p className={`post-row-desc ${listStyles.post_row_desc}`}>
            {description}
          </p>
        )}

        <div className={`post-row-meta ${listStyles.post_row_meta}`}>
          <PostDate value={date} />
          {updated && (
            <>
              <span aria-hidden="true">·</span>
              <span>
                {pathPrefix ? 'Updated' : '수정'} <PostDate value={updated} />
              </span>
            </>
          )}
          <span aria-hidden="true">·</span>
          <span>
            {pathPrefix ? `${readingTime} min read` : `${readingTime}분`}
          </span>
        </div>
      </div>

      <svg
        className={`post-row-arrow ${listStyles.post_row_arrow}`}
        width="18"
        height="18"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden="true"
      >
        <path d="M5 12h14" />
        <path d="M13 5l7 7-7 7" />
      </svg>
    </article>
  )
}
