import {stripTitleEmphasis} from '@yceffort/shared/utils'
import Image from 'next/image'
import Link from 'next/link'

import * as featuredStyles from '@/components/home/featured.styles'
import {EmphasizedTitle} from '@/components/post/EmphasizedTitle'
import * as listStyles from '@/components/post/list.styles'
import PostDate from '@/components/post/PostDate'
import type {Post} from '@/type'
export default function FeaturedPost({
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
      title: rawTitle,
      description,
      tags,
      thumbnail,
      series,
      published,
    },
    readingTime,
  } = post
  return (
    <article className={`featured-post ${featuredStyles.featured}`}>
      <Link
        href={`${pathPrefix}/${slug}`}
        aria-label={stripTitleEmphasis(rawTitle)}
        prefetch={false}
        className={listStyles.element_a}
      />
      <div className={featuredStyles.thumb}>
        {thumbnail && (
          <Image
            src={thumbnail}
            alt=""
            fill
            priority
            sizes="(min-width: 1024px) 560px, 100vw"
            className={featuredStyles.thumb_img}
          />
        )}
        {!published && <span className={listStyles.draft_badge}>Draft</span>}
      </div>
      <div className={featuredStyles.body}>
        {series && (
          <span className={`series ${listStyles.post_row_series}`}>
            ◆ {series}
          </span>
        )}
        <h3 className={featuredStyles.title}>
          <EmphasizedTitle title={rawTitle} />
        </h3>
        {description && <p className={featuredStyles.desc}>{description}</p>}
        <div className={`post-row-tags ${listStyles.post_row_tags}`}>
          {tags.slice(0, 3).map((tag) => (
            <span key={tag} className={`tag-chip ${listStyles.tag_chip}`}>
              #{tag}
            </span>
          ))}
        </div>
        <div className={featuredStyles.meta}>
          <PostDate value={date} />
          {pathPrefix
            ? ` · ${readingTime} min · read`
            : ` · ${readingTime}분 · read`}
        </div>
      </div>
    </article>
  )
}
