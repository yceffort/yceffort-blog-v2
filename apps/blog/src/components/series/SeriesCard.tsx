'use client'

import {stripTitleEmphasis} from '@yceffort/shared/utils'
import Image from 'next/image'
import Link from 'next/link'
import {useRef} from 'react'

import {EmphasizedTitle} from '@/components/post/EmphasizedTitle'
import * as listStyles from '@/components/post/list.styles'
import PostDate from '@/components/post/PostDate'
import type {Series} from '@/type'
export default function SeriesCard({
  series,
  thumbnail,
  pathPrefix = '',
}: {
  series: Series
  thumbnail: string
  pathPrefix?: string
}) {
  const {slug, title, description, posts} = series
  const latest = posts.reduce(
    (acc, p) => (p.frontMatter.date > acc ? p.frontMatter.date : acc),
    posts[0].frontMatter.date,
  )
  const cardRef = useRef<HTMLElement>(null)
  const onPointerMove = (e: React.PointerEvent<HTMLElement>) => {
    const el = cardRef.current
    if (!el) {
      return
    }
    const media = window.matchMedia('(prefers-reduced-motion: reduce)')
    if (media.matches) {
      return
    }
    const tiltAttr = Number(
      getComputedStyle(document.documentElement).getPropertyValue('--tilt') ||
        '8',
    )
    const tilt = Number.isFinite(tiltAttr) ? tiltAttr : 8
    const rect = el.getBoundingClientRect()
    const x = (e.clientX - rect.left) / rect.width
    const y = (e.clientY - rect.top) / rect.height
    const rx = (0.5 - y) * tilt
    const ry = (x - 0.5) * tilt
    el.style.transform = `perspective(1200px) rotateX(${rx}deg) rotateY(${ry}deg)`
    el.style.setProperty('--mx', `${x * 100}%`)
    el.style.setProperty('--my', `${y * 100}%`)
  }
  const onPointerLeave = () => {
    const el = cardRef.current
    if (!el) {
      return
    }
    el.style.transform = 'perspective(1200px) rotateX(0) rotateY(0)'
  }
  return (
    <article
      ref={cardRef}
      className={`post-card ${listStyles.post_card}`}
      onPointerMove={onPointerMove}
      onPointerLeave={onPointerLeave}
    >
      <Link
        href={`${pathPrefix}/series/${slug}`}
        aria-label={stripTitleEmphasis(title)}
        prefetch={false}
        className={listStyles.element_a}
      />
      <div className={`thumb ${listStyles.thumb}`}>
        <Image
          src={thumbnail}
          alt=""
          fill
          sizes="(min-width: 1280px) 25vw, (min-width: 768px) 50vw, 100vw"
          className={listStyles.post_card_thumb_img}
        />
      </div>
      <div className={`body ${listStyles.body}`}>
        <h3 className={listStyles.element_h3}>
          <EmphasizedTitle title={title} />
        </h3>
        {description && <p className={listStyles.desc}>{description}</p>}
        <div className={`meta ${listStyles.meta}`}>
          <span>
            {pathPrefix ? `${posts.length} posts` : `${posts.length}편의 글`}
          </span>
          <span aria-hidden="true">·</span>
          <span>
            {pathPrefix ? 'Updated ' : ''}
            <PostDate value={latest} />
            {pathPrefix ? '' : ' 업데이트'}
          </span>
        </div>
      </div>
    </article>
  )
}
