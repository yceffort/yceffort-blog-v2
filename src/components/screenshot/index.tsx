import React from 'react'

import styles from './index.module.css'

import SiteConfig from '#src/config'
export default function Screenshot({
  title,
  tags,
  url,
  imageSrc,
  imageCredit,
}: {
  title: string
  tags: string[]
  url: string
  imageSrc: string
  imageCredit: string
}) {
  return (
    <div className={styles.wrapper}>
      <div className={styles.preview}>
        <main className={styles.main}>
          <h1 className={styles.h1}>{title}</h1>
          <div className={styles.tag_list}>
            <ul className={styles.tag_ul}>
              {tags.map((tag, index) => (
                <li className={styles.tag_li} key={index}>
                  #{tag}
                </li>
              ))}
            </ul>
          </div>
          <span className={styles.url}>{url}</span>
          <span className={styles.author}>@{SiteConfig.author.name}</span>
          {imageCredit ? (
            <p className={styles.attribution}>
              <span className={styles.attribution_by}>
                image by: {imageCredit}
              </span>
            </p>
          ) : null}
        </main>
        <img
          className={styles.image}
          src={imageSrc || '/default-image.png'}
          alt="thumbnail"
        />
      </div>
    </div>
  )
}
