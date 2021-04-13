import React from 'react'
import Image from 'next/image'

import CustomLink from '#components/Link'

const Card = ({
  title,
  description,
  imgSrc,
  href,
}: {
  title: string
  description: string
  imgSrc: string
  href: string
}) => (
  <div className="p-4 md:w-1/2 md" style={{ maxWidth: '544px' }}>
    <div className="h-full border-2 border-gray-200 border-opacity-60 dark:border-gray-700 rounded-md overflow-hidden">
      {href ? (
        <CustomLink href={href} aria-label={`Link to ${title}`}>
          <Image
            alt={title}
            src={imgSrc}
            className="lg:h-48 md:h-36 object-cover object-center"
            width={544}
            height={306}
          />
        </CustomLink>
      ) : (
        <Image
          alt={title}
          src={imgSrc}
          className="lg:h-48 md:h-36 object-cover object-center"
          width={544}
          height={306}
        />
      )}
      <div className="p-6">
        <h2 className="text-2xl font-bold leading-8 tracking-tight mb-3">
          {href ? (
            <CustomLink href={href} aria-label={`Link to ${title}`}>
              {title}
            </CustomLink>
          ) : (
            title
          )}
        </h2>
        <p className="prose text-gray-500 max-w-none dark:text-gray-400 mb-3">
          {description}
        </p>
        {href && (
          <CustomLink
            href={href}
            className="text-base font-medium leading-6 text-blue-500 hover:text-blue-600 dark:hover:text-blue-400"
            aria-label={`Link to ${title}`}
          >
            Learn more &rarr;
          </CustomLink>
        )}
      </div>
    </div>
  </div>
)

export default Card
