import {createHash} from 'node:crypto'

import {generateRenderedMarp} from '@/lib/marp'
import type {OfflineDeck} from '@/lib/offline/types'
import {getSlideBySlug} from '@/lib/slidesIndex'

export async function GET(
  request: Request,
  {params}: {params: Promise<{slug: string}>},
) {
  const {slug} = await params
  // The slug is used as a filename. Never allow a path outside research/.
  if (!/^[a-zA-Z0-9_-]+$/.test(slug)) {
    return new Response('Not Found', {status: 404})
  }
  const entry = getSlideBySlug(slug)
  if (!entry) {
    return new Response('Not Found', {status: 404})
  }
  const {html, css, fonts, notes} = await generateRenderedMarp(entry.markdown)
  const deck: OfflineDeck = {
    schemaVersion: 1,
    slug,
    title: entry.title,
    description: entry.description,
    html,
    css,
    fonts,
    notes,
    post: entry.post,
    transition: entry.transition,
  }
  const body = JSON.stringify(deck)
  const etag = `"${createHash('sha256').update(body).digest('hex')}"`
  const headers = {'Cache-Control': 'no-store', ETag: etag}
  const matches = request.headers
    .get('if-none-match')
    ?.split(',')
    .some(
      (value) =>
        value.trim().replace(/^W\//, '') === etag || value.trim() === '*',
    )
  if (matches) return new Response(null, {status: 304, headers})
  return new Response(body, {
    headers: {...headers, 'Content-Type': 'application/json; charset=utf-8'},
  })
}
