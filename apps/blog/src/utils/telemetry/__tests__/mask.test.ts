import {describe, expect, it} from 'vitest'

import {maskSelector, stripQuery} from '../mask'

describe('maskSelector', () => {
  it('5자리 이상 숫자와 이메일을 지운다', () => {
    expect(maskSelector('div#member-0101234567>span.kim@example.co.kr')).toBe(
      'div#member-{n}>span.{email}',
    )
  })

  it('StyleX 원자 클래스를 지우고 이름 있는 클래스만 남긴다', () => {
    expect(
      maskSelector(
        'div.post-author.x1j9s4mx.x1pokpl6.xtvlv45>img.x8ufqv0.xau0sxb.xt1nusk',
      ),
    ).toBe('div.post-author>img')
    expect(maskSelector('div.thumb.x1234567.x18g2nej>img.x1pc235d')).toBe(
      'div.thumb>img',
    )
    expect(maskSelector('a.x19x9tm7.x1oov6x2.xqx2hci')).toBe('a')
  })

  it('x 로 시작해도 원자 클래스 모양이 아니면 남긴다', () => {
    expect(maskSelector('div.x-scroll>span.xl')).toBe('div.x-scroll>span.xl')
  })

  it('짧은 숫자는 남긴다', () => {
    expect(maskSelector('li.item-3>img')).toBe('li.item-3>img')
  })

  it('빈 값은 속성 자체를 빼도록 undefined 를 돌려준다', () => {
    expect(maskSelector('')).toBeUndefined()
    expect(maskSelector(undefined)).toBeUndefined()
  })
})

describe('stripQuery', () => {
  it('쿼리와 해시를 버린다', () => {
    expect(stripQuery('https://cdn.example.com/a/hero.webp?sig=secret#x')).toBe(
      'https://cdn.example.com/a/hero.webp',
    )
  })

  it('URL 이 아니면 undefined', () => {
    expect(stripQuery('not a url')).toBeUndefined()
  })
})
