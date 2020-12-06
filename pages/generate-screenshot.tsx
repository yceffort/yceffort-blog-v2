import { GetServerSideProps } from 'next'
import React from 'react'
import styled from 'styled-components'

const Wrapper = styled.div`
  height: 100vh;
  width: 100vw;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgb(15 15 15);
`

const Preview = styled.div`
  height: 630px;
  width: 1200px;
  background: rgba(83, 186, 266, 0.8);
  color: rgb(254 254 254);
  position: relative;
`

const Main = styled.main`
  position: relative;
  z-index: 2;

  font-size: 1.25rem;
  height: calc(100% - 8rem);
  width: calc(100% - 8rem);
  margin: 2rem;
  padding: 2rem;
  border: 3px solid white;
  border-color: white;
  display: grid;
  gap: 2rem;
  grid-template-rows: 1fr auto auto 1fr auto auto auto;
  grid-template-columns: 3fr 2fr 1fr;
  grid-template-areas:
    '. . .'
    'title title .'
    'tags tags .'
    'url url url'
    '. . .'
    '. attribution attribution'
    'author attribution attribution';
`

const Image = styled.img`
  position: absolute;
  z-index: 1;
  top: -1px;
  right: -1px;
  bottom: -1px;
  left: -1px;
  width: 100%;
  height: 100%;
  object-fit: cover;
  object-position: center center;
  filter: grayscale(100%);
  opacity: 0.5;
  mix-blend-mode: overlay;
  margin: 0;
`

const H1 = styled.h1`
  margin: 0;
  font-size: 3em;
  grid-area: title;
`

const TagList = styled.div`
  grid-area: tags;
  font-size: 1.25em;
`

const TagUl = styled.ul`
  font-size: 1.5em;

  color: white;
  list-style: none;
  padding: 0;
  margin: 0;
  display: inline-block;

  > li {
    margin: 0;
    display: inline-block;
  }
`

const URL = styled.span`
  grid-area: url;
  display: block;
  font-size: 1.5em;
`

const Author = styled.span`
  grid-area: author;
  display: block;
  font-size: 2em;
`

const Attribution = styled.p`
  grid-area: attribution;
  font-size: 0.8em;
  text-align: right;
  align-self: end;
  margin: 0;

  > span {
    display: block;
  }
`

// sample
// /generate-screenshot?title=hello&tags=react,javascript&categories=vscode&imageSrc=https://cdn.pixabay.com/photo/2015/04/23/22/00/tree-736885__340.jpg&imageCredit=sexy&url=https://yceffort.kr/2020/03/nextjs-02-data-fetching#4-getserversideprops
export default function GenerateScreenshot({
  title,
  tags,
  url,
  imageSrc,
  imageCredit,
}: {
  title: string
  tags: string
  url: string
  imageSrc: string
  imageCredit: string
}) {
  return (
    <Wrapper>
      <Preview>
        <Main>
          <H1>{title}</H1>
          <TagList>
            <TagUl>
              {tags.split(',').map((tag, index) => (
                <li key={index}>#{tag}</li>
              ))}
            </TagUl>
          </TagList>
          <URL>{url}</URL>
          <Author>@yceffort</Author>
          {imageCredit ? (
            <Attribution className="attribution">
              <span>image by: {imageCredit}</span>
            </Attribution>
          ) : null}
        </Main>
        <Image src={imageSrc || '/default-image.png'} alt="thumbnail" />
      </Preview>
    </Wrapper>
  )
}

export const getServerSideProps: GetServerSideProps = async (context) => {
  const {
    query: { title, tags, url, imageSrc = '', imageCredit = '' },
  } = context

  return {
    props: { title, tags, url, imageSrc, imageCredit },
  }
}
