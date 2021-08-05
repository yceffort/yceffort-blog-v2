import { GetServerSideProps } from 'next'

import Screenshot from '#components/screenshot'

// const credential = {
//   projectId: 'yceffort',
//   clientEmail: 'firebase-adminsdk-7sbu2@yceffort.iam.gserviceaccount.com',
//   privateKey:
//     '-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC0GPXubp1jEuAI\nqaztjk1h3+x4C+UKIF6Kj+TxIQABYcWwDYSto+GJYLcblbnD9Chx1vVxzMdC2GHV\naKka9EeZmUx55iaD7I4RwoloBI+4/qSBwVhpcLyFkAt46k4b3jqRwQf0V+mim7xK\nVD3I5BpIO8bvJS5zRhyKOcW6hCp/sV4tRXvoc3J3gQS4OCCe+btDDIiPERgSj29j\n7t7bFtOhe748o4bkezhAJEBYUlFCxQJM5dQxyAdeKwptJbbbubUeTAUUu6Vl52rQ\nhmMAYZd0xr3LTIVq0KLMP9eA1sk2z8IiSdzKgxoFpp8pZdgpJq5FboK4i2e39vdv\nAm4ddOEdAgMBAAECggEABoFoUNE6FNwXr6MWwLYFNuXfJYRNz7cqSG244rpDGnRX\nrxZYlpUhjn7UiRs4IfXt3WmFut30zLCtnVyHJJbHsgAG8oL3x92Bpu6UsWczcdvw\n6/xPSe/ClOiGduWv0J3ka7jB+szk1Eo1MPzTW9sG1Vkjbya0A4H4LJ4NoGvUb9b+\nxlzjFcS+7kVzsY8WkSB2Rl26kytSfMNF53kIs1Twh1G2ec/FA3Z6YEhJ6ic5blb1\n1LQdshswYrYRBdqbkG2VJ5/vNdSLEWkhKKFb0KHkbKKhonsElqwvkT8Yyfw8XlZF\nTj8P6fwEiPxdsweTJREmdJcDz5Znn9n/JKmrUw8UQQKBgQDrFrHoyzkF3sPTjhOy\nrJTFiIkHjUwnnCixt7HlOjGem4DW81eNbEDPHpQTjJXS9eEvFw6JAAxWGguw4nQg\nEdNeXEDq6mu4TADoW/yIu48gAV4eXlhidIhujLCApSQXN5ZuQ43aKbUpqUHApu0t\nq/qc+sIrHStpMLg2+eRAPedd9QKBgQDEHgndqKxTnWI3HG6rx2Ndz/8vTR3wPK5m\nk6AVWDFsgaocLnOuJN9Np13nwXK3OnbNjSaNWd1zWXwpUhGt0X1YbgKTvQv9eJvC\npLZbiAs7s4dgg03GpSqeykQSLmvcwgU+VKND5xoVmKNMloT5k7SEmHIxQsAIz05u\ntJ2RWqSViQKBgGuRF9if3DieZFYRhVvU8cGspp6I/ZaGMmyW09RCG2AqYPp5n877\nAHCE2lZTll5P2Th1wVXYasye4EiQZXgjD+b2KVIT7zQFusiXBmb+AxAu8ATPQHvU\nPHTw9PX4Ghpxeeh8CpUPTnCAnLBs8MtcDLD1YBDgKPPZsgCduN3YNVxdAoGBAJEY\nhEYZb/2g2DRb9cljiDG1HGB7lqXRz1oW6H5CNLbJq/iTqYRyxT9nj0NSzTOgrprf\nTmGP1hZsYz8S9/94mVsecQuq9z79x4eXY0+O9HikF4mhO563PjQjA3/MFoNKjKST\n7ALl7VeDCXY1eoZH8GuVeg7WCsu5zJZ9TIJo5JG5AoGAD+2zKEyx4QwAF+Ix5dJf\nMtwJ+5/5YqgzGrA5x09j//Q6w2m4aOclUdgLlous5H16Cmtyx3HWEv6MhpIGRvTU\nx9L21/T/U6SMa3nv8ujQyLGhSm6O7YHpPy/1huyK+YxZE7o93gPd0CzmMbZcdXpL\nGVF/cK9dltZFDPEN+qXGhpY=\n-----END PRIVATE KEY-----\n',
// }

// ${hostUrl}/generate-screenshot?title=hello&tags=react,javascript&categories=vscode&imageSrc=https://cdn.pixabay.com/photo/2015/04/23/22/00/tree-736885__340.jpg&imageCredit=sexy&url=https://yceffort.kr/2020/03/nextjs-02-data-fetching#4-getserversideprops
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
    <Screenshot
      title={title}
      tags={tags.split(',')}
      url={url}
      imageSrc={imageSrc}
      imageCredit={imageCredit}
    />
  )
}

export const getServerSideProps: GetServerSideProps = async (context) => {
  const {
    title,
    tags,
    url,
    imageSrc = '',
    imageCredit = '',
    slug,
  } = context.query
  let engTitle = ''
  if (slug) {
    const splitSlug = (slug as string).split('/')
    const tempTitle = splitSlug[splitSlug.length - 1].replace(/-/gi, ' ')
    engTitle = tempTitle.charAt(0).toUpperCase() + tempTitle.slice(1)
  }
  return {
    props: { title: engTitle || title, tags, url, imageSrc, imageCredit },
  }
}
/*
export const getServerSideProps: GetServerSideProps = async (context) => {
  const query = context.query

  const q = Object.keys(query).reduce<{ [key in keyof typeof query]: string }>(
    (result, key) => {
      result[key] = (query[key] || '').toString()
      return result
    },
    {},
  )

  if ((admin.apps || []).length === 0) {
    admin.initializeApp({
      credential: admin.credential.cert(credential),
    })
  }

  const db = admin.firestore()

  const screenshotRef = db.collection('screenshot')
  const title = new URL(q.url).pathname.replace(/\//gi, '-').slice(1)

  const ref = await screenshotRef.doc(title).get()

  if (ref.exists) {
    return {
      redirect: {
        destination: ref.data()?.url,
        // permanent: true,
      },
    }
  }

  // const {
  //   title,
  //   tags,
  //   url,
  //   imageSrc = '',
  //   imageCredit = '',
  //   slug,
  // } = context.query
  // let engTitle = ''
  // if (slug) {
  //   const splitSlug = (slug as string).split('/')
  //   const tempTitle = splitSlug[splitSlug.length - 1].replace(/-/gi, ' ')
  //   engTitle = tempTitle.charAt(0).toUpperCase() + tempTitle.slice(1)
  // }
  // return {
  //   props: { title: engTitle || title, tags, url, imageSrc, imageCredit },
  // }
}
*/
