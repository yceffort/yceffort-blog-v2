// @flow strict
import { useStaticQuery, graphql } from 'gatsby'

const useTagsList = () => {
  const { allMarkdownRemark } = useStaticQuery(
    graphql`
      query TagsListQuery {
        allMarkdownRemark(
          filter: {
            frontmatter: { template: { eq: "post" }, published: { ne: false } }
          }
        ) {
          group(field: frontmatter___tags) {
            fieldValue
            totalCount
          }
        }
      }
    `,
  )

  return allMarkdownRemark.group
}

export default useTagsList
