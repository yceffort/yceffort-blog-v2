import * as stylex from '@stylexjs/stylex'

const sx = stylex.create({
  container: {
    '@layer site': {
      marginInline: 'auto',
      maxWidth: '46rem',
      paddingInline: {
        default: '1rem',
        '@media (width >= 40rem)': '1.5rem',
      },
    },
  },
})

export const sectionContainer = stylex.props(sx.container).className ?? ''
