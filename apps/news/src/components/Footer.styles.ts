import * as stylex from '@stylexjs/stylex'

const sx = stylex.create({
  inner: {
    '@layer utilities': {
      marginTop: '5rem',
      paddingBottom: '2.5rem',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      gap: '0.5rem',
    },
  },
  line: {
    '@layer utilities': {
      display: 'flex',
      flexWrap: 'wrap',
      justifyContent: 'center',
      gap: '0.6rem',
      fontSize: '0.8125rem',
      color: 'var(--ink-3)',
    },
  },
  sep: {
    '@layer utilities': {
      color: 'var(--ink-4)',
    },
  },
  link: {
    '@layer utilities': {
      color: {
        default: 'var(--ink-3)',
        ':hover': 'var(--ink)',
      },
      transition: 'color 160ms ease',
    },
  },
})

export const inner = stylex.props(sx.inner).className ?? ''
export const line = stylex.props(sx.line).className ?? ''
export const sep = stylex.props(sx.sep).className ?? ''
export const link = stylex.props(sx.link).className ?? ''
