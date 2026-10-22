import * as stylex from '@stylexjs/stylex'
const styles = stylex.create({
  featured: {
    '@layer site': {
      position: 'relative',
      display: 'grid',
      gridTemplateColumns: {
        default: 'minmax(0, 1fr)',
        '@media (width >= 64rem)': 'minmax(0, 46fr) minmax(0, 54fr)',
      },
      overflow: 'hidden',
      borderWidth: '1px',
      borderStyle: 'solid',
      borderColor: {
        default: 'var(--border)',
        ':hover': 'var(--primary)',
      },
      borderRadius: 'var(--radius)',
      backgroundColor: 'var(--surface)',
      transition: 'border-color 200ms, box-shadow 200ms',
      boxShadow: {
        default: null,
        ':hover': '0 10px 40px -20px var(--primary)',
      },
      '--featured-thumb-transform': {
        default: null,
        ':hover': 'scale(1.04)',
      },
    },
  },
  thumb: {
    '@layer site': {
      position: 'relative',
      overflow: 'hidden',
      aspectRatio: '16 / 9',
      backgroundColor: 'var(--surface-2)',
      borderBottomWidth: {
        default: '1px',
        '@media (width >= 64rem)': '0',
      },
      borderBottomStyle: 'solid',
      borderBottomColor: 'var(--border)',
      borderRightWidth: {
        default: '0',
        '@media (width >= 64rem)': '1px',
      },
      borderRightStyle: 'solid',
      borderRightColor: 'var(--border)',
    },
  },
  thumb_img: {
    '@layer site': {
      objectFit: 'cover',
      transition: 'transform 700ms cubic-bezier(0.2, 0.9, 0.2, 1)',
      transform: 'var(--featured-thumb-transform, none)',
    },
  },
  body: {
    '@layer site': {
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'center',
      gap: '12px',
      padding: {
        default: '20px 20px 22px',
        '@media (width >= 64rem)': '30px 34px',
      },
    },
  },
  title: {
    '@layer site': {
      overflow: 'hidden',
      display: '-webkit-box',
      fontSize: 'clamp(22px, 2.4vw, 30px)',
      fontWeight: '800',
      lineHeight: '1.25',
      letterSpacing: '-0.025em',
      color: 'var(--ink)',
      WebkitLineClamp: '3',
      WebkitBoxOrient: 'vertical',
    },
  },
  desc: {
    '@layer site': {
      overflow: 'hidden',
      display: '-webkit-box',
      fontSize: '15px',
      lineHeight: '1.6',
      color: 'var(--ink-3)',
      WebkitLineClamp: '3',
      WebkitBoxOrient: 'vertical',
    },
  },
  meta: {
    '@layer site': {
      fontFamily: 'var(--font-mono), monospace',
      fontSize: '12px',
      letterSpacing: '0.05em',
      color: 'var(--ink-3)',
    },
  },
})

// Resolve locally so the compiler can erase style composition.
export const featured = stylex.props(styles.featured).className!
export const thumb = stylex.props(styles.thumb).className!
export const thumb_img = stylex.props(styles.thumb_img).className!
export const body = stylex.props(styles.body).className!
export const title = stylex.props(styles.title).className!
export const desc = stylex.props(styles.desc).className!
export const meta = stylex.props(styles.meta).className!
