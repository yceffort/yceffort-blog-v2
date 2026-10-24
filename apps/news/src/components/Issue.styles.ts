import * as stylex from '@stylexjs/stylex'

const sx = stylex.create({
  header: {
    '@layer site': {
      marginBottom: '8px',
    },
  },
  eyebrow: {
    '@layer site': {
      display: 'flex',
      alignItems: 'center',
      gap: '10px',
      fontFamily: 'var(--font-mono), monospace',
      fontSize: '12px',
      fontWeight: '600',
      letterSpacing: '0.3em',
      textTransform: 'uppercase',
      color: 'var(--primary)',
    },
  },
  eyebrow_dot: {
    '@layer site': {
      width: '4px',
      height: '4px',
      borderRadius: '50%',
      background: 'var(--ink-4)',
    },
  },
  title: {
    '@layer site': {
      marginTop: '14px',
      fontSize: 'clamp(36px, 7vw, 60px)',
      fontWeight: '900',
      lineHeight: '1.05',
      letterSpacing: '-0.04em',
      color: 'var(--ink)',
    },
  },
  subtitle: {
    '@layer site': {
      marginTop: '12px',
      fontSize: '15px',
      color: 'var(--ink-3)',
    },
  },
  nav: {
    '@layer site': {
      display: 'grid',
      gridTemplateColumns: '1fr 1fr',
      gap: '12px',
      marginTop: '56px',
    },
  },
  nav_link: {
    '@layer site': {
      display: 'flex',
      flexDirection: 'column',
      gap: '6px',
      padding: '14px 16px',
      border: '1px solid var(--border)',
      borderRadius: '12px',
      background: 'color-mix(in oklab, var(--surface) 55%, transparent)',
      borderColor: {
        default: 'var(--border)',
        ':hover': 'var(--border-2)',
      },
      transition: 'border-color 160ms ease, transform 160ms ease',
      transform: {
        default: null,
        ':hover': 'translateY(-1px)',
      },
    },
  },
  nav_link_next: {
    '@layer site': {
      alignItems: 'flex-end',
      textAlign: 'right',
    },
  },
  nav_label: {
    '@layer site': {
      fontFamily: 'var(--font-mono), monospace',
      fontSize: '10.5px',
      letterSpacing: '0.22em',
      textTransform: 'uppercase',
      color: 'var(--ink-4)',
    },
  },
  nav_title: {
    '@layer site': {
      fontSize: '14.5px',
      fontWeight: '600',
      color: 'var(--ink)',
    },
  },
  nav_sub: {
    '@layer site': {
      fontSize: '12.5px',
      color: 'var(--ink-3)',
    },
  },
})

export const header = stylex.props(sx.header).className ?? ''
export const eyebrow = stylex.props(sx.eyebrow).className ?? ''
export const eyebrow_dot = stylex.props(sx.eyebrow_dot).className ?? ''
export const title = stylex.props(sx.title).className ?? ''
export const subtitle = stylex.props(sx.subtitle).className ?? ''
export const nav = stylex.props(sx.nav).className ?? ''
export const nav_link = stylex.props(sx.nav_link).className ?? ''
export const nav_link_next =
  stylex.props(sx.nav_link, sx.nav_link_next).className ?? ''
export const nav_label = stylex.props(sx.nav_label).className ?? ''
export const nav_title = stylex.props(sx.nav_title).className ?? ''
export const nav_sub = stylex.props(sx.nav_sub).className ?? ''
