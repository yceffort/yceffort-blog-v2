import * as stylex from '@stylexjs/stylex'

const sx = stylex.create({
  section: {
    '@layer site': {
      marginTop: '56px',
    },
  },
  heading: {
    '@layer site': {
      display: 'flex',
      alignItems: 'center',
      gap: '12px',
      marginBottom: '6px',
      fontFamily: 'var(--font-mono), monospace',
      fontSize: '11.5px',
      fontWeight: '600',
      letterSpacing: '0.3em',
      textTransform: 'uppercase',
      color: 'var(--primary)',
    },
  },
  heading_rule: {
    '@layer site': {
      flex: 1,
      height: '1px',
      background: 'linear-gradient(90deg, var(--border-2), transparent)',
    },
  },
  heading_more: {
    '@layer site': {
      fontFamily: 'var(--font-sans)',
      fontSize: '13px',
      fontWeight: '500',
      letterSpacing: '0',
      textTransform: 'none',
      color: {
        default: 'var(--ink-3)',
        ':hover': 'var(--ink)',
      },
    },
  },
  month: {
    '@layer site': {
      marginTop: '28px',
      marginBottom: '4px',
      fontSize: '13px',
      fontWeight: '600',
      color: 'var(--ink-3)',
    },
  },
  row: {
    '@layer site': {
      display: 'grid',
      gridTemplateColumns: {
        default: '1fr auto',
        '@media (min-width: 640px)': '110px 1fr auto',
      },
      columnGap: '16px',
      rowGap: '2px',
      alignItems: 'baseline',
      padding: '13px 0',
      borderTop: '1px solid var(--border)',
      '--row-title-color': {
        default: 'var(--ink)',
        ':hover': 'var(--primary)',
      },
    },
  },
  row_date: {
    '@layer site': {
      fontFamily: 'var(--font-mono), monospace',
      fontSize: '12.5px',
      fontVariantNumeric: 'tabular-nums',
      color: 'var(--ink-3)',
      order: {
        default: 1,
        '@media (min-width: 640px)': 0,
      },
      gridColumn: {
        default: '1 / -1',
        '@media (min-width: 640px)': 'auto',
      },
    },
  },
  row_title: {
    '@layer site': {
      fontSize: '15px',
      fontWeight: '600',
      lineHeight: '1.45',
      color: 'var(--row-title-color)',
      transition: 'color 160ms ease',
    },
  },
  row_count: {
    '@layer site': {
      fontFamily: 'var(--font-mono), monospace',
      fontSize: '11.5px',
      color: 'var(--ink-4)',
      whiteSpace: 'nowrap',
    },
  },
})

export const section = stylex.props(sx.section).className ?? ''
export const heading = stylex.props(sx.heading).className ?? ''
export const heading_rule = stylex.props(sx.heading_rule).className ?? ''
export const heading_more = stylex.props(sx.heading_more).className ?? ''
export const month = stylex.props(sx.month).className ?? ''
export const row = stylex.props(sx.row).className ?? ''
export const row_date = stylex.props(sx.row_date).className ?? ''
export const row_title = stylex.props(sx.row_title).className ?? ''
export const row_count = stylex.props(sx.row_count).className ?? ''
