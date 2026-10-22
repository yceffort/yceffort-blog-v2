import * as stylex from '@stylexjs/stylex'
const styles = stylex.create({
  series_row: {
    '@layer site': {
      paddingLeft: '36px',
    },
  },
  rec_list: {
    '@layer site': {
      display: 'flex',
      flexDirection: 'column',
      borderTopWidth: '1px',
      borderTopStyle: 'solid',
      borderTopColor: 'var(--border)',
    },
  },
  rec_row: {
    '@layer site': {
      position: 'relative',
      display: 'grid',
      alignItems: 'center',
      padding: '20px 12px',
      borderBottomWidth: '1px',
      borderBottomStyle: 'solid',
      borderBottomColor: 'var(--border)',
      transition: 'background 200ms',
      cursor: 'pointer',
      gridTemplateColumns: {
        default: '56px 1.2fr 1fr 130px 28px',
        '@media (max-width: 900px)': '40px 1fr 24px',
      },
      gap: '20px',
      backgroundColor: {
        default: null,
        ':hover': 'var(--surface)',
        '@supports (color: color-mix(in lab, red, red))': {
          default: null,
          ':hover': 'color-mix(in oklab, var(--surface) 50%, transparent)',
        },
      },
      backgroundImage: {
        default: null,
        ':hover': 'none',
        '@supports (color: color-mix(in lab, red, red))': {
          default: null,
          ':hover': 'none',
        },
      },
      backgroundPosition: {
        default: null,
        ':hover': 'initial',
        '@supports (color: color-mix(in lab, red, red))': {
          default: null,
          ':hover': 'initial',
        },
      },
      backgroundSize: {
        default: null,
        ':hover': 'auto',
        '@supports (color: color-mix(in lab, red, red))': {
          default: null,
          ':hover': 'auto',
        },
      },
      backgroundRepeat: {
        default: null,
        ':hover': 'repeat',
        '@supports (color: color-mix(in lab, red, red))': {
          default: null,
          ':hover': 'repeat',
        },
      },
      backgroundOrigin: {
        default: null,
        ':hover': 'padding-box',
        '@supports (color: color-mix(in lab, red, red))': {
          default: null,
          ':hover': 'padding-box',
        },
      },
      backgroundClip: {
        default: null,
        ':hover': 'border-box',
        '@supports (color: color-mix(in lab, red, red))': {
          default: null,
          ':hover': 'border-box',
        },
      },
      backgroundAttachment: {
        default: null,
        ':hover': 'scroll',
        '@supports (color: color-mix(in lab, red, red))': {
          default: null,
          ':hover': 'scroll',
        },
      },
      '--rec-row-rarrow-color': {
        default: null,
        ':hover': 'var(--primary)',
      },
      '--rec-row-rarrow-transform': {
        default: null,
        ':hover': 'translateX(4px)',
      },
    },
  },
  // 홈 글 목록 변형: 순번 뒤에 썸네일 칸이 들어간다
  thumb_row: {
    '@layer site': {
      gridTemplateColumns: {
        default: '40px 128px 1.2fr 1fr 130px 28px',
        '@media (max-width: 900px)': '32px 88px 1fr 24px',
      },
    },
  },
  rthumb: {
    '@layer site': {
      position: 'relative',
      overflow: 'hidden',
      width: {
        default: '128px',
        '@media (max-width: 900px)': '88px',
      },
      aspectRatio: '16 / 9',
      borderWidth: '1px',
      borderStyle: 'solid',
      borderColor: 'var(--border)',
      borderRadius: '8px',
      backgroundColor: 'var(--surface-2)',
    },
  },
  rthumb_img: {
    '@layer site': {
      objectFit: 'cover',
    },
  },
  rn: {
    '@layer site': {
      fontFamily: 'var(--font-mono), monospace',
      fontSize: '12px',
      letterSpacing: '0.1em',
      color: 'var(--ink-4)',
    },
  },
  element_h4: {
    '@layer site': {
      overflow: 'hidden',
      display: '-webkit-box',
      fontSize: '16px',
      fontWeight: '700',
      lineHeight: '1.4',
      letterSpacing: '-0.01em',
      color: 'var(--ink)',
      WebkitLineClamp: '2',
      WebkitBoxOrient: 'vertical',
    },
  },
  rtags: {
    '@layer site': {
      display: 'flex',
      flexWrap: 'wrap',
      marginTop: '6px',
      fontFamily: 'var(--font-mono), monospace',
      fontSize: '10.5px',
      letterSpacing: '0.08em',
      textTransform: 'uppercase',
      color: 'var(--ink-3)',
      gap: '4px',
    },
  },
  rd: {
    '@layer site': {
      overflow: 'hidden',
      display: {
        default: '-webkit-box',
        '@media (max-width: 900px)': 'none',
      },
      fontSize: '13px',
      lineHeight: '1.55',
      color: 'var(--ink-3)',
      WebkitLineClamp: '2',
      WebkitBoxOrient: 'vertical',
    },
  },
  rmeta: {
    '@layer site': {
      fontFamily: 'var(--font-mono), monospace',
      fontSize: '11px',
      letterSpacing: '0.05em',
      textAlign: 'right',
      color: 'var(--ink-3)',
      display: {
        default: null,
        '@media (max-width: 900px)': 'none',
      },
    },
  },
  element_b: {
    '@layer site': {
      display: 'block',
      marginBottom: '2px',
      fontSize: '12px',
      fontWeight: '600',
      color: 'var(--ink-2)',
    },
  },
  rarrow: {
    '@layer site': {
      color: 'var(--rec-row-rarrow-color, var(--ink-3))',
      transition:
        'transform 220ms cubic-bezier(0.2, 0.9, 0.2, 1),\n    color 180ms',
      transform: 'var(--rec-row-rarrow-transform, none)',
    },
  },
  element_a: {
    '@layer site': {
      position: 'absolute',
      inset: '0',
      zIndex: '3',
    },
  },
})

// Resolve locally so the compiler can erase style composition.
export const rec_list = stylex.props(styles.rec_list).className!
export const rec_row = stylex.props(styles.rec_row).className!
export const series_row = stylex.props(
  styles.rec_row,
  styles.series_row,
).className!
export const thumb_row = stylex.props(
  styles.rec_row,
  styles.thumb_row,
).className!
export const rthumb = stylex.props(styles.rthumb).className!
export const rthumb_img = stylex.props(styles.rthumb_img).className!
export const rn = stylex.props(styles.rn).className!
export const element_h4 = stylex.props(styles.element_h4).className!
export const rtags = stylex.props(styles.rtags).className!
export const rd = stylex.props(styles.rd).className!
export const rmeta = stylex.props(styles.rmeta).className!
export const element_b = stylex.props(styles.element_b).className!
export const rarrow = stylex.props(styles.rarrow).className!
export const element_a = stylex.props(styles.element_a).className!
