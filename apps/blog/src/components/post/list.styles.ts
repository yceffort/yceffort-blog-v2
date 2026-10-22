import * as stylex from '@stylexjs/stylex'
const styles = stylex.create({
  post_row_list: {
    '@layer site': {
      display: 'flex',
      flexDirection: 'column',
      borderTopWidth: '1px',
      borderTopStyle: 'solid',
      borderTopColor: 'var(--border)',
    },
  },
  post_row_list_li: {
    '@layer site': {
      borderBottomWidth: '1px',
      borderBottomStyle: 'solid',
      borderBottomColor: 'var(--border)',
    },
  },
  post_row: {
    '@layer site': {
      position: 'relative',
      display: 'grid',
      gridTemplateColumns: {
        default: '96px 1fr 24px',
        '@media (max-width: 640px)': '72px 1fr',
      },
      alignItems: 'center',
      gap: {
        default: '20px',
        '@media (max-width: 640px)': '14px',
      },
      padding: {
        default: '18px 14px',
        '@media (max-width: 640px)': '14px 8px',
      },
      transition:
        'background-color 200ms ease,\n    transform 220ms cubic-bezier(0.2, 0.9, 0.2, 1)',
      backgroundColor: {
        default: null,
        ':hover': 'var(--surface)',
        '@supports (color: color-mix(in lab, red, red))': {
          default: null,
          ':hover': 'color-mix(in oklab, var(--surface) 60%, transparent)',
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
      '--post-row-title-color': {
        default: null,
        ':hover': 'var(--primary)',
      },
      '--post-row-arrow-color': {
        default: null,
        ':hover': 'var(--primary)',
      },
      '--post-row-arrow-transform': {
        default: null,
        ':hover': 'translateX(3px)',
      },
    },
  },
  post_row_link: {
    '@layer site': {
      position: 'absolute',
      inset: '0',
      zIndex: '3',
    },
  },
  post_row_thumb: {
    '@layer site': {
      overflow: 'hidden',
      position: 'relative',
      width: {
        default: '96px',
        '@media (max-width: 640px)': '72px',
      },
      borderWidth: '1px',
      borderStyle: 'solid',
      borderColor: 'var(--border)',
      borderRadius: '10px',
      backgroundColor: 'var(--surface-2)',
      aspectRatio: '3 / 2',
    },
  },
  // `.post-card .thumb img` 전용. `.post-row-thumb img`는 post_row_thumb_img를 쓴다
  post_card_thumb_img: {
    '@layer site': {
      objectFit: 'cover',
      transition: 'transform 700ms cubic-bezier(0.2, 0.9, 0.2, 1)',
      transform: 'var(--post-card-thumb-transform, none)',
    },
  },
  post_row_thumb_img: {
    '@layer site': {
      objectFit: 'cover',
    },
  },
  post_row_thumb_empty: {
    '@layer site': {
      display: 'grid',
      placeItems: 'center',
      backgroundColor: 'transparent',
      backgroundImage: {
        default:
          'linear-gradient(\n    135deg,\n    var(--primary),\n    var(--primary-3)\n  )',
        '@supports (color: color-mix(in lab, red, red))':
          'linear-gradient(\n    135deg,\n    color-mix(in oklab, var(--primary) 10%, var(--surface-2)),\n    color-mix(in oklab, var(--primary-3) 7%, var(--surface-2))\n  )',
      },
      color: 'var(--ink-3)',
    },
  },
  post_row_thumb_empty_svg: {
    '@layer site': {
      width: '60%',
      height: 'auto',
      color: 'var(--ink-4)',
    },
  },
  post_row_body: {
    '@layer site': {
      display: 'flex',
      flexDirection: 'column',
      minWidth: '0',
      gap: '6px',
    },
  },
  post_row_head: {
    '@layer site': {
      display: 'flex',
      flexWrap: 'wrap',
      alignItems: 'center',
      gap: '8px',
    },
  },
  series: {
    '@layer site': {
      fontWeight: '500',
    },
  },
  series_in_post_row: {
    '@layer site': {
      fontSize: '12px',
      lineHeight: '1.4',
      color: 'var(--ink-3)',
    },
  },
  post_row_tags: {
    '@layer site': {
      display: 'flex',
      flexWrap: 'wrap',
      gap: '6px',
    },
  },
  post_row_title: {
    '@layer site': {
      overflow: 'hidden',
      display: '-webkit-box',
      fontSize: {
        default: '17px',
        '@media (max-width: 640px)': '15.5px',
      },
      fontWeight: '700',
      lineHeight: '1.35',
      letterSpacing: '-0.015em',
      color: 'var(--post-row-title-color, var(--ink))',
      WebkitLineClamp: {
        default: '2',
        '@media (max-width: 640px)': '3',
      },
      WebkitBoxOrient: 'vertical',
    },
  },
  post_row_desc: {
    '@layer site': {
      overflow: 'hidden',
      display: {
        default: '-webkit-box',
        '@media (max-width: 640px)': 'none',
      },
      fontSize: '13px',
      lineHeight: '1.55',
      color: 'var(--ink-3)',
      WebkitLineClamp: '1',
      WebkitBoxOrient: 'vertical',
    },
  },
  post_row_meta: {
    '@layer site': {
      display: 'flex',
      alignItems: 'center',
      marginTop: '2px',
      fontFamily: 'var(--font-mono), monospace',
      fontSize: '11px',
      letterSpacing: '0.02em',
      color: 'var(--ink-4)',
      gap: '6px',
    },
  },
  post_row_arrow: {
    '@layer site': {
      color: 'var(--post-row-arrow-color, var(--ink-4))',
      transition:
        'transform 220ms cubic-bezier(0.2, 0.9, 0.2, 1),\n    color 180ms ease',
      transform: 'var(--post-row-arrow-transform, none)',
      display: {
        default: null,
        '@media (max-width: 640px)': 'none',
      },
    },
  },
  post_card: {
    '@layer site': {
      overflow: 'hidden',
      display: 'flex',
      flexDirection: 'column',
      borderTopWidth: '1px',
      borderTopStyle: 'solid',
      borderTopColor: {
        default: 'var(--border)',
        ':hover': 'var(--primary)',
        '@supports (color: color-mix(in lab, red, red))': {
          default: null,
          ':hover': 'color-mix(in oklab, var(--primary) 50%, var(--border))',
        },
      },
      borderRightWidth: '1px',
      borderRightStyle: 'solid',
      borderRightColor: {
        default: 'var(--border)',
        ':hover': 'var(--primary)',
        '@supports (color: color-mix(in lab, red, red))': {
          default: null,
          ':hover': 'color-mix(in oklab, var(--primary) 50%, var(--border))',
        },
      },
      borderBottomWidth: '1px',
      borderBottomStyle: 'solid',
      borderBottomColor: {
        default: 'var(--border)',
        ':hover': 'var(--primary)',
        '@supports (color: color-mix(in lab, red, red))': {
          default: null,
          ':hover': 'color-mix(in oklab, var(--primary) 50%, var(--border))',
        },
      },
      borderLeftWidth: '1px',
      borderLeftStyle: 'solid',
      borderLeftColor: {
        default: 'var(--border)',
        ':hover': 'var(--primary)',
        '@supports (color: color-mix(in lab, red, red))': {
          default: null,
          ':hover': 'color-mix(in oklab, var(--primary) 50%, var(--border))',
        },
      },
      transformStyle: 'preserve-3d',
      willChange: 'transform',
      boxShadow: {
        default: null,
        ':hover': '0 10px 40px -20px\n    var(--primary)',
        '@supports (color: color-mix(in lab, red, red))': {
          default: null,
          ':hover':
            '0 10px 40px -20px\n    color-mix(in oklab, var(--primary) 80%, transparent)',
        },
      },
      '--post-card-thumb-transform': {
        default: null,
        ':hover': 'scale(1.05)',
      },
    },
    position: {
      default: null,
      '@layer site': 'relative',
      '::after': {
        default: null,
        '@layer site': 'absolute',
      },
    },
    borderRadius: {
      default: null,
      '@layer site': 'var(--radius)',
      '::after': {
        default: null,
        '@layer site': 'inherit',
      },
    },
    backgroundColor: {
      default: null,
      '@layer site': 'var(--surface)',
      '::after': {
        default: null,
        '@layer site': 'transparent',
      },
    },
    backgroundImage: {
      default: null,
      '@layer site': 'none',
      '::after': {
        default: null,
        '@layer site': {
          default:
            'radial-gradient(\n    400px circle at var(--mx, 50%) var(--my, 50%),\n    var(--primary) 0%,\n    transparent 45%\n  )',
          '@supports (color: color-mix(in lab, red, red))':
            'radial-gradient(\n    400px circle at var(--mx, 50%) var(--my, 50%),\n    color-mix(in oklab, var(--primary) 10%, transparent) 0%,\n    transparent 45%\n  )',
        },
      },
    },
    backgroundPosition: {
      default: null,
      '@layer site': 'initial',
      '::after': {
        default: null,
        '@layer site': 'initial',
      },
    },
    backgroundSize: {
      default: null,
      '@layer site': 'auto',
      '::after': {
        default: null,
        '@layer site': 'auto',
      },
    },
    backgroundRepeat: {
      default: null,
      '@layer site': 'repeat',
      '::after': {
        default: null,
        '@layer site': 'repeat',
      },
    },
    backgroundOrigin: {
      default: null,
      '@layer site': 'padding-box',
      '::after': {
        default: null,
        '@layer site': 'padding-box',
      },
    },
    backgroundClip: {
      default: null,
      '@layer site': 'border-box',
      '::after': {
        default: null,
        '@layer site': 'border-box',
      },
    },
    backgroundAttachment: {
      default: null,
      '@layer site': 'scroll',
      '::after': {
        default: null,
        '@layer site': 'scroll',
      },
    },
    transition: {
      default: null,
      '@layer site': 'border-color 300ms,\n    box-shadow 300ms',
      '::after': {
        default: null,
        '@layer site': 'opacity 300ms',
      },
    },
    content: {
      default: null,
      '::after': {
        default: null,
        '@layer site': "''",
      },
    },
    opacity: {
      default: null,
      '::after': {
        default: null,
        '@layer site': {
          default: '0',
          ':hover': '1',
        },
      },
    },
    inset: {
      default: null,
      '::after': {
        default: null,
        '@layer site': '0',
      },
    },
    pointerEvents: {
      default: null,
      '::after': {
        default: null,
        '@layer site': 'none',
      },
    },
  },
  element_a: {
    '@layer site': {
      position: 'absolute',
      inset: '0',
      zIndex: '3',
    },
  },
  thumb: {
    '@layer site': {
      overflow: 'hidden',
      position: 'relative',
      borderBottomWidth: '1px',
      borderBottomStyle: 'solid',
      borderBottomColor: 'var(--border)',
      backgroundColor: 'var(--surface-2)',
      aspectRatio: '16 / 9',
    },
  },
  body: {
    '@layer site': {
      display: 'flex',
      flex: '1',
      flexDirection: 'column',
      padding: '18px 20px 20px',
      gap: '10px',
    },
  },
  element_h3: {
    '@layer site': {
      overflow: 'hidden',
      display: '-webkit-box',
      marginTop: '2px',
      fontSize: '17px',
      fontWeight: '700',
      lineHeight: '1.35',
      letterSpacing: '-0.015em',
      color: 'var(--ink)',
      transform: 'translateZ(20px)',
      WebkitLineClamp: '2',
      WebkitBoxOrient: 'vertical',
    },
  },
  desc: {
    '@layer site': {
      overflow: 'hidden',
      display: '-webkit-box',
      fontSize: '13px',
      lineHeight: '1.55',
      color: 'var(--ink-3)',
      transform: 'translateZ(8px)',
      WebkitLineClamp: '2',
      WebkitBoxOrient: 'vertical',
    },
  },
  meta: {
    '@layer site': {
      display: 'flex',
      alignItems: 'center',
      marginTop: 'auto',
      paddingTop: '10px',
      fontSize: '12px',
      color: 'var(--ink-4)',
      transform: 'translateZ(6px)',
      gap: '6px',
    },
  },
  draft_badge: {
    '@layer site': {
      position: 'absolute',
      top: 'calc(var(--spacing) * 2)',
      right: 'calc(var(--spacing) * 2)',
      zIndex: '10',
      borderRadius: 'var(--radius-md)',
      backgroundColor: 'var(--color-amber-500)',
      paddingInline: 'calc(var(--spacing) * 2)',
      paddingBlock: 'calc(var(--spacing) * 0.5)',
      fontSize: 'var(--text-xs)',
      lineHeight: 'var(--blog-leading, var(--text-xs--line-height))',
      fontWeight: 'var(--font-weight-bold)',
      color: 'var(--color-white)',
      textTransform: 'uppercase',
      '--blog-shadow':
        '0 1px 3px 0 var(--blog-shadow-color, rgb(0 0 0 / 0.1)), 0 1px 2px -1px var(--blog-shadow-color, rgb(0 0 0 / 0.1))',
      boxShadow:
        'var(--blog-inset-shadow), var(--blog-inset-ring-shadow), var(--blog-ring-offset-shadow), var(--blog-ring-shadow), var(--blog-shadow)',
    },
  },
  tag_chip: {
    '@layer site': {
      padding: '3px 9px',
      borderWidth: '1px',
      borderStyle: 'solid',
      borderColor: 'var(--border)',
      borderRadius: '999px',
      backgroundColor: 'var(--surface-2)',
      fontSize: '11px',
      fontWeight: '500',
      whiteSpace: 'nowrap',
      color: 'var(--ink-2)',
    },
  },
})

// Resolve locally so the compiler can erase style composition.
export const post_row_list = stylex.props(styles.post_row_list).className!
export const post_row_list_li = stylex.props(styles.post_row_list_li).className!
export const post_row = stylex.props(styles.post_row).className!
export const post_row_link = stylex.props(styles.post_row_link).className!
export const post_row_thumb = stylex.props(styles.post_row_thumb).className!
export const post_card_thumb_img = stylex.props(
  styles.post_card_thumb_img,
).className!
export const post_row_thumb_img = stylex.props(
  styles.post_row_thumb_img,
).className!
// Composed so the empty variant wins over the base thumb deterministically,
// instead of depending on the order the two class strings are emitted in.
export const post_row_thumb_empty_box = stylex.props(
  styles.post_row_thumb,
  styles.post_row_thumb_empty,
).className!
export const post_row_thumb_empty_svg = stylex.props(
  styles.post_row_thumb_empty_svg,
).className!
export const post_row_body = stylex.props(styles.post_row_body).className!
export const post_row_head = stylex.props(styles.post_row_head).className!
export const post_row_series = stylex.props(
  styles.series,
  styles.series_in_post_row,
).className!
export const post_row_tags = stylex.props(styles.post_row_tags).className!
export const post_row_title = stylex.props(styles.post_row_title).className!
export const post_row_desc = stylex.props(styles.post_row_desc).className!
export const post_row_meta = stylex.props(styles.post_row_meta).className!
export const post_row_arrow = stylex.props(styles.post_row_arrow).className!
export const post_card = stylex.props(styles.post_card).className!
export const element_a = stylex.props(styles.element_a).className!
export const thumb = stylex.props(styles.thumb).className!
export const body = stylex.props(styles.body).className!
export const element_h3 = stylex.props(styles.element_h3).className!
export const desc = stylex.props(styles.desc).className!
export const meta = stylex.props(styles.meta).className!
export const draft_badge = stylex.props(styles.draft_badge).className!
export const tag_chip = stylex.props(styles.tag_chip).className!
