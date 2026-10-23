import * as stylex from '@stylexjs/stylex'

const trace_sweep = stylex.keyframes({
  from: {
    transform: 'translateX(-40%)',
  },
  to: {
    transform: 'translateX(140%)',
  },
})
const styles = stylex.create({
  about_hero: {
    '@layer site': {
      padding: '24px 0 32px',
    },
  },
  about_socials: {
    '@layer site': {
      display: 'flex',
      flexWrap: 'wrap',
      gap: '16px',
      marginTop: '20px',
      fontFamily: 'var(--font-mono), monospace',
      fontSize: '13px',
      color: 'var(--ink-2)',
    },
  },
  about_social_link: {
    '@layer site': {
      display: 'inline-flex',
      alignItems: 'center',
      padding: '8px 14px',
      borderTopWidth: '1px',
      borderTopStyle: 'solid',
      borderTopColor: {
        default: 'var(--border)',
        ':hover': 'var(--primary)',
      },
      borderRightWidth: '1px',
      borderRightStyle: 'solid',
      borderRightColor: {
        default: 'var(--border)',
        ':hover': 'var(--primary)',
      },
      borderBottomWidth: '1px',
      borderBottomStyle: 'solid',
      borderBottomColor: {
        default: 'var(--border)',
        ':hover': 'var(--primary)',
      },
      borderLeftWidth: '1px',
      borderLeftStyle: 'solid',
      borderLeftColor: {
        default: 'var(--border)',
        ':hover': 'var(--primary)',
      },
      borderRadius: '8px',
      gap: '8px',
      color: {
        default: null,
        ':hover': 'var(--primary)',
      },
      backgroundColor: 'var(--surface)',
      backgroundImage: 'none',
      backgroundPosition: 'initial',
      backgroundSize: 'auto',
      backgroundRepeat: 'repeat',
      backgroundOrigin: 'padding-box',
      backgroundClip: 'border-box',
      backgroundAttachment: 'scroll',
      transition: 'all 200ms',
    },
  },
  about_tab_link: {
    '@layer site': {
      padding: '12px 24px',
      color: {
        default: 'var(--ink-3)',
        ':is([data-active="true"])': 'var(--ink)',
      },
      flex: '1',
      fontSize: '14px',
      fontWeight: '600',
      letterSpacing: '-0.01em',
      textAlign: 'center',
      textDecoration: 'none',
    },
    backgroundColor: {
      default: null,
      '::after': {
        default: null,
        '@layer site': 'var(--primary)',
      },
    },
    backgroundImage: {
      default: null,
      '::after': {
        default: null,
        '@layer site': 'none',
      },
    },
    backgroundPosition: {
      default: null,
      '::after': {
        default: null,
        '@layer site': 'initial',
      },
    },
    backgroundSize: {
      default: null,
      '::after': {
        default: null,
        '@layer site': 'auto',
      },
    },
    backgroundRepeat: {
      default: null,
      '::after': {
        default: null,
        '@layer site': 'repeat',
      },
    },
    backgroundOrigin: {
      default: null,
      '::after': {
        default: null,
        '@layer site': 'padding-box',
      },
    },
    backgroundClip: {
      default: null,
      '::after': {
        default: null,
        '@layer site': 'border-box',
      },
    },
    backgroundAttachment: {
      default: null,
      '::after': {
        default: null,
        '@layer site': 'scroll',
      },
    },
    transition: {
      default: null,
      '@layer site': 'color 180ms',
      '::after': {
        default: null,
        '@layer site': 'transform 240ms cubic-bezier(0.2, 0.9, 0.2, 1)',
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
    content: {
      default: null,
      '::after': {
        default: null,
        '@layer site': "''",
      },
    },
    right: {
      default: null,
      '::after': {
        default: null,
        '@layer site': '0',
      },
    },
    bottom: {
      default: null,
      '::after': {
        default: null,
        '@layer site': '-1px',
      },
    },
    left: {
      default: null,
      '::after': {
        default: null,
        '@layer site': '0',
      },
    },
    height: {
      default: null,
      '::after': {
        default: null,
        '@layer site': '2px',
      },
    },
    transform: {
      default: null,
      '::after': {
        default: null,
        '@layer site': {
          default: 'scaleX(0)',
          ':is([data-active="true"])': 'scaleX(1)',
        },
      },
    },
  },
  tabs: {
    '@layer site': {
      position: 'relative',
      display: 'flex',
      margin: '32px 0 24px',
      borderBottomWidth: '1px',
      borderBottomStyle: 'solid',
      borderBottomColor: 'var(--border)',
      gap: '0',
    },
  },
  about_title: {
    '@layer site': {
      margin: '4px 0 28px',
      fontSize: 'clamp(56px, 11vw, 128px)',
      fontWeight: '900',
      lineHeight: '0.9',
      letterSpacing: '-0.05em',
      color: 'var(--ink)',
    },
  },
  trace_figure: {
    '@layer site': {
      position: 'relative',
      margin: '28px 0 0',
    },
  },
  trace_detail: {
    '@layer site': {
      position: 'absolute',
      top: '14px',
      left: '14px',
      display: 'flex',
      flexDirection: 'column',
      width: 'min(340px, calc(100% - 28px))',
      maxHeight: {
        default: '312px',
        '@media (min-width: 768px)': '432px',
      },
      overflowY: 'auto',
      padding: '14px 16px',
      borderRadius: '10px',
      backgroundColor: 'rgba(22, 22, 31, 0.94)',
      boxShadow: '0 0 0 1px rgba(255, 255, 255, 0.1) inset',
      gap: '10px',
      fontFamily: 'var(--font-mono), monospace',
      fontSize: '12px',
      lineHeight: '1.5',
      color: '#c9c9d6',
    },
  },
  trace_detail_head: {
    '@layer site': {
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      gap: '12px',
    },
  },
  trace_detail_close: {
    '@layer site': {
      flexShrink: '0',
      padding: '3px 10px',
      borderWidth: '0',
      borderRadius: '6px',
      backgroundColor: {
        default: 'rgba(245, 245, 250, 0.08)',
        ':hover': 'rgba(245, 245, 250, 0.16)',
      },
      cursor: 'pointer',
      fontSize: '12px',
      color: '#f5f5fa',
    },
  },
  trace_detail_url: {
    '@layer site': {
      margin: '0',
      overflowWrap: 'anywhere',
      fontSize: '11px',
      color: '#8b8ba3',
    },
  },
  // 모든 행이 같은 열 폭을 쓰도록 dl 하나를 두 열 그리드로 두고 행은 풀어 놓는다
  trace_detail_rows: {
    '@layer site': {
      display: 'grid',
      gridTemplateColumns: 'max-content 1fr',
      margin: '0',
      gap: '3px 16px',
    },
  },
  trace_detail_row: {
    '@layer site': {
      display: 'contents',
      color: '#8b8ba3',
    },
  },
  trace_detail_value: {
    '@layer site': {
      margin: '0',
      color: '#f5f5fa',
    },
  },
  trace_phase_bar: {
    '@layer site': {
      display: 'flex',
      overflow: 'hidden',
      height: '8px',
      borderRadius: '4px',
      gap: '1px',
    },
  },
  trace_phase_list: {
    '@layer site': {
      display: 'flex',
      flexWrap: 'wrap',
      margin: '0',
      padding: '0',
      listStyle: 'none',
      gap: '4px 12px',
      fontSize: '11px',
      color: '#8b8ba3',
    },
  },
  trace_detail_note: {
    '@layer site': {
      margin: '0',
      fontSize: '11px',
      color: '#8b8ba3',
    },
  },
  // 빛 표현을 위해 라이트 모드에서도 어둡게 둔다
  trace_viewport: {
    '@layer site': {
      position: 'relative',
      overflow: 'hidden',
      height: {
        default: '340px',
        '@media (min-width: 768px)': '460px',
      },
      borderRadius: 'var(--radius)',
      backgroundColor: '#0a0a0f',
      boxShadow: '0 0 0 1px rgba(255, 255, 255, 0.06) inset',
      touchAction: 'pan-y',
      // Shift 드래그가 글자 선택이나 선택 끌기로 새지 않게 한다
      userSelect: 'none',
      cursor: 'grab',
    },
  },
  trace_grabbing: {
    '@layer site': {
      cursor: 'grabbing',
    },
  },
  trace_hint: {
    '@layer site': {
      position: 'absolute',
      bottom: '14px',
      left: '18px',
      margin: '0',
      fontSize: '11px',
      color: '#8b8ba3',
    },
  },
  trace_overlay: {
    '@layer site': {
      position: 'absolute',
      top: '0',
      right: '0',
      bottom: '0',
      left: '0',
      overflow: 'hidden',
      pointerEvents: 'none',
      fontFamily: 'var(--font-mono), monospace',
      color: '#c9c9d6',
    },
  },
  trace_hud: {
    '@layer site': {
      position: 'absolute',
      top: '14px',
      right: '18px',
      // 자릿수가 늘어도 왼쪽 끝이 움직이지 않게 폭을 고정한다(레이아웃 이동 방지)
      width: '9ch',
      textAlign: 'right',
      fontSize: 'clamp(22px, 3.4vw, 34px)',
      fontWeight: '600',
      fontVariantNumeric: 'tabular-nums',
      letterSpacing: '-0.02em',
      color: '#f5f5fa',
    },
  },
  trace_label: {
    '@layer site': {
      position: 'absolute',
      top: '0',
      left: '0',
      whiteSpace: 'nowrap',
      transitionProperty: 'opacity',
      transitionDuration: '300ms',
      willChange: 'transform',
    },
  },
  trace_label_text: {
    '@layer site': {
      display: 'block',
      transform: 'translate(-50%, -100%)',
      fontSize: '11px',
      color: '#8b8ba3',
    },
  },
  trace_label_mark: {
    '@layer site': {
      padding: '3px 7px',
      borderRadius: '6px',
      backgroundColor: 'rgba(245, 245, 250, 0.08)',
      fontSize: '12px',
      color: '#f5f5fa',
    },
  },
  trace_label_self: {
    '@layer site': {
      padding: '4px 9px',
      borderRadius: '6px',
      backgroundColor: '#2dd4bf',
      fontSize: '12px',
      fontWeight: '600',
      color: '#0a0a0f',
    },
  },
  trace_label_lane: {
    '@layer site': {
      transform: 'translate(-100%, -50%)',
      paddingRight: '10px',
      color: '#fb7185',
    },
  },
  trace_legend: {
    '@layer site': {
      position: 'absolute',
      right: '16px',
      bottom: '14px',
      display: 'flex',
      flexWrap: 'wrap',
      justifyContent: 'flex-end',
      maxWidth: '70%',
      margin: '0',
      padding: '0',
      listStyle: 'none',
      gap: '6px 14px',
      fontSize: '11px',
      color: '#8b8ba3',
    },
  },
  trace_legend_item: {
    '@layer site': {
      display: 'inline-flex',
      alignItems: 'center',
      gap: '6px',
    },
  },
  trace_swatch: {
    '@layer site': {
      display: 'inline-block',
      width: '8px',
      height: '8px',
      borderRadius: '2px',
    },
  },
  trace_tooltip: {
    '@layer site': {
      position: 'absolute',
      top: '0',
      left: '0',
      display: 'flex',
      flexDirection: 'column',
      maxWidth: '280px',
      padding: '10px 12px',
      borderRadius: '8px',
      backgroundColor: 'rgba(22, 22, 31, 0.92)',
      boxShadow: '0 0 0 1px rgba(255, 255, 255, 0.1) inset',
      gap: '2px',
      fontSize: '11px',
      lineHeight: '1.5',
      color: '#8b8ba3',
    },
  },
  trace_tooltip_title: {
    '@layer site': {
      overflow: 'hidden',
      textOverflow: 'ellipsis',
      whiteSpace: 'nowrap',
      fontSize: '12px',
      fontWeight: '600',
      color: '#f5f5fa',
    },
  },
  // 측정값에 따라 길이가 바뀌는 요약은 한 줄로 고정해, 로딩 전후 캡션 높이가 같게 한다
  trace_caption: {
    '@layer site': {
      display: 'grid',
      gridTemplateColumns: '1fr auto',
      alignItems: 'start',
      marginTop: '12px',
      gap: '4px 20px',
      fontSize: '14px',
      lineHeight: '1.6',
      color: 'var(--ink-3)',
    },
  },
  trace_caption_text: {
    '@layer site': {
      margin: '0',
    },
  },
  trace_stats: {
    '@layer site': {
      gridColumn: '1 / -1',
      overflow: 'hidden',
      margin: '0',
      textOverflow: 'ellipsis',
      whiteSpace: 'nowrap',
      fontFamily: 'var(--font-mono), monospace',
      fontSize: '13px',
      color: 'var(--ink-2)',
    },
  },
  trace_loading: {
    '@layer site': {
      position: 'absolute',
      top: '0',
      right: '0',
      bottom: '0',
      left: '0',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      overflow: 'hidden',
    },
  },
  trace_loading_line: {
    '@layer site': {
      position: 'absolute',
      top: '0',
      bottom: '0',
      left: '0',
      width: '40%',
      backgroundImage:
        'linear-gradient(90deg, transparent, rgba(45, 212, 191, 0.14) 45%, rgba(45, 212, 191, 0.5) 50%, rgba(45, 212, 191, 0.14) 55%, transparent)',
      animationName: {
        default: trace_sweep,
        '@media (prefers-reduced-motion: reduce)': 'none',
      },
      animationDuration: '1.6s',
      animationTimingFunction: 'ease-in-out',
      animationIterationCount: 'infinite',
    },
  },
  trace_loading_text: {
    '@layer site': {
      position: 'relative',
      fontFamily: 'var(--font-mono), monospace',
      fontSize: '12px',
      color: '#8b8ba3',
    },
  },
  trace_replay: {
    '@layer site': {
      padding: '6px 14px',
      borderTopWidth: '1px',
      borderTopStyle: 'solid',
      borderTopColor: {
        default: 'var(--border)',
        ':hover': 'var(--primary)',
      },
      borderRightWidth: '1px',
      borderRightStyle: 'solid',
      borderRightColor: {
        default: 'var(--border)',
        ':hover': 'var(--primary)',
      },
      borderBottomWidth: '1px',
      borderBottomStyle: 'solid',
      borderBottomColor: {
        default: 'var(--border)',
        ':hover': 'var(--primary)',
      },
      borderLeftWidth: '1px',
      borderLeftStyle: 'solid',
      borderLeftColor: {
        default: 'var(--border)',
        ':hover': 'var(--primary)',
      },
      borderRadius: '8px',
      backgroundColor: 'transparent',
      cursor: {
        default: 'pointer',
        ':disabled': 'default',
      },
      opacity: {
        default: '1',
        ':disabled': '0.5',
      },
      fontSize: '13px',
      color: {
        default: 'var(--ink-2)',
        ':hover': 'var(--primary)',
      },
    },
  },
  trace_fallback: {
    '@layer site': {
      padding: '24px',
      fontSize: '14px',
      color: '#c9c9d6',
    },
  },
})

// Resolve locally so the compiler can erase style composition.
export const about_hero = stylex.props(styles.about_hero).className!
export const about_socials = stylex.props(styles.about_socials).className!
export const about_social_link = stylex.props(
  styles.about_social_link,
).className!
export const about_tab_link = stylex.props(styles.about_tab_link).className!
export const tabs = stylex.props(styles.tabs).className!
export const about_title = stylex.props(styles.about_title).className!
export const trace_figure = stylex.props(styles.trace_figure).className!
export const trace_detail = stylex.props(styles.trace_detail).className!
export const trace_detail_head = stylex.props(
  styles.trace_detail_head,
).className!
export const trace_detail_close = stylex.props(
  styles.trace_detail_close,
).className!
export const trace_detail_url = stylex.props(styles.trace_detail_url).className!
export const trace_detail_rows = stylex.props(
  styles.trace_detail_rows,
).className!
export const trace_detail_row = stylex.props(styles.trace_detail_row).className!
export const trace_detail_value = stylex.props(
  styles.trace_detail_value,
).className!
export const trace_phase_bar = stylex.props(styles.trace_phase_bar).className!
export const trace_phase_list = stylex.props(styles.trace_phase_list).className!
export const trace_detail_note = stylex.props(
  styles.trace_detail_note,
).className!
export const trace_viewport = stylex.props(styles.trace_viewport).className!
export const trace_viewport_grabbing = stylex.props(
  styles.trace_viewport,
  styles.trace_grabbing,
).className!
export const trace_hint = stylex.props(styles.trace_hint).className!
export const trace_overlay = stylex.props(styles.trace_overlay).className!
export const trace_hud = stylex.props(styles.trace_hud).className!
export const trace_label = stylex.props(styles.trace_label).className!
// 라벨 종류마다 trace_label_text 위에 덮어쓰는 속성이 있어 합성한다
export const trace_label_text = {
  tick: stylex.props(styles.trace_label_text).className!,
  mark: stylex.props(styles.trace_label_text, styles.trace_label_mark)
    .className!,
  self: stylex.props(styles.trace_label_text, styles.trace_label_self)
    .className!,
  lane: stylex.props(styles.trace_label_text, styles.trace_label_lane)
    .className!,
}
export const trace_legend = stylex.props(styles.trace_legend).className!
export const trace_legend_item = stylex.props(
  styles.trace_legend_item,
).className!
export const trace_swatch = stylex.props(styles.trace_swatch).className!
export const trace_tooltip = stylex.props(styles.trace_tooltip).className!
export const trace_tooltip_title = stylex.props(
  styles.trace_tooltip_title,
).className!
export const trace_caption = stylex.props(styles.trace_caption).className!
export const trace_caption_text = stylex.props(
  styles.trace_caption_text,
).className!
export const trace_stats = stylex.props(styles.trace_stats).className!
export const trace_loading = stylex.props(styles.trace_loading).className!
export const trace_loading_line = stylex.props(
  styles.trace_loading_line,
).className!
export const trace_loading_text = stylex.props(
  styles.trace_loading_text,
).className!
export const trace_replay = stylex.props(styles.trace_replay).className!
export const trace_fallback = stylex.props(styles.trace_fallback).className!
