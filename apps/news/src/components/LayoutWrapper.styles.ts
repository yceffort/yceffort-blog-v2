import * as stylex from '@stylexjs/stylex'

const motion_logo_spin = stylex.keyframes({
  to: {transform: 'rotate(360deg)'},
})

const styles = stylex.create({
  site_header: {
    '@layer site': {
      position: 'sticky',
      top: 0,
      zIndex: 40,
      borderBottom: '1px solid var(--border)',
      background: 'color-mix(in oklab, var(--bg) 70%, transparent)',
      WebkitBackdropFilter: 'blur(14px) saturate(140%)',
      backdropFilter: 'blur(14px) saturate(140%)',
    },
  },
  site_header_inner: {
    '@layer site': {
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      maxWidth: '1440px',
      margin: '0 auto',
      padding: {
        default: '12px 16px',
        '@media (min-width: 640px)': '16px 32px',
      },
      gap: '16px',
    },
  },
  logo_link: {
    '@layer site': {
      display: 'flex',
      alignItems: 'center',
      gap: '0.75rem',
    },
  },
  logo_ring: {
    '@layer site': {
      display: 'grid',
      flexShrink: 0,
      width: '44px',
      height: '44px',
      padding: '2px',
      borderRadius: '50%',
      background:
        'conic-gradient(from 0deg, var(--primary), var(--primary-2), var(--primary-3), #fbbf24, var(--primary))',
      transition: 'transform 400ms cubic-bezier(0.2, 0.9, 0.2, 1)',
      transform: {
        default: null,
        ':hover': 'rotate(8deg) scale(1.06)',
      },
      animationName: {
        default: motion_logo_spin,
        '@media (prefers-reduced-motion: reduce)': 'none',
      },
      animationDuration: '14s',
      animationTimingFunction: 'linear',
      animationIterationCount: 'infinite',
      placeItems: 'center',
    },
  },
  logo_ring_inner: {
    '@layer site': {
      overflow: 'hidden',
      display: 'grid',
      width: '100%',
      height: '100%',
      borderRadius: '50%',
      background: 'var(--bg)',
      placeItems: 'center',
    },
  },
  avatar: {
    '@layer site': {
      height: '2.5rem',
      width: '2.5rem',
      borderRadius: '9999px',
    },
  },
  logo_name: {
    '@layer site': {
      display: 'flex',
      alignItems: 'center',
      gap: '8px',
      fontSize: '18px',
      fontWeight: '800',
      lineHeight: '1.1',
      letterSpacing: '-0.02em',
      color: 'var(--ink)',
    },
  },
  logo_text: {
    '@layer site': {
      display: {
        default: null,
        '@media (max-width: 639px)': 'none',
      },
    },
  },
  logo_badge: {
    '@layer site': {
      padding: '3px 9px',
      border: '1px solid color-mix(in oklab, var(--primary) 32%, transparent)',
      borderRadius: '999px',
      fontFamily: 'var(--font-mono), monospace',
      fontSize: '10.5px',
      fontWeight: '600',
      letterSpacing: '0.2em',
      textTransform: 'uppercase',
      color: 'var(--primary)',
      background: 'color-mix(in oklab, var(--primary) 12%, transparent)',
    },
  },
  header_right: {
    '@layer site': {
      display: 'flex',
      alignItems: 'center',
      gap: '10px',
    },
  },
  header_icons: {
    '@layer site': {
      display: 'flex',
      alignItems: 'center',
      gap: '2px',
    },
  },
  header_sep: {
    '@layer site': {
      display: {
        default: 'none',
        '@media (min-width: 640px)': 'inline-block',
      },
      width: '1px',
      height: '22px',
      background:
        'linear-gradient(180deg, transparent, var(--border-2), transparent)',
    },
  },
  nav_pills: {
    '@layer site': {
      position: 'relative',
      display: {
        default: 'none',
        '@media (min-width: 640px)': 'flex',
      },
      alignItems: 'center',
      padding: '4px',
      border: '1px solid var(--border)',
      borderRadius: '999px',
      boxShadow:
        'inset 0 1px 0 color-mix(in oklab, var(--surface) 80%, transparent), 0 6px 20px -14px color-mix(in oklab, var(--ink) 60%, transparent)',
      background: 'color-mix(in oklab, var(--surface) 55%, transparent)',
      gap: '2px',
      WebkitBackdropFilter: 'blur(10px) saturate(140%)',
      backdropFilter: 'blur(10px) saturate(140%)',
    },
  },
  nav_link: {
    '@layer site': {
      position: 'relative',
      display: 'inline-flex',
      alignItems: 'center',
      padding: '7px 14px',
      borderRadius: '999px',
      fontSize: '13.5px',
      fontWeight: '500',
      letterSpacing: '-0.005em',
      color: {
        default: 'var(--ink-3)',
        ':hover': 'var(--ink)',
      },
      background: {
        default: null,
        ':hover': 'color-mix(in oklab, var(--surface-2) 80%, transparent)',
      },
      outline: {
        default: null,
        ':focus-visible': 'none',
      },
      boxShadow: {
        default: null,
        ':focus-visible': '0 0 0 2px var(--bg), 0 0 0 4px var(--primary)',
      },
      transition:
        'color 180ms ease, background-color 220ms ease, box-shadow 220ms ease',
      gap: '6px',
    },
  },
  nav_link_active: {
    '@layer site': {
      boxShadow:
        'inset 0 0 0 1px color-mix(in oklab, var(--primary) 35%, transparent), 0 6px 18px -10px color-mix(in oklab, var(--primary) 60%, transparent)',
      background:
        'linear-gradient(180deg, color-mix(in oklab, var(--primary) 18%, var(--surface)), color-mix(in oklab, var(--primary) 10%, var(--surface)))',
      color: 'var(--ink)',
    },
  },
  nav_link_external: {
    '@layer site': {
      '--nav-link-ext-transform': {
        default: null,
        ':hover': 'translate(2px, -2px)',
      },
      '--nav-link-ext-opacity': {
        default: null,
        ':hover': '1',
      },
    },
  },
  nav_link_label_active: {
    '@layer site': {
      '::before': {
        content: '""',
        display: 'inline-block',
        width: '4px',
        height: '4px',
        marginRight: '8px',
        borderRadius: '50%',
        boxShadow: '0 0 10px var(--primary-3)',
        background: 'var(--primary-3)',
        verticalAlign: 'middle',
        transform: 'translateY(-1px)',
      },
    },
  },
  nav_link_ext: {
    '@layer site': {
      transition:
        'transform 220ms cubic-bezier(0.2, 0.9, 0.2, 1), opacity 180ms ease',
      transform: 'var(--nav-link-ext-transform, none)',
      opacity: 'var(--nav-link-ext-opacity, 0.55)',
    },
  },
  icon_btn: {
    '@layer site': {
      display: 'grid',
      width: '36px',
      height: '36px',
      borderRadius: '10px',
      color: {
        default: 'var(--ink-3)',
        ':hover': 'var(--ink)',
        ':focus-visible': 'var(--ink)',
      },
      background: {
        default: null,
        ':hover': 'color-mix(in oklab, var(--surface-2) 80%, transparent)',
      },
      transform: {
        default: null,
        ':hover': 'translateY(-1px)',
        ':active': 'translateY(0)',
      },
      outline: {
        default: null,
        ':focus-visible': 'none',
      },
      boxShadow: {
        default: null,
        ':focus-visible': '0 0 0 2px var(--bg), 0 0 0 4px var(--primary)',
      },
      transition:
        'color 180ms ease, background-color 180ms ease, transform 180ms cubic-bezier(0.2, 0.9, 0.2, 1)',
      placeItems: 'center',
    },
  },
  icon_svg_18: {
    '@layer site': {
      height: '18px',
      width: '18px',
    },
  },
  main: {
    '@layer site': {
      minHeight: 'calc(100vh - 260px)',
      paddingTop: '2rem',
    },
  },
})

export const site_header = stylex.props(styles.site_header).className!
export const site_header_inner = stylex.props(
  styles.site_header_inner,
).className!
export const logo_link = stylex.props(styles.logo_link).className!
export const logo_ring = stylex.props(styles.logo_ring).className!
export const logo_ring_inner = stylex.props(styles.logo_ring_inner).className!
export const avatar = stylex.props(styles.avatar).className!
export const logo_name = stylex.props(styles.logo_name).className!
export const logo_text = stylex.props(styles.logo_text).className!
export const logo_badge = stylex.props(styles.logo_badge).className!
export const header_right = stylex.props(styles.header_right).className!
export const header_icons = stylex.props(styles.header_icons).className!
export const header_sep = stylex.props(styles.header_sep).className!
export const nav_pills = stylex.props(styles.nav_pills).className!
export const nav_link = stylex.props(styles.nav_link).className!
export const nav_link_active = stylex.props(
  styles.nav_link,
  styles.nav_link_active,
).className!
export const nav_link_external = stylex.props(
  styles.nav_link,
  styles.nav_link_external,
).className!
export const nav_link_label_active = stylex.props(
  styles.nav_link_label_active,
).className!
export const nav_link_ext = stylex.props(styles.nav_link_ext).className!
export const icon_btn = stylex.props(styles.icon_btn).className!
export const icon_svg_18 = stylex.props(styles.icon_svg_18).className!
export const main = stylex.props(styles.main).className!
