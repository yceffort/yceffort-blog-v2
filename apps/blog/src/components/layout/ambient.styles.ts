import * as stylex from '@stylexjs/stylex'
const motion_page_fade = stylex.keyframes({
  from: {
    transform: 'translateY(10px)',
    opacity: '0',
  },
  to: {
    transform: 'none',
    opacity: '1',
  },
})
const styles = stylex.create({
  effect_grain: {
    '@layer site': {
      position: 'fixed',
      zIndex: '100',
      backgroundImage:
        "url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='160' height='160'><filter id='n'><feTurbulence baseFrequency='0.9' numOctaves='2' stitchTiles='stitch'/></filter><rect width='100%25' height='100%25' filter='url(%23n)'/></svg>\")",
      opacity: '0.055',
      inset: '0',
      pointerEvents: 'none',
      mixBlendMode: 'overlay',
      display: {
        default: null,
        '@media (prefers-reduced-motion: reduce)': 'none',
      },
    },
  },
  effect_anim_bg: {
    '@layer site': {
      position: 'fixed',
      zIndex: '0',
      backgroundColor: 'transparent',
      backgroundImage: {
        default:
          'radial-gradient(\n      800px circle at 10% 10%,\n      var(--primary),\n      transparent 60%\n    ),\n    radial-gradient(\n      600px circle at 90% 80%,\n      var(--primary-3),\n      transparent 60%\n    )',
        '@supports (color: color-mix(in lab, red, red))':
          'radial-gradient(\n      800px circle at 10% 10%,\n      color-mix(in oklab, var(--primary) 15%, transparent),\n      transparent 60%\n    ),\n    radial-gradient(\n      600px circle at 90% 80%,\n      color-mix(in oklab, var(--primary-3) 10%, transparent),\n      transparent 60%\n    )',
      },
      opacity: '0.6',
      inset: '0',
      pointerEvents: 'none',
    },
  },
  effect_cursor_glow: {
    '@layer site': {
      position: 'fixed',
      top: '0',
      left: '0',
      zIndex: '0',
      width: '500px',
      height: '500px',
      borderRadius: '50%',
      backgroundColor: 'transparent',
      backgroundImage:
        'radial-gradient(\n    circle,\n    var(--primary) 0%,\n    transparent 60%\n  )',
      transform: 'translate3d(50vw, 50vh, 0) translate(-50%, -50%)',
      transition: 'transform 400ms cubic-bezier(0.2, 0.9, 0.2, 1)',
      opacity: {
        default: '0.08',
        ':is(html.dark #cursor-glow)': '0.12',
      },
      pointerEvents: 'none',
      filter: 'blur(20px)',
      display: {
        default: null,
        '@media (prefers-reduced-motion: reduce)': 'none',
      },
    },
  },
  page_view: {
    '@layer site': {
      animationName: motion_page_fade,
      animationDuration: '450ms',
      animationTimingFunction: 'cubic-bezier(0.2, 0.9, 0.2, 1)',
      animationDelay: '0s',
      animationIterationCount: '1',
      animationDirection: 'normal',
      animationFillMode: 'both',
      animationPlayState: 'running',
    },
  },
})

// Resolve locally so the compiler can erase style composition.
export const effect_grain = stylex.props(styles.effect_grain).className!
export const effect_anim_bg = stylex.props(styles.effect_anim_bg).className!
export const effect_cursor_glow = stylex.props(
  styles.effect_cursor_glow,
).className!
export const page_view = stylex.props(styles.page_view).className!
