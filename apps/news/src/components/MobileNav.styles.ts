import * as stylex from '@stylexjs/stylex'

const TRANSITION_COLORS =
  'color, background-color, border-color, outline-color, text-decoration-color, fill, stroke'

const sx = stylex.create({
  root: {
    '@layer utilities': {
      display: {
        default: null,
        '@media (width >= 40rem)': 'none',
      },
    },
  },
  toggle: {
    '@layer utilities': {
      display: 'flex',
      height: '2.5rem',
      width: '2.5rem',
      alignItems: 'center',
      justifyContent: 'center',
      borderRadius: '0.5rem',
      transitionProperty: TRANSITION_COLORS,
      transitionTimingFunction: 'cubic-bezier(0.4, 0, 0.2, 1)',
      transitionDuration: '150ms',
    },
  },
  toggleIcon: {
    '@layer utilities': {
      height: '1.5rem',
      width: '1.5rem',
    },
  },
  visible: {
    '@layer utilities': {
      opacity: '100%',
    },
  },
  hidden: {
    '@layer utilities': {
      pointerEvents: 'none',
      opacity: '0%',
    },
  },
  backdrop: {
    '@layer utilities': {
      position: 'fixed',
      inset: '0px',
      zIndex: '100',
      transitionProperty: 'opacity',
      transitionTimingFunction: 'cubic-bezier(0.4, 0, 0.2, 1)',
      transitionDuration: '300ms',
    },
  },
  sheetVisible: {
    '@layer utilities': {
      translate: '0 0',
      transitionTimingFunction: 'cubic-bezier(0.32, 0.72, 0, 1)',
      transitionDuration: '400ms',
    },
  },
  sheetHidden: {
    '@layer utilities': {
      pointerEvents: 'none',
      // 그림자가 화면 아래에 걸리지 않도록 그림자 높이만큼 더 내린다
      translate: '0 calc(100% + 80px)',
      transitionTimingFunction: 'cubic-bezier(0.4, 0, 1, 1)',
      transitionDuration: '220ms',
    },
  },
  sheet: {
    '@layer utilities': {
      position: 'fixed',
      insetInline: '0px',
      bottom: '0px',
      zIndex: '101',
      transitionProperty: 'translate',
    },
  },
  dialog: {
    '@layer utilities': {
      display: 'block',
      width: '100%',
      borderTopLeftRadius: '28px',
      borderTopRightRadius: '28px',
      paddingInline: '1.25rem',
      paddingTop: '1.25rem',
      paddingBottom: '1.75rem',
    },
  },
  handleContainer: {
    '@layer utilities': {
      marginBottom: '1.25rem',
      display: 'flex',
      justifyContent: 'center',
    },
  },
  handle: {
    '@layer utilities': {
      height: '5px',
      width: '2.75rem',
      borderRadius: '9999px',
    },
  },
  heading: {
    '@layer utilities': {
      marginBottom: '0.75rem',
      paddingInline: '0.75rem',
      fontSize: '10px',
      fontWeight: '500',
      textTransform: 'uppercase',
    },
  },
  nav: {
    '@layer utilities': {
      display: 'flex',
      flexDirection: 'column',
      gap: '0.25rem',
    },
  },
  link: {
    '@layer utilities': {
      position: 'relative',
      display: 'flex',
      alignItems: 'center',
      gap: '1rem',
      borderRadius: '1rem',
      paddingInline: '1rem',
      paddingBlock: '0.875rem',
      transitionProperty: TRANSITION_COLORS,
      transitionTimingFunction: 'cubic-bezier(0.4, 0, 0.2, 1)',
      transitionDuration: '150ms',
    },
  },
  index: {
    '@layer utilities': {
      fontSize: '11px',
      fontVariantNumeric: 'tabular-nums',
    },
  },
  label: {
    '@layer utilities': {
      flex: '1',
      fontSize: '17px',
      fontWeight: '600',
    },
  },
  activeDot: {
    '@layer utilities': {
      height: '0.375rem',
      width: '0.375rem',
      borderRadius: '9999px',
    },
  },
  close: {
    '@layer utilities': {
      marginTop: '1.25rem',
      width: '100%',
      borderRadius: '1rem',
      paddingBlock: '0.875rem',
      fontSize: '0.875rem',
      lineHeight: 'calc(1.25 / 0.875)',
      fontWeight: '500',
      transitionProperty: TRANSITION_COLORS,
      transitionTimingFunction: 'cubic-bezier(0.4, 0, 0.2, 1)',
      transitionDuration: '150ms',
    },
  },
})

export const mobileNavClassNames = {
  root: stylex.props(sx.root).className ?? '',
  toggle: stylex.props(sx.toggle).className ?? '',
  toggleIcon: stylex.props(sx.toggleIcon).className ?? '',
  visible: stylex.props(sx.visible).className ?? '',
  hidden: stylex.props(sx.hidden).className ?? '',
  backdrop: stylex.props(sx.backdrop).className ?? '',
  sheetVisible: stylex.props(sx.sheetVisible).className ?? '',
  sheetHidden: stylex.props(sx.sheetHidden).className ?? '',
  sheet: stylex.props(sx.sheet).className ?? '',
  dialog: stylex.props(sx.dialog).className ?? '',
  handleContainer: stylex.props(sx.handleContainer).className ?? '',
  handle: stylex.props(sx.handle).className ?? '',
  heading: stylex.props(sx.heading).className ?? '',
  nav: stylex.props(sx.nav).className ?? '',
  link: stylex.props(sx.link).className ?? '',
  index: stylex.props(sx.index).className ?? '',
  label: stylex.props(sx.label).className ?? '',
  activeDot: stylex.props(sx.activeDot).className ?? '',
  close: stylex.props(sx.close).className ?? '',
}
