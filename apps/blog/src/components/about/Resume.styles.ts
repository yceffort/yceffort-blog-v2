import * as stylex from '@stylexjs/stylex'
const styles = stylex.create({
  resume_pageHeader: {
    '@layer site': {
      display: 'flex',
      flexWrap: 'wrap',
      alignItems: 'flex-end',
      justifyContent: 'space-between',
      gap: '24px',
      padding: '48px 0 12px',
    },
  },
  element_h1: {
    '@layer site': {
      margin: '12px 0',
      fontSize: 'clamp(48px, 8vw, 80px)',
      fontWeight: '800',
      lineHeight: '1.1',
      letterSpacing: '-0.06em',
      color: 'var(--ink)',
    },
  },
  resume_pageHeader_span: {
    '@layer site': {
      color: 'var(--primary)',
    },
  },
  resume_nav_span: {
    '@layer site': {
      fontFamily: 'var(--font-mono), monospace',
      fontSize: '10px',
      color: 'var(--primary)',
    },
  },
  resume_sectionHeading_span: {
    '@layer site': {
      fontFamily: 'var(--font-mono), monospace',
      fontSize: '11px',
      color: 'var(--ink-3)',
    },
  },
  resume_writing_item: {
    '@layer site': {
      display: 'flex',
      flexDirection: 'column',
      gap: '4px',
      margin: 0,
    },
  },
  resume_writing_span: {
    '@layer site': {
      fontSize: '12px',
      lineHeight: '1.8',
      color: 'var(--ink-3)',
    },
  },
  resume_status_span: {
    '@layer site': {
      width: '6px',
      height: '6px',
      borderRadius: '50%',
      backgroundColor: 'var(--primary)',
      backgroundImage: 'none',
      backgroundPosition: 'initial',
      backgroundSize: 'auto',
      backgroundRepeat: 'repeat',
      backgroundOrigin: 'padding-box',
      backgroundClip: 'border-box',
      backgroundAttachment: 'scroll',
    },
  },
  resume_headerNote: {
    '@layer site': {
      fontSize: '14px',
      lineHeight: '1.8',
      color: 'var(--ink-3)',
    },
  },
  resume_contacts: {
    '@layer site': {
      display: 'flex',
      flexWrap: 'wrap',
      gap: '12px 20px',
      paddingBottom: '4px',
      fontFamily: 'var(--font-mono), monospace',
      fontSize: '12px',
      color: 'var(--ink-2)',
    },
  },
  resume_link_hover: {
    '@layer site': {
      color: {
        default: null,
        ':hover': 'var(--primary)',
      },
      outline: {
        default: null,
        ':focus-visible': '2px solid var(--primary)',
      },
      outlineOffset: {
        default: null,
        ':focus-visible': '5px',
      },
    },
  },
  resume_nav_link: {
    '@layer site': {
      display: 'flex',
      alignItems: 'baseline',
      gap: '12px',
      padding: '10px 0',
      fontSize: '13px',
      color: {
        default: 'var(--ink-3)',
        ':hover': 'var(--primary)',
      },
      outline: {
        default: null,
        ':focus-visible': '2px solid var(--primary)',
      },
      outlineOffset: {
        default: null,
        ':focus-visible': '5px',
      },
    },
  },
  resume_writing_link: {
    '@layer site': {
      fontSize: '15px',
      fontWeight: '600',
      color: {
        default: 'var(--ink)',
        ':hover': 'var(--primary)',
      },
      outline: {
        default: null,
        ':focus-visible': '2px solid var(--primary)',
      },
      outlineOffset: {
        default: null,
        ':focus-visible': '5px',
      },
    },
  },
  resume_resume: {
    '@layer site': {
      padding: '24px 0 72px',
      wordBreak: 'keep-all',
      color: 'var(--ink-2)',
      overflowWrap: 'anywhere',
      paddingTop: {
        default: null,
        '@media (max-width: 600px)': '16px',
      },
    },
  },
  resume_summary: {
    '@layer site': {
      marginBottom: {
        default: '64px',
        '@media (max-width: 900px)': '24px',
      },
    },
  },
  resume_eyebrow: {
    '@layer site': {
      fontFamily: 'var(--font-mono), monospace',
      fontSize: '11px',
      fontWeight: '500',
      letterSpacing: '0.08em',
      color: 'var(--ink-3)',
    },
  },
  resume_summary_h2: {
    '@layer site': {
      maxWidth: '760px',
      margin: '16px 0 20px',
      fontSize: 'clamp(28px, 4vw, 42px)',
      fontWeight: '750',
      lineHeight: '1.3',
      letterSpacing: '-0.045em',
      color: 'var(--ink)',
    },
  },
  resume_sectionHeading_h2: {
    '@layer site': {
      fontSize: '22px',
      fontWeight: '700',
      letterSpacing: '-0.035em',
      color: 'var(--ink)',
    },
  },
  resume_lead: {
    '@layer site': {
      maxWidth: '720px',
      fontSize: '16px',
      lineHeight: '1.9',
    },
  },
  resume_status: {
    '@layer site': {
      display: 'inline-flex',
      alignItems: 'center',
      gap: '8px',
      marginTop: '20px',
      fontSize: '13px',
      color: 'var(--ink-3)',
    },
  },
  resume_expertise: {
    '@layer site': {
      display: 'grid',
      gridTemplateColumns: {
        default: 'repeat(3, minmax(0, 1fr))',
        '@media (max-width: 600px)': 'minmax(0, 1fr)',
      },
      gap: {
        default: '24px',
        '@media (max-width: 600px)': '18px',
      },
      marginTop: '32px',
      padding: '24px 0',
      borderTopWidth: '1px',
      borderTopStyle: 'solid',
      borderTopColor: 'var(--border)',
      borderBottomWidth: '1px',
      borderBottomStyle: 'solid',
      borderBottomColor: 'var(--border)',
    },
  },
  element_dt: {
    '@layer site': {
      marginBottom: {
        default: '8px',
        '@media (max-width: 600px)': '4px',
      },
      fontSize: '12px',
      color: 'var(--ink-3)',
    },
  },
  element_dd: {
    '@layer site': {
      fontSize: '13px',
      fontWeight: '500',
      lineHeight: '1.7',
    },
  },
  resume_layout: {
    '@layer site': {
      display: 'grid',
      gridTemplateColumns: {
        default: '180px minmax(0, 1fr)',
        '@media (max-width: 900px)': 'minmax(0, 1fr)',
      },
      alignItems: 'start',
      gap: {
        default: '48px',
        '@media (max-width: 900px)': '32px',
      },
    },
  },
  resume_nav: {
    '@layer site': {
      position: {
        default: 'sticky',
        '@media (max-width: 900px)': 'static',
      },
      top: '100px',
      display: 'flex',
      flexDirection: {
        default: 'column',
        '@media (max-width: 900px)': 'row',
      },
      gap: {
        default: '4px',
        '@media (max-width: 900px)': '4px 24px',
      },
      flexWrap: {
        default: null,
        '@media (max-width: 900px)': 'wrap',
      },
    },
  },
  resume_content: {
    '@layer site': {
      minWidth: '0',
    },
  },
  resume_section: {
    '@layer site': {
      marginBottom: {
        default: '64px',
        ':last-child': '0',
        '@media (max-width: 600px)': {
          default: '48px',
          ':last-child': '0',
        },
      },
      scrollMarginTop: '100px',
    },
  },
  resume_sectionHeading: {
    '@layer site': {
      display: 'flex',
      flexWrap: 'wrap',
      alignItems: 'baseline',
      justifyContent: 'space-between',
      gap: '8px 16px',
      paddingBottom: '18px',
      borderBottomWidth: '1px',
      borderBottomStyle: 'solid',
      borderBottomColor: 'var(--border-2)',
    },
  },
  resume_sectionNote: {
    '@layer site': {
      marginTop: '16px',
      fontSize: '12px',
      color: 'var(--ink-3)',
    },
  },
  resume_timeline: {
    '@layer site': {
      paddingLeft: '20px',
      borderLeftWidth: '1px',
      borderLeftStyle: 'solid',
      borderLeftColor: 'var(--border)',
    },
  },
  resume_job: {
    position: {
      default: null,
      '@layer site': 'relative',
      '::before': {
        default: null,
        '@layer site': 'absolute',
      },
    },
    '@layer site': {
      padding: '30px 0',
      paddingBottom: {
        default: null,
        ':last-child': '0',
      },
    },
    borderBottomWidth: {
      default: null,
      '@layer site': {
        default: '1px',
        ':last-child': '0',
      },
      '::before': {
        default: null,
        '@layer site': '2px',
      },
    },
    borderBottomStyle: {
      default: null,
      '@layer site': {
        default: 'solid',
        ':last-child': 'none',
      },
      '::before': {
        default: null,
        '@layer site': 'solid',
      },
    },
    borderBottomColor: {
      default: null,
      '@layer site': {
        default: 'var(--border)',
        ':last-child': 'currentColor',
      },
      '::before': {
        default: null,
        '@layer site': 'var(--border-2)',
      },
    },
    content: {
      default: null,
      '::before': {
        default: null,
        '@layer site': '""',
      },
    },
    top: {
      default: null,
      '::before': {
        default: null,
        '@layer site': '38px',
      },
    },
    left: {
      default: null,
      '::before': {
        default: null,
        '@layer site': '-25px',
      },
    },
    width: {
      default: null,
      '::before': {
        default: null,
        '@layer site': '9px',
      },
    },
    height: {
      default: null,
      '::before': {
        default: null,
        '@layer site': '9px',
      },
    },
    borderTopWidth: {
      default: null,
      '::before': {
        default: null,
        '@layer site': '2px',
      },
    },
    borderTopStyle: {
      default: null,
      '::before': {
        default: null,
        '@layer site': 'solid',
      },
    },
    borderTopColor: {
      default: null,
      '::before': {
        default: null,
        '@layer site': 'var(--border-2)',
      },
    },
    borderRightWidth: {
      default: null,
      '::before': {
        default: null,
        '@layer site': '2px',
      },
    },
    borderRightStyle: {
      default: null,
      '::before': {
        default: null,
        '@layer site': 'solid',
      },
    },
    borderRightColor: {
      default: null,
      '::before': {
        default: null,
        '@layer site': 'var(--border-2)',
      },
    },
    borderLeftWidth: {
      default: null,
      '::before': {
        default: null,
        '@layer site': '2px',
      },
    },
    borderLeftStyle: {
      default: null,
      '::before': {
        default: null,
        '@layer site': 'solid',
      },
    },
    borderLeftColor: {
      default: null,
      '::before': {
        default: null,
        '@layer site': 'var(--border-2)',
      },
    },
    borderRadius: {
      default: null,
      '::before': {
        default: null,
        '@layer site': '50%',
      },
    },
    backgroundColor: {
      default: null,
      '::before': {
        default: null,
        '@layer site': 'var(--bg)',
      },
    },
    backgroundImage: {
      default: null,
      '::before': {
        default: null,
        '@layer site': 'none',
      },
    },
    backgroundPosition: {
      default: null,
      '::before': {
        default: null,
        '@layer site': 'initial',
      },
    },
    backgroundSize: {
      default: null,
      '::before': {
        default: null,
        '@layer site': 'auto',
      },
    },
    backgroundRepeat: {
      default: null,
      '::before': {
        default: null,
        '@layer site': 'repeat',
      },
    },
    backgroundOrigin: {
      default: null,
      '::before': {
        default: null,
        '@layer site': 'padding-box',
      },
    },
    backgroundClip: {
      default: null,
      '::before': {
        default: null,
        '@layer site': 'border-box',
      },
    },
    backgroundAttachment: {
      default: null,
      '::before': {
        default: null,
        '@layer site': 'scroll',
      },
    },
  },
  resume_jobHeading: {
    '@layer site': {
      display: 'flex',
      flexWrap: 'wrap',
      alignItems: 'baseline',
      justifyContent: 'space-between',
      gap: '6px 16px',
    },
  },
  element_h3: {
    '@layer site': {
      fontSize: '17px',
      fontWeight: '650',
      lineHeight: '1.5',
      letterSpacing: '-0.025em',
      color: 'var(--ink)',
    },
  },
  resume_period: {
    '@layer site': {
      fontFamily: 'var(--font-mono), monospace',
      fontSize: '11px',
      lineHeight: '1.8',
      color: 'var(--ink-3)',
    },
  },
  resume_role: {
    '@layer site': {
      marginTop: '5px',
      fontSize: '12px',
      color: 'var(--ink-3)',
    },
  },
  resume_jobDescription: {
    '@layer site': {
      marginTop: '18px',
      fontSize: '14px',
      fontWeight: '550',
      lineHeight: '1.8',
    },
  },
  resume_contributions: {
    '@layer site': {
      display: 'grid',
      gap: '6px',
      margin: '12px 0 16px',
      paddingLeft: '17px',
      fontSize: '14px',
      lineHeight: '1.85',
      listStyle: 'disc',
    },
  },
  resume_contributions_li: {
    color: {
      default: null,
      '::marker': {
        default: null,
        '@layer site': 'var(--ink-3)',
      },
    },
  },
  resume_entries_li: {
    '@layer site': {
      display: 'grid',
      gridTemplateColumns: {
        default: '140px minmax(0, 1fr)',
        '@media (max-width: 600px)': 'minmax(0, 1fr)',
      },
      gap: {
        default: '20px',
        '@media (max-width: 600px)': '8px',
      },
      padding: '24px 0',
      borderBottomWidth: {
        default: '1px',
        ':last-child': '0',
      },
      borderBottomStyle: {
        default: 'solid',
        ':last-child': 'none',
      },
      borderBottomColor: {
        default: 'var(--border)',
        ':last-child': 'currentColor',
      },
      paddingBottom: {
        default: null,
        ':last-child': '0',
      },
    },
  },
  resume_stack: {
    '@layer site': {
      fontSize: '11px',
      lineHeight: '1.9',
      color: 'var(--ink-3)',
    },
  },
  resume_textLink: {
    '@layer site': {
      display: 'inline-block',
      marginTop: '14px',
      fontSize: '12px',
      textDecoration: 'underline',
      textUnderlineOffset: '4px',
    },
  },
  resume_books: {
    '@layer site': {
      display: 'grid',
      gridTemplateColumns: {
        default: 'repeat(2, minmax(0, 1fr))',
        '@media (max-width: 600px)': 'minmax(0, 1fr)',
      },
      gap: '16px',
      marginTop: '24px',
    },
  },
  resume_book: {
    '@layer site': {
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      padding: '24px',
      borderTopWidth: '1px',
      borderStyle: 'solid',
      borderTopColor: {
        default: 'var(--border)',
        ':hover': 'var(--primary)',
      },
      borderRightWidth: '1px',
      borderRightColor: {
        default: 'var(--border)',
        ':hover': 'var(--primary)',
      },
      borderBottomWidth: '1px',
      borderBottomColor: {
        default: 'var(--border)',
        ':hover': 'var(--primary)',
      },
      borderLeftWidth: '3px',
      borderLeftColor: 'var(--primary)',
      borderRadius: '2px 8px 8px 2px',
      backgroundColor: 'var(--surface)',
      transition: {
        default: 'border-color 160ms ease',
        '@media (prefers-reduced-motion: reduce)': 'none',
      },
    },
  },
  resume_bookRole: {
    '@layer site': {
      marginBottom: '16px',
      fontSize: '11px',
      color: 'var(--ink-3)',
    },
  },
  resume_book_p: {
    '@layer site': {
      margin: '10px 0 24px',
      fontSize: '12px',
      lineHeight: '1.8',
      color: 'var(--ink-3)',
    },
  },
  resume_entries_p: {
    '@layer site': {
      fontSize: '14px',
      lineHeight: '1.85',
      marginTop: '8px',
    },
  },
  resume_bookLink: {
    '@layer site': {
      marginTop: 'auto',
      fontSize: '11px',
      color: 'var(--ink-2)',
    },
  },
  resume_writing: {
    '@layer site': {
      display: 'flex',
      flexDirection: 'column',
      gap: '8px',
      marginTop: '24px',
      paddingLeft: '16px',
      borderLeftWidth: '2px',
      borderLeftStyle: 'solid',
      borderLeftColor: 'var(--border-2)',
    },
  },
  resume_honor: {
    '@layer site': {
      fontFamily: 'var(--font-mono), monospace',
      fontSize: '11px',
      lineHeight: '1.85',
      color: 'var(--ink-3)',
      marginTop: '8px',
    },
  },
})

// Resolve locally so the compiler can erase style composition.
export const resume_pageHeader = stylex.props(
  styles.resume_pageHeader,
).className!
export const element_h1 = stylex.props(styles.element_h1).className!
export const resume_pageHeader_span = stylex.props(
  styles.resume_pageHeader_span,
).className!
export const resume_nav_span = stylex.props(styles.resume_nav_span).className!
export const resume_sectionHeading_span = stylex.props(
  styles.resume_sectionHeading_span,
).className!
export const resume_writing_item = stylex.props(
  styles.resume_writing_item,
).className!
export const resume_writing_span = stylex.props(
  styles.resume_writing_span,
).className!
export const resume_status_span = stylex.props(
  styles.resume_status_span,
).className!
export const resume_headerNote = stylex.props(
  styles.resume_headerNote,
).className!
export const resume_contacts = stylex.props(styles.resume_contacts).className!
export const resume_link_hover = stylex.props(
  styles.resume_link_hover,
).className!
export const resume_nav_link = stylex.props(styles.resume_nav_link).className!
export const resume_writing_link = stylex.props(
  styles.resume_writing_link,
).className!
export const resume_resume = stylex.props(styles.resume_resume).className!
export const resume_summary = stylex.props(styles.resume_summary).className!
export const resume_eyebrow = stylex.props(styles.resume_eyebrow).className!
export const resume_summary_h2 = stylex.props(
  styles.resume_summary_h2,
).className!
export const resume_sectionHeading_h2 = stylex.props(
  styles.resume_sectionHeading_h2,
).className!
export const resume_lead = stylex.props(styles.resume_lead).className!
export const resume_status = stylex.props(styles.resume_status).className!
export const resume_expertise = stylex.props(styles.resume_expertise).className!
export const element_dt = stylex.props(styles.element_dt).className!
export const element_dd = stylex.props(styles.element_dd).className!
export const resume_layout = stylex.props(styles.resume_layout).className!
export const resume_nav = stylex.props(styles.resume_nav).className!
export const resume_content = stylex.props(styles.resume_content).className!
export const resume_section = stylex.props(styles.resume_section).className!
export const resume_sectionHeading = stylex.props(
  styles.resume_sectionHeading,
).className!
export const resume_sectionNote = stylex.props(
  styles.resume_sectionNote,
).className!
export const resume_timeline = stylex.props(styles.resume_timeline).className!
export const resume_job = stylex.props(styles.resume_job).className!
export const resume_jobHeading = stylex.props(
  styles.resume_jobHeading,
).className!
export const element_h3 = stylex.props(styles.element_h3).className!
export const resume_period = stylex.props(styles.resume_period).className!
export const resume_role = stylex.props(styles.resume_role).className!
export const resume_jobDescription = stylex.props(
  styles.resume_jobDescription,
).className!
export const resume_contributions = stylex.props(
  styles.resume_contributions,
).className!
export const resume_contributions_li = stylex.props(
  styles.resume_contributions_li,
).className!
export const resume_entries_li = stylex.props(
  styles.resume_entries_li,
).className!
export const resume_stack = stylex.props(styles.resume_stack).className!
export const resume_textLink = stylex.props(styles.resume_textLink).className!
export const resume_books = stylex.props(styles.resume_books).className!
export const resume_book = stylex.props(styles.resume_book).className!
export const resume_bookRole = stylex.props(styles.resume_bookRole).className!
export const resume_book_p = stylex.props(styles.resume_book_p).className!
export const resume_entries_p = stylex.props(styles.resume_entries_p).className!
export const resume_bookLink = stylex.props(styles.resume_bookLink).className!
export const resume_writing = stylex.props(styles.resume_writing).className!
export const resume_honor = stylex.props(styles.resume_honor).className!
