import * as stylex from '@stylexjs/stylex'

const styles = stylex.create({
  presenterView: {
    '@layer site': {
      display: 'grid',
      gridTemplateRows: 'auto minmax(0, 1fr) auto auto auto',
      height: '100dvh',
      background: '#1a1a1a',
      color: '#fff',
      fontFamily: 'system-ui, -apple-system, sans-serif',
      overflow: 'hidden',
    },
  },
  timerBar: {
    '@layer site': {
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      gap: '16px',
      padding: '12px 24px',
      background: '#2a2a2a',
      borderBottom: '1px solid #333',
    },
  },
  timer: {
    '@layer site': {
      fontSize: '32px',
      fontWeight: '600',
      fontVariantNumeric: 'tabular-nums',
      minWidth: '100px',
      textAlign: 'center',
    },
  },
  timerButton: {
    '@layer site': {
      padding: '8px 16px',
      background: {default: '#444', ':hover': '#555'},
      border: 'none',
      borderRadius: '6px',
      color: '#fff',
      fontSize: '14px',
      cursor: 'pointer',
      transition: 'background 0.2s',
    },
  },
  timerButtonRunning: {
    '@layer site': {
      background: {default: '#c53030', ':hover': '#e53e3e'},
    },
  },
  slidesContainer: {
    '@layer site': {
      display: 'grid',
      gridTemplateColumns: 'minmax(0, 2fr) minmax(0, 1fr)',
      gap: '16px',
      padding: '16px',
      minHeight: 0,
    },
  },
  slideWrapper: {
    '@layer site': {
      background: '#2a2a2a',
      borderRadius: '8px',
      overflow: 'hidden',
      display: 'flex',
      flexDirection: 'column',
      minWidth: 0,
      minHeight: 0,
    },
  },
  slideLabel: {
    '@layer site': {
      flexShrink: 0,
      padding: '8px 12px',
      background: '#333',
      fontSize: '12px',
      fontWeight: '600',
      textTransform: 'uppercase',
      letterSpacing: '0.5px',
      color: '#888',
    },
  },
  // 다음 슬라이드 쪽 라벨만 흐리다
  slideLabelNext: {
    '@layer site': {
      color: '#666',
    },
  },
  slideContent: {
    '@layer site': {
      flex: 1,
      minHeight: 0,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '8px',
      overflow: 'hidden',
    },
  },
  noNextSlide: {
    '@layer site': {
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      color: '#666',
      fontSize: '14px',
    },
  },
  notesResizer: {
    '@layer site': {
      height: '12px',
      margin: 0,
      border: 'none',
      backgroundColor: {default: '#1a1a1a', ':hover': '#333'},
      // hr 은 자식을 가질 수 없어서 가운데 손잡이 막대를 배경으로 그린다.
      backgroundImage: 'linear-gradient(#666, #666)',
      backgroundSize: '40px 4px',
      backgroundPosition: 'center',
      backgroundRepeat: 'no-repeat',
      cursor: 'row-resize',
      touchAction: 'none',
    },
  },
  notesPanel: {
    '@layer site': {
      display: 'grid',
      gridTemplateRows: 'auto minmax(0, 1fr) auto',
      padding: '12px 24px',
      background: '#2a2a2a',
      borderTop: '1px solid #333',
      minHeight: 0,
      maxHeight: 'min(240px, 32dvh)',
      overflow: 'hidden',
    },
  },
  notesBody: {
    '@layer site': {
      display: 'grid',
      gridTemplateColumns: 'minmax(0, 1fr) 18px',
      gridTemplateRows: 'minmax(0, 1fr)',
      gap: '12px',
      minHeight: 0,
    },
  },
  notesViewport: {
    '@layer site': {
      overflowY: 'auto',
      overscrollBehavior: 'contain',
      scrollbarWidth: 'none',
      minHeight: 0,
      outlineOffset: '-2px',
    },
  },
  notesTrack: {
    '@layer site': {
      position: 'relative',
      background: '#171717',
      borderRadius: '9px',
      touchAction: 'none',
      userSelect: 'none',
      cursor: 'pointer',
      outlineOffset: '2px',
    },
  },
  // 넘치는 글이 없으면 자리는 그대로 두고 트랙만 감춘다.
  notesTrackIdle: {
    '@layer site': {
      visibility: 'hidden',
    },
  },
  notesThumb: {
    '@layer site': {
      position: 'absolute',
      top: 0,
      left: '3px',
      right: '3px',
      height: '100%',
      borderRadius: '6px',
      background: {default: '#b8b8b8', ':hover': '#eee'},
      cursor: 'grab',
    },
  },
  notesStatus: {
    '@layer site': {
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      minHeight: '28px',
      marginTop: '6px',
      color: '#aaa',
      fontSize: '12px',
    },
  },
  notesMore: {
    '@layer site': {
      padding: '4px 12px',
      border: '1px solid #777',
      borderRadius: '14px',
      background: {default: '#383838', ':hover': '#484848'},
      color: '#fff',
      fontSize: '12px',
      fontWeight: 600,
      cursor: 'pointer',
    },
  },
  notesHeader: {
    '@layer site': {
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      gap: '12px',
      marginBottom: '8px',
    },
  },
  notesLabel: {
    '@layer site': {
      fontSize: '12px',
      fontWeight: '600',
      textTransform: 'uppercase',
      letterSpacing: '0.5px',
      color: '#888',
    },
  },
  notesFontControls: {
    '@layer site': {
      display: 'flex',
      alignItems: 'center',
      gap: '6px',
    },
  },
  notesFontButton: {
    '@layer site': {
      padding: '2px 10px',
      border: '1px solid #777',
      borderRadius: '6px',
      background: {default: '#383838', ':hover:not(:disabled)': '#484848'},
      color: '#fff',
      fontSize: '13px',
      fontWeight: 600,
      cursor: {default: 'pointer', ':disabled': 'not-allowed'},
      opacity: {default: null, ':disabled': 0.4},
    },
  },
  notesFontSize: {
    '@layer site': {
      minWidth: '40px',
      textAlign: 'center',
      color: '#aaa',
      fontSize: '12px',
      fontVariantNumeric: 'tabular-nums',
    },
  },
  notesContent: {
    '@layer site': {
      fontSize: '16px',
      lineHeight: '1.6',
      whiteSpace: 'pre-wrap',
      color: '#ccc',
    },
  },
  noNotes: {
    '@layer site': {
      color: '#666',
      fontStyle: 'italic',
    },
  },
  controlBar: {
    '@layer site': {
      display: 'grid',
      gridTemplateColumns: {
        default: 'minmax(0, 1fr) auto minmax(0, 1fr)',
        '@media (max-width: 640px)': '1fr',
      },
      alignItems: 'center',
      gap: '8px 16px',
      padding: '16px 24px',
      background: '#2a2a2a',
      borderTop: '1px solid #333',
    },
  },
  navigation: {
    '@layer site': {
      gridColumn: {default: '2', '@media (max-width: 640px)': '1'},
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      gap: '16px',
    },
  },
  navButton: {
    '@layer site': {
      padding: '12px 24px',
      background: {default: '#444', ':hover:not(:disabled)': '#555'},
      border: 'none',
      borderRadius: '6px',
      color: '#fff',
      fontSize: '16px',
      cursor: {default: 'pointer', ':disabled': 'not-allowed'},
      opacity: {default: null, ':disabled': 0.4},
      transition: 'background 0.2s',
    },
  },
  pageIndicator: {
    '@layer site': {
      fontSize: '18px',
      fontWeight: '600',
      minWidth: '80px',
      textAlign: 'center',
      fontVariantNumeric: 'tabular-nums',
    },
  },
  keyHints: {
    '@layer site': {
      gridColumn: {default: '3', '@media (max-width: 640px)': '1'},
      justifySelf: {default: 'end', '@media (max-width: 640px)': 'center'},
      display: 'flex',
      flexDirection: {default: 'column', '@media (max-width: 640px)': 'row'},
      flexWrap: 'wrap',
      justifyContent: 'center',
      columnGap: '16px',
      fontSize: '12px',
      color: '#999',
      lineHeight: '1.5',
    },
  },
  marpContainer: {
    '@layer site': {
      width: '100%',
      height: '100%',
    },
  },
})

export const presenterView = stylex.props(styles.presenterView).className!
export const timerBar = stylex.props(styles.timerBar).className!
export const timer = stylex.props(styles.timer).className!
export const timerButton = stylex.props(styles.timerButton).className!
export const timerButtonRunning = stylex.props(
  styles.timerButton,
  styles.timerButtonRunning,
).className!
export const slidesContainer = stylex.props(styles.slidesContainer).className!
export const slideWrapper = stylex.props(styles.slideWrapper).className!
export const slideLabel = stylex.props(styles.slideLabel).className!
export const slideLabelNext = stylex.props(
  styles.slideLabel,
  styles.slideLabelNext,
).className!
export const slideContent = stylex.props(styles.slideContent).className!
export const noNextSlide = stylex.props(styles.noNextSlide).className!
export const notesResizer = stylex.props(styles.notesResizer).className!
export const notesPanel = stylex.props(styles.notesPanel).className!
export const notesBody = stylex.props(styles.notesBody).className!
export const notesViewport = stylex.props(styles.notesViewport).className!
export const notesTrack = stylex.props(styles.notesTrack).className!
export const notesTrackIdle = stylex.props(
  styles.notesTrack,
  styles.notesTrackIdle,
).className!
export const notesThumb = stylex.props(styles.notesThumb).className!
export const notesStatus = stylex.props(styles.notesStatus).className!
export const notesMore = stylex.props(styles.notesMore).className!
export const notesHeader = stylex.props(styles.notesHeader).className!
export const notesLabel = stylex.props(styles.notesLabel).className!
export const notesFontControls = stylex.props(
  styles.notesFontControls,
).className!
export const notesFontButton = stylex.props(styles.notesFontButton).className!
export const notesFontSize = stylex.props(styles.notesFontSize).className!
export const notesContent = stylex.props(styles.notesContent).className!
export const noNotes = stylex.props(styles.noNotes).className!
export const controlBar = stylex.props(styles.controlBar).className!
export const navigation = stylex.props(styles.navigation).className!
export const navButton = stylex.props(styles.navButton).className!
export const pageIndicator = stylex.props(styles.pageIndicator).className!
export const keyHints = stylex.props(styles.keyHints).className!
export const marpContainer = stylex.props(styles.marpContainer).className!
