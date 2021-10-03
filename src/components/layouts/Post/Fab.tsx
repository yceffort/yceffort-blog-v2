import { Fab, Action } from 'react-tiny-fab'
import { BiPlus, BiArrowToTop } from 'react-icons/bi'
import { GoIssueOpened } from 'react-icons/go'

import 'react-tiny-fab/dist/styles.css'

export default function FloatingActionButton({ link }: { link: string }) {
  return (
    <Fab
      icon={<BiPlus />}
      event="click"
      mainButtonStyles={{ backgroundColor: '#00b7ff' }}
      style={{ bottom: 0, right: 0, border: 'none' }}
    >
      <Action
        text="tope"
        onClick={() => window.open(link)}
        style={{ backgroundColor: 'rgb(77, 139, 198)' }}
      >
        <GoIssueOpened />
      </Action>
      <Action
        text="issue"
        onClick={() => window.scrollTo(0, 0)}
        style={{ backgroundColor: 'rgb(77, 139, 198)' }}
      >
        <BiArrowToTop />
      </Action>
    </Fab>
  )
}
