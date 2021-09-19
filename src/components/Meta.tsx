export default function Meta(props: {
  name?: string
  content: string
  httpEquiv?: string
  property?: string
  media?: string
}) {
  return <meta {...props} />
}
