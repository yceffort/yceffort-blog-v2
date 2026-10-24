import Link from 'next/link'

export default function NotFound() {
  return (
    <p>
      그런 호는 없습니다. <Link href="/">최신 호로 돌아가기</Link>
    </p>
  )
}
