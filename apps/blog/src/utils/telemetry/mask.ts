const MAX_LENGTH = 120

/**
 * web-vitals 의 selector 는 태그, id, class 로만 만들어져 화면 글자는 담기지 않는다.
 * 그래도 id 나 class 에 회원 번호나 이메일을 박아 두는 화면이 있어 한 번 더 지운다.
 * 짧은 숫자(`item-3`)는 위치 정보라 남긴다.
 */
export function maskSelector(selector: string | undefined): string | undefined {
  if (!selector) return undefined

  const masked = selector
    // selector 에서 `.` 은 class 구분자라 local part 에 넣지 않는다. 넣으면 앞의 태그 이름까지 지운다.
    .replace(/[\w+-]+@[\w-]+(\.[\w-]+)+/g, '{email}')
    .replace(/\d{5,}/g, '{n}')
    .trim()

  return masked ? masked.slice(0, MAX_LENGTH) : undefined
}

/** 쿼리와 해시에는 서명 토큰이나 사용자 값이 실릴 수 있어 origin 과 path 만 남긴다. */
export function stripQuery(url: string | undefined): string | undefined {
  if (!url) return undefined

  try {
    const {origin, pathname} = new URL(url)
    return `${origin}${pathname}`.slice(0, MAX_LENGTH)
  } catch {
    return undefined
  }
}
