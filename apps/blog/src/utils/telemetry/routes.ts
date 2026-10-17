/**
 * 메트릭 라벨과 Tempo 대시보드의 화면 변수에 쓰는 화면 이름. URL 을 그대로 라벨로 쓰면
 * 글 수만큼 시리즈가 생겨서 정해진 다섯 값으로 줄인다. 영문판(`/en`)은 같은 화면으로 센다.
 */
export type Route = 'home' | 'post' | 'tag' | 'series' | 'other'

export function routeOf(pathname: string): Route {
  const path = pathname.replace(/^\/en(?=\/|$)/, '') || '/'

  if (path === '/' || /^\/pages\/\d+\/?$/.test(path)) return 'home'
  if (/^\/\d{4}\/\d{2}\/[^/]+/.test(path)) return 'post'
  if (/^\/tags(\/|$)/.test(path)) return 'tag'
  if (/^\/series(\/|$)/.test(path)) return 'series'
  return 'other'
}
