import {buildRecentPostsResponse} from '@/utils/recentPosts'

export function GET() {
  return buildRecentPostsResponse('ko')
}
