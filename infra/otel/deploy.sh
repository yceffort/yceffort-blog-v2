#!/usr/bin/env bash
# GitHub Actions(.github/workflows/deploy-otel.yml)가 SSH 로 부른다. 서버의 authorized_keys 에 묶인 forced command 가
# git pull 을 마친 뒤 pull 전 커밋 해시를 첫 인자로 넘겨 이 스크립트를 실행한다. 손으로 반영할 때는 인자 없이 실행한다.
#
# compose.yaml 이 바뀐 서비스는 `up -d` 가 다시 만든다. bind mount 한 설정 파일(collector/config.yaml 등)은 바뀌어도
# 컨테이너가 다시 만들어지지 않으므로, 바뀐 디렉터리의 서비스만 골라 재시작한다.
set -euo pipefail
cd "$(dirname "$0")"

before="${1:-}"
after="$(git rev-parse HEAD)"
echo "deploy ${before:0:8} -> ${after:0:8}"
if [[ -n "${SSH_ORIGINAL_COMMAND:-}" && "${SSH_ORIGINAL_COMMAND}" != "$after" ]]; then
  echo "warning: workflow asked for ${SSH_ORIGINAL_COMMAND:0:8} but HEAD is ${after:0:8}"
fi

changed=""
if [[ -n "$before" && "$before" != "$after" ]]; then
  changed="$(git diff --name-only "$before" "$after" -- infra/otel)"
  echo "$changed"
fi

docker compose -p blog-otel up -d --remove-orphans

for service in caddy collector tempo mimir grafana; do
  if grep -q "^infra/otel/$service/" <<<"$changed"; then
    echo "restart $service (config changed)"
    docker compose -p blog-otel restart "$service"
  fi
done

docker compose -p blog-otel ps --format '{{.Name}}\t{{.Status}}'
