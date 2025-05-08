#!/bin/bash
[ "$1" = -x ] && shift && set -x
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

set -e

cd "${DIR}"

docker compose build

docker compose up -d

docker compose logs -f claude-code-proxy
