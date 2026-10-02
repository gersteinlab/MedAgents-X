#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Adapted from the LF AI & Data foundation Milvus standalone startup script.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

name=${MILVUS_CONTAINER:-medagents-milvus}
data_dir=${MILVUS_DATA_DIR:-$PWD/volumes/milvus}

case "${1:-}" in
    start)
        if docker container inspect "$name" >/dev/null 2>&1; then
            docker start "$name" >/dev/null
        else
            mkdir -p -- "$data_dir"
            docker run -d --name "$name" --security-opt seccomp=unconfined \
                -e ETCD_USE_EMBED=true -e ETCD_DATA_DIR=/var/lib/milvus/etcd \
                -e ETCD_CONFIG_PATH=/milvus/configs/embedEtcd.yaml \
                -e COMMON_STORAGETYPE=local \
                -v "$data_dir:/var/lib/milvus" \
                -v "$PWD/embedEtcd.yaml:/milvus/configs/embedEtcd.yaml:ro" \
                -v "$PWD/user.yaml:/milvus/configs/user.yaml:ro" \
                -p 127.0.0.1:19530:19530 -p 127.0.0.1:9091:9091 \
                --health-cmd='curl -fsS http://localhost:9091/healthz' \
                --health-interval=5s --health-timeout=3s --health-retries=36 \
                milvusdb/milvus:v2.5.6 milvus run standalone >/dev/null
        fi
        for ((attempt=0; attempt<60; attempt++)); do
            state=$(docker inspect --format '{{.State.Status}} {{if .State.Health}}{{.State.Health.Status}}{{end}}' "$name")
            case "$state" in
                'running healthy') echo 'Milvus ready at http://127.0.0.1:19530'; exit 0 ;;
                exited*|dead*|*unhealthy) break ;;
            esac
            sleep 3
        done
        docker logs --tail 30 "$name" >&2
        echo 'Milvus did not become healthy within 180 seconds.' >&2
        exit 1
        ;;
    stop) docker stop "$name" ;;
    status) docker inspect --format '{{json .State}}' "$name" ;;
    remove) docker rm "$name" ;; # Refuses running containers; preserves data/config.
    -h|--help) echo 'Usage: retrieval/milvus.sh {start|stop|status|remove}' ;;
    *) echo 'Usage: retrieval/milvus.sh {start|stop|status|remove}' >&2; exit 2 ;;
esac
