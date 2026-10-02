#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

usage() {
    cat <<'HELP'
Usage: scripts/experiments.sh {medagents|medrag|hard|triage|orchestrate|search} [Hydra overrides...]
Environment: MODEL, DATASETS (comma separated), RUN_IDS (comma separated), SPLIT,
             PYTHON (executable), DRY_RUN=1 (print commands without running).
Example: DRY_RUN=1 MODEL=gpt-4o DATASETS=medqa RUN_IDS=0 scripts/experiments.sh search
HELP
}
case "${1:-}" in
    -h|--help) usage; exit 0 ;;
    medagents|medrag|hard) datasets=medqa,medbullets,medexqa,medmcqa,medxpertqa-r,medxpertqa-u,mmlu,mmlu-pro,pubmedqa ;;
    triage|orchestrate|search) datasets=medqa ;;
    *) usage >&2; exit 2 ;;
esac
suite=$1
shift
extra=("$@")
model=gpt-4o-mini
runs=0,1,2
[[ $suite != medrag ]] || model=gpt-4o
[[ $suite != hard ]] || model=o3-mini
[[ $suite != search ]] || runs=0

run() {
    local name=$1
    shift
    local command=("${PYTHON:-python3}" run_experiments.py --multirun
        "execution.dataset.name=${DATASETS:-$datasets}"
        "execution.dataset.split=${SPLIT:-test_hard}"
        "execution.model.name=${MODEL:-$model}"
        "execution.experiments.run_id=${RUN_IDS:-$runs}"
        "execution.experiment_name=$name" "$@" "${extra[@]}")
    printf '%q ' "${command[@]}"
    printf '\n'
    if [[ ${DRY_RUN:-0} != 1 ]]; then
        "${command[@]}"
    fi
}

case "$suite" in
    medagents)
        run medagents/medagents
        ;;
    medrag)
        run medrag/medrag triage.disable_triage=true triage.forced_level=easy triage.easy.num_experts=1 triage.easy.max_rounds=1 triage.easy.search_mode=required search.rewrite=false search.review=false search.search_mode=vector orchestrate.discussion_mode=group_chat_voting_only
        ;;
    hard)
        run medagents/new triage.forced_level=hard triage.disable_triage=true search.search_mode=vector
        ;;
    triage)
        for variant in \
            1_agent_1_round_no_search \
            1_agent_1_round_with_search \
            1_agent_2_round_with_search \
            1_agent_3_rounds_with_search \
            2_agents_1_round_with_search \
            2_agents_1_round_no_search \
            2_agents_2_rounds_with_search \
            2_agents_3_rounds_with_search \
            2_agents_3_rounds_no_search \
            3_agents_1_round_no_search \
            3_agents_1_round_with_search \
            3_agents_2_rounds_no_search \
            3_agents_2_rounds_with_search \
            3_agents_3_rounds_with_search \
            3_agents_3_rounds_no_search \
            5_agents_1_round_with_search \
            5_agents_1_round_no_search \
            5_agents_2_rounds_with_search \
            5_agents_2_rounds_no_search \
            5_agents_3_rounds_with_search \
            5_agents_3_rounds_no_search \
        ; do
            IFS=_ read -r -a parts <<< "$variant"
            search=required
            [[ $variant != *_no_search ]] || search=none
            run "agent_configuration/$variant" triage.disable_triage=true \
                triage.forced_level=medium "triage.medium.num_experts=${parts[0]}" \
                "triage.medium.max_rounds=${parts[2]}" "triage.medium.search_mode=$search" \
                search.search_mode=web
        done
        run triage_configuration/enable_triage triage.disable_triage=false search.search_mode=web
        run triage_configuration/disable_triage triage.disable_triage=true triage.forced_level=hard search.search_mode=web
        ;;
    orchestrate)
        run discussion_mode_ablation/independent orchestrate.discussion_mode=independent triage.disable_triage=true triage.forced_level=custom triage.forced_level_custom.num_experts=1 triage.forced_level_custom.max_rounds=3 search.search_mode=web
        run discussion_mode_ablation/group_chat_with_orchestrator orchestrate.discussion_mode=group_chat_with_orchestrator search.search_mode=web
        run discussion_mode_ablation/group_chat_voting_only orchestrate.discussion_mode=group_chat_voting_only search.search_mode=web
        run discussion_mode_ablation/one_on_one_sync orchestrate.discussion_mode=one_on_one_sync search.search_mode=web
        ;;
    search)
        run search_features/baseline search.topk.retrieve=10 search.topk.rerank=5 search.rewrite=true search.review=true search.search_mode=both
        run search_features/no_query_rewrite search.topk.retrieve=10 search.topk.rerank=5 search.rewrite=false search.review=true search.search_mode=both
        run search_features/no_document_review search.topk.retrieve=10 search.topk.rerank=5 search.rewrite=true search.review=false search.search_mode=both
        run search_features/no_rewrite_no_review search.topk.retrieve=10 search.topk.rerank=5 search.rewrite=false search.review=false search.search_mode=both
        run search_modality/web_only search.search_mode=web
        run search_modality/vector_only search.search_mode=vector
        run search_modality/both search.search_mode=both
        run search_history/individual search.search_history=individual search.search_mode=both
        run search_history/shared search.search_history=shared search.search_mode=both
        run search_source_depth/cpg_only 'search.allowed_sources=[cpg]' search.topk.retrieve=100 search.topk.rerank=25 search.search_mode=both
        run search_source_depth/textbooks_only 'search.allowed_sources=[textbooks]' search.topk.retrieve=100 search.topk.rerank=25 search.search_mode=both
        run search_source_depth/fewer_docs search.allowed_sources=all search.topk.retrieve=10 search.topk.rerank=5 search.search_mode=both
        run search_source_depth/more_docs search.allowed_sources=all search.topk.retrieve=200 search.topk.rerank=50 search.search_mode=both
        ;;
esac
