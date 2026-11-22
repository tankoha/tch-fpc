#!/usr/bin/env bash
# Hand one task of DESIGN 8.18 to the Hermes SDLC process (main agent
# GLM 5 on Bedrock; the sub-agent is delegation.* of ~/.hermes/config.yaml,
# GLM 5 on Bedrock 2026-11-22 T05b3–T13, local qwen3.8-27b-128k from T14) and keep the transcript.
# Usage: tasks/run-task.sh <Txx>      (reads tasks/<Txx>.md)
#   TURNS  max turns of the main agent (default 150)
#   MAIN_MODEL  main agent model id (default zai.glm-5: from T03 on it reported
#               honestly, while GPT-OSS-120B and Mistral leaked tool calls as
#               text; "config" uses model.default of ~/.hermes/config.yaml)
#   MAIN_PROVIDER  its provider (default bedrock)
# Output: work/tasks/<Txx>-<time>.txt; then run the oracle of the task by hand.
set -euo pipefail
proj=$(cd "$(dirname "$0")/.." && pwd)
t=${1:?usage: run-task.sh <Txx>}
spec=$proj/tasks/$t.md
[ -f "$spec" ] || { echo "no $spec" >&2; exit 2; }
mkdir -p "$proj/work/tasks"
log=$proj/work/tasks/$t-$(date +%Y%m%d-%H%M%S).txt
prompt="You work in the SDLC project $proj. Read $proj/AGENTS.md first and follow it.
The design is $proj/../../docs/DESIGN.md (sections 8.9-8.19); read the sections the task names.

$(sed "s|{proj}|$proj|g" "$spec")"
export HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
# A resident Hermes gateway keeps the environment of the run that started it,
# and tools and hooks may run in it. Refuse to start when its GPU hiding or
# QLORA_* values differ from this run's (stop the gateway, then retry).
for gw in $(pgrep -f 'hermes_cli.main.* gateway run' || true); do
  genv=$(tr '\0' '\n' < "/proc/$gw/environ" 2>/dev/null || true)
  for v in HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES $(compgen -e | grep '^QLORA_' || true) \
           $(printf '%s\n' "$genv" | sed -n 's/^\(QLORA_[A-Za-z0-9_]*\)=.*/\1/p'); do
    g=$(printf '%s\n' "$genv" | sed -n "s/^$v=//p")
    if [ "$g" != "${!v:-}" ]; then
      echo "Hermes gateway (pid $gw) has $v='$g', this run '${!v:-}'." >&2
      echo "Stop it first: hermes gateway stop (or kill $gw)." >&2
      exit 3
    fi
  done
done
# Bedrock (the main agent) needs a live `aws login` session; T01b1 lost a run
# to an expired one. The identity printed is discarded.
if ! aws sts get-caller-identity > /dev/null 2>&1; then
  echo "AWS session expired or missing: run 'aws login', then retry." >&2
  exit 4
fi
# sdlc.bb delegate (pre_tool_call on delegate_task) puts "First read <this
# file>" at the head of every delegated goal, so the sub-agent reads the task
# itself instead of the main agent's summary.
# The hook must be in ~/.hermes/shell-hooks-allowlist.json (approved once by the user).
# It reads the task file from ~/.hermes/sdlc/current-task, not from the
# environment: hooks may run in a resident Hermes gateway whose environment
# is that of the run that started it. One run at a time.
cur=${HERMES_HOME:-$HOME/.hermes}/sdlc/current-task
mkdir -p "$(dirname "$cur")"
printf '%s\n' "$spec" > "$cur"
# A review left pending by an earlier, interrupted run does not carry over.
rm -f "$(dirname "$cur")/review-pending.tsv"
trap 'rm -f "$cur"' EXIT
# Toolsets: no code_execution (execute_code runs Python, which this project
# forbids; Hermes refused it 21 times in T06 and each refusal cost a re-think).
# Sub-agents never get tools the main agent lacks.
main_args=()
main_model=${MAIN_MODEL:-zai.glm-5}
[ "$main_model" != config ] && main_args=(-m "$main_model" --provider "${MAIN_PROVIDER:-bedrock}")
hermes chat -Q "${main_args[@]}" -t terminal,file,skills,todo,delegation --max-turns "${TURNS:-150}" --in "$proj" -q "$prompt" < /dev/null > "$log" 2>&1 || true
echo "$log"
# On Bedrock both GPT-OSS and Mistral sometimes print a tool call as text
# (`to=functions.x`, `<|message|>`, `delegate_task{`) instead of making it;
# the run then ends early. Say so, so the run is not taken as finished.
if grep -qE 'to=functions\.|<\|message\|>|(delegate_task|write_file|patch|terminal|read_file|search_files)\{"|^\{"(tasks|goal|path|command|content)":' "$log"; then
  echo "WARNING: the transcript ends with a tool call printed as text; the run stopped early. Re-run the task." >&2
  exit 5
fi
