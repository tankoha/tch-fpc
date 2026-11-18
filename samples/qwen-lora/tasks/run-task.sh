#!/usr/bin/env bash
# Hand one task of DESIGN 8.18 to the Hermes SDLC process (main agent
# Mistral, sub-agent Qwen) and keep the transcript.
# Usage: tasks/run-task.sh <Txx>      (reads tasks/<Txx>.md)
#   TURNS  max turns of the main agent (default 150)
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
trap 'rm -f "$cur"' EXIT
hermes chat -Q --max-turns "${TURNS:-150}" --in "$proj" -q "$prompt" < /dev/null > "$log" 2>&1 || true
echo "$log"
