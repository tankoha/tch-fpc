#!/usr/bin/env bash
# H39 (DESIGN 7.5): give the ten tasks of tests/skill/tasks to the local
# Qwen through Hermes, with and without the tch-fpc skill, and check each
# answer with check-task.sh.
# Usage: run-h39.sh <with|without> [task...]
#   OUT      result directory (default: build/h39/<date-time>)
#   LIBTCH   libtch build directory (default: build/libtch)
#   HERMES   hermes binary (default: hermes)
#   MODEL    Ollama model (default: qwen3.8-27b-64k)
#   TURNS    max turns per task (default: 40)
# Writes <OUT>/<mode>/<task>/{answer.pas,hermes.txt,check.txt} and appends
# "<mode> <task> PASS|FAIL <seconds> <skill-used>" to <OUT>/summary.tsv.
# The programs run on the CPU (GPUs hidden); Ollama serves the model.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
mode=${1:?usage: run-h39.sh <with|without> [task...]}; shift
case $mode in with|without) ;; *) echo "mode is with or without" >&2; exit 2 ;; esac
out=${OUT:-$root/build/h39/$(date +%Y%m%d-%H%M%S)}
libtch=$(cd "${LIBTCH:-$root/build/libtch}" && pwd)
hermes=${HERMES:-hermes}
model=${MODEL:-qwen3.8-27b-64k}
turns=${TURNS:-40}
export HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
tasks=${*:-$(cd "$root/tests/skill/tasks" && ls)}
mkdir -p "$out/$mode"

for t in $tasks; do
  dir=$out/$mode/$t
  mkdir -p "$dir"
  # The Hermes Pascal verify hook compiles a written .pas with plain fpc and
  # only the file's own directory as unit and library path. Links to the
  # tch-fpc units and libtch let that check pass, so the agent is not sent
  # after a phantom "Can't find unit Tch".
  ln -sf "$root"/src/*.pas "$root"/src/*.inc "$libtch/libtch.so" "$dir/"
  build="TCH_FPC_ROOT=$root TCH_OUT=$dir/bin TCH_UNITS=$dir/units $root/plugin/skills/tch-fpc/scripts/build-example.sh $dir/answer.pas $libtch"
  if [ "$mode" = with ]; then
    skill_note="Read the tch-fpc skill first (skill_view name=\"tch-fpc\") and follow it."
    skill_args=(-s tch-fpc)
  else
    skill_note="Do not load the tch-fpc skill and do not read anything under $root/plugin; use the sources in $root/src instead."
    skill_args=()
  fi
  prompt="$(cat "$root/tests/skill/tasks/$t/prompt.md")

Rules for this job:
- Write the complete program to $dir/answer.pas, in {\$mode objfpc}{\$H+} (load the pascal-fpc-objfpc skill before writing).
- tch-fpc is checked out at $root; its units are in $root/src.
- $skill_note
- Build with: $build
  It prints the path of the executable. Run that executable with $root as the current directory and compare its output with the required output. Repeat until it matches exactly.
- The tch-fpc units linked into $dir are for the verify hook; do not edit them.
- Do not change files outside $dir."
  start=$(date +%s)
  "$hermes" chat -Q --max-turns "$turns" --provider custom \
      --base-url http://localhost:11434/v1 --api-key ollama -m "$model" \
      -t terminal,file,skills "${skill_args[@]}" --in "$dir" -q "$prompt" \
      < /dev/null > "$dir/hermes.txt" 2>&1 || true
  secs=$(( $(date +%s) - start ))
  if [ "$mode" = with ]; then used=preloaded; else
    used=no; grep -qiE 'skill_view[^\n]*tch-fpc' "$dir/hermes.txt" && used=yes; fi
  LIBTCH=$libtch "$root/tests/skill/check-task.sh" "$root/tests/skill/tasks/$t" "$dir/answer.pas" > "$dir/check.txt" 2>&1 || true
  printf '%s\t%s\t%s\t%s\t%s\n' "$mode" "$t" "$(cut -d' ' -f1 "$dir/check.txt")" "$secs" "$used" | tee -a "$out/summary.tsv"
done
