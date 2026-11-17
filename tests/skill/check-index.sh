#!/usr/bin/env bash
# Static checks of the tch-fpc skill (DESIGN 7.1, 7.3); needs no libtorch.
#   - size limits: SKILL.md <= 160 lines, reference/*.md <= 120,
#     each example, pitfall and trap <= 80
#   - the tables of SKILL.md and the files on disk agree both ways
#   - every row of "Where to look" names an existing file and a heading
#     (or, for a .pas file, a line) that is still in it
#   - the version in SKILL.md equals the one in plugin.json
set -euo pipefail
shopt -s nullglob
root=$(cd "$(dirname "$0")/../.." && pwd)
skill=$root/plugin/skills/tch-fpc
md=$skill/SKILL.md
fail=0
ng() { echo "NG  $*"; fail=1; }

limit() {   # <max> <file...>
  local max=$1; shift
  for f in "$@"; do
    n=$(wc -l < "$f")
    [ "$n" -le "$max" ] || ng "${f#$root/}: $n lines, limit $max"
  done
}
limit 160 "$md"
limit 120 "$skill"/reference/*.md
limit 80 "$skill"/examples/*.pas "$skill"/pitfalls/*.pas "$skill"/traps/*.pas

for f in "$skill"/examples/*.pas "$skill"/pitfalls/*.pas "$skill"/traps/*.pas \
         "$skill"/reference/*.md; do
  rel=${f#$skill/}
  grep -qF -- "$(basename "$rel")" "$md" || ng "$rel is not mentioned in SKILL.md"
done
for f in "$skill"/examples/*.pas "$skill"/traps/*.pas; do
  [ -f "${f%.pas}.expected" ] || ng "${f#$skill/}: no .expected"
done
grep -oE '`(examples|pitfalls|traps)/[A-Za-z0-9_]+\.pas`' "$md" | tr -d '`' | sort -u |
  while read -r rel; do [ -f "$skill/$rel" ] || echo "NG  SKILL.md names missing $rel"; done |
  { grep . && exit 1 || true; } || fail=1
for name in $(sed -n '/^## Reference/,/^## /p' "$md" | grep -oE '`[a-z]+\.md`' | tr -d '`'); do
  [ -f "$skill/reference/$name" ] || ng "SKILL.md names missing reference/$name"
done

# "Where to look": | topic | `path` "heading" | link |
rows=0
while IFS= read -r line; do
  path=$(sed -n 's/^| [^|]* | `\([^`]*\)` "\(.*\)" | .*/\1/p' <<< "$line")
  head=$(sed -n 's/^| [^|]* | `\([^`]*\)` "\(.*\)" | .*/\2/p' <<< "$line")
  [ -n "$path" ] || continue
  rows=$((rows + 1))
  if [ ! -f "$root/$path" ]; then ng "index: $path does not exist"; continue; fi
  case $path in
    *.md) grep -E '^#+ ' "$root/$path" | grep -qF -- "$head" \
            || ng "index: no heading \"$head\" in $path" ;;
    *)    grep -qF -- "$head" "$root/$path" || ng "index: no \"$head\" in $path" ;;
  esac
done < <(sed -n '/^## Where to look/,/^## /p' "$md")
[ "$rows" -gt 0 ] || ng "index: no rows found under \"## Where to look\""

v_md=$(sed -n 's/^version: *//p' "$md" | head -1)
v_plugin=$(sed -n 's/.*"version": *"\([^"]*\)".*/\1/p' "$root/plugin/.claude-plugin/plugin.json")
[ "$v_md" = "$v_plugin" ] || ng "version: SKILL.md $v_md, plugin.json $v_plugin"

[ $fail -eq 0 ] && echo "OK  skill index ($rows index rows)"
exit $fail
