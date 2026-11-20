# T01 ITa test cases

Run every case from a fresh directory outside the project
(`w=$(mktemp -d)`; `cd "$w"`), with `QLORA_*` other than `QLORA_MODEL`
unset. `$QLORA_MODEL` is the absolute path of the rehearsal model,
`$P` the project root, `$O=$P/work/oracle/T01`, `$B=$P/work/bin/qlora_inspect`.
Keep the outputs under `$P/work/ita/T01/`. A comparison passes only when
`diff` prints nothing.

| Case ID | Title | Expected (short) |
|---|---|---|
| ITa-T01-001 | Summary lines match the oracle | the first 12 lines (summary + type lines) equal `summary.txt` |
| ITa-T01-002 | Per-type counts match the oracle | the `type` lines equal the `type` rows of `summary.txt` (f32 133, q8_0 36, q4_K 98, q5_K 36, q6_K 17) |
| ITa-T01-003 | KV lines match the oracle | the `kv` lines without `kv<TAB>` equal `kv-types.tsv` (46 lines) |
| ITa-T01-004 | Tensor lines match the oracle | the `tensor` lines without `tensor<TAB>` equal `tensors.tsv` (320 lines) |
| ITa-T01-005 | Per-type counts match the llama-server load log | the counts of the `type` lines equal the `llama_model_loader: - type` lines of `llama-server-load.log` |
| ITa-T01-006 | Exit 0 on the model | exit code 0, stderr empty |
| ITa-T01-007 | Not a GGUF file | exit 1, stderr `error: ...`, stdout empty |
| ITa-T01-008 | The model cut short by one byte | exit 1, stderr `error: ...` naming the last tensor, stdout empty (`Open` fails) |
| ITa-T01-009 | No argument | exit 2, usage on stderr, stdout empty |
| ITa-T01-010 | Unrecognised option | exit 2, usage on stderr, stdout empty |
| ITa-T01-011 | The model by a relative path from another directory | exit 0, stdout equal to ITa-T01-001..004's output |
| ITa-T01-012 | The model with one byte appended | exit 1, all lines on stdout (`data_end 532517120`, `file_size 532517121`), then stderr `error: data ends at 532517120, file size 532517121` |

## Commands

```sh
r=$P/work/ita/T01; mkdir -p "$r"
# ITa-T01-001..006
"$B" --kv --tensors "$QLORA_MODEL" > "$r/full.out" 2> "$r/full.err"; echo $? > "$r/full.rc"
head -12 "$r/full.out" | diff - "$O/summary.txt"                              # 001
grep '^type	' "$r/full.out" | diff - <(grep '^type	' "$O/summary.txt")       # 002
sed -n 's/^kv	\(.*	.*\)$/\1/p' "$r/full.out" | diff - "$O/kv-types.tsv"      # 003
sed -n 's/^tensor	//p' "$r/full.out" | diff - "$O/tensors.tsv"               # 004
sed -n 's/.*llama_model_loader: - type *\([A-Za-z0-9_]*\): *\([0-9]*\) tensors$/type\t\1\t\2/p' \
  "$O/llama-server-load.log" | diff - <(grep '^type	' "$r/full.out")          # 005
cat "$r/full.rc"; wc -c < "$r/full.err"                                       # 006: 0 and 0
# ITa-T01-007
"$B" "$O/summary.txt" > "$r/007.out" 2> "$r/007.err"; echo $?
# ITa-T01-008 (the copy stays in the mktemp directory, never in the repository)
head -c 532517119 "$QLORA_MODEL" > short.gguf
"$B" short.gguf > "$r/008.out" 2> "$r/008.err"; echo $?
# ITa-T01-009, 010
"$B" > "$r/009.out" 2> "$r/009.err"; echo $?
"$B" --foo "$QLORA_MODEL" > "$r/010.out" 2> "$r/010.err"; echo $?
# ITa-T01-011 (the relative path is computed, run from the mktemp directory)
"$B" --kv --tensors "$(realpath --relative-to=. "$QLORA_MODEL")" | diff - "$r/full.out"
# ITa-T01-012
{ cat "$QLORA_MODEL"; printf 'x'; } > long.gguf
"$B" long.gguf > "$r/012.out" 2> "$r/012.err"; echo $?
```

(`<TAB>` in the `grep`/`sed` patterns above is a literal tab.)
