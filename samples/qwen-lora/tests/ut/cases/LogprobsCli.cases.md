# UT cases — unit `LogprobsCli` (T11)

| Case ID | Title | Target function | Input (short) | Expected result | Notes |
|---|---|---|---|---|---|
| UT-LogprobsCli-201 | TopLine, quoting and format | TopLine | `TopLine('a"b', 3, [5, 7], [-0.5, -1.25])` | exact: `{"id":"a\"b","n":3,"top":[{"id":5,"logprob":-0.5000000},{"id":7,"logprob":-1.2500000}]}` | |
| UT-LogprobsCli-202 | no arguments | RunLogprobsCLI | `CheckUsage(Self, [], ..)` | exit 2; E two lines: `qlora_logprobs: MODEL and FILE expected` and `Usage: qlora_logprobs [--top N] [--split S] MODEL FILE`; out empty | |
| UT-LogprobsCli-203 | `--top` without a value | RunLogprobsCLI | `['--top']` | exit 2; reason `--top needs a value`; usage line | |
| UT-LogprobsCli-204 | `--top 0`, below range | ParseCount | `['--top', '0', TinyPath, PromptPath]` | exit 2; reason `--top must be 1..100, got 0`; usage line | |
| UT-LogprobsCli-205 | `--top 101`, above range | ParseCount | `['--top', '101', TinyPath, PromptPath]` | exit 2; reason `--top must be 1..100, got 101`; usage line | |
| UT-LogprobsCli-206 | `--split x`, not an integer | ParseCount | `['--split', 'x', TinyPath, PromptPath]` | exit 2; reason `--split must be 0..1000, got x`; usage line | |
| UT-LogprobsCli-207 | `--TOP` (case) is unknown | RunLogprobsCLI | `['--TOP', '3', TinyPath, PromptPath]` | exit 2; reason `unknown option --TOP`; usage line | options are compared literally |
| UT-LogprobsCli-208 | one positional argument | RunLogprobsCLI | `[TinyPath]` | exit 2; reason `MODEL and FILE expected`; usage line | |
| UT-LogprobsCli-209 | `--top 3`, token records | RunLogprobsCLI | `RunCli(['--top', '3', TinyPath, PromptPath], O, E)` | exit 0; E empty; 3 lines; `O[0]` = `Want('a', [5, 17, 2], 3)`; `O[1]` = `Want('2', [8], 3)` | the record without `id` gets its line number |
| UT-LogprobsCli-210 | `text` record, tokenized | RunLogprobsCLI | the run of row 209; `Ids := TQwenTokenizer.Create(G).Encode('hello world')` (G on `TinyPath`; free both) | `O[2]` = `Want('t', Ids, 3)` | same run as row 209 |
| UT-LogprobsCli-211 | default top (20) | RunLogprobsCLI | `RunCli([TinyPath, PromptPath], O, E)` (default top) | exit 0; 3 lines; each has 20 occurrences of `"logprob"` | |
| UT-LogprobsCli-212 | invalid JSON line | RunLogprobsCLI | `WriteText(TmpPath, 'x')`; `CheckError(Self, TmpPath, ..)` | exit 1; one line `qlora_logprobs: line 1: not JSON (Error at line 1, Pos 1: Unexpected token (x) encountered.)` | `CheckError` = `[TinyPath, File]` |
| UT-LogprobsCli-213 | JSON array, not an object | RunLogprobsCLI | `WriteText(TmpPath, '[1, 2]')` | exit 1; one line `qlora_logprobs: line 1: not an object` | via `CheckError` |
| UT-LogprobsCli-214 | neither `tokens` nor `text` | RunLogprobsCLI | `WriteText(TmpPath, '{"id": "x"}')` | exit 1; one line `qlora_logprobs: line 1: needs "tokens" (array) or "text" (string)` | via `CheckError` |
| UT-LogprobsCli-215 | missing model file | RunLogprobsCLI | `RunCli([TinyPath + '.missing', PromptPath], O, E)` | exit 1; one line `'qlora_logprobs: Unable to open file "' + TinyPath + '.missing": No such file or directory'` | |
| UT-LogprobsCli-216 | token id out of range | RunLogprobsCLI | `WriteText(TmpPath, '{"tokens": [274]}')` | exit 1; one line `qlora_logprobs: token id 274 outside 0..273` | message of `QwenModel.Embed`, via `CheckError` |
| UT-LogprobsCli-217 | `--split 1` without GPUs | RunLogprobsCLI | `RunCli(['--split', '1', TinyPath, PromptPath], O, E)` | exit 1; no output; one line `qlora_logprobs: No CUDA GPUs are available` | the GPUs are hidden |
