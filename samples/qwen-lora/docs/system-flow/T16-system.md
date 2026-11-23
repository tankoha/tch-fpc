# System flow — unit `Checkpoint` (T16)

## Callers

### T17 (training loop)

The trainer saves a checkpoint:
- Every 50 updates (DESIGN 8.14).
- At the end of each epoch.
- Before exit (normal or interrupted).

It resumes from `Latest` when the checkpoint directory is not empty.

The trainer computes `config_hash` and `base_sha256`; T16 only stores them.

### T18 (qlora_merge)

`qlora_merge` reads the adapters through `ReadRunState` to obtain the rank
and scale, then loads `lora.ot` into a set of that rank and scale.

### Unit tests

UT runs on the CPU without a model file. A small LoRA set with two pairs
is created, trained, saved, loaded and compared. Coverage: 100% of the
lines in `src/Checkpoint.pas`.

### ITa driver

`tests/ita/T16/ckpt_check.pas` runs on the rehearsal model (the environment
variable `QLORA_MODEL` gives its path). Four processes:
1. `full`: 10 updates in one go, a checkpoint after each.
2. `part1`: updates 1-5, a checkpoint.
3. `part2`: a fresh run with another seed, resumed from the latest
   checkpoint, updates 6-10, a checkpoint.
4. `compare`: compares the results of `full` and `part`.

One update: minus the log-likelihood of prompt chat-065 of
`work/oracle/T11` through the whole graph, Backward, clip 1, AdamW at
`ScheduleLR(1e-4, U, 2, 10, 0.1)`. Rank 16, scale 2.

The driver runs from a fresh directory outside the project, each process
its own working directory. About two minutes; the checkpoints (about
0.5 GB) are deleted afterwards.

## Data flow

```
TCheckpoints.Save(state, set, optim)
    |
    +--[CheckOptim]-- set and optimiser match?
    |
    +--[cleanup]-- remove stale tmp-ckpt-* entries
    |
    +-- mkdir tmp-ckpt-<update>
    |
    +-- TLoraSet.Save -> tmp-ckpt-<update>/lora.ot
    |
    +-- SaveMoments -> tmp-ckpt-<update>/moments.ot
    |       (by variable name, not by store index)
    |
    +-- WriteText -> tmp-ckpt-<update>/state.json
    |
    +-- rename tmp-ckpt-<update> -> ckpt-<update>
    |
    +--[Prune]-- delete old checkpoints, keep last Keep and best

TCheckpoints.Load(dir, set, optim)
    |
    +-- ReadRunState(dir) -> state
    |       (checks dir/state.json exists, parses JSON)
    |
    +--[rank/scale check]-- state matches set?
    |
    +--[CheckOptim]-- set and optimiser match?
    |
    +-- TLoraSet.Load <- dir/lora.ot
    |
    +-- LoadMoments <- dir/moments.ot
    |       (by variable name, into the optimiser's moment stores)
    |
    +-- optimiser.StepCount := state.Update
    |
    +-- return state
```

## File format

`state.json`: JSON object with the keys listed in the external design.

`lora.ot`: TVarStore archive (tch-fpc), variables named by their LoRA path
(e.g. `blk.0.attn_qkv.lora_a`).

`moments.ot`: TVarStore archive, variables named `<path>.adam_m` and
`<path>.adam_v`.
