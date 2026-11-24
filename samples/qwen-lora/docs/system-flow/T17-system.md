# System flow — program `qlora_train` (T17)

## Callers

### Harness owner: M2 acceptance (CPU)

Runs `qlora_train` on the rehearsal model (M2) with the test data set of
T17's ITa (8 short records). Acceptance: overfits 32 samples (loss falls
below 10% of the first loss in 8 updates), validation loss falls on a small
real set.

### User: M2g and M4 production (GPU)

Runs `qlora_train` on the GPUs with `--split` and production data sets.
The output checkpoints feed T18's merge program.

## Dependencies

T17 uses everything from M1 and M2:

- `QwenModel` (T11): loads the base model, embeds tokens, computes logits.
- `Lora` (T12): the LoRA adapters attached to the model.
- `Recompute` (T13): block-wise recomputation for memory-efficient backward.
- `LoraOptim` (T14): AdamW optimizer with LoRA gradient clipping and step.
- `TrainData` (T15): data set loading, epoch ordering, batches.
- `Checkpoint` (T16): checkpoint write, read, and state management.

## Outputs

- `OUTDIR/train-log.tsv`: the training log (update lines, checkpoint lines).
- `OUTDIR/ckpt/`: the checkpoint directory.
  - `ckpt-NNNNNNNN/`: checkpoint files (adapters, optimizer state).
  - `latest`: symlink to the latest checkpoint.
  - `best`: symlink to the checkpoint with the lowest validation loss.
- The best checkpoint's path for T18's `qlora_merge`.

## Unit tests

- Tiny model fixture (T10/T11): two blocks, hidden 64, tiny tokenizer.
- `TrainStep_test.pas`: loss computation, slice loss, gradient accumulation,
  validation loss, error messages.
- `TrainCli_test.pas`: argument parsing, configuration hash, run loop,
  checkpoint save and load, resume, non-finite loss, exit codes.
- GPUs hidden (`HIP_VISIBLE_DEVICES=-1`, `ROCR_VISIBLE_DEVICES=-1`).

## ITa

Four runs with the rehearsal model on the CPU:

1. `ovf`: 8 epochs of one update, lr 1e-3, warm-up 1; checks overfitting.
2. `full`: 2 epochs, budget 256, 5 updates per epoch, checkpoint every 3.
3. `part`: same as `full` with `--max-updates 3`, then `--resume`.
4. `nan`: lr 1e30, checks non-finite loss handling.

Driver: `tests/ita/T17-run.sh` builds `qlora_train.pas` and
`train_check.pas`, runs both, and writes `summary.tsv`.

## Program flow

```
ParseTrainArgs
  check options and positionals
  return config or usage error

Load model and data
  TGgufFile.Open(MODEL)
  TQwenTokenizer.Create
  TTrainSet.Create(TRAIN, classes, budget)
  TTrainSet.Create(VALID) if given
  total = TotalUpdates(set, config)
  check warmup < total
  TQwenModel.Load with split
  ManualSeed(seed)
  TLoraSet.Create(rank, alpha/rank)
  AttachLora
  TLoraAdamW.Create
  TCheckpoints.Create

Resume or start
  if --resume:
    load latest checkpoint
    check ConfigHash and BaseSha256
    restore epoch, position, update
  else:
    epoch = 0, position = 0, update = 0

Training loop
  while epoch < epochs:
    batches = EpochBatches(set, seed, epoch)
    skip batches to position
    for each batch:
      for each sample:
        SampleLossBackward(denom, slice)
        accumulate loss
        increment position
      ClipGlobalNorm(clip)
      Step(ScheduleLR(lr, update, warmup, total, min ratio))
      ZeroGrad
      log update
      if at checkpoint:
        compute validation loss
        save checkpoint (epoch+1, 0) if epoch end
    epoch++, position = 0

Exit
  print best checkpoint
  exit 0
```
