# System flow — unit `TrainData` (T15)

## Callers

### T17: training loop

1. Creates a `TTrainSet` from the data set file with the admitted classes
   and maximum length from the configuration.
2. For each epoch:
   - Calls `EpochBatches` with the seed, epoch, pool and budget.
   - Skips the batches of a resumed checkpoint's position (`SkipBatches`,
     T16's `Position`).
   - For each batch:
     - Calls `BatchTensors` to obtain `Ids`, `Targets`, `Mask`.
     - Runs the forward and backward pass.
     - Accumulates the loss (sum of masked cross-entropies / mask count).

### T19: `qlora_dataset`

Writes the JSONL records with their `class` field. The records are read
by T17's training loop via `TTrainSet`.

## Open question for T17

`QwenModel` (T11) processes one sequence `[T, H]`. A batch of B samples
requires either B separate forwards (each paying the dequantisation cost)
or a batched model. T15 provides both the sample indices and the padded
tensors; T17 chooses the approach.

## Unit tests

- Use T05's tiny tokenizer (`TokenizerFixture.NewTinyTokenizer`).
- Bytes are tokens except the merges of T05 and specials.
- `im_start`, `im_end`, `think` are one token each; `HasWideEscape` refusing `\u00e9`;
  `GetStr` raising `missing or bad <key>` messages; `TTrainSet.Create`
  parsing JSONL, checking ids, admitting by class and length, encoding
  samples.

### Stage 2 (T15b2): Epoch ordering and batches

`EpochOrder` via `Shuffle` with SplitMix64; `EpochBatches` with pool
sorting and greedy packing; `SkipBatches` position validation;
`BatchTensors` building [B, T] tensors with the loss mask.

## Program flow

```
TTrainSet.Create
  for each non-blank line:
    HasWideEscape? → error
    parse JSON, check keys
    check duplicate id
    admit by class
    EncodeSample(tokens, loss span)
    admit by length
  set sample array

EpochBatches
  EpochOrder(n, seed, epoch)
  for each pool:
    sort by length
    greedy batches: count × longest ≤ budget
    shuffle batches
  return all batches

BatchTensors
  find max length T
  allocate [B, T] arrays
  for each sample:
    copy tokens, shift for targets
    set mask 1 where target in [LossFrom, LossTo)
  create tensors

SkipBatches
  sum batch lengths until position
  position must equal sum
  return remaining batches
```
