# External design — unit `TrainData` (T15)

## Purpose

The unit `TrainData` transforms a data set file into training samples,
organises an epoch into batches within a position budget, and renders a
batch as padded tensors with the loss mask. It bridges T19's record
writer and T17's training loop.

## Record format

- JSONL: one record per line.
- Required string keys: `id`, `dialect`, `prompt`, `answer`, `class`.
- Other keys are ignored; blank lines are skipped.
- Text is raw UTF-8. A `\u` escape of a character above U+007F is refused
  (fpjson turns such an escape into `?`). T19 writes UTF-8 directly.

## Admission

A record is admitted when:
- its `class` is in the list of admitted classes (compared literally;
  default `compiled-run`, `fpc-approx`; see DESIGN 8.16);
- its tokenised length does not exceed the maximum length (the budget).

Samples skipped by class or by length are counted separately.

## Rendering and loss span

- User = prompt, assistant = answer, rendered without thinking:
  `QwenChat.EncodeTraining` with `EnableThinking` False.
- No system message. The empty think block precedes the answer (DESIGN 8.14).
- The loss span `[LossFrom, LossTo)` covers the trimmed answer and its
  closing `im_end`. The newline after `im_end` is not in the loss.

## No token cache

DESIGN 8.12 planned a token cache. Measured on the rehearsal tokenizer,
`EncodeTraining` of a 2000-token sample takes 4 ms, so the 4000 samples
of data set v1 tokenise in seconds at every start. No cache is needed.

## Epoch order

- A permutation from (seed, epoch) by Fisher-Yates with SplitMix64.
- No libtorch random state: the order is the same in every process.
- Supports deterministic resumption from a checkpoint.

## Batches (pools and budget)

- The epoch's order is cut into pools of `Pool` samples.
- Each pool is sorted by length (ties keep the epoch's order).
- Each pool is cut greedily into batches: count × longest ≤ `Budget`.
- The batches of a pool are shuffled (SplitMix64 from seed, epoch, pool).
- Every batch is a function of (seed, epoch), so a run resumes by
  skipping the batches of the samples already consumed (`SkipBatches`,
  T16's `Position`).

## Tensors

- `Ids` [B, T] int64: right-padded with `Pad`.
- `Targets` [B, T] int64: `Ids` shifted left by one; `Pad` at the end.
- `Mask` [B, T] float: 1 where the target `t + 1` is in `[LossFrom, LossTo)`.
- T is the longest sample of the batch.
- The loss (DESIGN 8.14) is the sum of masked cross-entropies divided by
  the mask count.

## Errors

- `ETrainData` with messages for: missing file, parse errors, missing or
  bad fields, duplicate ids, negative max length, budget too low,
  position not at batch end, empty batch, sample index out of range.

## Open for T17

`QwenModel` (T11) runs one sequence `[T, H]`. A batch of B samples needs
either B forwards (each paying the dequantisation) or a batched model.
T15 delivers both views: the batch as sample indices and as padded tensors.
