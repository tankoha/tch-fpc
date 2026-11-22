# T07 System Flow: QwenChat

## Callers

### T15: TrainData

T15 builds the training data for LoRA fine-tuning. It uses `EncodeTraining` to
produce tokenized samples with the loss span:

1. Reads supervised conversation pairs from a JSONL corpus
2. For each conversation, builds `TChatMessages` and calls `EncodeTraining`
3. Stores the token ids and the `[LossFrom, LossTo)` span in the dataset

The loss covers only the last assistant answer and its `<e>` token.

### T20: Evaluation Prompts

T20 renders evaluation prompts (for generation, not training) using
`RenderChat`:

1. Builds the conversation with the evaluation prompt
2. Calls `RenderChat` with appropriate options (EnableThinking, Effort)
3. The result goes to the model for inference

## Unit Tests

The UT (`tests/ut/QwenChat_test.pas`) verifies the unit against hand-computed
expected texts:

- Uses the tag constants from the test skeleton (`TagS`, `TagE`, `TagThink`,
  `TagEndThink`, `TagTr`, `TagEndTr`)
- Uses `XH` and `LOW` constants for reasoning instructions
- Uses `Msgs([...])` to build message arrays
- EncodeTraining tests use `NewTinyTokenizer` from T05's UT vocabulary:
  - ids 0..255 are the bytes
  - 259 = `hello`, 264 = ` world`
  - 266 = `<s>`, 267 = `<e>` (EOS), 268 = `{think}`
  - `{/think}` is not a token, spelled out in bytes

## ITa Driver

The ITa driver (`tests/ita/T07/chat_check.pas`) compares the unit against the
bundled `llama-server`'s `/apply-template` and `/tokenize` endpoints:

1. Started with `QLORA_MODEL=<rehearsal GGUF> tests/ita/T07-run.sh`
2. The server is started with `--chat-template-file` of the 27B template
   (the rehearsal 0.8B GGUF's own template is different)
3. Reads `work/oracle/T07/corpus.jsonl` (14 conversations, 5 training samples,
   4 rejected conversations)
4. Compares renderings with `/apply-template` and encodings with `/tokenize`
5. Writes results to `work/ita/T07/summary.tsv`

## Data Flow

```
T15 (TrainData)
    |
    v
EncodeTraining(Msgs, Opt, LossFrom, LossTo) --> Token ids + loss span
    |
    v
Dataset for training


T20 (Evaluation)
    |
    v
RenderChat(Msgs, Opt) --> Rendered prompt text
    |
    v
Model inference
```
