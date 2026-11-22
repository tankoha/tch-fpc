# T07 External Design: QwenChat (Chat Template Rendering)

## Purpose

Unit `QwenChat` renders the chat template of the deployed model (`qwen3.8:27b`)
for text turns (system, user, assistant messages) and produces training samples
with the span the loss covers. It reproduces exactly the GGUF's Jinja template
(`tokenizer.chat_template`, 8,952 bytes) for text-only conversations (tools and
images are out of scope).

T15 (`TrainData`) builds training data using this unit; T20 renders evaluation
prompts with it.

## Tags (read this first)

The template's tags are special tokens of Qwen (and GLM 5 uses the think and
tool tags itself). **Never write a tag literally** in code, tests, documents
or messages. In documents they are written as placeholders; in Pascal they are
constants built from pieces:

| Placeholder | Unit constant (src) | Test constant (UT) |
|---|---|---|
| `<s>` (im_start) | `ImStart` | `TagS` |
| `<e>` (im_end) | `ImEnd` | `TagE` |
| `{think}` | `ThinkOpen` | `TagThink` |
| `{/think}` | `ThinkClose` | `TagEndThink` |
| `{tr}` (tool_response) | `ToolRespOpen` | `TagTr` |
| `{/tr}` | `ToolRespClose` | `TagEndTr` |

## Rendering Rules

### Constants

`XH` and `LOW` are reasoning instruction sentences:
- XH = `Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.`
- LOW = `Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.`

`R` (the reasoning instruction) is XH for effort xhigh, LOW for low, empty for
medium, and empty whenever thinking is off.

### Rule 1: System Turn

If the first message is a system message with a non-empty trimmed content C:

```
<s>system\n
```
+ (R + `\n\n` if R is not empty) + C + `<e>\n`

Otherwise (no system message, or its trimmed content is empty) and R is not
empty:

```
<s>system\n
```
+ R + `<e>\n`

Otherwise nothing.

### Rule 2: Last Query

The last query is the last user message whose trimmed content does not both
start with `{tr}` and end with `{/tr}`.

### Rule 3: Each Message in Order

- A system message other than the first is an error.
- User: `<s>user\n` + trimmed content + `<e>\n`
- Assistant: if PreserveThinking or the message comes after the last query:
  `<s>assistant\n{think}\n` + trimmed reasoning + `\n{/think}\n\n` + trimmed
  content + `<e>\n`, else `<s>assistant\n` + trimmed content + `<e>\n`

### Rule 4: Generation Prompt

If AddGenerationPrompt:

```
<s>assistant\n
```
+ (`{think}\n` if thinking is on, else `{think}\n\n{/think}\n\n`)

## Trimming

Trimming removes the ASCII white space `' '`, #9, #10, #11, #12, #13 at both
ends and nothing else. U+3000 and U+00A0 are not removed.

## Interface

```pascal
type
  EChatError = class(Exception);
  TChatRole = (crSystem, crUser, crAssistant);
  TChatMessage = record
    Role: TChatRole;
    Content: RawByteString;
    Reasoning: RawByteString;
  end;
  TChatMessages = array of TChatMessage;
  TReasoningEffort = (reXHigh, reMedium, reLow);
  TChatOptions = record
    EnableThinking: Boolean;
    Effort: TReasoningEffort;
    PreserveThinking: Boolean;
    AddGenerationPrompt: Boolean;
  end;

function DefaultChatOptions: TChatOptions;
function ChatMsg(ARole: TChatRole; const AContent: RawByteString;
  const AReasoning: RawByteString = ''): TChatMessage;
function TrimChat(const S: RawByteString): RawByteString;
function ReasoningInstruction(const Opt: TChatOptions): RawByteString;
function RenderChat(const Msgs: TChatMessages; const Opt: TChatOptions): RawByteString;
function RenderTraining(const Msgs: TChatMessages; const Opt: TChatOptions;
  out AnswerStart, AnswerEnd: Integer): RawByteString;
function EncodeTraining(Tok: TQwenTokenizer; const Msgs: TChatMessages;
  const Opt: TChatOptions; out LossFrom, LossTo: Integer): TTokenArray;
```

### Offsets

- `AnswerStart`, `AnswerEnd`: 0-based byte offsets in the result of the last
  assistant message's trimmed content and the end of its `<e>`.
- `LossFrom`, `LossTo`: token indices, `[LossFrom, LossTo)`.

## Errors (EChatError)

Raised in this order:

1. `Render` (so `RenderChat` and `RenderTraining`): no messages: `no messages`
2. `Render`: no last query: `no user query in the messages`
3. `Render`: in the loop, a system message at index I > 0:
   `message %d: a system message must be the first` (with I)
4. `RenderTraining` first: no messages, or the last message is not an assistant
   message: `a training sample must end with an assistant message`
