# QwenChat Detailed Design (T07)

Based on: `docs/external-design/T07-external.md`, `docs/system-flow/T07-system.md`,
`tasks/T07-facts.md` (authoritative; the facts are final, measured against the
bundled `llama-server`'s `/apply-template` and `/tokenize` under the 27B
template). This design restates and organizes the facts; it adds no behavior.

## 0. Tags (read this first)

The template's tags are special tokens of Qwen (and GLM 5 uses the think and
tool tags itself). **Never write a tag literally**, in code, tests, documents
or tables — an agent that writes one may end its own reply or get rejected.
In these documents the tags appear only as the placeholders below; in Pascal
they are the unit constants already present in the skeleton, built from
pieces (so the tag text never appears as a literal anywhere):

| Placeholder | Unit constant (src) | Test constant (UT) |
|---|---|---|
| `<s>` (im_start) | `ImStart` | `TagS` |
| `<e>` (im_end) | `ImEnd` | `TagE` |
| `{think}` | `ThinkOpen` | `TagThink` |
| `{/think}` | `ThinkClose` | `TagEndThink` |
| `{tr}` (tool_response) | `ToolRespOpen` | `TagTr` |
| `{/tr}` | `ToolRespClose` | `TagEndTr` |

In the formula fragments below, `\n` means #10 and the tags are written by
their placeholders.

## 1. Module composition

| File | Kind | Responsibility |
|---|---|---|
| `src/QwenChat.pas` | unit | All of T07: rendering the 27B chat template for text turns (rules 1–4), the training sample with the span the loss covers |

- No program file: the callers are T15 (`TrainData`, via
  `RenderTraining` / `EncodeTraining`) and T20 (evaluation prompts, via
  `RenderChat`).
- Depends on unit `QwenTokenizer` (T05) only for `EncodeTraining`; the
  tokenizer is always supplied by the caller (tests use
  `NewTinyTokenizer`, the UT vocabulary of T05).
- No file I/O, no environment variables, no GPU; pure string construction
  plus optional tokenizer calls.
- Free Pascal 3.2, `{$mode objfpc}{$H+}`.
- UT stages (fpcunit, `tests/ut/QwenChat_test.pas`):
  - Stage 1 (T07b1): the helpers — rows UT-QwenChat-001 to 010.
  - Stage 2 (T07b2): `Render`, `RenderChat`, `IsToolResponse` — rows UT-QwenChat-011 to 027.
  - Stage 3 (T07b3): `RenderTraining`, `EncodeTraining` — rows UT-QwenChat-101 to 107.
  - ITa (T07c): ITa-T07-001 to 004, compared with the bundled
    `llama-server` (`/apply-template`, `/tokenize`) under the 27B template.

## 2. Function specifications

The interface (final, as in the external design):

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

All strings are `RawByteString`; offsets are byte offsets computed with
`Length` / `Copy` (1-based in `Copy`, 0-based as reported by the caller).

### 2.1 Helpers (stage 1, rows UT-QwenChat-001 to 010)

#### `function DefaultChatOptions: TChatOptions;`
- No parameters.
- Returns: `EnableThinking := True`, `Effort := reXHigh`,
  `PreserveThinking := True`, `AddGenerationPrompt := True`.
- No exceptions.

#### `function ChatMsg(ARole: TChatRole; const AContent: RawByteString; const AReasoning: RawByteString = ''): TChatMessage;`
- No internal state.
- Returns: the record with `Role := ARole`, `Content := AContent`,
  `Reasoning := AReasoning` (the second argument from its arguments,
  defaulting to the empty string).
- No exceptions.

#### `function TrimChat(const S: RawByteString): RawByteString;`
- Behavior: removes leading and trailing bytes that are ASCII white space and
  returns the rest; the middle is untouched. An all-space string trims to
  the empty string.
- The space set is exactly `' '`, #9, #10, #11, #12, #13 (see
  `IsChatSpace`). Nothing else is removed: U+3000 and U+00A0 at the ends
  stay (measured).
- No exceptions.

#### Private: `function IsChatSpace(C: AnsiChar): Boolean;`
- Used only by `TrimChat`.
- Returns True iff `C in [' ', #9, #10, #11, #12, #13]`, else False.

#### `function ReasoningInstruction(const Opt: TChatOptions): RawByteString;`
- Returns, in this order of conditions:
  1. `''` when `Opt.EnableThinking` is False (whichever the effort);
  2. `XHighText` (the XH sentence) when the effort is `reXHigh`;
  3. `LowText` (the LOW sentence) when the effort is `reLow`;
  4. `''` when the effort is `reMedium`.
- No exceptions.

### 2.2 Private: `function IsToolResponse(const C: RawByteString): Boolean;`
- Returns True iff `C` both starts with `{tr}` (`ToolRespOpen`) and ends
  with `{/tr}` (`ToolRespClose`).
- Called only on trimmed user content (rendering rule 2; rows 022, 027).

### 2.3 `Render` (private; stage 2 — the main rendering function)

Signature: `function Render(const Msgs: TChatMessages; const Opt: TChatOptions; out AnswerStart, AnswerEnd: Integer): RawByteString;`

- `AnswerStart`, `AnswerEnd` are 0-based byte offsets into the returned
  text: the first byte of the **last** assistant message's trimmed content,
  and the byte just after the same message's `<e>`. They are set only for
  `crAssistant` messages (overwritten on each, so the final values are
  those of the last one). No assistant message → both stay `-1`.
- `RenderChat` calls it and ignores the offsets; `RenderTraining` uses them.
- Raises `EChatError` as listed in 2.6 (checks).

Algorithm, in code order (R = `ReasoningInstruction(Opt)`, C = trimmed
content `TrimChat(Msgs[I].Content)`, Rc = trimmed reasoning):

1. **Init.** `AnswerStart := -1; AnswerEnd := -1;`
2. **Empty check.** `Length(Msgs) = 0` → raise `EChatError('no messages')`.
3. **Reasoning instruction.** `R := ReasoningInstruction(Opt);`
   `Result := ''`.
4. **System turn (rule 1)** — decided purely from `Msgs[0]` and R:
   - first message is `crSystem` and C <> '':
     - R <> '': `<s>system\n` + R + `\n\n` + C + `<e>\n`
       (Pascal: `ImStart + 'system'#10 + R + #10#10 + C + ImEnd + #10`)
     - R = '': `<s>system\n` + C + `<e>\n`
   - first message is `crSystem` and C = '', or first message is not
     system:
     - R <> '': `<s>system\n` + R + `<e>\n`
       (`ImStart + 'system'#10 + R + ImEnd + #10`)
     - R = '': nothing (the result stays empty).
   - Note: with a real system message and R = '', the result is **not**
     the empty string — only the combination "no non-empty system turn"
     yields nothing (rows 013–017, 103).
5. **Last query (rule 2).** `LastQuery := -1;` scan `I := High(Msgs) downto
   0`: the first (i.e. last in order) index where `Msgs[I].Role = crUser`
   **and** `not IsToolResponse(TrimChat(Msgs[I].Content))` becomes
   `LastQuery`; stop at the first hit.
   Content that both starts with `{tr}` and ends with `{/tr}` is **not**
   a query (rows 022, 027).
6. **Query check.** `LastQuery < 0` → raise
   `EChatError('no user query in the messages')`.
7. **Message loop (rule 3).** For `I := 0 to High(Msgs)` (in order):
   - `C := TrimChat(Msgs[I].Content);`
   - `crSystem`: `I > 0` → raise
     `EChatError.CreateFmt('message %d: a system message must be the
     first', [I])` (I is the 0-based index); `I = 0` → nothing is
     appended (the system turn is already in the result from step 4).
   - `crUser`: append `<s>user\n` + C + `<e>\n`
     (`Result := Result + ImStart + 'user'#10 + C + ImEnd + #10`).
   - `crAssistant`:
     1. `Rc := TrimChat(Msgs[I].Reasoning);`
     2. If `Opt.PreserveThinking or (I > LastQuery)`: append
        `<s>assistant\n{think}\n` + Rc + `\n{/think}\n\n`
        (`Result + ImStart + 'assistant'#10 + ThinkOpen + #10 + Rc + #10 + ThinkClose + #10#10`);
        the empty think block `{think}\n\n{/think}\n\n` appears when Rc is
        empty (rows 019, 101).
        Otherwise append only `<s>assistant\n`.
     3. `AnswerStart := Length(Result);`
     4. Append C + `<e>` (`Result := Result + C + ImEnd;`) — for the last
        assistant this is the answer span, see step 8 below.
     5. `AnswerEnd := Length(Result);`
     6. Append one `\n` (trailing newline of the turn).
   - So the think block is kept for an assistant message **after the last
     query even when PreserveThinking is False** (row 021), and dropped
     **before the last query** when PreserveThinking is False (row 020).
8. **Answer span.** After the loop, `AnswerStart`/`AnswerEnd` hold the last
   assistant message's span over its trimmed content and its `<e>`; the
   answer is `Copy(Text, AnswerStart + 1, AnswerEnd - AnswerStart)` (it
   includes `<e>`, not the trailing `\n` of step 7.6).
9. **Generation prompt (rule 4).** If `Opt.AddGenerationPrompt`:
   - append `<s>assistant\n` (`Result + ImStart + 'assistant'#10`);
   - if `Opt.EnableThinking`: append `{think}\n`
     (`Result + ThinkOpen + #10`);
   - else append an empty think block `{think}\n\n{/think}\n\n`
     (`Result + ThinkOpen + #10#10 + ThinkClose + #10#10`).
   - `RenderTraining` always passes `AddGenerationPrompt = False`, so the
     prompt is never part of a training sample (rows 101–103).

### 2.4 `RenderChat` (stage 2, rows UT-QwenChat-011 to 023)

- `function RenderChat(const Msgs: TChatMessages; const Opt: TChatOptions): RawByteString;`
- Calls `Render(Msgs, Opt, AnswerStart, AnswerEnd)` and returns its
  Result; the two out offsets are ignored.
- All checks and messages of `Render` apply (2.6), raised unchanged.

### 2.5 `RenderTraining` and `EncodeTraining` (stage 3, rows UT-QwenChat-101 to 107)

#### `function RenderTraining(const Msgs: TChatMessages; const Opt: TChatOptions; out AnswerStart, AnswerEnd: Integer): RawByteString;`
- **Check first, before anything else** (2.6, item 1): `Length(Msgs) = 0`,
  or the last message is not an assistant message → raise
  `EChatError('a training sample must end with an assistant message')`.
  For an empty array **this** message (not `no messages`) is raised
  (rows 104, 105).
- Then render with a copy of the options whose generation prompt is off:
  `O := Opt; O.AddGenerationPrompt := False;`
  `Result := Render(Msgs, O, AnswerStart, AnswerEnd);`
- The answer span semantics of 2.3.8 apply: `AnswerStart`, `AnswerEnd` are
  the 0-based byte offsets of the last assistant's trimmed content and the
  end of its `<e>`; the caller recomputes the answer as
  `Copy(Text, AnswerStart + 1, AnswerEnd - AnswerStart)` (rows 101–103).

#### `function EncodeTraining(Tok: TQwenTokenizer; const Msgs: TChatMessages; const Opt: TChatOptions; out LossFrom, LossTo: Integer): TTokenArray;`
- Steps (locals `Text: RawByteString; A, B: Integer`):
  1. `Text := RenderTraining(Msgs, Opt, A, B);` (A = AnswerStart, B =
     AnswerEnd).
  2. `Result := Tok.Encode(Text);`
  3. `LossFrom := Length(Tok.Encode(Copy(Text, 1, A)));`
  4. `LossTo := Length(Tok.Encode(Copy(Text, 1, B)));`
- `LossFrom`/`LossTo` are token indices, the span `[LossFrom, LossTo)`;
  it equals the last answer plus its `<e>` token (rows 106, 107).
- Why the prefix-encodings give the token indices: the answer always
  follows a `\n` and ends with the special token `<e>`, so the
  pre-tokenizer splits there, and both prefixes (`Text[1..A]`, `Text[1..B]`)
  are prefixes of the whole text — encoding a prefix yields the first
  N ids of the full encoding (ITa checks the spans; the ITa oracle is T05's
  vocabulary plus byte ids).
- Raises: `EChatError` from `RenderTraining` (2.6, item 1), unchanged;
  tokenizer errors from `Tok.Encode` (T05) propagate.

### 2.6 Checks and error messages (EChatError), in order

All messages are `EChatError` and are raised in exactly this execution
order. A `RenderChat` call can only reach items 2–4; a `RenderTraining`
call can reach item 1 or, inside `Render`, items 2–4.

1. **`RenderTraining` (before it calls `Render`)** — `Msgs` empty, or the
   last message is not an assistant message:
   `a training sample must end with an assistant message`
   (rows 104, 105).
2. **`Render` (so `RenderChat` and `RenderTraining`)** — `Msgs` empty:
   `no messages` (row 024).
3. **`Render`** — no user message exists whose trimmed content is **not**
   a tool response (no last query found):
   `no user query in the messages` (rows 026, 027).
4. **`Render`, in the message loop** — a `crSystem` message at index
   `I > 0` (0-based):
   `message %d: a system message must be the first`, with `%d` filled from
   `I` via `CreateFmt([...], [I])`; literal result for `I = 1`:
   `message 1: a system message must be the first` (row 025).

Nothing else in the unit raises `EChatError`; the tokenizer's own errors
are T05's.

## 3. Data structures and constants

- Types (interface section, as above): `EChatError`, `TChatRole`,
  `TChatMessage`, `TChatMessages`, `TReasoningEffort`, `TChatOptions`;
  the offset types in the signatures (`Integer`).
- Constants (implementation section, already in the skeleton):
  - The six tag constants `ImStart`, `ImEnd`, `ThinkOpen`, `ThinkClose`,
    `ToolRespOpen`, `ToolRespClose` — each **built from pieces** so no tag
    text appears as a literal (tag table in section 0).
  - `XHighText` — the XH sentence; `LowText` — the LOW sentence (their full
    texts are in the external design's "Constants" section; the tests carry
    the same text as the skeleton constants `XH` and `LOW`).
- No persistent state, no files.

## 4. Business-rule → implementation map

Every rule of the external design is realized here (rows = UT-QwenChat-*):

| Rule (external design / facts) | Where it is realized | UT rows |
|---|---|---|
| Default options: thinking on, xhigh, preserve thinking, generation prompt | `DefaultChatOptions` | 001 |
| Message construction from arguments | `ChatMsg` | 002, 003 |
| Trimming: `' '`, #9–#13 at both ends, nothing else (U+3000/U+00A0 stay) | `TrimChat` + `IsChatSpace` | 004–006, 016, 017, 023 |
| Reasoning instruction: XH / LOW / `''` per effort; `''` when thinking off | `ReasoningInstruction` | 007–010, 013–015 |
| Rule 1: system turn (content + R, content only, R only, nothing) | `Render` step 4 | 011, 013–017, 103 |
| Rule 2: last query = last non-tool user message | `Render` step 5 + `IsToolResponse` | 019–022, 027 |
| Rule 3: user turn `<s>user\n` + C + `<e>\n` | `Render` step 7 | 012, 018–023, 101, 102 |
| Rule 3: assistant think block when PreserveThinking or after the last query | `Render` step 7.2 | 019–021, 101–103 |
| Rule 3: a system message at `I > 0` is an error | `Render` step 7, check 2.6.4 | 025 |
| Rule 4: generation prompt (on: open think; off: empty think block) | `Render` step 9 | 011, 012, 018–022 |
| `no messages` | check 2.6.2 | 024 |
| `no user query in the messages` | check 2.6.3 | 026, 027 |
| training must end with an assistant message | check 2.6.1 | 104, 105 |
| training: `AddGenerationPrompt` ignored | `RenderTraining`, step 2.5 | 103 |
| `AnswerStart`/`AnswerEnd` = last assistant content + `<e>` | `Render` step 8 | 101–103 |
| `LossFrom`/`LossTo` = last answer + `<e>` token | `EncodeTraining` steps 3–4 | 106, 107 |

## 5. Change history

| Date | Change | Reason |
|---|---|---|
| 2026-11-22 | Initial detailed design (T07a) | first version, from `tasks/T07-facts.md` and `docs/external-design/T07-external.md` |
