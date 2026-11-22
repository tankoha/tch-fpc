# T07 facts (checked by the harness owner)

T07 is the unit `QwenChat`: the chat template of the deployed model
(`qwen3.8:27b`) for text turns (DESIGN 8.10 "Chat template", 8.14 "Loss
mask"). It renders system, user and assistant messages exactly as the
GGUF's Jinja template (`tokenizer.chat_template`, 8,952 bytes; tools and
images are out of scope), and gives a training sample with the span the
loss covers: the last assistant answer and its `<e>`. T15
(`TrainData`) builds the training data with it; T20 renders evaluation
prompts with it.

Measured by the harness owner with a prototype written exactly as below:
on 14 conversations it renders byte for byte what the bundled
`llama-server`'s `/apply-template` renders with the 27B template (the
rehearsal 0.8B GGUF's own template is a different one: it has no
reasoning-effort sentence; the server is started with
`--chat-template-file` of the 27B template); the 5 training samples
encode to the ids of `/tokenize`; the loss spans are exact; the 4
conversations the template rejects raise. Every row of "Unit tests" gives
the value in its row; the messages are the prototype's. These facts are
final; do not investigate further or write probes.

## Tags (read this first)

The template's tags are special tokens of Qwen (and GLM 5 uses the think
and tool tags itself): an agent that writes one literally may end its own
reply or have its request rejected. **Never write a tag literally**, in
code, tests, documents or tables. In these facts and in every document
they are written as placeholders; in Pascal they are constants built from
pieces (`'<|' + 'im_end|>'`), already in the skeletons:

| Placeholder | Unit constant (src) | Test constant (UT) |
|---|---|---|
| `<s>` (im_start) | `ImStart` | `TagS` |
| `<e>` (im_end) | `ImEnd` | `TagE` |
| `{think}` | `ThinkOpen` | `TagThink` |
| `{/think}` | `ThinkClose` | `TagEndThink` |
| `{tr}` (tool_response) | `ToolRespOpen` | `TagTr` |
| `{/tr}` | `ToolRespClose` | `TagEndTr` |

For example the expected text `<s>user\nQ<e>\n` is written
`TagS + 'user'#10'Q' + TagE + #10` in a test.

## The template (what the unit reproduces)

`XH` and `LOW` are the two reasoning sentences (constants of the test
skeleton; in the unit as constants of the implementation section):
- XH = `Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.`
- LOW = `Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.`

`R` (the reasoning instruction) is XH for effort xhigh, LOW for low, empty
for medium, and empty whenever thinking is off. Below `\n` is #10; the
tags are written as in "Tags".

1. System turn: if the first message is a system message with a non-empty
   trimmed content C: `<s>system\n` + (R + `\n\n` if R is not empty) + C +
   `<e>\n`. Otherwise (no system message, or its trimmed content is empty)
   and R is not empty: `<s>system\n` + R + `<e>\n`. Otherwise nothing.
2. The last query is the last user message whose trimmed content does not
   both start with `{tr}` and end with `{/tr}`.
3. Each message in order: a system message other than the first is an
   error; user: `<s>user\n` + trimmed content + `<e>\n`; assistant: if
   PreserveThinking or the message comes after the last query:
   `<s>assistant\n{think}\n` + trimmed reasoning + `\n{/think}\n\n` +
   trimmed content + `<e>\n`, else `<s>assistant\n` + trimmed content +
   `<e>\n`.
4. Generation prompt (if AddGenerationPrompt): `<s>assistant\n` +
   (`{think}\n` if thinking is on, else `{think}\n\n{/think}\n\n`).

Trimming removes the ASCII white space `' '`, #9, #10, #11, #12, #13 at
both ends and nothing else (measured: U+3000 and U+00A0 stay).

## Unit `QwenChat` (interface, final; in the skeleton)

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

`AnswerStart`, `AnswerEnd`: 0-based byte offsets in the result of the
last assistant message's trimmed content and the end of its `<e>`
(the answer is `Copy(Text, AnswerStart + 1, AnswerEnd - AnswerStart)`).
`LossFrom`, `LossTo`: token indices, `[LossFrom, LossTo)`.

## Implementation (write it exactly so)

The skeleton's implementation section already has the constants `ImStart`,
`ImEnd`, `ThinkOpen`, `ThinkClose`, `ToolRespOpen`, `ToolRespClose`
(the tags, built from pieces) and `XHighText`, `LowText`. Use them;
**never write a tag literally** (see "Tags").

- `DefaultChatOptions`: EnableThinking True, Effort reXHigh,
  PreserveThinking True, AddGenerationPrompt True.
- `ChatMsg`: the three fields from the arguments.
- `TrimChat`: a private `IsChatSpace(C: AnsiChar): Boolean` (`C in [' ',
  #9, #10, #11, #12, #13]`); skip such bytes from the start and from the
  end, `Copy` the rest.
- `ReasoningInstruction`: `''` if not EnableThinking; else XHighText for
  reXHigh, LowText for reLow, `''` for reMedium.
- A private `IsToolResponse(const C: RawByteString): Boolean`: C starts
  with `{tr}` and ends with `{/tr}`.
- A private `Render(const Msgs; const Opt; out AnswerStart, AnswerEnd:
  Integer): RawByteString` does the work (below); `RenderChat` calls it
  and ignores the offsets.

```pascal
function Render(const Msgs: TChatMessages; const Opt: TChatOptions;
  out AnswerStart, AnswerEnd: Integer): RawByteString;
var
  R, C, Rc: RawByteString;
  I, LastQuery: Integer;
begin
  AnswerStart := -1;
  AnswerEnd := -1;
  if Length(Msgs) = 0 then
    raise EChatError.Create('no messages');
  R := ReasoningInstruction(Opt);
  Result := '';
  if Msgs[0].Role = crSystem then
  begin
    C := TrimChat(Msgs[0].Content);
    if C <> '' then
    begin
      if R <> '' then
        Result := ImStart + 'system'#10 + R + #10#10 + C + ImEnd + #10
      else
        Result := ImStart + 'system'#10 + C + ImEnd + #10;
    end
    else if R <> '' then
      Result := ImStart + 'system'#10 + R + ImEnd + #10;
  end
  else if R <> '' then
    Result := ImStart + 'system'#10 + R + ImEnd + #10;
  LastQuery := -1;
  for I := High(Msgs) downto 0 do
    if (Msgs[I].Role = crUser) and not IsToolResponse(TrimChat(Msgs[I].Content)) then
    begin
      LastQuery := I;
      Break;
    end;
  if LastQuery < 0 then
    raise EChatError.Create('no user query in the messages');
  for I := 0 to High(Msgs) do
  begin
    C := TrimChat(Msgs[I].Content);
    case Msgs[I].Role of
      crSystem:
        if I > 0 then
          raise EChatError.CreateFmt('message %d: a system message must be the first', [I]);
      crUser:
        Result := Result + ImStart + 'user'#10 + C + ImEnd + #10;
      crAssistant:
        begin
          Rc := TrimChat(Msgs[I].Reasoning);
          if Opt.PreserveThinking or (I > LastQuery) then
            Result := Result + ImStart + 'assistant'#10 + ThinkOpen + #10 + Rc + #10 + ThinkClose + #10#10
          else
            Result := Result + ImStart + 'assistant'#10;
          AnswerStart := Length(Result);
          Result := Result + C + ImEnd;
          AnswerEnd := Length(Result);
          Result := Result + #10;
        end;
    end;
  end;
  if Opt.AddGenerationPrompt then
  begin
    Result := Result + ImStart + 'assistant'#10;
    if Opt.EnableThinking then
      Result := Result + ThinkOpen + #10
    else
      Result := Result + ThinkOpen + #10#10 + ThinkClose + #10#10;
  end;
end;
```

`RenderTraining`: its check, then render with a copy of Opt whose
AddGenerationPrompt is False (`O := Opt; O.AddGenerationPrompt := False;
Result := Render(Msgs, O, AnswerStart, AnswerEnd)`).

`EncodeTraining` (`Text: RawByteString; A, B: Integer`):

```pascal
  Text := RenderTraining(Msgs, Opt, A, B);
  Result := Tok.Encode(Text);
  LossFrom := Length(Tok.Encode(Copy(Text, 1, A)));
  LossTo := Length(Tok.Encode(Copy(Text, 1, B)));
```

(The answer always follows a #10 and ends with the special token
`<e>`, so the pre-tokenizer splits there and both prefixes are
prefixes of the whole encoding; the ITa checks the spans.)

## Checks and messages (EChatError; in this order)

- `Render` (so `RenderChat` and `RenderTraining`): 1. no messages:
  `no messages`; 2. no last query: `no user query in the messages`; 3. in
  the loop, a system message at index I > 0:
  `CreateFmt('message %d: a system message must be the first', [I])`.
- `RenderTraining` first: no messages, or the last message is not an
  assistant message: `a training sample must end with an assistant message`.

## UT helpers

The test skeleton `tests/ut/QwenChat_test.pas` has the constants `XH`,
`LOW` and the tag constants of "Tags", and `Msgs([...])` (an array of `TChatMessage` as `TChatMessages`;
`Msgs([])` is empty). EncodeTraining rows use `NewTinyTokenizer` and
`IdsText` of `tests/ut/TokenizerFixture.pas` (T05's UT vocabulary: ids
0..255 are the bytes, 259 `hello`, 264 ` world`, 266 `<s>`, 267
`<e>` = EOS, 268 `{think}`; `{/think}` is not a token there, so it
is spelled out in bytes). Free the tokenizer in `try ... finally`.

## Unit tests (rows; every row has one method, every method belongs to a row)

In the expected texts, `\n` is #10 (write `#10` in Pascal) and the tags
are the placeholders of "Tags" (write the constants); XH and LOW are the skeleton's
constants. "D" is `DefaultChatOptions`; "D off" is D with EnableThinking
False. Text rows compare the whole string (`AssertEquals`). Error rows
catch `EChatError` and compare the whole message; a row fails if no
exception comes (`Fail('no EChatError')`).

Stage 1, helpers (001–010):

| Row | Input | Expected |
|---|---|---|
| 001 | `DefaultChatOptions` | EnableThinking True, Effort reXHigh, PreserveThinking True, AddGenerationPrompt True |
| 002 | `ChatMsg(crUser, 'c')` | Role crUser, Content `'c'`, Reasoning `''` |
| 003 | `ChatMsg(crAssistant, 'c', 'r')` | Role crAssistant, Content `'c'`, Reasoning `'r'` |
| 004 | `TrimChat(' '#9#10#11#12#13'a b'#13#10)` | `'a b'` |
| 005 | `TrimChat(#$E3#$80#$80'Q'#$C2#$A0)` (U+3000, Q, U+00A0) | unchanged, the same 6 bytes |
| 006 | `TrimChat('   ')` | `''` |
| 007 | `ReasoningInstruction(D)` | XH |
| 008 | D with Effort reLow | LOW |
| 009 | D with Effort reMedium | `''` |
| 010 | D with EnableThinking False and Effort reLow | `''` |

Stage 2, `RenderChat` (011–027):

| Row | Messages; options | Expected |
|---|---|---|
| 011 | user `Hello`; D | `<s>system\n` XH `<e>\n<s>user\nHello<e>\n<s>assistant\n{think}\n` |
| 012 | user `Hello`; D off | `<s>user\nHello<e>\n<s>assistant\n{think}\n\n{/think}\n\n` |
| 013 | system `S`, user `Q`; D | `<s>system\n` XH `\n\nS<e>\n<s>user\nQ<e>\n<s>assistant\n{think}\n` |
| 014 | as 013; D with Effort reMedium | `<s>system\nS<e>\n<s>user\nQ<e>\n<s>assistant\n{think}\n` |
| 015 | as 013; D with Effort reLow | `<s>system\n` LOW `\n\nS<e>\n<s>user\nQ<e>\n<s>assistant\n{think}\n` |
| 016 | system `'   '`, user `Q`; D | `<s>system\n` XH `<e>\n<s>user\nQ<e>\n<s>assistant\n{think}\n` |
| 017 | as 016; D off | `<s>user\nQ<e>\n<s>assistant\n{think}\n\n{/think}\n\n` |
| 018 | user `Q`; D off with AddGenerationPrompt False | `<s>user\nQ<e>\n` |
| 019 | user `Q1`, assistant `A1` reasoning `r1`, user `Q2`; D off | `<s>user\nQ1<e>\n<s>assistant\n{think}\nr1\n{/think}\n\nA1<e>\n<s>user\nQ2<e>\n<s>assistant\n{think}\n\n{/think}\n\n` |
| 020 | as 019; D off with PreserveThinking False | `<s>user\nQ1<e>\n<s>assistant\nA1<e>\n<s>user\nQ2<e>\n<s>assistant\n{think}\n\n{/think}\n\n` |
| 021 | user `Q`, assistant `A` reasoning `r`; D off, PreserveThinking False, AddGenerationPrompt False | `<s>user\nQ<e>\n<s>assistant\n{think}\nr\n{/think}\n\nA<e>\n` (after the last query: the think block stays) |
| 022 | user `Q`, assistant `A` reasoning `r`, user `TagTr + #10'ok'#10 + TagEndTr`; D off with PreserveThinking False | `<s>user\nQ<e>\n<s>assistant\n{think}\nr\n{/think}\n\nA<e>\n<s>user\n{tr}\nok\n{/tr}<e>\n<s>assistant\n{think}\n\n{/think}\n\n` (the tool response is not a query) |
| 023 | system `'  '#10' S '#9`, user `' '#13#10' Q '#10`, assistant `#10' A  '` reasoning `'  r '#10`; D off with AddGenerationPrompt False | `<s>system\nS<e>\n<s>user\nQ<e>\n<s>assistant\n{think}\nr\n{/think}\n\nA<e>\n` |
| 024 | `Msgs([])`; D | `no messages` |
| 025 | user `Q`, system `S`; D | `message 1: a system message must be the first` |
| 026 | system `S`; D | `no user query in the messages` |
| 027 | user `TagTr + 'x' + TagEndTr`; D | `no user query in the messages` |

Stage 3, `RenderTraining` and `EncodeTraining` (101–107):

| Row | Input | Expected |
|---|---|---|
| 101 | `RenderTraining`: user `Q`, assistant `A`; D off (AddGenerationPrompt True is ignored) | text `<s>user\nQ<e>\n<s>assistant\n{think}\n\n{/think}\n\nA<e>\n` (82 bytes); AnswerStart 70, AnswerEnd 81; `Copy(Text, AnswerStart + 1, AnswerEnd - AnswerStart)` = `'A' + TagE` |
| 102 | user `Q1`, assistant `A1` reasoning `r1`, user `Q2`, assistant `A2` reasoning `r2`; D off | text as 019 up to `<s>user\nQ2<e>\n`, then `<s>assistant\n{think}\nr2\n{/think}\n\nA2<e>\n` (172 bytes); AnswerStart 159, AnswerEnd 171; the answer `'A2' + TagE` |
| 103 | user `Q`, assistant `A` reasoning `r`; D | 320 bytes, starting `<s>system\n` XH `<e>\n`; AnswerStart 308, AnswerEnd 319; the answer `'A' + TagE` |
| 104 | user `Q`; D off | `a training sample must end with an assistant message` |
| 105 | `Msgs([])`; D off | `a training sample must end with an assistant message` |
| 106 | `EncodeTraining(NewTinyTokenizer, ..)`: user `hello`, assistant `hello world`; D off | `IdsText` = `266 117 115 101 114 10 259 267 10 266 97 115 115 105 115 116 97 110 116 10 268 10 10 60 47 116 104 105 110 107 62 10 10 259 264 267 10`; LossFrom 33, LossTo 36; `Tok.Decode(Copy(Ids, LossFrom, LossTo - LossFrom))` = `'hello world' + TagE` |
| 107 | user `hello`, assistant `world` reasoning `hello`; D | 250 ids; LossFrom 245, LossTo 249; the decoded span `'world' + TagE`; `Ids[LossTo - 1]` 267 |

Build and run (no GPU, no libtorch):
`fpc -Mobjfpc -Sh -vew -l- -Fu{proj}/src -Fu{proj}/tests/ut -FU{proj}/work/bin {proj}/tests/ut/QwenChat_test.pas -o{proj}/work/bin/QwenChat_test`
then `{proj}/work/bin/QwenChat_test --all --format=plain`.
The warnings `QwenTokenizer.pas(71,33)` / `(86,31) Comparison might be
always true` and `generics.dictionaries.inc ... abstract method` come from
T05's unit and the RTL; they are expected.

## ITa (prepared by the harness owner)

`QLORA_MODEL=<rehearsal GGUF> {proj}/tests/ita/T07-run.sh` builds
`tests/ita/T07/chat_check.pas` against `src/`, runs it from a fresh
directory outside the project on `work/oracle/T07/corpus.jsonl` and
compares with `expected.jsonl` (the bundled `llama-server`'s
`/apply-template` and `/tokenize` with the 27B template, made once by the
harness owner). It writes `{proj}/work/ita/T07/summary.tsv` (case, OK/NG,
value, limit, note; then the driver's exit code and stderr line count)
and `details.txt` (one line per record). OK when value ≤ limit. A few
seconds.

| Case | Compared | Limit |
|---|---|---|
| ITa-T07-001 | renderings different from `/apply-template`, of 14 conversations (thinking on/off, the three efforts, empty system, multi-turn with and without PreserveThinking, a tool response, white space incl. U+3000, Pascal and Japanese text) | 0 |
| ITa-T07-002 | training samples whose `EncodeTraining` ids differ from `/tokenize` of the oracle text, of 5 | 0 |
| ITa-T07-003 | training samples whose loss span is not exactly the last answer and its `<e>` (decoded, last id EOS, only `\n` after it), of 5 | 0 |
| ITa-T07-004 | conversations the template rejects that did not raise `EChatError`, of 4 | 0 |

Measured with the prototype: 0, 0, 0, 0. With the skeleton every case is
NG; mutants are caught (trimming only space, tab and newline: 001, 002;
the think block only with PreserveThinking: 001; a wrong span end: 003).

## Stages and case IDs

- Stage 1 (task T07b1): `DefaultChatOptions`, `ChatMsg`, `TrimChat`,
  `ReasoningInstruction`; rows UT-QwenChat-001 to 010.
- Stage 2 (task T07b2): `Render`, `RenderChat`, `IsToolResponse`; rows
  UT-QwenChat-011 to 027.
- Stage 3 (task T07b3): `RenderTraining`, `EncodeTraining`; rows
  UT-QwenChat-101 to 107.
- ITa: ITa-T07-001 to 004.
