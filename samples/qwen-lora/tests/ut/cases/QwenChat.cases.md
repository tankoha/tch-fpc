# UT cases — unit `QwenChat` (T07)

Legend: D = `DefaultChatOptions`; XH / LOW are the reasoning-instruction
constants of the test skeleton; TagS, TagE, TagThink, TagEndThink, TagTr,
TagEndTr are the tag constants (never written literally); `#10` is the
newline. Text rows compare the whole string (`AssertEquals`); error rows
catch `EChatError` and compare the whole message, and fail with
`Fail('no EChatError')` if no exception comes.

| Case ID | Title | Target function | Input (short) | Expected result | Notes |
|---|---|---|---|---|---|
| UT-QwenChat-001 | default options | DefaultChatOptions | `DefaultChatOptions` | EnableThinking True, Effort reXHigh, PreserveThinking True, AddGenerationPrompt True | |
| UT-QwenChat-002 | user record | ChatMsg | `ChatMsg(crUser, 'c')` | Role crUser, Content `'c'`, Reasoning `''` | |
| UT-QwenChat-003 | assistant record with reasoning | ChatMsg | `ChatMsg(crAssistant, 'c', 'r')` | Role crAssistant, Content `'c'`, Reasoning `'r'` | |
| UT-QwenChat-004 | trim all six chat spaces | TrimChat | `TrimChat(' '#9#10#11#12#13'a b'#13#10)` | `'a b'` | |
| UT-QwenChat-005 | trim leaves U+3000 and U+00A0 | TrimChat | `TrimChat(#$E3#$80#$80'Q'#$C2#$A0)` (U+3000, Q, U+00A0) | unchanged, the same 6 bytes | |
| UT-QwenChat-006 | trim all-whitespace | TrimChat | `TrimChat('   ')` | `''` | |
| UT-QwenChat-007 | instruction, XHigh | ReasoningInstruction | `ReasoningInstruction(D)` | XH | |
| UT-QwenChat-008 | instruction, Low | ReasoningInstruction | D with Effort reLow | LOW | |
| UT-QwenChat-009 | instruction, Medium | ReasoningInstruction | D with Effort reMedium | `''` | |
| UT-QwenChat-010 | instruction, thinking off | ReasoningInstruction | D with EnableThinking False and Effort reLow | `''` | |
| UT-QwenChat-011 | user only, thinking on | RenderChat | user `Hello`; D | TagS + 'system'#10 + XH + TagE + #10 + TagS + 'user'#10 + 'Hello' + TagE + #10 + TagS + 'assistant'#10 + TagThink + #10 | |
| UT-QwenChat-012 | user only, thinking off | RenderChat | user `Hello`; D off | TagS + 'user'#10 + 'Hello' + TagE + #10 + TagS + 'assistant'#10 + TagThink + #10#10 + TagEndThink + #10#10 | |
| UT-QwenChat-013 | system + user, thinking on | RenderChat | system `S`, user `Q`; D | TagS + 'system'#10 + XH + #10#10 + 'S' + TagE + #10 + TagS + 'user'#10 + 'Q' + TagE + #10 + TagS + 'assistant'#10 + TagThink + #10 | |
| UT-QwenChat-014 | system + user, Medium | RenderChat | as 013; D with Effort reMedium | TagS + 'system'#10 + 'S' + TagE + #10 + TagS + 'user'#10 + 'Q' + TagE + #10 + TagS + 'assistant'#10 + TagThink + #10 | |
| UT-QwenChat-015 | system + user, Low | RenderChat | as 013; D with Effort reLow | TagS + 'system'#10 + LOW + #10#10 + 'S' + TagE + #10 + TagS + 'user'#10 + 'Q' + TagE + #10 + TagS + 'assistant'#10 + TagThink + #10 | |
| UT-QwenChat-016 | empty system content, thinking on | RenderChat | system `'   '`, user `Q`; D | TagS + 'system'#10 + XH + TagE + #10 + TagS + 'user'#10 + 'Q' + TagE + #10 + TagS + 'assistant'#10 + TagThink + #10 | |
| UT-QwenChat-017 | empty system content, thinking off | RenderChat | as 016; D off | TagS + 'user'#10 + 'Q' + TagE + #10 + TagS + 'assistant'#10 + TagThink + #10#10 + TagEndThink + #10#10 | |
| UT-QwenChat-018 | no generation prompt | RenderChat | user `Q`; D off with AddGenerationPrompt False | TagS + 'user'#10 + 'Q' + TagE + #10 | |
| UT-QwenChat-019 | multi-turn, PreserveThinking | RenderChat | user `Q1`, assistant `A1` reasoning `r1`, user `Q2`; D off | TagS + 'user'#10 + 'Q1' + TagE + #10 + TagS + 'assistant'#10 + TagThink + #10 + 'r1' + #10 + TagEndThink + #10#10 + 'A1' + TagE + #10 + TagS + 'user'#10 + 'Q2' + TagE + #10 + TagS + 'assistant'#10 + TagThink + #10#10 + TagEndThink + #10#10 | |
| UT-QwenChat-020 | multi-turn, without PreserveThinking | RenderChat | as 019; D off with PreserveThinking False | TagS + 'user'#10 + 'Q1' + TagE + #10 + TagS + 'assistant'#10 + 'A1' + TagE + #10 + TagS + 'user'#10 + 'Q2' + TagE + #10 + TagS + 'assistant'#10 + TagThink + #10#10 + TagEndThink + #10#10 | |
| UT-QwenChat-021 | answer after the last query | RenderChat | user `Q`, assistant `A` reasoning `r`; D off, PreserveThinking False, AddGenerationPrompt False | TagS + 'user'#10 + 'Q' + TagE + #10 + TagS + 'assistant'#10 + TagThink + #10 + 'r' + #10 + TagEndThink + #10#10 + 'A' + TagE + #10 | after the last query: the think block stays |
| UT-QwenChat-022 | tool response is not a query | RenderChat | user `Q`, assistant `A` reasoning `r`, user `TagTr + #10'ok'#10 + TagEndTr`; D off with PreserveThinking False | TagS + 'user'#10 + 'Q' + TagE + #10 + TagS + 'assistant'#10 + TagThink + #10 + 'r' + #10 + TagEndThink + #10#10 + 'A' + TagE + #10 + TagS + 'user'#10 + TagTr + #10 + 'ok' + #10 + TagEndTr + TagE + #10 + TagS + 'assistant'#10 + TagThink + #10#10 + TagEndThink + #10#10 | the tool response is not a query (IsToolResponse) |
| UT-QwenChat-023 | trimming in every role | RenderChat | system `'  '#10' S '#9`, user `' '#13#10' Q '#10`, assistant `#10' A  '` reasoning `'  r '#10`; D off with AddGenerationPrompt False | TagS + 'system'#10 + 'S' + TagE + #10 + TagS + 'user'#10 + 'Q' + TagE + #10 + TagS + 'assistant'#10 + TagThink + #10 + 'r' + #10 + TagEndThink + #10#10 + 'A' + TagE + #10 | |
| UT-QwenChat-024 | no messages | RenderChat | `Msgs([])`; D | EChatError `no messages` | whole message compared; `Fail('no EChatError')` if no exception |
| UT-QwenChat-025 | system message not first | RenderChat | user `Q`, system `S`; D | EChatError `message 1: a system message must be the first` | whole message compared |
| UT-QwenChat-026 | no user query | RenderChat | system `S`; D | EChatError `no user query in the messages` | whole message compared |
| UT-QwenChat-027 | only a tool response | RenderChat | user `TagTr + 'x' + TagEndTr`; D | EChatError `no user query in the messages` | IsToolResponse counts it as a tool response |
| UT-QwenChat-101 | training sample, D off | RenderTraining | user `Q`, assistant `A`; D off (AddGenerationPrompt True is ignored) | text TagS + 'user'#10 + 'Q' + TagE + #10 + TagS + 'assistant'#10 + TagThink + #10#10 + TagEndThink + #10#10 + 'A' + TagE + #10 (82 bytes); AnswerStart 70, AnswerEnd 81; `Copy(Text, AnswerStart + 1, AnswerEnd - AnswerStart)` = `'A' + TagE` | |
| UT-QwenChat-102 | training sample, multi-turn | RenderTraining | user `Q1`, assistant `A1` reasoning `r1`, user `Q2`, assistant `A2` reasoning `r2`; D off | text as 019 up to TagS + 'user'#10 + 'Q2' + TagE + #10, then TagS + 'assistant'#10 + TagThink + #10 + 'r2' + #10 + TagEndThink + #10#10 + 'A2' + TagE + #10 (172 bytes); AnswerStart 159, AnswerEnd 171; the answer `'A2' + TagE` | |
| UT-QwenChat-103 | training sample, thinking on | RenderTraining | user `Q`, assistant `A` reasoning `r`; D | 320 bytes, starting TagS + 'system'#10 + XH + TagE + #10; AnswerStart 308, AnswerEnd 319; the answer `'A' + TagE` | |
| UT-QwenChat-104 | sample does not end with assistant | RenderTraining | user `Q`; D off | EChatError `a training sample must end with an assistant message` | whole message compared |
| UT-QwenChat-105 | training with no messages | RenderTraining | `Msgs([])`; D off | EChatError `a training sample must end with an assistant message` | whole message compared |
| UT-QwenChat-106 | encode, thinking off | EncodeTraining | `EncodeTraining(NewTinyTokenizer, ..)`: user `hello`, assistant `hello world`; D off | `IdsText` = 266 117 115 101 114 10 259 267 10 266 97 115 115 105 115 116 97 110 116 10 268 10 10 60 47 116 104 105 110 107 62 10 10 259 264 267 10; LossFrom 33, LossTo 36; `Tok.Decode(Copy(Ids, LossFrom, LossTo - LossFrom))` = `'hello world' + TagE` | `NewTinyTokenizer` and `IdsText` of `tests/ut/TokenizerFixture.pas` (T05's UT vocabulary) |
| UT-QwenChat-107 | encode, thinking on | EncodeTraining | user `hello`, assistant `world` reasoning `hello`; D | 250 ids; LossFrom 245, LossTo 249; the decoded span `'world' + TagE`; `Ids[LossTo - 1]` 267 | same tokenizer fixture as 106; free the tokenizer in `try ... finally` |
