{ UT program for src/QwenChat.pas: the rows UT-QwenChat-NNN of
  tests/ut/cases/QwenChat.cases.md (stage 1 from 001: DefaultChatOptions,
  ChatMsg, TrimChat, ReasoningInstruction; stage 2 from 011: RenderChat;
  stage 3 from 101: RenderTraining, EncodeTraining). EncodeTraining rows use
  the UT vocabulary of tests/ut/TokenizerFixture.pas (NewTinyTokenizer,
  IdsText). No GGUF and no libtorch are needed.
  Invoked with `--all --format=plain` (see .sdlc/ut-coverage.sh).

  Skeleton by the harness owner; the test methods are written in T07b. }
program QwenChat_test;

{$mode objfpc}{$H+}

uses
  SysUtils, QwenTokenizer, QwenChat, TokenizerFixture,
  fpcunit, testregistry, consoletestrunner;

const
  { The template's two reasoning sentences (the facts' XH and LOW). }
  XH = 'Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.';
  LOW = 'Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.';
  { The template's tags, built from pieces: never write them literally
    (GLM 5 and Qwen read them as their own special tokens). In the facts
    they are <s>, <e>, {think}, {/think}, {tr} and {/tr}. }
  TagS = '<|' + 'im_start|>';
  TagE = '<|' + 'im_end|>';
  TagThink = '<' + 'think>';
  TagEndThink = '</' + 'think>';
  TagTr = '<' + 'tool_response>';
  TagEndTr = '</' + 'tool_response>';

type
  TQwenChatTest = class(TTestCase)
  public
    { The messages as a TChatMessages (Msgs([]) is the empty array). }
    class function Msgs(const A: array of TChatMessage): TChatMessages; static;
  published
    procedure TestUT_QwenChat_001;
    procedure TestUT_QwenChat_002;
    procedure TestUT_QwenChat_003;
    procedure TestUT_QwenChat_004;
    procedure TestUT_QwenChat_005;
    procedure TestUT_QwenChat_006;
    procedure TestUT_QwenChat_007;
    procedure TestUT_QwenChat_008;
    procedure TestUT_QwenChat_009;
    procedure TestUT_QwenChat_010;
    procedure TestUT_QwenChat_011;
    procedure TestUT_QwenChat_012;
    procedure TestUT_QwenChat_013;
    procedure TestUT_QwenChat_014;
    procedure TestUT_QwenChat_015;
    procedure TestUT_QwenChat_016;
    procedure TestUT_QwenChat_017;
    procedure TestUT_QwenChat_018;
    procedure TestUT_QwenChat_019;
    procedure TestUT_QwenChat_020;
    procedure TestUT_QwenChat_021;
    procedure TestUT_QwenChat_022;
    procedure TestUT_QwenChat_023;
    procedure TestUT_QwenChat_024;
    procedure TestUT_QwenChat_025;
    procedure TestUT_QwenChat_026;
    procedure TestUT_QwenChat_027;
    procedure TestUT_QwenChat_101;
    procedure TestUT_QwenChat_102;
    procedure TestUT_QwenChat_103;
    procedure TestUT_QwenChat_104;
    procedure TestUT_QwenChat_105;
    procedure TestUT_QwenChat_106;
    procedure TestUT_QwenChat_107;
  end;

class function TQwenChatTest.Msgs(const A: array of TChatMessage): TChatMessages;
var
  I: Integer;
begin
  Result := nil;
  SetLength(Result, Length(A));
  for I := 0 to High(A) do
    Result[I] := A[I];
end;

procedure TQwenChatTest.TestUT_QwenChat_001;
var
  D: TChatOptions;
begin
  D := DefaultChatOptions;
  AssertTrue((D.EnableThinking = True)
    and (D.Effort = reXHigh)
    and (D.PreserveThinking = True)
    and (D.AddGenerationPrompt = True));
end;

procedure TQwenChatTest.TestUT_QwenChat_002;
var
  M: TChatMessage;
begin
  M := ChatMsg(crUser, 'c');
  AssertTrue(M.Role = crUser);
  AssertEquals('002 Content', 'c', M.Content);
  AssertEquals('002 Reasoning', '', M.Reasoning);
end;

procedure TQwenChatTest.TestUT_QwenChat_003;
var
  M: TChatMessage;
begin
  M := ChatMsg(crAssistant, 'c', 'r');
  AssertTrue(M.Role = crAssistant);
  AssertEquals('003 Content', 'c', M.Content);
  AssertEquals('003 Reasoning', 'r', M.Reasoning);
end;

procedure TQwenChatTest.TestUT_QwenChat_004;
begin
  AssertEquals('004', 'a b', TrimChat(' '#9#10#11#12#13'a b'#13#10));
end;

procedure TQwenChatTest.TestUT_QwenChat_005;
var
  S: RawByteString;
begin
  { U+3000, Q, U+00A0: the same six bytes, none of the six chat spaces. }
  S := #$E3#$80#$80'Q'#$C2#$A0;
  AssertEquals('005 6 bytes', 6, Length(S));
  AssertEquals('005 unchanged', S, TrimChat(S));
end;

procedure TQwenChatTest.TestUT_QwenChat_006;
begin
  AssertEquals('006', '', TrimChat('   '));
end;

procedure TQwenChatTest.TestUT_QwenChat_007;
begin
  AssertEquals('007', XH, ReasoningInstruction(DefaultChatOptions));
end;

procedure TQwenChatTest.TestUT_QwenChat_008;
var
  D: TChatOptions;
begin
  D := DefaultChatOptions;
  D.Effort := reLow;
  AssertEquals('008', LOW, ReasoningInstruction(D));
end;

procedure TQwenChatTest.TestUT_QwenChat_009;
var
  D: TChatOptions;
begin
  D := DefaultChatOptions;
  D.Effort := reMedium;
  AssertEquals('009', '', ReasoningInstruction(D));
end;

procedure TQwenChatTest.TestUT_QwenChat_010;
var
  D: TChatOptions;
begin
  D := DefaultChatOptions;
  D.EnableThinking := False;
  D.Effort := reLow;
  AssertEquals('010', '', ReasoningInstruction(D));
end;

procedure TQwenChatTest.TestUT_QwenChat_011;
begin
  { user only, thinking on (D). }
  AssertEquals('011',
    TagS + 'system'#10 + XH + TagE + #10
    + TagS + 'user'#10 + 'Hello' + TagE + #10
    + TagS + 'assistant'#10 + TagThink + #10,
    RenderChat(Msgs([ChatMsg(crUser, 'Hello')]), DefaultChatOptions));
end;

procedure TQwenChatTest.TestUT_QwenChat_012;
var
  D: TChatOptions;
begin
  { user only, thinking off (D off). }
  D := DefaultChatOptions;
  D.EnableThinking := False;
  AssertEquals('012',
    TagS + 'user'#10 + 'Hello' + TagE + #10
    + TagS + 'assistant'#10 + TagThink + #10#10 + TagEndThink + #10#10,
    RenderChat(Msgs([ChatMsg(crUser, 'Hello')]), D));
end;

procedure TQwenChatTest.TestUT_QwenChat_013;
begin
  { system + user, thinking on (D). }
  AssertEquals('013',
    TagS + 'system'#10 + XH + #10#10 + 'S' + TagE + #10
    + TagS + 'user'#10 + 'Q' + TagE + #10
    + TagS + 'assistant'#10 + TagThink + #10,
    RenderChat(Msgs([ChatMsg(crSystem, 'S'), ChatMsg(crUser, 'Q')]), DefaultChatOptions));
end;

procedure TQwenChatTest.TestUT_QwenChat_014;
var
  D: TChatOptions;
begin
  { system + user, D with Effort reMedium (no instruction sentence). }
  D := DefaultChatOptions;
  D.Effort := reMedium;
  AssertEquals('014',
    TagS + 'system'#10 + 'S' + TagE + #10
    + TagS + 'user'#10 + 'Q' + TagE + #10
    + TagS + 'assistant'#10 + TagThink + #10,
    RenderChat(Msgs([ChatMsg(crSystem, 'S'), ChatMsg(crUser, 'Q')]), D));
end;

procedure TQwenChatTest.TestUT_QwenChat_015;
var
  D: TChatOptions;
begin
  { system + user, D with Effort reLow. }
  D := DefaultChatOptions;
  D.Effort := reLow;
  AssertEquals('015',
    TagS + 'system'#10 + LOW + #10#10 + 'S' + TagE + #10
    + TagS + 'user'#10 + 'Q' + TagE + #10
    + TagS + 'assistant'#10 + TagThink + #10,
    RenderChat(Msgs([ChatMsg(crSystem, 'S'), ChatMsg(crUser, 'Q')]), D));
end;

procedure TQwenChatTest.TestUT_QwenChat_016;
begin
  { system content '   ' (trimmed empty), thinking on (D). }
  AssertEquals('016',
    TagS + 'system'#10 + XH + TagE + #10
    + TagS + 'user'#10 + 'Q' + TagE + #10
    + TagS + 'assistant'#10 + TagThink + #10,
    RenderChat(Msgs([ChatMsg(crSystem, '   '), ChatMsg(crUser, 'Q')]), DefaultChatOptions));
end;

procedure TQwenChatTest.TestUT_QwenChat_017;
var
  D: TChatOptions;
begin
  { system content '   ' (trimmed empty), thinking off (D off). }
  D := DefaultChatOptions;
  D.EnableThinking := False;
  AssertEquals('017',
    TagS + 'user'#10 + 'Q' + TagE + #10
    + TagS + 'assistant'#10 + TagThink + #10#10 + TagEndThink + #10#10,
    RenderChat(Msgs([ChatMsg(crSystem, '   '), ChatMsg(crUser, 'Q')]), D));
end;

procedure TQwenChatTest.TestUT_QwenChat_018;
var
  D: TChatOptions;
begin
  { user only, D off with AddGenerationPrompt False. }
  D := DefaultChatOptions;
  D.EnableThinking := False;
  D.AddGenerationPrompt := False;
  AssertEquals('018',
    TagS + 'user'#10 + 'Q' + TagE + #10,
    RenderChat(Msgs([ChatMsg(crUser, 'Q')]), D));
end;

procedure TQwenChatTest.TestUT_QwenChat_019;
var
  D: TChatOptions;
begin
  { multi-turn, D off (PreserveThinking True keeps the think block). }
  D := DefaultChatOptions;
  D.EnableThinking := False;
  AssertEquals('019',
    TagS + 'user'#10 + 'Q1' + TagE + #10
    + TagS + 'assistant'#10 + TagThink + #10 + 'r1' + #10 + TagEndThink + #10#10 + 'A1' + TagE + #10
    + TagS + 'user'#10 + 'Q2' + TagE + #10
    + TagS + 'assistant'#10 + TagThink + #10#10 + TagEndThink + #10#10,
    RenderChat(Msgs([ChatMsg(crUser, 'Q1'),
                     ChatMsg(crAssistant, 'A1', 'r1'),
                     ChatMsg(crUser, 'Q2')]), D));
end;

procedure TQwenChatTest.TestUT_QwenChat_020;
var
  D: TChatOptions;
begin
  { multi-turn, D off with PreserveThinking False (think block dropped). }
  D := DefaultChatOptions;
  D.EnableThinking := False;
  D.PreserveThinking := False;
  AssertEquals('020',
    TagS + 'user'#10 + 'Q1' + TagE + #10
    + TagS + 'assistant'#10 + 'A1' + TagE + #10
    + TagS + 'user'#10 + 'Q2' + TagE + #10
    + TagS + 'assistant'#10 + TagThink + #10#10 + TagEndThink + #10#10,
    RenderChat(Msgs([ChatMsg(crUser, 'Q1'),
                     ChatMsg(crAssistant, 'A1', 'r1'),
                     ChatMsg(crUser, 'Q2')]), D));
end;

procedure TQwenChatTest.TestUT_QwenChat_021;
var
  D: TChatOptions;
begin
  { answer after the last query: D off, PreserveThinking False,
    AddGenerationPrompt False (the think block stays). }
  D := DefaultChatOptions;
  D.EnableThinking := False;
  D.PreserveThinking := False;
  D.AddGenerationPrompt := False;
  AssertEquals('021',
    TagS + 'user'#10 + 'Q' + TagE + #10
    + TagS + 'assistant'#10 + TagThink + #10 + 'r' + #10 + TagEndThink + #10#10 + 'A' + TagE + #10,
    RenderChat(Msgs([ChatMsg(crUser, 'Q'),
                     ChatMsg(crAssistant, 'A', 'r')]), D));
end;

procedure TQwenChatTest.TestUT_QwenChat_022;
var
  D: TChatOptions;
begin
  { the tool response is not a query (IsToolResponse): LastQuery stays 0. }
  D := DefaultChatOptions;
  D.EnableThinking := False;
  D.PreserveThinking := False;
  AssertEquals('022',
    TagS + 'user'#10 + 'Q' + TagE + #10
    + TagS + 'assistant'#10 + TagThink + #10 + 'r' + #10 + TagEndThink + #10#10 + 'A' + TagE + #10
    + TagS + 'user'#10 + TagTr + #10 + 'ok' + #10 + TagEndTr + TagE + #10
    + TagS + 'assistant'#10 + TagThink + #10#10 + TagEndThink + #10#10,
    RenderChat(Msgs([ChatMsg(crUser, 'Q'),
                     ChatMsg(crAssistant, 'A', 'r'),
                     ChatMsg(crUser, TagTr + #10'ok'#10 + TagEndTr)]), D));
end;

procedure TQwenChatTest.TestUT_QwenChat_023;
var
  D: TChatOptions;
begin
  { trimming in every role: D off with AddGenerationPrompt False. }
  D := DefaultChatOptions;
  D.EnableThinking := False;
  D.AddGenerationPrompt := False;
  AssertEquals('023',
    TagS + 'system'#10 + 'S' + TagE + #10
    + TagS + 'user'#10 + 'Q' + TagE + #10
    + TagS + 'assistant'#10 + TagThink + #10 + 'r' + #10 + TagEndThink + #10#10 + 'A' + TagE + #10,
    RenderChat(Msgs([ChatMsg(crSystem, '  '#10' S '#9),
                     ChatMsg(crUser, ' '#13#10' Q '#10),
                     ChatMsg(crAssistant, #10' A  ', '  r '#10)]), D));
end;

procedure TQwenChatTest.TestUT_QwenChat_024;
begin
  { no messages. }
  try
    RenderChat(Msgs([]), DefaultChatOptions);
    Fail('no EChatError');
  except
    on E: EChatError do
      AssertEquals('024', 'no messages', E.Message);
  end;
end;

procedure TQwenChatTest.TestUT_QwenChat_025;
begin
  { a system message at index 1. }
  try
    RenderChat(Msgs([ChatMsg(crUser, 'Q'), ChatMsg(crSystem, 'S')]), DefaultChatOptions);
    Fail('no EChatError');
  except
    on E: EChatError do
      AssertEquals('025', 'message 1: a system message must be the first', E.Message);
  end;
end;

procedure TQwenChatTest.TestUT_QwenChat_026;
begin
  { system only: no user query. }
  try
    RenderChat(Msgs([ChatMsg(crSystem, 'S')]), DefaultChatOptions);
    Fail('no EChatError');
  except
    on E: EChatError do
      AssertEquals('026', 'no user query in the messages', E.Message);
  end;
end;

procedure TQwenChatTest.TestUT_QwenChat_027;
begin
  { only a tool response: IsToolResponse counts it as a tool response. }
  try
    RenderChat(Msgs([ChatMsg(crUser, TagTr + 'x' + TagEndTr)]), DefaultChatOptions);
    Fail('no EChatError');
  except
    on E: EChatError do
      AssertEquals('027', 'no user query in the messages', E.Message);
  end;
end;

procedure TQwenChatTest.TestUT_QwenChat_101;
var
  D: TChatOptions;
  Text: RawByteString;
  A, B: Integer;
begin
  { training sample, D off: the generation prompt (AddGenerationPrompt True)
    is ignored; the span is the last answer and its end tag. }
  D := DefaultChatOptions;
  D.EnableThinking := False;
  Text := RenderTraining(Msgs([ChatMsg(crUser, 'Q'), ChatMsg(crAssistant, 'A')]), D, A, B);
  AssertEquals('101 text',
    TagS + 'user'#10 + 'Q' + TagE + #10
    + TagS + 'assistant'#10 + TagThink + #10#10 + TagEndThink + #10#10 + 'A' + TagE + #10,
    Text);
  AssertEquals('101 82 bytes', 82, Length(Text));
  AssertEquals('101 AnswerStart', 70, A);
  AssertEquals('101 AnswerEnd', 81, B);
  AssertEquals('101 answer', 'A' + TagE, Copy(Text, A + 1, B - A));
end;

procedure TQwenChatTest.TestUT_QwenChat_102;
var
  D: TChatOptions;
  Text: RawByteString;
  A, B: Integer;
begin
  { training sample, multi-turn, D off: the span is the last answer
    ('A2' + TagE), not the first ('A1'). }
  D := DefaultChatOptions;
  D.EnableThinking := False;
  Text := RenderTraining(Msgs([ChatMsg(crUser, 'Q1'), ChatMsg(crAssistant, 'A1', 'r1'),
                               ChatMsg(crUser, 'Q2'), ChatMsg(crAssistant, 'A2', 'r2')]), D, A, B);
  AssertEquals('102 text',
    TagS + 'user'#10 + 'Q1' + TagE + #10
    + TagS + 'assistant'#10 + TagThink + #10 + 'r1' + #10 + TagEndThink + #10#10 + 'A1' + TagE + #10
    + TagS + 'user'#10 + 'Q2' + TagE + #10
    + TagS + 'assistant'#10 + TagThink + #10 + 'r2' + #10 + TagEndThink + #10#10 + 'A2' + TagE + #10,
    Text);
  AssertEquals('102 172 bytes', 172, Length(Text));
  AssertEquals('102 AnswerStart', 159, A);
  AssertEquals('102 AnswerEnd', 171, B);
  AssertEquals('102 answer', 'A2' + TagE, Copy(Text, A + 1, B - A));
end;

procedure TQwenChatTest.TestUT_QwenChat_103;
var
  Text: RawByteString;
  A, B: Integer;
begin
  { training sample, thinking on (D): the system turn carries the
    reasoning-effort sentence. }
  Text := RenderTraining(Msgs([ChatMsg(crUser, 'Q'), ChatMsg(crAssistant, 'A', 'r')]),
    DefaultChatOptions, A, B);
  AssertEquals('103 text',
    TagS + 'system'#10 + XH + TagE + #10
    + TagS + 'user'#10 + 'Q' + TagE + #10
    + TagS + 'assistant'#10 + TagThink + #10 + 'r' + #10 + TagEndThink + #10#10 + 'A' + TagE + #10,
    Text);
  AssertEquals('103 320 bytes', 320, Length(Text));
  AssertEquals('103 AnswerStart', 308, A);
  AssertEquals('103 AnswerEnd', 319, B);
  AssertEquals('103 answer', 'A' + TagE, Copy(Text, A + 1, B - A));
end;

procedure TQwenChatTest.TestUT_QwenChat_104;
var
  D: TChatOptions;
  A, B: Integer;
begin
  { the sample ends with a user message, not an assistant message. }
  D := DefaultChatOptions;
  D.EnableThinking := False;
  try
    RenderTraining(Msgs([ChatMsg(crUser, 'Q')]), D, A, B);
    Fail('no EChatError');
  except
    on E: EChatError do
      AssertEquals('104', 'a training sample must end with an assistant message', E.Message);
  end;
end;

procedure TQwenChatTest.TestUT_QwenChat_105;
var
  D: TChatOptions;
  A, B: Integer;
begin
  { the sample has no messages at all. }
  D := DefaultChatOptions;
  D.EnableThinking := False;
  try
    RenderTraining(Msgs([]), D, A, B);
    Fail('no EChatError');
  except
    on E: EChatError do
      AssertEquals('105', 'a training sample must end with an assistant message', E.Message);
  end;
end;

procedure TQwenChatTest.TestUT_QwenChat_106;
var
  Tok: TQwenTokenizer;
  D: TChatOptions;
  Ids: TTokenArray;
  LossFrom, LossTo: Integer;
begin
  { encode the training sample (D off): the loss span is the last answer
    and its end tag, id for id. }
  D := DefaultChatOptions;
  D.EnableThinking := False;
  Tok := NewTinyTokenizer;
  try
    Ids := EncodeTraining(Tok, Msgs([ChatMsg(crUser, 'hello'),
        ChatMsg(crAssistant, 'hello world')]), D, LossFrom, LossTo);
    AssertEquals('106 IdsText',
      '266 117 115 101 114 10 259 267 10 266 97 115 115 105 115 116 97 110 116 10 268 10 10 60 47 116 104 105 110 107 62 10 10 259 264 267 10',
      IdsText(Ids));
    AssertEquals('106 LossFrom', 33, LossFrom);
    AssertEquals('106 LossTo', 36, LossTo);
    AssertEquals('106 span', 'hello world' + TagE, Tok.Decode(Copy(Ids, LossFrom, LossTo - LossFrom)));
  finally
    Tok.Free;
  end;
end;

procedure TQwenChatTest.TestUT_QwenChat_107;
var
  Tok: TQwenTokenizer;
  Ids: TTokenArray;
  LossFrom, LossTo: Integer;
begin
  { encode with thinking on (D, xhigh effort): the instruction sentence
    dominates the id count. }
  Tok := NewTinyTokenizer;
  try
    Ids := EncodeTraining(Tok, Msgs([ChatMsg(crUser, 'hello'),
        ChatMsg(crAssistant, 'world', 'hello')]), DefaultChatOptions, LossFrom, LossTo);
    AssertEquals('107 250 ids', 250, Length(Ids));
    AssertEquals('107 LossFrom', 245, LossFrom);
    AssertEquals('107 LossTo', 249, LossTo);
    AssertEquals('107 span', 'world' + TagE, Tok.Decode(Copy(Ids, LossFrom, LossTo - LossFrom)));
    AssertEquals('107 last id of the span is the end tag', 267, Ids[LossTo - 1]);
  finally
    Tok.Free;
  end;
end;

var
  App: TTestRunner;
begin
  RegisterTest(TQwenChatTest);
  App := TTestRunner.Create(nil);
  App.Initialize;
  App.Title := 'QwenChat UT';
  App.Run;
  App.Free;
end.
