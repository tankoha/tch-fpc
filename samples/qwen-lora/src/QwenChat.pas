{ T07: the Qwen3.8 chat template for text turns (DESIGN 8.10 "Chat
  template", 8.14 "Loss mask").

  Renders system, user and assistant messages exactly as the qwen3.8:27b
  GGUF's Jinja template does (tools and images are out of scope): the
  reasoning-effort sentence in the system turn while thinking is on, the
  think block of assistant turns (kept for every assistant turn while
  PreserveThinking, else only after the last user query), the generation
  prompt with thinking on or off. Contents are trimmed of ASCII white
  space only. RenderTraining / EncodeTraining give a training sample and
  the span the loss covers: the last assistant answer and its end-of-turn tag.

  Interface by the harness owner (tasks/T07-facts.md); the bodies are
  written in T07b. }

unit QwenChat;

{$mode objfpc}{$H+}

interface

uses
  SysUtils, QwenTokenizer;

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

implementation

const
  { The template's tags, built from pieces: an agent must never write them
    literally (GLM 5 and Qwen read them as their own special tokens). }
  ImStart = '<|' + 'im_start|>';
  ImEnd = '<|' + 'im_end|>';
  ThinkOpen = '<' + 'think>';
  ThinkClose = '</' + 'think>';
  ToolRespOpen = '<' + 'tool_response>';
  ToolRespClose = '</' + 'tool_response>';
  { The two reasoning sentences of the template. }
  XHighText = 'Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.';
  LowText = 'Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.';

function DefaultChatOptions: TChatOptions;
begin
  Result := Default(TChatOptions);
  Result.EnableThinking := True;
  Result.Effort := reXHigh;
  Result.PreserveThinking := True;
  Result.AddGenerationPrompt := True;
end;

function ChatMsg(ARole: TChatRole; const AContent: RawByteString;
  const AReasoning: RawByteString): TChatMessage;
begin
  Result := Default(TChatMessage);
  Result.Role := ARole;
  Result.Content := AContent;
  Result.Reasoning := AReasoning;
end;

function IsChatSpace(C: AnsiChar): Boolean;
begin
  Result := C in [' ', #9, #10, #11, #12, #13];
end;

function TrimChat(const S: RawByteString): RawByteString;
var
  Lo, Hi: Integer;
begin
  Lo := 1;
  Hi := Length(S);
  while (Lo <= Hi) and IsChatSpace(S[Lo]) do
    Inc(Lo);
  while (Lo <= Hi) and IsChatSpace(S[Hi]) do
    Dec(Hi);
  if Lo > Hi then
    Result := ''
  else
    Result := Copy(S, Lo, Hi - Lo + 1);
end;

function ReasoningInstruction(const Opt: TChatOptions): RawByteString;
begin
  if not Opt.EnableThinking then
  begin
    Result := '';
    Exit;
  end;
  case Opt.Effort of
    reXHigh: Result := XHighText;
    reLow: Result := LowText;
    reMedium: Result := '';
  end;
end;

function IsToolResponse(const C: RawByteString): Boolean;
begin
  Result := (Copy(C, 1, Length(ToolRespOpen)) = ToolRespOpen)
    and (Copy(C, Length(C) - Length(ToolRespClose) + 1, Length(ToolRespClose))
         = ToolRespClose);
end;

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

function RenderChat(const Msgs: TChatMessages; const Opt: TChatOptions): RawByteString;
var
  A, B: Integer;
begin
  Result := Render(Msgs, Opt, A, B);
end;

function RenderTraining(const Msgs: TChatMessages; const Opt: TChatOptions;
  out AnswerStart, AnswerEnd: Integer): RawByteString;
var
  O: TChatOptions;
begin
  if (Length(Msgs) = 0) or (Msgs[High(Msgs)].Role <> crAssistant) then
    raise EChatError.Create('a training sample must end with an assistant message');
  O := Opt;
  O.AddGenerationPrompt := False;
  Result := Render(Msgs, O, AnswerStart, AnswerEnd);
end;

function EncodeTraining(Tok: TQwenTokenizer; const Msgs: TChatMessages;
  const Opt: TChatOptions; out LossFrom, LossTo: Integer): TTokenArray;
var
  Text: RawByteString;
  A, B: Integer;
begin
  Text := RenderTraining(Msgs, Opt, A, B);
  Result := Tok.Encode(Text);
  LossFrom := Length(Tok.Encode(Copy(Text, 1, A)));
  LossTo := Length(Tok.Encode(Copy(Text, 1, B)));
end;

end.
