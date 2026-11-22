{ ITa driver of T07 (harness owner's): QwenChat against the bundled
  llama-server's /apply-template with the 27B model's chat template
  (work/oracle/T07/expected.jsonl, made once by the harness owner).
  Usage: chat_check <rehearsal GGUF> <corpus.jsonl> <expected.jsonl>
  Writes one detail line per record ("# id<TAB>OK|NG<TAB>what"), then one
  line per case:
  id, value, limit, note (OK when value <= limit). }
program chat_check;

{$mode objfpc}{$H+}

uses
  Classes, SysUtils, fpjson, jsonparser, GgufFile, QwenTokenizer, QwenChat;

function ReadLines(const AFile: string): TStringList;
begin
  Result := TStringList.Create;
  Result.LoadFromFile(AFile);
end;

function Options(O: TJSONObject; Gen: Boolean): TChatOptions;
var
  E: string;
begin
  Result := DefaultChatOptions;
  Result.EnableThinking := O.Get('enable_thinking', True);
  Result.PreserveThinking := O.Get('preserve_thinking', True);
  E := O.Get('reasoning_effort', 'xhigh');
  if E = 'medium' then
    Result.Effort := reMedium
  else if E = 'low' then
    Result.Effort := reLow
  else
    Result.Effort := reXHigh;
  Result.AddGenerationPrompt := Gen;
end;

function Messages(A: TJSONArray): TChatMessages;
var
  I: Integer;
  M: TJSONObject;
  R: string;
begin
  Result := nil;
  SetLength(Result, A.Count);
  for I := 0 to A.Count - 1 do
  begin
    M := A.Objects[I];
    R := M.Get('role', '');
    if R = 'system' then
      Result[I].Role := crSystem
    else if R = 'user' then
      Result[I].Role := crUser
    else
      Result[I].Role := crAssistant;
    Result[I].Content := M.Get('content', '');
    Result[I].Reasoning := M.Get('reasoning_content', '');
  end;
end;

procedure Emit(const Id: string; Value, Limit: Double; const Note: string);
begin
  WriteLn(Id, #9, Format('%.0f', [Value]), #9, Format('%.0f', [Limit]), #9, Note);
end;

var
  G: TGgufFile;
  Tok: TQwenTokenizer;
  Corpus, Expected: TStringList;
  C, E: TJSONObject;
  Id: string;
  Msgs: TChatMessages;
  Opt: TChatOptions;
  Ours, Want, Span: RawByteString;
  Ids: TTokenArray;
  WantIds: TJSONArray;
  I, J, LossFrom, LossTo, NRender, BadRender, NTok, BadTok, BadSpan, NErr, BadErr: Integer;
  Same: Boolean;
begin
  G := TGgufFile.Open(ParamStr(1));
  Tok := TQwenTokenizer.Create(G);
  Corpus := ReadLines(ParamStr(2));
  Expected := ReadLines(ParamStr(3));
  NRender := 0; BadRender := 0; NTok := 0; BadTok := 0; BadSpan := 0; NErr := 0; BadErr := 0;
  try
    for I := 0 to Corpus.Count - 1 do
    begin
      C := GetJSON(Corpus[I], False) as TJSONObject;  { measured: UseUTF8 = True corrupts non-ASCII in FPC 3.2.2 }
      E := GetJSON(Expected[I], False) as TJSONObject;
      try
        Id := C.Get('id', '');
        Msgs := Messages(C.Arrays['messages']);
        Opt := Options(C.Objects['options'], C.Get('gen', True));
        if C.Get('error', False) then
        begin
          Inc(NErr);
          try
            RenderChat(Msgs, Opt);
            Inc(BadErr);
            WriteLn('# ', Id, #9'NG'#9'no EChatError');
          except
            on X: EChatError do
              WriteLn('# ', Id, #9'OK'#9'EChatError: ', X.Message);
          end;
          Continue;
        end;
        Inc(NRender);
        Want := E.Get('prompt', '');
        if C.Get('training', False) then
          Ours := RenderTraining(Msgs, Opt, LossFrom, LossTo)
        else
          Ours := RenderChat(Msgs, Opt);
        if Ours = Want then
          WriteLn('# ', Id, #9'OK'#9'rendering')
        else
        begin
          Inc(BadRender);
          WriteLn('# ', Id, #9'NG'#9'rendering differs');
        end;
        if C.Get('training', False) then
        begin
          Inc(NTok);
          Ids := EncodeTraining(Tok, Msgs, Opt, LossFrom, LossTo);
          WantIds := E.Arrays['tokens'];
          Same := Length(Ids) = WantIds.Count;
          if Same then
            for J := 0 to High(Ids) do
              if Ids[J] <> WantIds.Integers[J] then
                Same := False;
          if Same then
            WriteLn('# ', Id, #9'OK'#9'token ids (', Length(Ids), ')')
          else
          begin
            Inc(BadTok);
            WriteLn('# ', Id, #9'NG'#9'token ids differ');
          end;
          { the loss span: the last answer and its end-of-turn tag, then only #10 }
          Span := Tok.Decode(Copy(Ids, LossFrom, LossTo - LossFrom));
          if (Span = TrimChat(Msgs[High(Msgs)].Content) + '<|' + 'im_end|>')
             and (Ids[LossTo - 1] = Tok.EosId) and (LossTo = Length(Ids) - 1) then
            WriteLn('# ', Id, #9'OK'#9'loss span ', LossFrom, '..', LossTo)
          else
          begin
            Inc(BadSpan);
            WriteLn('# ', Id, #9'NG'#9'loss span ', LossFrom, '..', LossTo);
          end;
        end;
      finally
        C.Free;
        E.Free;
      end;
    end;
    Emit('ITa-T07-001', BadRender, 0, Format('renderings different from /apply-template, of %d conversations', [NRender]));
    Emit('ITa-T07-002', BadTok, 0, Format('training samples whose token ids differ from /tokenize, of %d', [NTok]));
    Emit('ITa-T07-003', BadSpan, 0, Format('training samples whose loss span is not exactly the last answer and its end-of-turn token, of %d', [NTok]));
    Emit('ITa-T07-004', BadErr, 0, Format('conversations the template rejects that did not raise EChatError, of %d', [NErr]));
  finally
    Corpus.Free;
    Expected.Free;
    Tok.Free;
    G.Free;
  end;
end.
