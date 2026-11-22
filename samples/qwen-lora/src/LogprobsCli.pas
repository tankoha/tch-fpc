(* LogprobsCli (T11): the behaviour of qlora_logprobs. Command line:
    qlora_logprobs [--top N] [--split S] MODEL FILE
  FILE: JSONL; every non-empty line is {"id": ..., "tokens": [ids]} or
  {"id": ..., "text": "..."} (text encoded by QwenTokenizer). One output
  line per record: {"id":"..","n":T,"top":[{"id":I,"logprob":L},...]}, the
  N most likely next tokens after the whole prompt (natural log of the
  softmax over the vocabulary; default N 20, 1..100). Without --split the
  model runs on the CPU; with --split S blocks 0..S-1 run on cuda:0 and the
  rest and the head on cuda:1.
  Exit codes: 0 done, 1 error (message on stderr), 2 usage.
  tasks/T11-facts.md.

  Skeleton by the harness owner (interface final); the routines are
  written in T11b3. *)
unit LogprobsCli;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils;

{ AOut / AErr are filled (caller-created, caller-owned); the result is the
  exit code. }
function RunLogprobsCLI(const AArgs: array of string; AOut, AErr: TStrings): Integer;

{ One output line (the format above) from ids and log-probabilities. }
function TopLine(const AId: string; ACount: Integer; const AIds: array of Int64;
  const ALogProbs: array of Double): string;

implementation

uses
  fpjson, jsonparser, Tch, GgufFile, QwenTokenizer, QwenModel;

const
  K_USAGE = 'Usage: qlora_logprobs [--top N] [--split S] MODEL FILE';

type
  EUsage = class(Exception);

function TopLine(const AId: string; ACount: Integer; const AIds: array of Int64;
  const ALogProbs: array of Double): string;
var
  I: Integer;
  FS: TFormatSettings;
begin
  FS := DefaultFormatSettings;
  FS.DecimalSeparator := '.';
  Result := '{"id":"' + StringToJSONString(AId) + '","n":' + IntToStr(ACount) + ',"top":[';
  for I := 0 to High(AIds) do
  begin
    if I > 0 then
      Result := Result + ',';
    Result := Result + Format('{"id":%d,"logprob":%.7f}', [AIds[I], ALogProbs[I]], FS);
  end;
  Result := Result + ']}';
end;

function ParseCount(const AText, AFlag: string; ALow, AHigh: Integer): Integer;
begin
  if not TryStrToInt(AText, Result) or (Result < ALow) or (Result > AHigh) then
    raise EUsage.CreateFmt('%s must be %d..%d, got %s', [AFlag, ALow, AHigh, AText]);
end;

function RunLogprobsCLI(const AArgs: array of string; AOut, AErr: TStrings): Integer;
var
  Top, Split, I, J, LineNo: Integer;
  Pos: array of string;
  G: TGgufFile;
  Tok: TQwenTokenizer;
  M: TQwenModel;
  Lines: TStringList;
  Js: TJSONData;
  O: TJSONObject;
  Arr: TJSONArray;
  Ids: TTokenArray;
  Id: string;
  LP, Vals, Idx: TTensor;
  TopIds: array of Int64;
  TopLp: array of Double;
begin
  Top := 20;
  Split := -1;
  Pos := nil;
  try
    I := 0;
    while I <= High(AArgs) do
    begin
      if (AArgs[I] = '--top') or (AArgs[I] = '--split') then
      begin
        if I = High(AArgs) then
          raise EUsage.Create(AArgs[I] + ' needs a value');
        if AArgs[I] = '--top' then
          Top := ParseCount(AArgs[I + 1], '--top', 1, 100)
        else
          Split := ParseCount(AArgs[I + 1], '--split', 0, 1000);
        Inc(I, 2);
      end
      else if (AArgs[I] <> '') and (AArgs[I][1] = '-') then
        raise EUsage.Create('unknown option ' + AArgs[I])
      else
      begin
        SetLength(Pos, Length(Pos) + 1);
        Pos[High(Pos)] := AArgs[I];
        Inc(I);
      end;
    end;
    if Length(Pos) <> 2 then
      raise EUsage.Create('MODEL and FILE expected');
  except
    on E: EUsage do
    begin
      AErr.Add('qlora_logprobs: ' + E.Message);
      AErr.Add(K_USAGE);
      Exit(2);
    end;
  end;
  G := nil;
  Tok := nil;
  M := nil;
  Lines := TStringList.Create;
  try
    try
      G := TGgufFile.Open(Pos[0]);
      Lines.LoadFromFile(Pos[1]);
      if Split < 0 then
        M := TQwenModel.Load(G, TDevice.Cpu, TDevice.Cpu, 0)
      else
        M := TQwenModel.Load(G, TDevice.Cuda(0), TDevice.Cuda(1), Split);
      for LineNo := 1 to Lines.Count do
      begin
        if Trim(Lines[LineNo - 1]) = '' then
          Continue;
        try
          Js := GetJSON(Lines[LineNo - 1], False);
        except
          on E: Exception do
            raise EQwenModel.CreateFmt('line %d: not JSON (%s)', [LineNo, E.Message]);
        end;
        try
          if not (Js is TJSONObject) then
            raise EQwenModel.CreateFmt('line %d: not an object', [LineNo]);
          O := TJSONObject(Js);
          Id := O.Get('id', IntToStr(LineNo));
          if (O.IndexOfName('tokens') >= 0) and (O.Elements['tokens'] is TJSONArray) then
          begin
            Arr := O.Arrays['tokens'];
            Ids := nil;
            SetLength(Ids, Arr.Count);
            for J := 0 to Arr.Count - 1 do
              Ids[J] := Arr.Integers[J];
          end
          else if (O.IndexOfName('text') >= 0) and (O.Elements['text'] is TJSONString) then
          begin
            if Tok = nil then
              Tok := TQwenTokenizer.Create(G);
            Ids := Tok.Encode(RawByteString(UTF8String(O.Strings['text'])));
          end
          else
            raise EQwenModel.CreateFmt('line %d: needs "tokens" (array) or "text" (string)', [LineNo]);
          LP := M.LastLogProbs(Ids);
          LP.topk(Top, 0, True, True, Vals, Idx);
          TopIds := nil;
          TopLp := nil;
          SetLength(TopIds, Top);
          SetLength(TopLp, Top);
          for J := 0 to Top - 1 do
          begin
            TopIds[J] := Idx.Int64Value([J]);
            TopLp[J] := Vals.DoubleValue([J]);
          end;
          AOut.Add(TopLine(Id, Length(Ids), TopIds, TopLp));
        finally
          Js.Free;
        end;
      end;
      Result := 0;
    except
      on E: Exception do
      begin
        AErr.Add('qlora_logprobs: ' + E.Message);
        Result := 1;
      end;
    end;
  finally
    M.Free;
    Tok.Free;
    G.Free;
    Lines.Free;
  end;
end;

end.
