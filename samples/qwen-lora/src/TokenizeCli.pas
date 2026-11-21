{ TokenizeCli — the logic of qlora_tokenize (T05), per tasks/T05-facts.md
  (authoritative): parses the arguments, loads the tokenizer from the GGUF,
  reads FILE as bytes and writes token ids. Every behaviour line lives
  here, measured in process by tests/ut/TokenizeCli_test.pas; the program
  src/qlora_tokenize.pas only forwards.
  Skeleton by the harness owner; stage 3 (T05b3) writes RunTokenizeCLI. }
unit TokenizeCli;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils;

{ AArgs: [--jsonl] MODEL FILE. AOut / AErr are filled (caller-created,
  caller-owned); the result is the exit code (0 done, 1 error, 2 usage). }
function RunTokenizeCLI(const AArgs: array of string; AOut, AErr: TStrings): Integer;

implementation

uses
  fpjson, jsonparser, GgufFile, QwenTokenizer;

const
  K_USAGE = 'Usage: qlora_tokenize [--jsonl] MODEL FILE';

function ReadBytes(const APath: string): RawByteString;
var
  S: TFileStream;
begin
  S := TFileStream.Create(APath, fmOpenRead or fmShareDenyNone);
  try
    SetLength(Result, S.Size);
    if S.Size > 0 then S.ReadBuffer(Result[1], S.Size);
  finally
    S.Free;
  end;
  SetCodePage(Result, CP_UTF8, False);
end;

function IdsText(const A: TTokenArray): string;
var
  I: Integer;
begin
  Result := '';
  for I := 0 to High(A) do
  begin
    if I > 0 then Result := Result + ' ';
    Result := Result + IntToStr(A[I]);
  end;
end;

function RunTokenizeCLI(const AArgs: array of string; AOut, AErr: TStrings): Integer;
var
  Jsonl: Boolean;
  First, I, LineNo, P: Integer;
  G: TGgufFile;
  T: TQwenTokenizer;
  Data, Line: RawByteString;
  J: TJSONData;
  Id: string;
  Txt: UTF8String;
begin
  Jsonl := (Length(AArgs) > 0) and (AArgs[0] = '--jsonl');
  if Jsonl then First := 1 else First := 0;
  if Length(AArgs) - First <> 2 then
  begin
    AErr.Add(K_USAGE);
    Exit(2);
  end;
  for I := First to High(AArgs) do
    if (AArgs[I] <> '') and (AArgs[I][1] = '-') then
    begin
      AErr.Add(K_USAGE);
      Exit(2);
    end;
  G := nil; T := nil;
  try
    try
      G := TGgufFile.Open(AArgs[First]);
      T := TQwenTokenizer.Create(G);
      Data := ReadBytes(AArgs[First + 1]);
      if not Jsonl then
        AOut.Add(IdsText(T.Encode(Data)))
      else
      begin
        LineNo := 0;
        while Data <> '' do
        begin
          P := Pos(#10, Data);
          if P = 0 then P := Length(Data) + 1;
          Line := Copy(Data, 1, P - 1);
          Delete(Data, 1, P);
          Inc(LineNo);
          if (Line <> '') and (Line[Length(Line)] = #13) then SetLength(Line, Length(Line) - 1);
          if Line = '' then Continue;
          SetCodePage(Line, CP_UTF8, False);
          try
            J := GetJSON(Line, False);
          except
            on E: Exception do
              raise ETokenizerError.CreateFmt('line %d: not JSON (%s)', [LineNo, E.Message]);
          end;
          try
            if not (J is TJSONObject) or (TJSONObject(J).IndexOfName('text') < 0) or
               not (TJSONObject(J).Elements['text'] is TJSONString) then
              raise ETokenizerError.CreateFmt('line %d: no string "text"', [LineNo]);
            Txt := TJSONObject(J).Strings['text'];
            Id := TJSONObject(J).Get('id', IntToStr(LineNo));
            AOut.Add(Id + #9 + IdsText(T.Encode(RawByteString(Txt))));
          finally
            J.Free;
          end;
        end;
      end;
      Result := 0;
    except
      on E: Exception do
      begin
        AErr.Add('qlora_tokenize: ' + E.Message);
        Result := 1;
      end;
    end;
  finally
    T.Free;
    G.Free;
  end;
end;

end.
