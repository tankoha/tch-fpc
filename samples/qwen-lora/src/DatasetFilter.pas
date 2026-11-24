{ DatasetFilter — the pure parts of the data set filter (T19), per
  DESIGN.md 8.16 and tasks/T19-facts.md (authoritative): the record
  schema, the fenced blocks of an answer, the normalised source and its
  key, lexical tokens, 5-gram shingles and their Jaccard similarity, the
  valid split's size and order.
  Interface by the harness owner (tasks/T19-facts.md); the bodies are
  written in T19b1. }
unit DatasetFilter;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpjson, jsonparser, md5;

const
  K_DIALECTS = 'objfpc,fpc-delphi,unleashed,delphi,oxygene';
  K_TASK_TYPES = 'write,extend,fix,convert,unit';
  K_NGRAM = 5;
  K_NEAR = 0.5;

type
  { The fields of a raw record the filter uses. }
  TRawRecord = record
    Id, Dialect, TaskType, Prompt, Answer, Expected: string;
    Blocks: TStringArray;   { the fenced blocks of Answer, in order }
  end;
  TShingles = array of QWord;   { ascending, unique }

{ '' when ALine is a valid record (R filled), else the reason
  'schema: <what>'. Checks, in order: a JSON object; the strings id
  (non-empty), dialect (one of K_DIALECTS), task_type (one of
  K_TASK_TYPES), topic, language, prompt (non-empty), answer (non-empty),
  expected_stdout; the array features; the object generator; at least one
  fenced block in answer. }
function ParseRecord(const ALine: string; out R: TRawRecord): string;

{ The closed fenced blocks of AText: a line starting with ``` opens a
  block, a line equal to ``` (trailing spaces ignored) closes it; the
  lines between, each ended by #10. An unclosed block is not returned. }
function ExtractBlocks(const AText: string): TStringArray;

{ ASrc without comments (braces, (* *), // to the line's end; not inside
  '...' strings), lower case outside strings, every run of white space
  one space, trimmed. }
function NormalizeSource(const ASrc: string): string;

{ MD5 (hex) of the normalised blocks joined by #10. }
function SourceKey(const ABlocks: TStringArray): string;

{ The tokens of LowerCase(AText): words ([a-z_][a-z0-9_]*), numbers
  ([0-9]+), every other character but white space alone. }
function LexTokens(const AText: string): TStringArray;

{ The FNV-1a 64 hashes of the K_NGRAM-grams of ATokens (joined by one
  space), ascending and unique; empty when there are fewer tokens. }
function Shingles(const ATokens: TStringArray): TShingles;

{ |A ∩ B| / |A ∪ B|; 0 when both are empty. }
function Jaccard(const A, B: TShingles): Double;

{ The records of a cell of AN that go to valid: (AN · APercent + 50) div 100. }
function ValidCount(AN, APercent: Integer): Integer;

{ The order inside a cell: MD5 (hex) of '<seed>:<id>'; the first
  ValidCount by this key go to valid. }
function SplitKey(ASeed: Int64; const AId: string): string;

implementation

function InList(const AValue, AList: string): Boolean;
begin
  Result := Pos(',' + AValue + ',', ',' + AList + ',') > 0;
end;

function ParseRecord(const ALine: string; out R: TRawRecord): string;
var
  D: TJSONData;
  O: TJSONObject;

  function Str(const AKey: string; ANonEmpty: Boolean; out AValue: string): Boolean;
  var
    V: TJSONData;
  begin
    V := O.Find(AKey);
    Result := (V <> nil) and (V.JSONType = jtString);
    if Result then
      AValue := V.AsString
    else
      AValue := '';
    if Result and ANonEmpty then
      Result := AValue <> '';
  end;

var
  Topic, Lang: string;
  V: TJSONData;
begin
  R := Default(TRawRecord);
  try
    D := GetJSON(ALine, False);
  except
    on Exception do   { EJSONParser for broken text }
      Exit('schema: not a JSON object');
  end;
  try
    if D.JSONType <> jtObject then
      Exit('schema: not a JSON object');
    O := TJSONObject(D);
    if not Str('id', True, R.Id) then
      Exit('schema: id');
    if not Str('dialect', True, R.Dialect) or not InList(R.Dialect, K_DIALECTS) then
      Exit('schema: dialect');
    if not Str('task_type', True, R.TaskType) or not InList(R.TaskType, K_TASK_TYPES) then
      Exit('schema: task_type');
    if not Str('topic', False, Topic) then
      Exit('schema: topic');
    if not Str('language', False, Lang) then
      Exit('schema: language');
    if not Str('prompt', True, R.Prompt) then
      Exit('schema: prompt');
    if not Str('answer', True, R.Answer) then
      Exit('schema: answer');
    if not Str('expected_stdout', False, R.Expected) then
      Exit('schema: expected_stdout');
    V := O.Find('features');
    if (V = nil) or (V.JSONType <> jtArray) then
      Exit('schema: features');
    V := O.Find('generator');
    if (V = nil) or (V.JSONType <> jtObject) then
      Exit('schema: generator');
    R.Blocks := ExtractBlocks(R.Answer);
    if Length(R.Blocks) = 0 then
      Exit('schema: answer has no fenced block');
    Result := '';
  finally
    D.Free;
  end;
end;

function ExtractBlocks(const AText: string): TStringArray;
var
  L: TStringList;
  I, N: Integer;
  Inside: Boolean;
  Cur: string;
begin
  Result := nil;
  N := 0;
  Inside := False;
  Cur := '';
  L := TStringList.Create;
  try
    L.Text := AText;
    for I := 0 to L.Count - 1 do
      if not Inside then
      begin
        if Copy(L[I], 1, 3) = '```' then
        begin
          Inside := True;
          Cur := '';
        end;
      end
      else if TrimRight(L[I]) = '```' then
      begin
        Inside := False;
        SetLength(Result, N + 1);
        Result[N] := Cur;
        Inc(N);
      end
      else
        Cur := Cur + L[I] + #10;
  finally
    L.Free;
  end;
end;

function NormalizeSource(const ASrc: string): string;
var
  I, N: Integer;
  C: Char;
  Space: Boolean;

  procedure Put(AC: Char);
  begin
    if Space and (Result <> '') then
      Result := Result + ' ';
    Space := False;
    Result := Result + AC;
  end;

begin
  Result := '';
  Space := False;
  N := Length(ASrc);
  I := 1;
  while I <= N do
  begin
    C := ASrc[I];
    if C = '''' then
    begin
      Put(C);
      Inc(I);
      while (I <= N) and (ASrc[I] <> '''') do
      begin
        Result := Result + ASrc[I];
        Inc(I);
      end;
      if I <= N then
        Result := Result + '''';
      Inc(I);
    end
    else if C = '{' then
    begin
      while (I <= N) and (ASrc[I] <> '}') do
        Inc(I);
      Inc(I);
      Space := True;
    end
    else if (C = '(') and (I < N) and (ASrc[I + 1] = '*') then
    begin
      Inc(I, 2);
      while (I < N) and not ((ASrc[I] = '*') and (ASrc[I + 1] = ')')) do
        Inc(I);
      Inc(I, 2);
      Space := True;
    end
    else if (C = '/') and (I < N) and (ASrc[I + 1] = '/') then
    begin
      while (I <= N) and (ASrc[I] <> #10) do
        Inc(I);
      Space := True;
    end
    else if C in [#9, #10, #13, ' '] then
    begin
      Space := True;
      Inc(I);
    end
    else
    begin
      Put(LowerCase(C));
      Inc(I);
    end;
  end;
end;

function SourceKey(const ABlocks: TStringArray): string;
var
  I: Integer;
  S: string;
begin
  S := '';
  for I := 0 to High(ABlocks) do
  begin
    if I > 0 then
      S := S + #10;
    S := S + NormalizeSource(ABlocks[I]);
  end;
  Result := MD5Print(MD5String(S));
end;

function LexTokens(const AText: string): TStringArray;
var
  T: string;
  I, J, N, K: Integer;
begin
  Result := nil;
  K := 0;
  T := LowerCase(AText);
  N := Length(T);
  I := 1;
  while I <= N do
  begin
    J := I;
    if T[I] in ['a'..'z', '_'] then
      while (J <= N) and (T[J] in ['a'..'z', '0'..'9', '_']) do
        Inc(J)
    else if T[I] in ['0'..'9'] then
      while (J <= N) and (T[J] in ['0'..'9']) do
        Inc(J)
    else if T[I] in [#9, #10, #13, ' '] then
    begin
      Inc(I);
      Continue;
    end
    else
      J := I + 1;
    SetLength(Result, K + 1);
    Result[K] := Copy(T, I, J - I);
    Inc(K);
    I := J;
  end;
end;

{$push}{$Q-}{$R-}
function Fnv1a(const S: string): QWord;
var
  I: Integer;
begin
  Result := QWord($CBF29CE484222325);
  for I := 1 to Length(S) do
    Result := (Result xor Ord(S[I])) * QWord($100000001B3);
end;
{$pop}

function Shingles(const ATokens: TStringArray): TShingles;
var
  I, J, K, N: Integer;
  S: string;
  H: QWord;
begin
  Result := nil;
  N := 0;
  for I := 0 to Length(ATokens) - K_NGRAM do
  begin
    S := ATokens[I];
    for J := 1 to K_NGRAM - 1 do
      S := S + ' ' + ATokens[I + J];
    H := Fnv1a(S);
    { insertion into the ascending array, duplicates dropped }
    J := N;
    while (J > 0) and (Result[J - 1] > H) do
      Dec(J);
    if (J > 0) and (Result[J - 1] = H) then
      Continue;
    SetLength(Result, N + 1);
    for K := N downto J + 1 do
      Result[K] := Result[K - 1];
    Result[J] := H;
    Inc(N);
  end;
end;

function Jaccard(const A, B: TShingles): Double;
var
  I, J, Both: Integer;
begin
  if (Length(A) = 0) and (Length(B) = 0) then
    Exit(0);
  I := 0;
  J := 0;
  Both := 0;
  while (I < Length(A)) and (J < Length(B)) do
    if A[I] = B[J] then
    begin
      Inc(Both);
      Inc(I);
      Inc(J);
    end
    else if A[I] < B[J] then
      Inc(I)
    else
      Inc(J);
  Result := Both / (Length(A) + Length(B) - Both);
end;

function ValidCount(AN, APercent: Integer): Integer;
begin
  Result := (AN * APercent + 50) div 100;
end;

function SplitKey(ASeed: Int64; const AId: string): string;
begin
  Result := MD5Print(MD5String(IntToStr(ASeed) + ':' + AId));
end;

end.
