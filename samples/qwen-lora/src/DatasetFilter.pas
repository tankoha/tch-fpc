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

function ParseRecord(const ALine: string; out R: TRawRecord): string;
begin
  Result := '';
  raise Exception.Create('ParseRecord: not implemented yet');
end;

function ExtractBlocks(const AText: string): TStringArray;
begin
  Result := nil;
  raise Exception.Create('ExtractBlocks: not implemented yet');
end;

function NormalizeSource(const ASrc: string): string;
begin
  Result := '';
  raise Exception.Create('NormalizeSource: not implemented yet');
end;

function SourceKey(const ABlocks: TStringArray): string;
begin
  Result := '';
  raise Exception.Create('SourceKey: not implemented yet');
end;

function LexTokens(const AText: string): TStringArray;
begin
  Result := nil;
  raise Exception.Create('LexTokens: not implemented yet');
end;

function Shingles(const ATokens: TStringArray): TShingles;
begin
  Result := nil;
  raise Exception.Create('Shingles: not implemented yet');
end;

function Jaccard(const A, B: TShingles): Double;
begin
  Result := 0;
  raise Exception.Create('Jaccard: not implemented yet');
end;

function ValidCount(AN, APercent: Integer): Integer;
begin
  Result := 0;
  raise Exception.Create('ValidCount: not implemented yet');
end;

function SplitKey(ASeed: Int64; const AId: string): string;
begin
  Result := '';
  raise Exception.Create('SplitKey: not implemented yet');
end;

end.
