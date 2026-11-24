{ DatasetCli — the data set builder qlora_dataset (T19), per DESIGN.md
  8.16 and tasks/T19-facts.md (authoritative): reads the generator's raw
  JSONL, rejects by schema, id, dialect, leak, duplicate, tokenizer and
  verification, splits the accepted records into train and valid per
  dialect × task type, and writes the records, the manifest, the rejects
  and the card.
  Interface by the harness owner (tasks/T19-facts.md); the bodies are
  written in T19b2. }
unit DatasetCli;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Math, fpjson, jsonparser, md5, GgufFile, QwenTokenizer, DialectVerify,
  DatasetFilter;

const
  K_DATASET_USAGE = 'usage: qlora_dataset [options] RAW OUTDIR (see the unit header)';

type
  TDatasetConfig = record
    Raw, OutDir: string;          { OutDir ends with the path delimiter }
    Model, RefTokens: string;     { both or neither }
    Evals, LeakDirs: TStringArray;
    Seed: Int64;
    ValidPercent, Timeout: Integer;
    SpecVersion: string;
  end;

  { A text the leak check compares with: its name and shingles. }
  TLeakText = record
    Name: string;
    Sh: TShingles;
  end;
  TLeakTexts = array of TLeakText;

{ Seed 1, ValidPercent 5, Timeout 10, SpecVersion 'v1', the rest empty. }
function DefaultDatasetConfig: TDatasetConfig;

{ '' or the usage error. Options: --model F, --ref-tokens F, --eval F
  (repeatable), --leak-dir D (repeatable), --seed N, --valid-percent P
  (0..100), --timeout S (1..3600), --spec-version V; then RAW OUTDIR. }
function ParseDatasetArgs(const AArgs: array of string; out C: TDatasetConfig): string;

{ The program block of ABlocks: the first whose normalised text does not
  start with 'unit '; -1 when there is none. }
function MainBlock(const ABlocks: TStringArray): Integer;

{ The unit name of a block that starts with 'unit <name>;' (normalised),
  as written in the source; '' otherwise. }
function UnitNameOf(const ABlock: string): string;

{ Writes the blocks into ADir (main.pas for the program block, <unit>.pas
  for the others, main.expected with AExpected) and verifies main.pas as
  ADialect. AOutcome and ADetail are VerifyFile's; the result is True when
  the outcome is 'accepted', the class compiled-run or fpc-approx and the
  program block demands no compile error. AClass gets the class. }
function VerifyRecord(const R: TRawRecord; const ADir: string; ATimeout: Integer;
  const AEnv: TEnvRec; out AClass, AOutcome, ADetail: string): Boolean;

{ AOut / AErr caller-owned; AWorkRoot: where the verifier works; the
  result is the exit code (0 done, 1 error, 2 usage). }
function RunDatasetCLI(const AArgs: array of string; const AEnv: TEnvRec;
  const AWorkRoot: string; AOut, AErr: TStrings): Integer;

implementation

function DefaultDatasetConfig: TDatasetConfig;
begin
  Result := Default(TDatasetConfig);
  raise Exception.Create('DefaultDatasetConfig: not implemented yet');
end;

function ParseDatasetArgs(const AArgs: array of string; out C: TDatasetConfig): string;
begin
  Result := '';
  raise Exception.Create('ParseDatasetArgs: not implemented yet');
end;

function MainBlock(const ABlocks: TStringArray): Integer;
begin
  Result := 0;
  raise Exception.Create('MainBlock: not implemented yet');
end;

function UnitNameOf(const ABlock: string): string;
begin
  Result := '';
  raise Exception.Create('UnitNameOf: not implemented yet');
end;

function VerifyRecord(const R: TRawRecord; const ADir: string; ATimeout: Integer;
  const AEnv: TEnvRec; out AClass, AOutcome, ADetail: string): Boolean;
begin
  Result := False;
  raise Exception.Create('VerifyRecord: not implemented yet');
end;

function RunDatasetCLI(const AArgs: array of string; const AEnv: TEnvRec;
  const AWorkRoot: string; AOut, AErr: TStrings): Integer;
begin
  Result := 0;
  raise Exception.Create('RunDatasetCLI: not implemented yet');
end;

end.
