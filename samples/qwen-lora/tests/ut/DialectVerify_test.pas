{ UT for unit DialectVerify (T06) — stage 1: text functions.
  One published Test method per case row of
  tests/ut/cases/DialectVerify.cases.md (stage-1 rows), named
  TestUT_DialectVerify_NNN so cases and results match.
  Invoked with `--all --format=plain` (see .sdlc/ut-coverage.sh). }
program DialectVerify_test;

{$mode objfpc}{$H+}

uses
  SysUtils, Classes, process, fpcunit, testregistry, consoletestrunner, DialectVerify;

type
  TTestDialectVerify = class(TTestCase)
  public
    procedure WriteTestFile(const APath, AContent: string);
    { stage-2 helpers: executable bit, fake compiler/exe factory, TSV cell }
    procedure ChmodExec(const APath: string);
    function WriteFakeFpc(const APath: string; const AExeLines, AErrLine,
      AArgsLogTo: string; ACompilerSleep: Integer): string;
    function Cell(const ALn: string; ACol: Integer): string;
  published
    procedure TestUT_DialectVerify_001;
    procedure TestUT_DialectVerify_002;
    procedure TestUT_DialectVerify_003;
    procedure TestUT_DialectVerify_006;
    procedure TestUT_DialectVerify_007;
    procedure TestUT_DialectVerify_008;
    procedure TestUT_DialectVerify_011;
    procedure TestUT_DialectVerify_012;
    procedure TestUT_DialectVerify_013;
    procedure TestUT_DialectVerify_027;
    procedure TestUT_DialectVerify_028;
    procedure TestUT_DialectVerify_038;
    procedure TestUT_DialectVerify_046;
    procedure TestUT_DialectVerify_051;
    procedure TestUT_DialectVerify_055;
    procedure TestUT_DialectVerify_056;
    procedure TestUT_DialectVerify_057;
    procedure TestUT_DialectVerify_058;
    procedure TestUT_DialectVerify_059;
    procedure TestUT_DialectVerify_060;
    procedure TestUT_DialectVerify_061;
    procedure TestUT_DialectVerify_062;
    procedure TestUT_DialectVerify_063;
    procedure TestUT_DialectVerify_064;
    procedure TestUT_DialectVerify_065;
    procedure TestUT_DialectVerify_066;
    procedure TestUT_DialectVerify_067;
    procedure TestUT_DialectVerify_068;
    procedure TestUT_DialectVerify_069;
    procedure TestUT_DialectVerify_070;
    procedure TestUT_DialectVerify_071;
    procedure TestUT_DialectVerify_072;
    procedure TestUT_DialectVerify_073;
    procedure TestUT_DialectVerify_074;
    procedure TestUT_DialectVerify_075;
    procedure TestUT_DialectVerify_076;
    procedure TestUT_DialectVerify_077;
    procedure TestUT_DialectVerify_078;
    procedure TestUT_DialectVerify_079;
    procedure TestUT_DialectVerify_080;
    procedure TestUT_DialectVerify_081;
    procedure TestUT_DialectVerify_082;
    procedure TestUT_DialectVerify_083;
    procedure TestUT_DialectVerify_084;
    procedure TestUT_DialectVerify_085;
    procedure TestUT_DialectVerify_086;
    procedure TestUT_DialectVerify_087;
    procedure TestUT_DialectVerify_088;
    procedure TestUT_DialectVerify_089;
    procedure TestUT_DialectVerify_132;
    procedure TestUT_DialectVerify_133;
    procedure TestUT_DialectVerify_134;
    procedure TestUT_DialectVerify_135;
    procedure TestUT_DialectVerify_136;
    { T06c2 defect fixes: BUG-001 (case of expect-error text), BUG-002 (bare name) }
    procedure TestUT_DialectVerify_140;
    procedure TestUT_DialectVerify_141;
    { T06c4 defect fix: BUG-003 (bare name not resolved in PATH by RunLimited) }
    procedure TestUT_DialectVerify_142;
    procedure TestUT_DialectVerify_090;
    procedure TestUT_DialectVerify_091;
    procedure TestUT_DialectVerify_092;
    procedure TestUT_DialectVerify_093;
    procedure TestUT_DialectVerify_094;
    procedure TestUT_DialectVerify_095;
    procedure TestUT_DialectVerify_096;
    procedure TestUT_DialectVerify_097;
    procedure TestUT_DialectVerify_098;
    procedure TestUT_DialectVerify_099;
    procedure TestUT_DialectVerify_100;
    procedure TestUT_DialectVerify_101;
    procedure TestUT_DialectVerify_102;
    procedure TestUT_DialectVerify_103;
    procedure TestUT_DialectVerify_104;
    procedure TestUT_DialectVerify_105;
    procedure TestUT_DialectVerify_106;
    procedure TestUT_DialectVerify_109;
    procedure TestUT_DialectVerify_110;
    procedure TestUT_DialectVerify_118;
    procedure TestUT_DialectVerify_119;
    procedure TestUT_DialectVerify_120;
    procedure TestUT_DialectVerify_121;
    procedure TestUT_DialectVerify_122;
    procedure TestUT_DialectVerify_123;
    procedure TestUT_DialectVerify_124;
    procedure TestUT_DialectVerify_125;
    procedure TestUT_DialectVerify_126;
    procedure TestUT_DialectVerify_127;
    procedure TestUT_DialectVerify_128;
    procedure TestUT_DialectVerify_129;
    procedure TestUT_DialectVerify_130;
    procedure TestUT_DialectVerify_131;
    { stage 3: CollectPascalFiles / VerifyMain (+ DialectValidateMain) }
    procedure TestUT_DialectVerify_036;
    procedure TestUT_DialectVerify_037;
    procedure TestUT_DialectVerify_107;
    procedure TestUT_DialectVerify_108;
    procedure TestUT_DialectVerify_111;
    procedure TestUT_DialectVerify_112;
    procedure TestUT_DialectVerify_113;
    procedure TestUT_DialectVerify_114;
    procedure TestUT_DialectVerify_115;
    procedure TestUT_DialectVerify_116;
    procedure TestUT_DialectVerify_117;
    { extra coverage of defensive CLI branches (not a contract row) }
    procedure TestUT_DialectVerify_200;
    procedure TestUT_DialectVerify_201;
    { T22: Oxygene compiled (EBuild in the Elements flatpak) and run }
    procedure TestUT_DialectVerify_301;
    procedure TestUT_DialectVerify_302;
    procedure TestUT_DialectVerify_303;
    procedure TestUT_DialectVerify_304;
    procedure TestUT_DialectVerify_305;
  end;

procedure TTestDialectVerify.WriteTestFile(const APath, AContent: string);
var
  D: string;
  F: TFileStream;
  R: TRawByteSearchRec;
begin
  D := ExtractFilePath(APath);
  if (D <> '') then
  begin
    if FindFirst(D + '/', faAnyFile, R) = 0 then
    begin
      FindClose(R);
    end
    else
      ForceDirectories(D);
  end;
  F := TFileStream.Create(APath, fmCreate);
  try
    if AContent <> '' then
      F.Write(AContent[1], Length(AContent));
  finally
    F.Free;
  end;
end;

procedure TTestDialectVerify.TestUT_DialectVerify_001;
begin
  AssertEquals('objfpc', DetectDialect('{$mode objfpc}{$H+}', 'x.pas'));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_002;
begin
  AssertEquals('fpc-delphi', DetectDialect('{ a }' + #10 + '{$MODE Delphi}', 'x.pas'));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_003;
begin
  AssertEquals('unleashed', DetectDialect('{$MODE UNLEASHED}' + #10 + 'program x; begin end.', 'x.pas'));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_006;
begin
  AssertEquals('objfpc', DetectDialect('// MYNAMESPACE foo' + #10 + 'program x; begin end.', 'x.pas'));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_007;
begin
  AssertEquals('objfpc', DetectDialect('program x; begin end.', 'x.pas'));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_008;
begin
  AssertEquals('fpc-delphi', DetectDialect('{$MODE Delphi}' + #10 + '{$mode objfpc}', 'x.pas'));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_011;
var
  E: TEnvRec;
  Cls: string;
begin
  E := ReadEnvRec('', '', '/h');
  AssertEquals('/h/unleashed/fpc/bin/fpc', ChooseCompiler('unleashed', E, Cls));
  AssertEquals('compiled-run', Cls);
end;

procedure TTestDialectVerify.TestUT_DialectVerify_012;
var
  E: TEnvRec;
  Cls: string;
begin
  E := ReadEnvRec('', '', '/h');
  AssertEquals('fpc', ChooseCompiler('objfpc', E, Cls));
  AssertEquals('compiled-run', Cls);
end;

procedure TTestDialectVerify.TestUT_DialectVerify_013;
var
  E: TEnvRec;
  Cls: string;
begin
  E := ReadEnvRec('', '', '/h');
  AssertEquals(ExeSearch('flatpak'), ChooseCompiler('oxygene', E, Cls));
  AssertEquals('compiled-run', Cls);
end;

procedure TTestDialectVerify.TestUT_DialectVerify_038;
begin
  AssertEquals('broken proc', ExtractExpectErrorText('// expect-error: broken proc'));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_046;
begin
  AssertEquals('', ExtractExpectErrorText('program x; begin end.'));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_051;
var
  A: TStringArray;
begin
  SetLength(A, 0);
  SortStrings(A);
  AssertEquals(0, Length(A));
  SetLength(A, 1);
  A[0] := 'solo';
  SortStrings(A);
  AssertEquals('solo', A[0]);
  SetLength(A, 3);
  A[0] := 'c';
  A[1] := 'a';
  A[2] := 'b';
  SortStrings(A);
  AssertEquals('a', A[0]);
  AssertEquals('b', A[1]);
  AssertEquals('c', A[2]);
end;

procedure TTestDialectVerify.TestUT_DialectVerify_055;
begin
  AssertEquals('s/unit.expected', ExpectedPathFor('s/unit.pas'));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_056;
begin
  AssertEquals('s/main.expected', ExpectedPathFor('s/main.dpr'));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_057;
begin
  AssertEquals('fpc-delphi', DetectDialect('{$MODE Delphi} uses x;', 'x.pas'));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_058;
begin
  AssertEquals('unleashed', DetectDialect('{$MODE UNLEASHED}' + #10 + 'begin end.', 'x.pas'));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_059;
begin
  AssertEquals('delphi', DetectDialect('program x; begin end.', 'main.dpr'));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_060;
begin
  AssertEquals('delphi', DetectDialect('program x;' + #10 + '{$APPTYPE CONSOLE}' + #10 + 'begin end.', 'x.pas'));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_061;
begin
  AssertEquals('objfpc', DetectDialect('// {$APPTYPE CONSOLE}' + #10 + 'program x; begin end.', 'x.pas'));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_062;
begin
  AssertEquals('oxygene', DetectDialect('namespace Demo;' + #10 + 'end.', 'x.pas'));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_063;
begin
  AssertEquals('objfpc', DetectDialect('program x; begin namespace foo; end.', 'x.pas'));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_064;
var
  E: TEnvRec;
  Cls: string;
begin
  E := ReadEnvRec('', '', '/h');
  E.Fpc := '/fp';
  AssertEquals('/fp', ChooseCompiler('fpc-delphi', E, Cls));
  AssertEquals('compiled-run', Cls);
end;

procedure TTestDialectVerify.TestUT_DialectVerify_065;
var
  E: TEnvRec;
  Cls: string;
begin
  E := ReadEnvRec('', '', '/h');
  E.FpcUnleashed := '/up';
  AssertEquals('/up', ChooseCompiler('delphi', E, Cls));
  AssertEquals('fpc-approx', Cls);
end;

procedure TTestDialectVerify.TestUT_DialectVerify_066;
var
  E: TEnvRec;
  Cls: string;
begin
  E := ReadEnvRec('', '', '/h');
  AssertEquals('', ChooseCompiler('wat', E, Cls));
  AssertEquals('', Cls);
end;

procedure TTestDialectVerify.TestUT_DialectVerify_067;
var
  S: string;
  P1, P2, P3, P4: Integer;
begin
  S := BuildCompileCmdLine('/fp', 's.pas', '/tmp/o', False);
  P1 := Pos(Chr(39) + '-vew' + Chr(39), S);
  P2 := Pos(Chr(39) + '-l-' + Chr(39), S);
  P3 := Pos(Chr(39) + '-FE/tmp/o' + Chr(39), S);
  P4 := Pos(Chr(39) + '-FU/tmp/o' + Chr(39), S);
  AssertTrue('flag order -vew, -l-, -FE, -FU: ' + S,
    (P1 > 0) and (P1 < P2) and (P2 < P3) and (P3 < P4));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_068;
var
  S: string;
  P1, P2, P3, P4: Integer;
begin
  S := BuildCompileCmdLine('/up', 's.pas', '/tmp/o', True);
  P1 := Pos(Chr(39) + '-Mdelphiunicode' + Chr(39), S);
  P2 := Pos(Chr(39) + '-Manonymousfunctions' + Chr(39), S);
  P3 := Pos(Chr(39) + '-Mfunctionreferences' + Chr(39), S);
  P4 := Pos(Chr(39) + '-Minlinevars' + Chr(39), S);
  AssertTrue('delphi -M flag order: ' + S,
    (P1 > 0) and (P1 < P2) and (P2 < P3) and (P3 < P4));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_069;
begin
  AssertEquals('uses SysUtils;', RewriteDelphiUnits('uses System.SysUtils;'));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_070;
begin
  AssertEquals('uses SysUtils, Math, StrUtils, DateUtils, Classes, Types, '
    + 'Variants, TypInfo, Rtti;',
    RewriteDelphiUnits('uses System.SysUtils, System.Math, System.StrUtils, '
      + 'System.DateUtils, System.Classes, System.Types, System.Variants, '
      + 'System.TypInfo, System.Rtti;'));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_071;
begin
  AssertEquals('uses Generics.Collections;', RewriteDelphiUnits('uses System.Generics.Collections;'));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_072;
begin
  AssertEquals('uses SysUtils, Generics.Collections;',
    RewriteDelphiUnits('uses System.SysUtils, System.Generics.Collections;'));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_073;
begin
  AssertEquals('uses XSystem.SysUtils;', RewriteDelphiUnits('uses XSystem.SysUtils;'));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_074;
begin
  AssertEquals('uses System.SysUtilsX;', RewriteDelphiUnits('uses System.SysUtilsX;'));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_075;
begin
  AssertEquals('program x; begin end.', RewriteDelphiUnits('program x; begin end.'));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_076;
begin
  AssertEquals('broken proc', ExtractExpectErrorText('{ bad call expect-error: broken proc }'));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_077;
var
  Log: string;
begin
  Log := '{ a expect-error: first text }' + #10 + '{ b expect-error: second text }';
  AssertEquals('first text', ExtractExpectErrorText(Log));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_078;
var
  Log: string;
begin
  Log := 'banner one'#10'banner two'#10'x.pas(1): Error: something'#10'note one'#10'note two';
  AssertEquals('x.pas(1): Error: something', FirstCompilerErrorLine(Log));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_079;
begin
  AssertEquals('Fatal: cannot open unit',
    FirstCompilerErrorLine('some log'#10'Fatal: cannot open unit'#10'more'));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_080;
begin
  AssertEquals('', FirstCompilerErrorLine(
    'Free Pascal Compiler version 3.2.2'#10'note: reading x'#10'Linking y'));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_081;
begin
  AssertEquals('a b', NormalizeDetail('a'#9'b'));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_082;
begin
  AssertEquals(StringOfChar('x', 200), NormalizeDetail(StringOfChar('x', 200)));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_083;
begin
  AssertEquals(StringOfChar('x', 200), NormalizeDetail(StringOfChar('x', 201)));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_084;
var
  Txt: string;
begin
  WriteTestFile('work/ut/dv-084.txt', 'hello 084');
  AssertTrue(ReadTextFile('work/ut/dv-084.txt', Txt));
  AssertEquals('hello 084', Txt);
end;

procedure TTestDialectVerify.TestUT_DialectVerify_085;
var
  Txt: string;
begin
  AssertTrue(not ReadTextFile('work/ut/definitely-missing-085.txt', Txt));
  AssertEquals('', Txt);
end;

procedure TTestDialectVerify.TestUT_DialectVerify_086;
begin
  WriteTestFile('work/ut/dv-086.txt', 'x');
  AssertTrue(IsPathUsable('work/ut/dv-086.txt'));
  AssertTrue(IsPathUsable('work/ut'));
  AssertTrue(not IsPathUsable('/no/such/dir-086'));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_109;
begin
  AssertTrue(IsGoodOutcome('accepted'));
  AssertTrue(IsGoodOutcome('static-ok'));
  AssertTrue(IsGoodOutcome('skipped'));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_110;
begin
  AssertTrue(not IsGoodOutcome('compiled-but-should-fail'));
  AssertTrue(not IsGoodOutcome('wrong-error'));
  AssertTrue(not IsGoodOutcome('compile-error'));
  AssertTrue(not IsGoodOutcome('run-error'));
  AssertTrue(not IsGoodOutcome('timeout'));
  AssertTrue(not IsGoodOutcome('wrong-output'));
  AssertTrue(not IsGoodOutcome('no-expected'));
  AssertTrue(not IsGoodOutcome('static-reject'));
  AssertTrue(not IsGoodOutcome('bad-token'));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_121;
var
  R: TEnvRec;
begin
  R := ReadEnvRec('/fp', '', '/h');
  AssertEquals('/fp', R.Fpc);
end;

procedure TTestDialectVerify.TestUT_DialectVerify_122;
var
  R: TEnvRec;
begin
  R := ReadEnvRec('', '', '/h');
  AssertEquals('fpc', R.Fpc);
  AssertEquals('/h/unleashed/fpc/bin/fpc', R.FpcUnleashed);
end;

procedure TTestDialectVerify.TestUT_DialectVerify_123;
var
  R: TEnvRec;
begin
  R := ReadEnvRec('', '/up', '/h');
  AssertEquals('/up', R.FpcUnleashed);
end;

procedure TTestDialectVerify.TestUT_DialectVerify_124;
begin
  { 124: leading blanks before the first word are skipped; 'hello' is not
    'namespace', no .dpr, no APPTYPE -> default objfpc }
  AssertEquals('objfpc', DetectDialect('   hello world', 'f.pas'));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_125;
begin
  { 125: a CRLF-terminated mode directive line is still detected }
  AssertEquals('fpc-delphi',
    DetectDialect('{$MODE Delphi}' + #13#10 + 'program x; begin end.', 'x.pas'));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_126;
begin
  { 126: the CR of a CRLF-terminated directive text is stripped }
  AssertEquals('broken proc',
    ExtractExpectErrorText('// expect-error: broken proc' + #13#10 + 'next line'));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_127;
begin
  { 127: the marker embedded in a longer word is not a directive }
  AssertEquals('', ExtractExpectErrorText('xexpect-error: broken proc'));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_128;
begin
  { 128: an Error: line ending in CRLF is returned with the CR stripped }
  AssertEquals('Error: boom',
    FirstCompilerErrorLine('banner line' + #13#10 + 'Error: boom' + #13#10));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_129;
begin
  { 129: a bare file name (no directory part) }
  AssertEquals('unit.expected', ExpectedPathFor('unit.pas'));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_130;
begin
  { 130: an empty path is not usable }
  AssertTrue('empty path must not be usable', not IsPathUsable(''));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_131;
const
  P = 'work/ut/dv-131.txt';
var
  Pr: TProcess;

  procedure Chmod(const AMode: string);
  var T: Integer;
  begin
    T := 0;
    Pr := TProcess.Create(nil);
    try
      Pr.Executable := 'chmod';
      Pr.Parameters.Add(AMode);
      Pr.Parameters.Add(P);
      Pr.Execute;
    finally
      Pr.Free;
    end;
    { TProcess.Execute does not block on POSIX, so poll the file state the
      chmod is meant to produce (bounded), rather than an exit-status field }
    if AMode = '000' then
      while IsPathUsable(P) do begin Inc(T); if T >= 500 then Break; Sleep(10); end
    else
      while not IsPathUsable(P) do begin Inc(T); if T >= 500 then Break; Sleep(10); end;
  end;
begin
  { 131: an existing file without read permission is not usable }
  WriteTestFile(P, 'no-read');
  Chmod('000');
  AssertTrue('chmod 000 file must not be usable', not IsPathUsable(P));
  { restore readability for any later run }
  Chmod('644');
end;

procedure TTestDialectVerify.ChmodExec(const APath: string);
var
  I: Integer;
  Pr: TProcess;
begin
  Pr := TProcess.Create(nil);
  try
    Pr.Executable := 'chmod';
    Pr.Parameters.Add('755');
    Pr.Parameters.Add(APath);
    Pr.Execute;
    for I := 0 to 499 do   { POSIX Execute is non-blocking; bounded wait }
      if not Pr.Running then
        Break
      else
        Sleep(10);
  finally
    Pr.Free;
  end;
end;

{ Fake fpc/shell compiler: ok mode writes <FE dir>/<src base> as a shell
  script made of AExeLines (each line verbatim in the exe); err mode echoes
  AErrLine and exits 3; sleep mode sleeps ACompilerSleep seconds. }
function TTestDialectVerify.WriteFakeFpc(const APath: string;
  const AExeLines, AErrLine, AArgsLogTo: string;
  ACompilerSleep: Integer): string;
var
  L: string;
  I: Integer;
  S: string;
begin
  S := '#!/bin/sh'#10;
  if AArgsLogTo <> '' then
    S := S + 'printf "%s\\n" "$@" >> ' + AArgsLogTo + #10;
  if AErrLine <> '' then
    S := S + 'echo ' + Chr(39) + AErrLine + Chr(39) + #10 + 'exit 3' + #10
  else if ACompilerSleep > 0 then
    S := S + 'sleep ' + IntToStr(ACompilerSleep) + #10
  else
  begin
    S := S +
      'fe=' + Chr(39) + Chr(39) + #10 +
      'src=' + Chr(39) + Chr(39) + #10 +
      'for a in "$@"; do case "$a" in -FE*) fe=${a#-FE};; *) src=$a;; esac; done' + #10 +
      'base=$(basename "$src"); base=${base%.*}' + #10 +
      'printf ' + Chr(39) + '#!/bin/sh\n' + Chr(39) + ' > "$fe/$base"' + #10;
    L := AExeLines;
    while L <> '' do
    begin
      I := Pos(#10, L);
      if I = 0 then
      begin
        S := S + 'echo ' + Chr(39) + L + Chr(39) + ' >> "$fe/$base"' + #10;
        L := '';
      end
      else
      begin
        S := S + 'echo ' + Chr(39) + Copy(L, 1, I - 1) + Chr(39) +
             ' >> "$fe/$base"' + #10;
        Delete(L, 1, I);
      end;
    end;
    S := S + 'chmod +x "$fe/$base"' + #10 + 'exit 0' + #10;
  end;
  WriteTestFile(APath, S);
  ChmodExec(APath);
  Result := APath;
end;

function TTestDialectVerify.Cell(const ALn: string; ACol: Integer): string;
var
  L: TStringList;
begin
  Result := '';
  L := TStringList.Create;
  try
    L.Delimiter := #9;
    L.StrictDelimiter := True;
    L.DelimitedText := ALn;
    if (ACol >= 1) and (ACol <= L.Count) then
      Result := L[ACol - 1];
  finally
    L.Free;
  end;
end;

procedure TTestDialectVerify.TestUT_DialectVerify_027;
var
  C: Integer;
  Out: string;
  T: Boolean;
begin
  Out := '';
  C := -1;
  T := False;
  AssertFalse(RunLimited('/no/such/binary', ['arg'], '', 1000, Out, C, T));
  AssertTrue(Pos('process error', Out) > 0);
end;

procedure TTestDialectVerify.TestUT_DialectVerify_028;
var
  C: Integer;
  Out: string;
  T: Boolean;
begin
  Out := '';
  C := -1;
  T := False;
  { clean start/read path; a RELATIVE non-empty work dir is exercised here
    so IsRelativePath(Dir) + the chdir branch of RunLimited are covered
    (binaries exec fine with a cwd change) }
  ForceDirectories('work/ut/dv-028');
  AssertTrue(RunLimited('/usr/bin/echo', ['hello'], 'work/ut/dv-028',
    5000, Out, C, T));
  AssertFalse(T);
  AssertEquals(0, C);
  AssertEquals('hello', Trim(Out));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_087;
var
  L: string;
  P: string;
  St: TCompileState;
begin
  WriteTestFile('work/ut/dv-087/s.pas', 'program s; begin end.');
  WriteFakeFpc('work/ut/dv-087/fpc', 'echo ok', '', '', 0);
  St := CompileStep('work/ut/dv-087/fpc', 'work/ut/dv-087/s.pas',
    'work/ut/dv-087/out', 5000, False, L, P);
  AssertTrue(St = csOk);
  AssertEquals(GetCurrentDir + '/work/ut/dv-087/out/s', P);
end;

procedure TTestDialectVerify.TestUT_DialectVerify_088;
var
  L: string;
  P: string;
  St: TCompileState;
begin
  WriteTestFile('work/ut/dv-088/s.pas', 'program s; begin end.');
  WriteFakeFpc('work/ut/dv-088/fpc', '', 'Error: something', '', 0);
  St := CompileStep('work/ut/dv-088/fpc', 'work/ut/dv-088/s.pas',
    'work/ut/dv-088/out', 5000, False, L, P);
  AssertTrue(St = csFail);
  AssertEquals('', P);
  AssertTrue(Pos('Error: something', L) > 0);
end;

procedure TTestDialectVerify.TestUT_DialectVerify_132;
var
  L: string;
  P: string;
  St: TCompileState;
begin
  { delphi rewrite with an unreadable source → csFail (never csNoCompiler,
    the fake compiler itself is present) }
  WriteFakeFpc('work/ut/dv-132/fpc', '', '', '', 0);
  St := CompileStep('work/ut/dv-132/fpc', 'work/ut/dv-132/nosuch.pas',
    'work/ut/dv-132/out', 5000, True, L, P);
  AssertTrue(St = csFail);
  AssertEquals('', P);
end;

procedure TTestDialectVerify.TestUT_DialectVerify_133;
var
  E: TEnvRec;
  OK: Boolean;
  W, S: string;
begin
  { T06c: the program must run INSIDE the fresh <out> dir. The exe reads
    data.txt by RELATIVE name; the fake compiler writes it next to the
    exe in <out>. With the inherited working dir (pre-fix) the relative
    read fails (cat error in the captured output → wrong-output). }
  S := '#!/bin/sh'#10
    + 'fe=' + Chr(39) + Chr(39) + #10
    + 'for a in "$@"; do case "$a" in -FE*) fe=${a#-FE};; esac; done' + #10
    + 'printf ' + Chr(39) + 'FROM-OWN-DIR\n' + Chr(39)
    + ' > "$fe/data.txt"' + #10
    + 'printf ' + Chr(39) + '#!/bin/sh\ncat data.txt\necho read-ok\n'
    + Chr(39) + ' > "$fe/s"' + #10
    + 'chmod +x "$fe/s"' + #10
    + 'exit 0';
  WriteTestFile('work/ut/dv-133/s.pas', 'program s; begin end.');
  WriteTestFile('work/ut/dv-133/s.expected',
    'FROM-OWN-DIR'#10 + 'read-ok'#10);
  WriteTestFile('work/ut/dv-133/fpc', S);
  ChmodExec('work/ut/dv-133/fpc');
  E.Fpc := 'work/ut/dv-133/fpc';
  W := VerifyFile('work/ut/dv-133/s.pas', '', 5, E, 'work/ut/dv-133/w',
    OK);
  AssertTrue(OK);
  AssertEquals('accepted', Cell(W, 4));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_134;
var
  Args: array of string;
  E: TEnvRec;
  L: TStringArray;
  EC: Integer;
begin
  { T06c: --help → usage printed, code 0 (VerifyMain returns True). }
  E := ReadEnvRec('', '', '');
  SetLength(Args, 1);
  Args[0] := '--help';
  AssertEquals(0, RunVerifyCLI(Args, 'work/ut/dv-134', E));
  { stdout went to the runner log; pin its content through the same
    VerifyMain path RunVerifyCLI calls: --help must be usage, code 0 }
  AssertTrue(VerifyMain(Args, 'work/ut/dv-134/m', E, L, EC));
  AssertEquals(K_USAGE, L[0]);
  AssertEquals(0, EC);
end;

procedure TTestDialectVerify.TestUT_DialectVerify_135;
var
  Args: array of string;
  E: TEnvRec;
  L: TStringArray;
  EC: Integer;
begin
  { T06c: no args → usage + the `if not VerifyMain` branch (Result := 2). }
  E := ReadEnvRec('', '', '');
  SetLength(Args, 0);
  AssertEquals(2, RunVerifyCLI(Args, 'work/ut/dv-135', E));
  AssertFalse(VerifyMain(Args, 'work/ut/dv-135/m', E, L, EC));
  AssertEquals(2, EC);
  AssertEquals(K_USAGE, L[0]);
end;

procedure TTestDialectVerify.TestUT_DialectVerify_136;
var
  Args: array of string;
  E: TEnvRec;
  S, P: string;
  L: TStringArray;
  EC: Integer;
begin
  { T06c: one real file + fake fpc → `accepted` line, code 0. }
  S := GetCurrentDir + '/work/ut/dv-136/s.pas';
  WriteTestFile('work/ut/dv-136/s.pas', 'program s; begin end.');
  WriteTestFile('work/ut/dv-136/s.expected', 'ok'#10);
  P := WriteFakeFpc('work/ut/dv-136/fpc', 'echo ok', '', '', 0);
  E := ReadEnvRec(P, '', '');
  SetLength(Args, 3);
  Args[0] := '--dialect';
  Args[1] := 'objfpc';
  Args[2] := S;
  AssertEquals(0, RunVerifyCLI(Args, 'work/ut/dv-136', E));
  AssertTrue(VerifyMain(Args, 'work/ut/dv-136/m', E, L, EC));
  AssertEquals(0, EC);
  AssertTrue(Pos('accepted', L[0]) > 0);
end;

procedure TTestDialectVerify.TestUT_DialectVerify_140;
begin
  { 140 (T06c2, BUG-001): the pitfall text keeps its letter case — the
    compiler-log match in VerifyFile is case-sensitive (grep -F semantics,
    T06-facts), so a lowercased extraction misses texts with capitals.
    Before the fix this returned the lowercased 'syntax error'. }
  AssertEquals('SyNtAx ErRoR',
    ExtractExpectErrorText('{ expect-error: SyNtAx ErRoR }'));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_141;
var
  Cls: string;
  E: TEnvRec;
begin
  { 141 (T06c2, BUG-002): a bare compiler name like `fpc` is resolved
    through PATH like a shell would. The pipeline gate is IsPathUsable:
    before the fix it demanded an existing file ./fpc, so the bare name
    was `skipped` ("compiler not found"). Case-table input: `fpc` is in
    $PATH (this UT suite's own runner invokes fpc by bare name). }
  E := ReadEnvRec('', '', '/h');
  AssertEquals('fpc', ChooseCompiler('objfpc', E, Cls));
  AssertEquals('compiled-run', Cls);
  AssertTrue('bare fpc in PATH must be usable', IsPathUsable('fpc'));
  AssertTrue('bare name not in PATH must not be usable',
    not IsPathUsable('qlora-ut-missing-bare-name'));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_142;
var
  Out: string;
  C: Integer;
  T: Boolean;
begin
  { 142 (T06c4, BUG-003): a BARE name with no '/' is a program NAME, not a
    path — like a shell, `RunLimited` must resolve it in PATH (SysUtils
    ExeSearch) BEFORE its relative→absolute rebase. Before the fix the bare
    name was rebased to <cwd>/echo (ENOENT → `process error`, Result=False);
    after the fix ExeSearch resolves it and it runs. A relative exe WITH '/'
    is unchanged (cwd prefix: 027/028).
    `echo` is the exemplar: guaranteed on PATH (028 already runs
    /usr/bin/echo; this suite's runner invokes fpc by bare name) yet NOT
    present in the working directory — so it exercises the ExeSearch branch,
    not the cwd-file or cwd-prefix fallbacks. }
  AssertFalse('echo: a bare name must NOT exist in the current dir',
    FileExists('echo'));
  Out := '';
  C := -1;
  T := False;
  AssertTrue('bare name resolvable in PATH must run (BUG-003): ' + Out,
    RunLimited('echo', ['dv142-marker'], '', 5000, Out, C, T));
  AssertEquals(0, C);
  AssertEquals('dv142-marker', Trim(Out));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_089;
var
  L: string;
  P: string;
  St: TCompileState;
begin
  WriteFakeFpc('work/ut/dv-089/fpc', '', '', '', 5);
  St := CompileStep('work/ut/dv-089/fpc', 'work/ut/dv-089/s.pas',
    'work/ut/dv-089/out', 1000, False, L, P);
  AssertTrue(St = csTimeout);
end;

procedure TTestDialectVerify.TestUT_DialectVerify_090;
var
  L: string;
  P: string;
  St: TCompileState;
begin
  St := CompileStep('/no/fpc', 'work/ut/dv-090/s.pas', 'work/ut/dv-090/out',
    5000, False, L, P);
  AssertTrue(St = csNoCompiler);
  AssertEquals('', P);
end;

procedure TTestDialectVerify.TestUT_DialectVerify_091;
var
  Args, L, P, St: string;
  CS: TCompileState;
begin
  WriteTestFile('work/ut/dv-091/s.pas',
    'program s; uses System.SysUtils, System.Generics.Collections;' + #10 +
    'begin end.');
  WriteFakeFpc('work/ut/dv-091/fpc', '', '', 'work/ut/dv-091/args.log', 0);
  CS := CompileStep('work/ut/dv-091/fpc', 'work/ut/dv-091/s.pas',
    'work/ut/dv-091/out', 5000, True, L, P);
  AssertTrue(CS = csOk);
  AssertTrue(ReadTextFile('work/ut/dv-091/out/s.pas', St));
  AssertTrue(Pos('Generics.Collections', St) > 0);
  AssertTrue(ReadTextFile('work/ut/dv-091/args.log', Args));
  AssertTrue(Pos('work/ut/dv-091/out/s.pas', Args) > 0);
end;

procedure TTestDialectVerify.TestUT_DialectVerify_092;
var
  E: TEnvRec;
  OK: Boolean;
  O, W: string;
begin
  WriteTestFile('work/ut/dv-092/s.pas', 'program s; begin end.');
  WriteTestFile('work/ut/dv-092/s.expected', 'ok'#10);
  WriteFakeFpc('work/ut/dv-092/fpc', 'echo ok', '', '', 0);
  E.Fpc := 'work/ut/dv-092/fpc';
  W := VerifyFile('work/ut/dv-092/s.pas', '', 5, E, 'work/ut/dv-092/w', OK);
  AssertTrue(OK);
  AssertEquals('work/ut/dv-092/s.pas'#9'objfpc'#9'compiled-run'#9 +
    'accepted'#9, W);
end;
procedure TTestDialectVerify.TestUT_DialectVerify_093;
var
  E: TEnvRec;
  OK: Boolean;
  O, W: string;
begin
  WriteTestFile('work/ut/dv-093/s.pas', 'program s; begin end.');
  WriteTestFile('work/ut/dv-093/s.expected', 'expected');
  WriteFakeFpc('work/ut/dv-093/fpc', 'echo other', '', '', 0);
  E.Fpc := 'work/ut/dv-093/fpc';
  W := VerifyFile('work/ut/dv-093/s.pas', '', 5, E, 'work/ut/dv-093/w', OK);
  AssertFalse(OK);
  AssertEquals('wrong-output', Cell(W, 4));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_094;
var
  E: TEnvRec;
  OK: Boolean;
  O, W: string;
begin
  WriteTestFile('work/ut/dv-094/s.pas', 'program s; begin end.');
  WriteFakeFpc('work/ut/dv-094/fpc', 'echo ok', '', '', 0);
  E.Fpc := 'work/ut/dv-094/fpc';
  W := VerifyFile('work/ut/dv-094/s.pas', '', 5, E, 'work/ut/dv-094/w', OK);
  AssertFalse(OK);
  AssertEquals('no-expected', Cell(W, 4));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_095;
var
  E: TEnvRec;
  OK: Boolean;
  O, W: string;
begin
  WriteTestFile('work/ut/dv-095/s.pas', 'program s; begin end.');
  WriteTestFile('work/ut/dv-095/s.expected', 'whatever');
  WriteFakeFpc('work/ut/dv-095/fpc', 'exit 3', '', '', 0);
  E.Fpc := 'work/ut/dv-095/fpc';
  W := VerifyFile('work/ut/dv-095/s.pas', '', 5, E, 'work/ut/dv-095/w', OK);
  AssertFalse(OK);
  AssertEquals('run-error', Cell(W, 4));
  AssertEquals('3', Cell(W, 5));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_096;
var
  E: TEnvRec;
  OK: Boolean;
  O, W: string;
begin
  { compile succeeds; the exe hangs → run timeout }
  WriteTestFile('work/ut/dv-096/s.pas', 'program s; begin end.');
  WriteTestFile('work/ut/dv-096/s.expected', 'whatever');
  WriteFakeFpc('work/ut/dv-096/fpc', 'sleep 5', '', '', 0);
  E.Fpc := 'work/ut/dv-096/fpc';
  W := VerifyFile('work/ut/dv-096/s.pas', '', 1, E, 'work/ut/dv-096/w', OK);
  AssertFalse(OK);
  AssertEquals('timeout', Cell(W, 4));
  AssertEquals('1', Cell(W, 5));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_097;
var
  E: TEnvRec;
  OK: Boolean;
  O, W: string;
begin
  WriteTestFile('work/ut/dv-097/s.pas', 'program s; begin end.');
  WriteFakeFpc('work/ut/dv-097/fpc', '', 'Error: boom', '', 0);
  E.Fpc := 'work/ut/dv-097/fpc';
  W := VerifyFile('work/ut/dv-097/s.pas', '', 5, E, 'work/ut/dv-097/w', OK);
  AssertFalse(OK);
  AssertEquals('compile-error', Cell(W, 4));
  AssertEquals('Error: boom', Cell(W, 5));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_098;
var
  E: TEnvRec;
  OK: Boolean;
  O, W: string;
begin
  WriteTestFile('work/ut/dv-098/s.pas',
    'program s; begin { expect-error: boom } end.');
  WriteFakeFpc('work/ut/dv-098/fpc', '', 'Error: boom', '', 0);
  E.Fpc := 'work/ut/dv-098/fpc';
  W := VerifyFile('work/ut/dv-098/s.pas', '', 5, E, 'work/ut/dv-098/w', OK);
  AssertTrue(OK);
  AssertEquals('accepted', Cell(W, 4));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_099;
var
  E: TEnvRec;
  OK: Boolean;
  O, W: string;
begin
  WriteTestFile('work/ut/dv-099/s.pas',
    'program s; begin { expect-error: other } end.');
  WriteFakeFpc('work/ut/dv-099/fpc', '', 'Error: boom', '', 0);
  E.Fpc := 'work/ut/dv-099/fpc';
  W := VerifyFile('work/ut/dv-099/s.pas', '', 5, E, 'work/ut/dv-099/w', OK);
  AssertFalse(OK);
  AssertEquals('wrong-error', Cell(W, 4));
  AssertEquals('Error: boom', Cell(W, 5));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_100;
var
  E: TEnvRec;
  OK: Boolean;
  O, W: string;
begin
  WriteTestFile('work/ut/dv-100/s.pas',
    'program s; begin { expect-error: boom } end.');
  WriteFakeFpc('work/ut/dv-100/fpc', 'echo ok', '', '', 0);
  E.Fpc := 'work/ut/dv-100/fpc';
  W := VerifyFile('work/ut/dv-100/s.pas', '', 5, E, 'work/ut/dv-100/w', OK);
  AssertFalse(OK);
  AssertEquals('compiled-but-should-fail', Cell(W, 4));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_101;
var
  E: TEnvRec;
  OK: Boolean;
  O, W: string;
begin
  E.Fpc := '/no/fpc';
  O := VerifyFile('nonexistent.pas', '', 5, E, 'work/ut/dv-101/w', OK);
  AssertTrue(OK);
  AssertEquals('skipped', Cell(O, 4));
  AssertEquals('compiler not found', Cell(O, 5));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_102;
var
  E: TEnvRec;
  OK: Boolean;
  O, W: string;
begin
  WriteTestFile('work/ut/dv-102/s.pas', 'program s; begin end.');
  WriteFakeFpc('work/ut/dv-102/fpc', '', '', '', 5);
  E.Fpc := 'work/ut/dv-102/fpc';
  O := VerifyFile('work/ut/dv-102/s.pas', '', 1, E, 'work/ut/dv-102/w', OK);
  AssertTrue(OK);
  AssertEquals('skipped', Cell(O, 4));
  AssertEquals('timeout', Cell(O, 5));
end;

const
  { a correct Oxygene console program; prints "Hello from Oxygene" and 4 }
  OXY_OK = 'namespace Hello;' + #10 + #10 + 'type' + #10 + '  Program = class' + #10 +
    '  public' + #10 + '    class method Main(args: array of String): Int32;' + #10 +
    '    begin' + #10 + '      writeLn(''Hello from Oxygene'');' + #10 + '      writeLn(2 + 2);' + #10 +
    '    end;' + #10 + '  end;' + #10 + #10 + 'end.' + #10;
  OXY_OUT = 'Hello from Oxygene' + #10 + '4' + #10;

{ OXY_OK with Main's body replaced by ABody }
function OxyWith(const ABody: string): string;
begin
  Result := 'namespace Hello;' + #10 + #10 + 'type' + #10 + '  Program = class' + #10 +
    '  public' + #10 + '    class method Main(args: array of String): Int32;' + #10 +
    '    begin' + #10 + ABody + '    end;' + #10 + '  end;' + #10 + #10 + 'end.' + #10;
end;

{ the environment of the T22 rows: Oxygene work under work/ut/<ADir>/oxy }
function OxyEnv(const ADir: string): TEnvRec;
begin
  Result := ReadEnvRec('', '', GetEnvironmentVariable('HOME'));
  Result.OxyRoot := ExpandFileName('work/ut/' + ADir + '/oxy');
end;

{ how many entries ADir holds (0 when it does not exist) }
function EntryCount(const ADir: string): Integer;
var
  R: TSearchRec;
begin
  Result := 0;
  if FindFirst(ADir + '/*', faAnyFile or faDirectory, R) = 0 then
  begin
    repeat
      if (R.Name <> '.') and (R.Name <> '..') then
        Inc(Result);
    until FindNext(R) <> 0;
    FindClose(R);
  end;
end;

procedure TTestDialectVerify.TestUT_DialectVerify_103;
var
  E: TEnvRec;
  OK: Boolean;
  O: string;
begin
  WriteTestFile('work/ut/dv-103/o.pas', 'namespace Demo; type T = class of object end; end.');
  E := OxyEnv('dv-103');
  O := VerifyFile('work/ut/dv-103/o.pas', '', 10, E, '', OK);
  AssertFalse(OK);
  AssertEquals('compile-error', Cell(O, 4));
  AssertEquals('oxygene', Cell(O, 2));
  AssertEquals('compiled-run', Cell(O, 3));
  AssertEquals('E: semicolon (;) expected, got "end" [Program.pas (1)]', Cell(O, 5));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_104;
var
  E: TEnvRec;
  OK: Boolean;
  O: string;
begin
  WriteTestFile('work/ut/dv-104/p.pas', 'program p; begin end.');
  E := OxyEnv('dv-104');
  O := VerifyFile('work/ut/dv-104/p.pas', 'oxygene', 10, E, '', OK);
  AssertFalse(OK);
  AssertEquals('compile-error', Cell(O, 4));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_105;
var
  Args, Env, O, Line: string;
  En: TEnvRec;
  OK: Boolean;
begin
  WriteTestFile('work/ut/dv-105/d.pas', 'program d; begin end.');
  WriteTestFile('work/ut/dv-105/d.expected', 'ok'#10);
  WriteFakeFpc('work/ut/dv-105/fpu', 'echo ok', '', 'work/ut/dv-105/args.log', 0);
  En.FpcUnleashed := 'work/ut/dv-105/fpu';
  Line := VerifyFile('work/ut/dv-105/d.pas', 'delphi', 5, En,
    'work/ut/dv-105/w', OK);
  AssertTrue(OK);
  AssertEquals('delphi', Cell(Line, 2));
  AssertEquals('fpc-approx', Cell(Line, 3));
  AssertEquals('accepted', Cell(Line, 4));
  AssertTrue(ReadTextFile('work/ut/dv-105/args.log', Args));
  AssertTrue(Pos('-Mdelphiunicode', Args) > 0);
  { delphi compiles the rewritten copy in the fresh out dir
    <workroot>/<n>/d.pas (design 2.14.4), never the original source }
  AssertTrue(Pos('/d.pas', Args) > 0);
  AssertTrue(Pos('work/ut/dv-105/d.pas', Args) = 0);
end;

procedure TTestDialectVerify.TestUT_DialectVerify_106;
var
  L1, L2, P: string;
  N: Integer;
  OK, OK2: Boolean;
  En: TEnvRec;
  W: string;
begin
  WriteTestFile('work/ut/dv-106/da.pas', 'program a; begin end.');
  WriteTestFile('work/ut/dv-106/db.pas', 'program b; begin end.');
  WriteTestFile('work/ut/dv-106/da.expected', 'ok'#10);
  WriteTestFile('work/ut/dv-106/db.expected', 'ok'#10);
  { T06c: the exe now runs with cwd=<out> (a fresh out dir), so its log
    target must be ABSOLUTE, otherwise it would land inside <out> }
  WriteFakeFpc('work/ut/dv-106/fpu',
    'echo ok'#10 + 'echo "$0" >> '
    + QuotedStr(GetCurrentDir + '/work/ut/dv-106/paths.log'), '', '', 0);
  En.FpcUnleashed := 'work/ut/dv-106/fpu';
  DeleteFile('work/ut/dv-106/paths.log');
  W := VerifyFile('work/ut/dv-106/da.pas', 'delphi', 5, En,
    'work/ut/dv-106/w', OK);
  W := W + #10 + VerifyFile('work/ut/dv-106/db.pas', 'delphi', 5, En,
    'work/ut/dv-106/w', OK2);
  AssertTrue(OK and OK2);
  L1 := Copy(W, 1, Pos(#10, W) - 1);
  L2 := Copy(W, Pos(#10, W) + 1, MaxInt);
  AssertEquals('accepted', Cell(L1, 4));
  AssertEquals('accepted', Cell(L2, 4));
  AssertTrue(ReadTextFile('work/ut/dv-106/paths.log', P));
  N := 0;
  if Pos(#10, P) > 0 then
    N := 2
  else
    N := 1;
  AssertEquals(2, N);
  L1 := Copy(P, 1, Pos(#10, P) - 1);
  L2 := Copy(P, Pos(#10, P) + 1, MaxInt);
  AssertTrue(L1 <> L2);
end;

procedure TTestDialectVerify.TestUT_DialectVerify_118;
var
  E: TEnvRec;
  OK: Boolean;
  O, W: string;
begin
  WriteTestFile('work/ut/dv-118/s.pas', 'program s; begin end.');
  WriteTestFile('work/ut/dv-118/s.expected', '');
  WriteFakeFpc('work/ut/dv-118/fpc', '', '', '', 0);
  E.Fpc := 'work/ut/dv-118/fpc';
  W := VerifyFile('work/ut/dv-118/s.pas', '', 5, E, 'work/ut/dv-118/w', OK);
  AssertTrue(OK);
  AssertEquals('accepted', Cell(W, 4));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_119;
var
  E: TEnvRec;
  OK: Boolean;
  O, W: string;
begin
  WriteTestFile('work/ut/dv-119/s.pas', 'program s; begin end.');
  WriteTestFile('work/ut/dv-119/s.expected', '');
  WriteFakeFpc('work/ut/dv-119/fpc', 'echo x', '', '', 0);
  E.Fpc := 'work/ut/dv-119/fpc';
  W := VerifyFile('work/ut/dv-119/s.pas', '', 5, E, 'work/ut/dv-119/w', OK);
  AssertFalse(OK);
  AssertEquals('wrong-output', Cell(W, 4));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_120;
var
  E: TEnvRec;
  OK: Boolean;
  O, W: string;
begin
  WriteTestFile('work/ut/dv-120/s.pas', 'program s; begin end.');
  WriteTestFile('work/ut/dv-120/s.expected', 'ok'#10);
  WriteFakeFpc('work/ut/dv-120/fpc', 'printf "ok\r\n"', '', '', 0);
  E.Fpc := 'work/ut/dv-120/fpc';
  W := VerifyFile('work/ut/dv-120/s.pas', '', 5, E, 'work/ut/dv-120/w', OK);
  AssertFalse(OK);
  AssertEquals('wrong-output', Cell(W, 4));
end;

{ ---------- stage 3: CollectPascalFiles / VerifyMain ---------- }

procedure TTestDialectVerify.TestUT_DialectVerify_036;
var
  E: TEnvRec;
  L: TStringArray;
  EC: Integer;
  R: Boolean;
begin
  { --timeout abc (not an integer) -> usage, exit 2, not run }
  E.Fpc := 'fpc';
  E.FpcUnleashed := 'fpc-unleashed';
  R := VerifyMain(['--timeout', 'abc', 'file.pas'], '', E, L, EC);
  AssertFalse(R);
  AssertEquals(2, EC);
  AssertEquals(1, Length(L));
  AssertTrue(Pos('usage:', L[0]) > 0);
end;

procedure TTestDialectVerify.TestUT_DialectVerify_037;
var
  E: TEnvRec;
  L: TStringArray;
  EC: Integer;
  R: Boolean;
begin
  { unknown -- option -> usage, exit 2 }
  E.Fpc := 'fpc';
  E.FpcUnleashed := 'fpc-unleashed';
  R := VerifyMain(['--frobnicate', 'a.pas'], '', E, L, EC);
  AssertFalse(R);
  AssertEquals(2, EC);
  AssertEquals(1, Length(L));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_107;
var
  D: string;
  F: TStringArray;
  R: Boolean;
begin
  { file + nested dir, mixed extensions -> only .pas/.dpr, recursively,
    sorted (root before sub) }
  D := 'work/ut/dv107';
  WriteTestFile(D + '/a.pas', 'program a; begin end.'#10);
  WriteTestFile(D + '/c.pp', 'program c; begin end.'#10);
  WriteTestFile(D + '/d.txt', 'data'#10);
  WriteTestFile(D + '/sub/b.dpr', 'program b; begin end.'#10);
  R := CollectPascalFiles(D, F);
  AssertTrue(R);
  AssertEquals(2, Length(F));
  AssertEquals(D + '/a.pas', F[0]);
  AssertEquals(D + '/sub/b.dpr', F[1]);
end;

procedure TTestDialectVerify.TestUT_DialectVerify_108;
var
  D: string;
  F: TStringArray;
  I: Integer;
  Found: Boolean;
begin
  { uppercase .DPR is collected (.pas/.dpr are case-insensitive) }
  D := 'work/ut/dv108';
  WriteTestFile(D + '/UPPER.DPR', 'program u; begin end.'#10);
  WriteTestFile(D + '/lower.pas', 'program l; begin end.'#10);
  AssertTrue(CollectPascalFiles(D, F));
  AssertEquals(2, Length(F));
  Found := False;
  for I := 0 to Length(F) - 1 do
    if F[I] = D + '/UPPER.DPR' then
      Found := True;
  AssertTrue(Found);
end;

procedure TTestDialectVerify.TestUT_DialectVerify_111;
var
  D: string;
  E: TEnvRec;
  L: TStringArray;
  EC: Integer;
  R: Boolean;
  Fpc: string;
begin
  { one path a file, one a good directory -> one line per file, exit 0 }
  D := 'work/ut/dv111';
  WriteTestFile(D + '/x.pas', 'program x; begin end.'#10);
  WriteTestFile(D + '/x.expected', 'hello'#10);
  WriteTestFile(D + '/d/s.pas', 'program s; begin end.'#10);
  WriteTestFile(D + '/d/s.expected', 'hello'#10);
  Fpc := WriteFakeFpc('work/ut/build/dv111-fpc', 'echo hello', '', '', 0);
  E.Fpc := Fpc;
  E.FpcUnleashed := Fpc;
  R := VerifyMain([D + '/x.pas', D + '/d'], D + '/w', E, L, EC);
  AssertTrue(R);
  AssertEquals(0, EC);
  AssertEquals(2, Length(L));
  AssertEquals('accepted', Cell(L[0], 4));
  AssertEquals('accepted', Cell(L[1], 4));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_112;
var
  E: TEnvRec;
  L: TStringArray;
  EC: Integer;
  R: Boolean;
begin
  { --timeout 5 with no path -> usage, exit 2 }
  E.Fpc := 'fpc';
  E.FpcUnleashed := 'fpc-unleashed';
  R := VerifyMain(['--timeout', '5'], '', E, L, EC);
  AssertFalse(R);
  AssertEquals(2, EC);
  AssertEquals(1, Length(L));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_113;
var
  E: TEnvRec;
  L: TStringArray;
  EC: Integer;
  R: Boolean;
begin
  { --dialect with an unknown dialect -> usage, exit 2 }
  E.Fpc := 'fpc';
  E.FpcUnleashed := 'fpc-unleashed';
  R := VerifyMain(['--dialect', 'wat', 'x.pas'], '', E, L, EC);
  AssertFalse(R);
  AssertEquals(2, EC);
  AssertEquals(1, Length(L));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_114;
var
  D: string;
  E: TEnvRec;
  L: TStringArray;
  EC: Integer;
  R: Boolean;
  Fpu: string;
begin
  { forced delphi dialect: the line says delphi / fpc-approx; the fake
    unleashed compiler runs and the outcome decides the exit code }
  D := 'work/ut/dv114';
  WriteTestFile(D + '/d.pas', 'program d; uses System.SysUtils; begin end.'#10);
  WriteTestFile(D + '/d.expected', 'ok'#10);
  Fpu := WriteFakeFpc('work/ut/build/dv114-fpu', 'echo ok', '', '', 0);
  E.Fpc := Fpu;
  E.FpcUnleashed := Fpu;
  R := VerifyMain(['--dialect', 'delphi', D + '/d.pas'], D + '/w', E, L, EC);
  AssertTrue(R);
  AssertEquals(0, EC);
  AssertEquals(1, Length(L));
  AssertEquals('delphi', Cell(L[0], 2));
  AssertEquals('fpc-approx', Cell(L[0], 3));
  AssertEquals('accepted', Cell(L[0], 4));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_115;
var
  D: string;
  E: TEnvRec;
  L: TStringArray;
  EC: Integer;
  R: Boolean;
  Fpc: string;
begin
  { the same file twice -> two identical lines, exit per the outcome }
  D := 'work/ut/dv115';
  WriteTestFile(D + '/x.pas', 'program x; begin end.'#10);
  WriteTestFile(D + '/x.expected', 'hello'#10);
  Fpc := WriteFakeFpc('work/ut/build/dv115-fpc', 'echo hello', '', '', 0);
  E.Fpc := Fpc;
  E.FpcUnleashed := Fpc;
  R := VerifyMain([D + '/x.pas', D + '/x.pas'], D + '/w', E, L, EC);
  AssertTrue(R);
  AssertEquals(0, EC);
  AssertEquals(2, Length(L));
  AssertEquals(L[0], L[1]);
end;

procedure TTestDialectVerify.TestUT_DialectVerify_116;
var
  D: string;
  E: TEnvRec;
  L: TStringArray;
  EC: Integer;
  R: Boolean;
  Fpc: string;
begin
  { one accepted file + one wrong-output file -> 2 lines, exit 1 }
  D := 'work/ut/dv116';
  WriteTestFile(D + '/g.pas', 'program g; begin end.'#10);
  WriteTestFile(D + '/g.expected', 'hello'#10);   { matches exe -> accepted }
  WriteTestFile(D + '/b.pas', 'program b; begin end.'#10);
  WriteTestFile(D + '/b.expected', 'nope'#10);    { mismatch  -> wrong-output }
  Fpc := WriteFakeFpc('work/ut/build/dv116-fpc', 'echo hello', '', '', 0);
  E.Fpc := Fpc;
  E.FpcUnleashed := Fpc;
  R := VerifyMain([D + '/g.pas', D + '/b.pas'], D + '/w', E, L, EC);
  AssertTrue(R);
  AssertEquals(1, EC);
  AssertEquals(2, Length(L));
  AssertEquals('accepted', Cell(L[0], 4));
  AssertEquals('wrong-output', Cell(L[1], 4));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_117;
var
  D: string;
  E: TEnvRec;
  L: TStringArray;
  EC: Integer;
  R: Boolean;
begin
  { no usable compiler -> one skipped line, still a status line, exit 0 }
  D := 'work/ut/dv117';
  WriteTestFile(D + '/s.pas', 'program s; begin end.'#10);
  E.Fpc := '/no/fpc';                 { non-existent -> compiler not found }
  E.FpcUnleashed := '/no/fpc-unleashed';
  R := VerifyMain([D + '/s.pas'], '', E, L, EC);
  AssertTrue(R);
  AssertEquals(0, EC);
  AssertEquals(1, Length(L));
  AssertEquals('skipped', Cell(L[0], 4));
  AssertTrue(Pos('compiler not found', L[0]) > 0);
end;

{ ---------- extra coverage: defensive VerifyMain CLI branches ---------- }

procedure TTestDialectVerify.TestUT_DialectVerify_200;
var
  E: TEnvRec;
  L: TStringArray;
  EC: Integer;
  R: Boolean;
begin
  E.Fpc := 'fpc';
  E.FpcUnleashed := 'fpc-unleashed';
  { --help: usage text, exit 0, returns True }
  R := VerifyMain(['--help'], '', E, L, EC);
  AssertTrue(R);
  AssertEquals(0, EC);
  AssertEquals(1, Length(L));
  AssertTrue(Pos('usage:', L[0]) > 0);
  { --dialect as the last argument (no value) -> usage, exit 2 }
  R := VerifyMain(['--dialect'], '', E, L, EC);
  AssertFalse(R);
  AssertEquals(2, EC);
  { --timeout as the last argument (no value) -> usage, exit 2 }
  R := VerifyMain(['--timeout'], '', E, L, EC);
  AssertFalse(R);
  AssertEquals(2, EC);
  { -- ends options; the non-existent path after it -> usage, exit 2 }
  R := VerifyMain(['--', 'work/ut/dv200/does-not-exist.pas'], '', E, L, EC);
  AssertFalse(R);
  AssertEquals(2, EC);
  AssertEquals(1, Length(L));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_201;
var
  D: string;
  E: TEnvRec;
  L: TStringArray;
  EC: Integer;
  R: Boolean;
begin
  { a directory with no .pas/.dpr as the only path -> no lines -> usage, 2 }
  D := 'work/ut/dv201';
  WriteTestFile(D + '/note.txt', 'not pascal'#10);
  E.Fpc := 'fpc';
  E.FpcUnleashed := 'fpc-unleashed';
  R := VerifyMain([D], '', E, L, EC);
  AssertFalse(R);
  AssertEquals(2, EC);
  AssertEquals(1, Length(L));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_301;
var
  E: TEnvRec;
begin
  E := ReadEnvRec('', '', '/h');
  AssertEquals('com.remobjects.earth', E.Elements);
  AssertEquals('/h/.cache/qlora-verify', E.OxyRoot);
end;

procedure TTestDialectVerify.TestUT_DialectVerify_302;
begin
  AssertEquals('E: Unknown identifier "Foo" [Program.pas (7)]',
    FirstCompilerErrorLine('Building'#10 +
      'XE: not this'#10 +
      'E: Unknown identifier "Foo" [Program.pas (7)]'#13#10 +
      'E: second'#10));
  AssertEquals('', FirstCompilerErrorLine(' E: indented'#10 + 'e: lower'#10));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_303;
var
  E: TEnvRec;
  Fp, Log: string;
  S: TCompileState;
begin
  E := OxyEnv('dv-303');
  Fp := ExeSearch('flatpak');
  WriteTestFile('work/ut/dv-303/src/ok.pas', OXY_OK);
  WriteTestFile('work/ut/dv-303/src/bad.pas', OxyWith('      writeLn(Foo);' + #10));
  S := OxygeneStep(Fp, E.Elements, 'work/ut/dv-303/src/ok.pas', E.OxyRoot + '/a', 60000, Log);
  AssertEquals('good build must be csOk', Ord(csOk), Ord(S));
  AssertTrue('Program.pas missing in /a', FileExists(E.OxyRoot + '/a/Program.pas'));
  AssertTrue('Ex.elements missing in /a', FileExists(E.OxyRoot + '/a/Ex.elements'));
  AssertTrue('Bin/Ex.dll missing in /a', FileExists(E.OxyRoot + '/a/Bin/Ex.dll'));
  S := OxygeneStep(Fp, E.Elements, 'work/ut/dv-303/src/bad.pas', E.OxyRoot + '/b', 60000, Log);
  AssertEquals('bad build must be csFail', Ord(csFail), Ord(S));
  AssertEquals('E: Unknown identifier "Foo" [Program.pas (8)]',
    FirstCompilerErrorLine(Log));
  AssertTrue('work directory left in log', Pos(E.OxyRoot, Log) = 0);
  S := OxygeneStep('', E.Elements, 'work/ut/dv-303/src/ok.pas', E.OxyRoot + '/c', 60000, Log);
  AssertEquals('empty flatpak path must be csNoCompiler', Ord(csNoCompiler), Ord(S));
  S := OxygeneStep('/nonexistent/flatpak', E.Elements, 'work/ut/dv-303/src/ok.pas',
    E.OxyRoot + '/d', 60000, Log);
  AssertEquals('missing flatpak must be csNoCompiler', Ord(csNoCompiler), Ord(S));
  S := OxygeneStep(Fp, E.Elements, 'work/ut/dv-303/src/ok.pas', E.OxyRoot + '/e', 1, Log);
  AssertEquals('1 ms limit must be csTimeout', Ord(csTimeout), Ord(S));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_304;
begin
  WriteTestFile('work/ut/dv-304/t/a.txt', 'a'#10);
  WriteTestFile('work/ut/dv-304/t/s/b.txt', 'b'#10);
  WriteTestFile('work/ut/dv-304/t/s/u/c.txt', 'c'#10);
  RemoveTree('work/ut/dv-304/t');
  AssertTrue('t still there', not DirectoryExists('work/ut/dv-304/t'));
  AssertTrue('parent work/ut/dv-304 gone', DirectoryExists('work/ut/dv-304'));
end;

procedure TTestDialectVerify.TestUT_DialectVerify_305;
var
  E: TEnvRec;
  OK: Boolean;
  O: string;
begin
  E := OxyEnv('dv-305');
  WriteTestFile('work/ut/dv-305/s/ok.pas', OXY_OK);
  WriteTestFile('work/ut/dv-305/s/ok.expected', OXY_OUT);
  WriteTestFile('work/ut/dv-305/s/wrong.pas', OXY_OK);
  WriteTestFile('work/ut/dv-305/s/wrong.expected', 'Hello'#10);
  WriteTestFile('work/ut/dv-305/s/noexp.pas', OXY_OK);
  WriteTestFile('work/ut/dv-305/s/code.pas',
    OxyWith('      writeLn(''x'');'#10 + '      exit 3;'#10));
  WriteTestFile('work/ut/dv-305/s/code.expected', 'x'#10);
  WriteTestFile('work/ut/dv-305/s/pit.pas',
    OxyWith('      writeLn(Foo); // expect-error: Unknown identifier "Foo"'#10));
  WriteTestFile('work/ut/dv-305/s/pit2.pas',
    OxyWith('      writeLn(Foo); // expect-error: Type mismatch'#10));
  O := VerifyFile('work/ut/dv-305/s/ok.pas', '', 10, E, 'work/ut/dv-305/w', OK);
  AssertTrue('ok.pas must be OK', OK);
  AssertEquals('oxygene', Cell(O, 2));
  AssertEquals('compiled-run', Cell(O, 3));
  AssertEquals('accepted', Cell(O, 4));
  O := VerifyFile('work/ut/dv-305/s/wrong.pas', '', 10, E, 'work/ut/dv-305/w', OK);
  AssertFalse('wrong.pas must not be OK', OK);
  AssertEquals('wrong-output', Cell(O, 4));
  O := VerifyFile('work/ut/dv-305/s/noexp.pas', '', 10, E, 'work/ut/dv-305/w', OK);
  AssertFalse('noexp.pas must not be OK', OK);
  AssertEquals('no-expected', Cell(O, 4));
  O := VerifyFile('work/ut/dv-305/s/code.pas', '', 10, E, 'work/ut/dv-305/w', OK);
  AssertFalse('code.pas must not be OK', OK);
  AssertEquals('run-error', Cell(O, 4));
  AssertEquals('3', Cell(O, 5));
  O := VerifyFile('work/ut/dv-305/s/pit.pas', '', 10, E, 'work/ut/dv-305/w', OK);
  AssertTrue('pit.pas must be accepted', OK);
  AssertEquals('accepted', Cell(O, 4));
  O := VerifyFile('work/ut/dv-305/s/pit2.pas', '', 10, E, 'work/ut/dv-305/w', OK);
  AssertFalse('pit2.pas must not be OK', OK);
  AssertEquals('wrong-error', Cell(O, 4));
  AssertEquals('E: Unknown identifier "Foo" [Program.pas (8)]', Cell(O, 5));
  AssertEquals(0, EntryCount(E.OxyRoot));
  AssertTrue('work/ut/dv-305/w exists', not DirectoryExists('work/ut/dv-305/w'));
end;

var
  App: TTestRunner;
begin
  RegisterTest(TTestDialectVerify);
  App := TTestRunner.Create(nil);
  App.Initialize;
  App.Title := 'DialectVerify UT (T06)';
  App.Run;
  App.Free;
end.
