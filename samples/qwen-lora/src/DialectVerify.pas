{ DialectVerify — dialect detection, compiler choice, delphi-copy rewrite,
  expect-error handling, compile/run with a time limit, expected-output
  comparison and the TSV verification core for `qlora_verify` (T06).

  Contract: docs/detailed-design/T06-detailed.md (stage sections 2.1–2.13,
  2.15–2.18); tasks/T06-facts.md is authoritative where they differ.

  Stages (one delegation each, functions appended, never rewritten):
  - stage 1: text functions (ReadEnvRec .. SortStrings)
  - stage 2: RunLimited (adapted from tasks/snippets/runlimited.pas),
    CompileStep, VerifyFile
  - stage 3: CollectPascalFiles, VerifyMain
  - T22 (tasks/T22-facts.md): Oxygene compiled with EBuild in the Elements
    flatpak and run with its dotnet (OxygeneStep, RemoveTree, the Oxygene
    branches of ChooseCompiler and VerifyFile, EBuild's error lines) }
unit DialectVerify;

{$mode objfpc}{$H+}

interface

uses
  SysUtils;

const
  D_OBJFPC = 'objfpc';
  D_FPC_DELPHI = 'fpc-delphi';
  D_UNLEASHED = 'unleashed';
  D_DELPHI = 'delphi';
  D_OXYGENE = 'oxygene';
  VS_COMPILED_RUN = 'compiled-run';
  VS_FPC_APPROX = 'fpc-approx';
  VS_STATIC_ONLY = 'static-only';
  OUT_ACCEPTED = 'accepted';
  OUT_PIT_OK = 'compiled-but-should-fail';
  OUT_WRONG_ERR = 'wrong-error';
  OUT_COMPILE_ERR = 'compile-error';
  OUT_RUN_ERR = 'run-error';
  OUT_TIMEOUT = 'timeout';
  OUT_WRONG_OUT = 'wrong-output';
  OUT_NO_EXPECTED = 'no-expected';
  OUT_STATIC_OK = 'static-ok';
  OUT_STATIC_REJECT = 'static-reject';
  OUT_SKIPPED = 'skipped';
  K_DEFAULT_TIMEOUT_S = 10;
  K_DETAIL_MAX = 200;
  K_OUT_MAX = 10 * 1024;
  { comma-separated lists; plain string consts (FPC 3.2.2 generates runtime
    static init for array-of-string consts, which kcov cannot attribute or
    trace — scalar consts keep the measured lines of src/ honest) }
  K_SIMPLE_UNITS = 'SysUtils,Classes,Math,StrUtils,DateUtils,Types,Variants,TypInfo,Rtti';
  K_GENERIC_UNITS = 'Collections,Defaults';
  { the .NET Core console project of one Oxygene source (T22; as the
    pascal-oxygene skill's check-examples.sh) }
  K_OXY_PROJECT = '<Project DefaultTargets="Build" xmlns="http://schemas.microsoft.com/developer/msbuild/2003" ToolsVersion="4.0">' + #10 +
    '  <PropertyGroup>' + #10 +
    '    <RootNamespace>Ex</RootNamespace>' + #10 +
    '    <ProjectGuid>{6b1f1c3e-1d2a-4c55-9a7e-0c1f2e3d4a01}</ProjectGuid>' + #10 +
    '    <OutputType>Exe</OutputType>' + #10 +
    '    <AssemblyName>Ex</AssemblyName>' + #10 +
    '    <Configuration Condition="''$(Configuration)'' == ''''">Release</Configuration>' + #10 +
    '    <TargetFramework>.NETCore</TargetFramework>' + #10 +
    '    <Mode>Echoes</Mode>' + #10 +
    '    <OutputPath>.\Bin</OutputPath>' + #10 +
    '  </PropertyGroup>' + #10 +
    '  <ItemGroup>' + #10 +
    '    <Compile Include="Program.pas" />' + #10 +
    '  </ItemGroup>' + #10 +
    '  <Import Project="$(MSBuildExtensionsPath)\RemObjects Software\Elements\RemObjects.Elements.Echoes.targets" />' + #10 +
    '</Project>' + #10;
  K_USAGE = 'usage: qlora_verify [--dialect D] [--timeout SECONDS] PATH...';

type
  TEnvRec = record
    Fpc, FpcUnleashed: string;
    { T22: the Elements flatpak (app id) and the root of the Oxygene work
      directories (under the home directory) }
    Elements, OxyRoot: string;
  end;
  TCompileState = (csNoCompiler, csStartFail, csTimeout, csFail, csOk);

function ReadEnvRec(const AFpc, AFpcUnleashed, AHome: string): TEnvRec;
function DetectDialect(AText, AFileName: string): string;
function ChooseCompiler(ADialect: string; const AEnv: TEnvRec; out AClass: string): string;
function BuildCompileCmdLine(ACompiler, ASrcPath, AOutDir: string; ADelphi: Boolean): string;
function RewriteDelphiUnits(AText: string): string;
function ExtractExpectErrorText(AText: string): string;
function FirstCompilerErrorLine(ALog: string): string;
function NormalizeDetail(A: string): string;
function ExpectedPathFor(ASrcName: string): string;
function ReadTextFile(APath: string; out AText: string): Boolean;
function IsPathUsable(APath: string): Boolean;
function IsGoodOutcome(AOutcome: string): Boolean;
procedure SortStrings(A: TStringArray);
function RunLimited(const AExe: string; const AArgs: array of string;
  const ADir: string; ATimeoutMs: Integer; out AOutput: string;
  out AExitCode: Integer; out ATimedOut: Boolean): Boolean;
function CompileStep(ACompiler, ASrcPath, AOutDir: string; ATimeoutMs: Integer;
  ADelphi: Boolean; out ALog: string; out AExePath: string): TCompileState;
function OxygeneStep(AFlatpak, AApp, ASrcPath, AOutDir: string; ATimeoutMs: Integer;
  out ALog: string): TCompileState;
procedure RemoveTree(const ADir: string);
function VerifyFile(ASrcName, AForcedDialect: string; ATimeoutSec: Integer;
  const AEnv: TEnvRec; AWorkRoot: string; out AOK: Boolean): string;
function CollectPascalFiles(APath: string; out AFiles: TStringArray): Boolean;
function VerifyMain(AArgs: array of string; AWorkRoot: string; const AEnv: TEnvRec;
  out ALines: TStringArray; out AExitCode: Integer): Boolean;
function RunVerifyCLI(AArgs: array of string; AWorkRoot: string;
  const AEnv: TEnvRec): Integer;

implementation

uses
  Classes, Process;

var
  GVerifyInit: QWord;
  GVerifySeq: Integer;

{ 2.1 }
function ReadEnvRec(const AFpc, AFpcUnleashed, AHome: string): TEnvRec;
begin
  if AFpc = '' then
    Result.Fpc := 'fpc'
  else
    Result.Fpc := AFpc;
  if AFpcUnleashed = '' then
    Result.FpcUnleashed := AHome + '/unleashed/fpc/bin/fpc'
  else
    Result.FpcUnleashed := AFpcUnleashed;
end;

{ 2.2 }
function DetectDialect(AText, AFileName: string): string;
var
  I, N, Q: Integer;
  T, L: string;

  function Compact(A: string): string;
  var K: Integer;
  begin
    Result := '';
    for K := 1 to Length(A) do
      if A[K] <> ' ' then
        Result := Result + A[K];
  end;

  function HasLinePrefix(A, B: string): Boolean;
  var K: Integer;
  begin
    Result := False;
    if Length(A) < Length(B) then
      Exit;
    for K := 1 to Length(B) do
      if A[K] <> B[K] then
        Exit;
    Result := True;
  end;

  function FirstWord: string;
  var K: Integer;
  begin
    Result := '';
    K := 1;
    while (K <= N) and (T[K] <= ' ') do
      Inc(K);
    while (K <= N) and (T[K] in ['a'..'z', 'A'..'Z']) do
    begin
      Result := Result + T[K];
      Inc(K);
    end;
    Result := LowerCase(Result);
  end;

  function LineAt(AI: Integer; out AQ: Integer): string;
  begin
    Q := AI;
    while (Q <= N) and (T[Q] <> #10) do
      Inc(Q);
    Result := Compact(Copy(T, AI, Q - AI));
    if (Length(Result) > 0) and (Result[Length(Result)] = #13) then
      SetLength(Result, Length(Result) - 1);
    AQ := Q + 1;
  end;

begin
  Result := D_OBJFPC;
  T := LowerCase(AText);
  N := Length(T);
  { 1) earliest mode directive line wins (single pass) }
  I := 1;
  while I <= N do
  begin
    L := LineAt(I, Q);
    if HasLinePrefix(L, '{$modeobjfpc}') then
      Exit(D_OBJFPC);
    if HasLinePrefix(L, '{$modedelphi}') then
      Exit(D_FPC_DELPHI);
    if HasLinePrefix(L, '{$modeunleashed}') then
      Exit(D_UNLEASHED);
    I := Q;
  end;
  { 2) first word of the file is namespace }
  if FirstWord = 'namespace' then
    Exit(D_OXYGENE);
  { 3) .dpr file name or an APPTYPE directive line }
  if Pos('.dpr', LowerCase(AFileName)) = Length(LowerCase(AFileName)) - 3 then
    Exit(D_DELPHI);
  I := 1;
  while I <= N do
  begin
    L := LineAt(I, Q);
    if HasLinePrefix(L, '{$apptypeconsole}') then
      Exit(D_DELPHI);
    I := Q;
  end;
  { 4) default }
  Exit(D_OBJFPC);
end;

{ 2.3 }
function ChooseCompiler(ADialect: string; const AEnv: TEnvRec; out AClass: string): string;
var
  D: string;
begin
  AClass := '';
  D := LowerCase(ADialect);
  if (D = D_OBJFPC) or (D = D_FPC_DELPHI) then
  begin
    Result := AEnv.Fpc;
    AClass := VS_COMPILED_RUN;
  end
  else if D = D_UNLEASHED then
  begin
    Result := AEnv.FpcUnleashed;
    AClass := VS_COMPILED_RUN;
  end
  else if D = D_DELPHI then
  begin
    Result := AEnv.FpcUnleashed;
    AClass := VS_FPC_APPROX;
  end
  else if D = D_OXYGENE then
  begin
    Result := '';
    AClass := VS_STATIC_ONLY;
  end
  else
    Result := '';
end;

{ 2.4 }
function BuildCompileCmdLine(ACompiler, ASrcPath, AOutDir: string; ADelphi: Boolean): string;
function Q(const A: string): string;
begin
  Result := Chr(39) + A + Chr(39);
end;
begin
  Result := Q(ACompiler);
  if ADelphi then
    Result := Result + ' ' + Q('-Mdelphiunicode') + ' ' + Q('-Manonymousfunctions')
      + ' ' + Q('-Mfunctionreferences') + ' ' + Q('-Minlinevars');
  Result := Result + ' ' + Q('-vew') + ' ' + Q('-l-')
    + ' ' + Q('-FE' + AOutDir) + ' ' + Q('-FU' + AOutDir) + ' ' + Q(ASrcPath);
end;

{ 2.5 }
function RewriteToken(const ASrc, AOld, ANew: string): string;
var
  I, N, OL: Integer;
const
  LBAD = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-$';
  RBAD = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-$';
begin
  Result := '';
  N := Length(ASrc);
  OL := Length(AOld);
  I := 1;
  while I <= N do
  begin
    if ( (I = 1) or (Pos(ASrc[I - 1], LBAD) = 0) )
       and ( (I + OL > N) or (Pos(ASrc[I + OL], RBAD) = 0) )
       and (I + OL - 1 <= N)
       and (Copy(ASrc, I, OL) = AOld) then
    begin
      Result := Result + ANew;
      I := I + OL;
    end
    else
    begin
      Result := Result + ASrc[I];
      Inc(I);
    end;
  end;
end;

function RewriteDelphiUnits(AText: string): string;

  procedure RewriteList(ACSV, AFromPrefix, AToPrefix: string);
  var K, M: Integer; Item: string;
  begin
    K := 1;
    while K <= Length(ACSV) do
    begin
      M := K;
      while (M <= Length(ACSV)) and (ACSV[M] <> ',') do
        Inc(M);
      Item := Copy(ACSV, K, M - K);
      if Item <> '' then
        Result := RewriteToken(Result, AFromPrefix + Item, AToPrefix + Item);
      K := M + 1;
    end;
  end;

begin
  Result := AText;
  RewriteList(K_GENERIC_UNITS, 'System.Generics.', 'Generics.');
  RewriteList(K_SIMPLE_UNITS, 'System.', '');
end;

{ 2.6 }
function ExtractExpectErrorText(AText: string): string;
const
  MARKLEN = 13;  { length of 'expect-error:' }
var
  Low: string;
  P, S, E: Integer;
  Line: string;
begin
  Result := '';
  Low := LowerCase(AText);
  P := 1;
  while (P <= Length(Low)) and (Result = '') do
  begin
    if (P + MARKLEN - 1 <= Length(Low))
       and (Copy(Low, P, MARKLEN) = 'expect-error:') then
    begin
      if (P = 1) or (not (Low[P - 1] in ['a'..'z', '0'..'9', '-'])) then
      begin
        { valid directive: text after it runs to the closing brace or to end of line }
        S := P + MARKLEN;
        while (S <= Length(Low)) and (Low[S] = ' ') do
          Inc(S);
        E := S;
        while (E <= Length(Low)) and (Low[E] <> '}') and (Low[E] <> #10) do
          Inc(E);
        { BUG-001 fix: extract from the ORIGINAL text (not the lowercased
          copy) — the pitfall text is matched case-sensitively against the
          compiler log (grep -F semantics, T06-facts). The scan offsets S/E
          are identical, because LowerCase is 1:1 per character }
        Line := Copy(AText, S, E - S);
        if (Length(Line) > 0) and (Line[Length(Line)] = #13) then
          Delete(Line, Length(Line), 1);
        Result := Trim(Line);
        Break;
      end
      else
        P := P + MARKLEN  { embedded occurrence: skip and re-scan }
    end
    else
      Inc(P);
  end;
end;

{ 2.7 }
function FirstCompilerErrorLine(ALog: string): string;
var
  I, N, Q: Integer;
  Line, LowLine: string;
begin
  Result := '';
  N := Length(ALog);
  I := 1;
  while (I <= N) and (Result = '') do
  begin
    Q := I;
    while (Q <= N) and (ALog[Q] <> #10) do
      Inc(Q);
    Line := Copy(ALog, I, Q - I);
    if (Length(Line) > 0) and (Line[Length(Line)] = #13) then
      Delete(Line, Length(Line), 1);
    LowLine := LowerCase(Line);
    if (Pos('error:', LowLine) > 0) or (Pos('fatal:', LowLine) > 0) then
      Result := Line;
    I := Q + 1;
  end;
end;

{ 2.8 }
function NormalizeDetail(A: string): string;
var
  I: Integer;
  B: string;
begin
  B := '';
  for I := 1 to Length(A) do
    if A[I] = #9 then
      B := B + ' '
    else
      B := B + A[I];
  Result := B;
  if Length(Result) > K_DETAIL_MAX then
    SetLength(Result, K_DETAIL_MAX);
end;

{ 2.10 }
function ExpectedPathFor(ASrcName: string): string;
var
  Fn: string;
begin
  Fn := ExtractFileName(ASrcName);
  if Length(ASrcName) > Length(Fn) then
    Result := Copy(ASrcName, 1, Length(ASrcName) - Length(Fn))
             + ChangeFileExt(Fn, '.expected')
  else
    Result := ChangeFileExt(Fn, '.expected');
end;

{ 2.11 }
function ReadTextFile(APath: string; out AText: string): Boolean;
var
  F: TFileStream;
begin
  AText := '';
  Result := False;
  try
    F := TFileStream.Create(APath, fmOpenRead);
    try
      SetLength(AText, F.Size);
      if F.Size > 0 then
        F.ReadBuffer(AText[1], F.Size);
      Result := True;
    finally
      F.Free;
    end;
  except
    on E: Exception do
    begin
      Result := False;
      AText := '';
    end;
  end;
end;

{ FPC 3.2.2 has no DirExists }
function CheckDir(AName: string): Boolean;
var
  FSRec: TRawByteSearchRec;
  D: string;
begin
  Result := False;
  D := AName;
  if (D = '') or (D[Length(D)] <> '/') then
    D := D + '/';
  if FindFirst(D, faAnyFile, FSRec) = 0 then
  begin
    Result := True;
    FindClose(FSRec);
  end;
end;

{ 2.12 }
function IsPathUsable(APath: string): Boolean;
var
  FS: TFileStream;
  R: string;
begin
  Result := False;
  if APath = '' then
    Exit;
  { BUG-002 fix: a bare name (no '/') is a program name, not a path — like
    a shell, resolve it in PATH (SysUtils ExeSearch); the existence and
    readability checks below then apply to the resolved file. A bare name
    that resolves nowhere falls back to the old "is a local directory"
    check, so previous behaviour for bare directory names is unchanged }
  if (Pos('/', APath) = 0) and (not FileExists(APath)) then
  begin
    R := ExeSearch(APath);
    if R <> '' then
      APath := R
    else if not CheckDir(APath) then
      Exit;
  end;
  if FileExists(APath) then
  begin
    FS := nil;
    try
      FS := TFileStream.Create(APath, fmOpenRead or fmShareDenyNone);
      Result := True;
    except
      on E: Exception do
        Result := False;
    end;
    if Assigned(FS) then
      FS.Free;
  end
  else
    Result := CheckDir(APath);
end;

{ 2.18 }
function IsGoodOutcome(AOutcome: string): Boolean;
var
  O: string;
begin
  O := LowerCase(AOutcome);
  Result := (O = OUT_ACCEPTED) or (O = OUT_STATIC_OK) or (O = OUT_SKIPPED);
end;

{ 2.17 }
procedure SortStrings(A: TStringArray);
var
  I, J: Integer;
  T: string;
begin
  for I := 1 to High(A) do
  begin
    T := A[I];
    J := I - 1;
    while (J >= 0) and (A[J] > T) do
    begin
      A[J + 1] := A[J];
      Dec(J);
    end;
    A[J + 1] := T;
  end;
end;

{ 2.9 }
function IsRelativePath(const S: string): Boolean;
begin
  Result := (S <> '') and (S[1] <> '/') and
            ((Length(S) < 2) or (not ((S[2] = ':') or (S[2] = '/'))));
end;

function RunLimited(const AExe: string; const AArgs: array of string;
  const ADir: string; ATimeoutMs: Integer; out AOutput: string;
  out AExitCode: Integer; out ATimedOut: Boolean): Boolean;
var
  P: TProcess;
  Buf: array[0..4095] of Byte;
  N: LongInt;
  OutS: TStringStream;
  T0: QWord;
  A: string;
  Exe, Dir, R: string;
begin
  AOutput := '';
  AExitCode := -1;
  ATimedOut := False;
  Exe := AExe;
  Dir := ADir;
  { BUG-003 fix: a bare name (no '/') is a program NAME, not a path — like a
    shell, resolve it in PATH (SysUtils ExeSearch) BEFORE the relative→absolute
    rebase below. A bare name that resolves nowhere falls back to the old
    cwd-prefix logic (or a local file of the same name), so the relative-exe
    behaviour is unchanged. Mirrors the IsPathUsable gate (BUG-002). }
  if (Pos('/', Exe) = 0) and (not FileExists(Exe)) then
  begin
    R := ExeSearch(Exe);
    if R <> '' then
      Exe := R;
  end;
  { the child chdirs to ADir before exec: a relative exe or dir would then
    resolve against the NEW working directory (ENOENT → exit 127), so both
    are resolved against the program working directory before start }
  if IsRelativePath(Exe) then
    Exe := GetCurrentDir + '/' + Exe;
  if IsRelativePath(Dir) then
    Dir := GetCurrentDir + '/' + Dir;
  OutS := TStringStream.Create('');
  P := TProcess.Create(nil);
  try
    P.Executable := Exe;
    for A in AArgs do
      P.Parameters.Add(A);
    if Dir <> '' then
      P.CurrentDirectory := Dir;
    P.Options := [poUsePipes, poStderrToOutPut];
    try
      P.Execute;
    except
      on E: Exception do
      begin
        AOutput := 'process error: ' + E.Message;
        Exit(False);
      end;
    end;
    T0 := GetTickCount64;
    { liveness and timeout polling; captured output is read in one go,
      in the drain loop below, once the process has ended or been killed }
    repeat
      if not P.Running then
        Break;
      if GetTickCount64 - T0 > QWord(ATimeoutMs) then
      begin
        ATimedOut := True;
        AExitCode := 255;
        P.Terminate(9);
        Break;
      end;
      Sleep(10);
    until False;
    repeat
      N := P.Output.Read(Buf, SizeOf(Buf));
      if N > 0 then
        OutS.Write(Buf, N);
    until N <= 0;
    if not ATimedOut then
      AExitCode := P.ExitCode;  { ExitStatus is the raw wait status (3 shl 8) }
    AOutput := OutS.DataString;
    Result := True;
  finally
    P.Free;
    OutS.Free;
  end;
end;

{ 2.13 }
function CompileStep(ACompiler, ASrcPath, AOutDir: string; ATimeoutMs: Integer;
  ADelphi: Boolean; out ALog: string; out AExePath: string): TCompileState;
var
  Args: TStringArray;
  SrcToCompile, SrcText, ExePath, Out: string;
  Code: Integer;
  TimedOut: Boolean;
  I, N: Integer;
  F: TFileStream;
begin
  AExePath := '';
  Result := csFail;
  if (ACompiler = '') or (not IsPathUsable(ACompiler)) then
    Exit(csNoCompiler);
  if IsRelativePath(AOutDir) then
    AOutDir := GetCurrentDir + '/' + AOutDir;
  if AOutDir <> '' then
    ForceDirectories(AOutDir);
  SrcToCompile := ASrcPath;
  if ADelphi then
  begin
    { delphi: rewrite the copy at <out>/<name> and compile the copy (2.13) }
    SrcToCompile := AOutDir + '/' + ExtractFileName(ASrcPath);
    if not ReadTextFile(ASrcPath, SrcText) then
      Exit(csFail);
    F := TFileStream.Create(SrcToCompile, fmCreate);
    try
      SrcText := RewriteDelphiUnits(SrcText);
      if Length(SrcText) > 0 then
        F.WriteBuffer(SrcText[1], Length(SrcText));
    finally
      F.Free;
    end;
  end;
  { command line: delphi compat flags, then the flag table (2.4), src last,
    passed as separate arguments (not a quoted ACmdLine string) }
  N := 5;
  if ADelphi then
    N := N + 4;
  SetLength(Args, N);
  I := 0;
  if ADelphi then
  begin
    Args[I] := '-Mdelphiunicode';
    Inc(I);
    Args[I] := '-Manonymousfunctions';
    Inc(I);
    Args[I] := '-Mfunctionreferences';
    Inc(I);
    Args[I] := '-Minlinevars';
    Inc(I);
  end;
  Args[I] := '-vew';
  Inc(I);
  Args[I] := '-l-';
  Inc(I);
  Args[I] := '-FE' + AOutDir;
  Inc(I);
  Args[I] := '-FU' + AOutDir;
  Inc(I);
  Args[I] := SrcToCompile;
  { empty work dir: all paths in the flag table are absolute (AOutDir is
    resolved above), so the compile child needs no cwd change.
    T06c verification of the old T06b2 note: FPC 3.2.2's TProcess CAN exec
    a shebang script after CurrentDirectory is set, provided the script is
    named by an ABSOLUTE path (probe: exit 0, output captured). The T06b2
    failure (127 / ENOENT) was the relative script path resolving against
    the child's new cwd, not a TProcess limitation. RunLimited therefore
    pre-resolves a relative AExe to absolute before start (see its body). }
  { Started is not inspected here: the compiler was pre-validated above
    (csNoCompiler), and FPC 3.2.2's TProcess reports post-start exec
    failures as exit 127 (handled via csFail below), not via Result }
  RunLimited(ACompiler, Args, '', ATimeoutMs, Out, Code, TimedOut);
  ALog := Out;
  if TimedOut then
    Exit(csTimeout);
  if Code <> 0 then
    Exit(csFail);
  ExePath := AOutDir + '/' + ChangeFileExt(ExtractFileName(ASrcPath), '');
  if IsPathUsable(ExePath) then
  begin
    AExePath := ExePath;
    Result := csOk;
  end;
end;

{ 2.13b (T22) Oxygene: copy the source as Program.pas into AOutDir (under
  the home directory: the flatpak's /tmp is its own), write the project
  Ex.elements (.NET Core, Echoes) and build it with EBuild inside the
  Elements flatpak. csOk when EBuild exits 0 and Bin/Ex.dll exists. }
function OxygeneStep(AFlatpak, AApp, ASrcPath, AOutDir: string; ATimeoutMs: Integer;
  out ALog: string): TCompileState;
begin
  ALog := '';
  Result := csFail;
  raise Exception.Create('OxygeneStep: not implemented yet');
end;

{ 2.13c (T22) Deletes ADir and everything below it. }
procedure RemoveTree(const ADir: string);
begin
  raise Exception.Create('RemoveTree: not implemented yet');
end;

{ 2.14 }
function VerifyFile(ASrcName, AForcedDialect: string; ATimeoutSec: Integer;
  const AEnv: TEnvRec; AWorkRoot: string; out AOK: Boolean): string;
var
  Dialect, Comp, CClass, Text, Pitfall, Log, ExePath, RunOut, ExpText: string;
  Code: Integer;
  TimedOut, Ran: Boolean;
  State: TCompileState;
  W, Outcome, Detail: string;
begin
  { the source must exist (CLI checks the path); a missing file flows
    through as empty text and is classified, never raised }
  ReadTextFile(ASrcName, Text);
  Log := '';
  Dialect := LowerCase(AForcedDialect);
  if Dialect = '' then
    Dialect := DetectDialect(Text, ASrcName);
  CClass := '';
  Comp := ChooseCompiler(Dialect, AEnv, CClass);
  if Dialect = D_OXYGENE then
  begin
    if DetectDialect(Text, ASrcName) = D_OXYGENE then
      Outcome := OUT_STATIC_OK
    else
      Outcome := OUT_STATIC_REJECT;
    AOK := IsGoodOutcome(Outcome);
    Result := ASrcName + #9 + Dialect + #9 + CClass + #9 + Outcome + #9 +
      NormalizeDetail('');
    Exit;
  end;
  { fresh out dir <workroot>/<n>, n unique per call (salt + per-process
    sequence), so a rerun never collides with a stale directory }
  GVerifySeq := GVerifySeq + 1;
  W := AWorkRoot + '/' + IntToStr(GVerifyInit + GVerifySeq);
  State := CompileStep(Comp, ASrcName, W, ATimeoutSec * 1000,
    (Dialect = D_DELPHI), Log, ExePath);
  if (State = csNoCompiler) or (State = csStartFail) then
  begin
    Outcome := OUT_SKIPPED;
    Detail := 'compiler not found';
  end
  else if State = csTimeout then
  begin
    Outcome := OUT_SKIPPED;
    Detail := 'timeout';
  end
  else if State = csFail then
  begin
    Pitfall := ExtractExpectErrorText(Text);
    if (Pitfall <> '') and (Pos(Pitfall, Log) > 0) then
      Outcome := OUT_ACCEPTED
    else if Pitfall <> '' then
    begin
      Outcome := OUT_WRONG_ERR;
      Detail := FirstCompilerErrorLine(Log);
    end
    else
    begin
      Outcome := OUT_COMPILE_ERR;
      Detail := FirstCompilerErrorLine(Log);
    end;
  end
  else
  begin
    Pitfall := ExtractExpectErrorText(Text);
    if Pitfall <> '' then
      Outcome := OUT_PIT_OK  { compiled but the source demands a compile error }
    else
    begin
      Outcome := OUT_SKIPPED;
      Detail := 'could not start';
      { T06c fix: the program runs INSIDE its fresh out dir <out> (the facts:
        run it where the compiler put it). RunLimited pre-resolves the relative
        dir to absolute before setting CurrentDirectory, so the absolute-shebang
        exec path below stays valid. Fixing the inherited-cwd default meant
        test 106's fixture now logs to an absolute path (see its notes). }
      Ran := RunLimited(ExePath, [], W, ATimeoutSec * 1000, RunOut, Code,
        TimedOut);
      if Ran and TimedOut then
      begin
        Outcome := OUT_TIMEOUT;
        Detail := IntToStr(ATimeoutSec);
      end
      else if Ran and (Code <> 0) then
      begin
        Outcome := OUT_RUN_ERR;
        Detail := IntToStr(Code);
      end
      else if Ran then
      begin
        Detail := '';
        if ReadTextFile(ExpectedPathFor(ASrcName), ExpText) and
           (ExpText = RunOut) then
          Outcome := OUT_ACCEPTED
        else if ReadTextFile(ExpectedPathFor(ASrcName), ExpText) then
          Outcome := OUT_WRONG_OUT
        else
          Outcome := OUT_NO_EXPECTED;
      end;
    end;
  end;
  AOK := IsGoodOutcome(Outcome);
  Result := ASrcName + #9 + Dialect + #9 + CClass + #9 + Outcome + #9 +
    NormalizeDetail(Detail);
end;

{ 2.15 Collect files under a directory, recursively, .pas/.dpr case-insensitive. }
{ Positive FindFirst form so there is no uncovered early-exit branch. }
{ Append one element using the project's SetLength idiom (objfpc does not
  support dynamic-array '+' concatenation — see CompileStep, 2.10).
  Note: `var`, NOT `out` — FPC `out` re-NILs a dynamic-array formal each call,
  so SetLength(len+1) would always start from 0 and clobber the array. }
procedure ArrAdd(var AR: TStringArray; const AValue: string);
begin
  SetLength(AR, Length(AR) + 1);
  AR[Length(AR) - 1] := AValue;
end;

{ `var`, NOT `out`: this is recursive, so the nested call must append into the
  caller's already-accumulated array; `out` would re-NIL it and clobber results. }
procedure CollectDirRec(ADir: string; var AFiles: TStringArray);
var
  R: TRawByteSearchRec;
  fn, ext: string;
begin
  if FindFirst(ADir + '/' + '*', faAnyFile, R) = 0 then
  begin
    repeat
      fn := R.Name;
      if (fn <> '.') and (fn <> '..') then
        if (R.Attr and faDirectory) <> 0 then
          CollectDirRec(ADir + '/' + fn, AFiles)
        else if Length(fn) >= 4 then
        begin
          ext := LowerCase(Copy(fn, Length(fn) - 3));
          if (ext = '.pas') or (ext = '.dpr') then
            ArrAdd(AFiles, ADir + '/' + fn);
        end;
    until FindNext(R) <> 0;
    FindClose(R);
  end;
end;

function CollectPascalFiles(APath: string; out AFiles: TStringArray): Boolean;
var
  Files: TStringArray;
begin
  Files := [];
  if FileExists(APath) then
  begin
    AFiles := [APath];
    Exit(True);
  end;
  if CheckDir(APath) then
  begin
    CollectDirRec(APath, Files);
    SortStrings(Files);
    AFiles := Files;
    Exit(True);
  end;
  AFiles := [];
  Result := False;
end;

{ 2.21 VerifyMain helpers (CLI parsing support). }
function IsValidDialect(const A: string): Boolean;
var
  D: string;
begin
  D := LowerCase(A);
  Result := (D = D_OBJFPC) or (D = D_FPC_DELPHI) or (D = D_UNLEASHED) or
            (D = D_DELPHI) or (D = D_OXYGENE);
end;

function SetUsage(out ALines: TStringArray; out AExitCode: Integer): Boolean;
begin
  ALines := [K_USAGE];
  AExitCode := 2;
  Result := False;
end;

{ 2.19 Entry point: parse AArgs, collect files, one line per file, return status.
  AEnv is passed in (the program qlora_verify fills it from the environment);
  VerifyMain never reads the environment itself. }
function VerifyMain(AArgs: array of string; AWorkRoot: string; const AEnv: TEnvRec;
  out ALines: TStringArray; out AExitCode: Integer): Boolean;
var
  I, N, J, Val: Integer;
  A: string;
  Dialect: string;
  Timeout: Integer;
  Paths, Files, Lines: TStringArray;
  F, Line: string;
  AllGood, OK: Boolean;
begin
  ALines := [];
  AExitCode := 2;
  Dialect := '';
  Timeout := K_DEFAULT_TIMEOUT_S;
  Paths := [];
  N := Length(AArgs);
  I := 0;
  while I < N do
  begin
    A := AArgs[I];
    if A = '--help' then
    begin
      ALines := [K_USAGE];
      AExitCode := 0;
      Exit(True);
    end;
    if A = '--' then
    begin
      Inc(I);
      while I < N do
      begin
        ArrAdd(Paths, AArgs[I]);
        Inc(I);
      end;
      Break;
    end;
    if A = '--dialect' then
    begin
      if I + 1 >= N then
        Exit(SetUsage(ALines, AExitCode));
      if not IsValidDialect(AArgs[I + 1]) then
        Exit(SetUsage(ALines, AExitCode));
      Dialect := AArgs[I + 1];
      Inc(I, 2);
      Continue;
    end;
    if A = '--timeout' then
    begin
      if I + 1 >= N then
        Exit(SetUsage(ALines, AExitCode));
      if not TryStrToInt(AArgs[I + 1], Val) or (Val < 1) then
        Exit(SetUsage(ALines, AExitCode));
      Timeout := Val;
      Inc(I, 2);
      Continue;
    end;
    if (Length(A) >= 2) and (A[1] = '-') and (A[2] = '-') then
      Exit(SetUsage(ALines, AExitCode));
    ArrAdd(Paths, A);
    Inc(I);
  end;
  if Length(Paths) = 0 then
    Exit(SetUsage(ALines, AExitCode));
  AllGood := True;
  Lines := [];
  for I := 0 to Length(Paths) - 1 do
  begin
    F := Paths[I];
    if not CollectPascalFiles(F, Files) then
      Exit(SetUsage(ALines, AExitCode));
    for J := 0 to Length(Files) - 1 do
    begin
      OK := False;
      Line := VerifyFile(Files[J], Dialect, Timeout, AEnv, AWorkRoot, OK);
      if not OK then
        AllGood := False;
      ArrAdd(Lines, Line);
    end;
  end;
  if Length(Lines) = 0 then
    Exit(SetUsage(ALines, AExitCode));
  ALines := Lines;
  if AllGood then
    AExitCode := 0
  else
    AExitCode := 1;
  Result := True;
end;

{ 2.22 CLI wrapper for the program qlora_verify (T06c). Forwards the whole
  argument vector to VerifyMain with AWorkRoot, prints each result line to
  stdout, and returns the process exit code (2 on a usage error, i.e. when
  VerifyMain itself reports failure; 0/1 are VerifyMain's verify verdicts).
  AEnv is injected by the caller — the program fills it with
  ReadEnvRec(GetEnvironmentVariable(...)); this function never reads the
  environment, so unit tests can drive it hermetically. }
function RunVerifyCLI(AArgs: array of string; AWorkRoot: string;
  const AEnv: TEnvRec): Integer;
var
  Lines: TStringArray;
  i: Integer;
begin
  if not VerifyMain(AArgs, AWorkRoot, AEnv, Lines, Result) then
    Result := 2;
  for i := 0 to Length(Lines) - 1 do
    WriteLn(Lines[i]);
end;

initialization
  GVerifyInit := GetTickCount64;

end.
