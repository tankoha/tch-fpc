# T22 facts (checked by the harness owner)

T22 writes the Oxygene row of the filter (DESIGN 8.16, "The filter" and
"Oxygene: compiled and run, in the first run"): `qlora_verify` compiles an
Oxygene source with RemObjects Elements 14.0.0.3157 (EBuild in the Earth
flatpak, a .NET Core (Echoes) `.elements` project) and runs it with the
flatpak's `dotnet`, like the other dialects; its class is `compiled-run`.
It changes one unit, `DialectVerify` (T06); `qlora_verify`, `qlora_dataset`
and `qlora_eval` use it unchanged.

Decisions of the harness owner for this task (recorded in DESIGN 8.16 as
"As written for T22"):
- The build follows the `pascal-oxygene` skill's `check-examples.sh`: the
  source is copied as `Program.pas` next to the project `Ex.elements`
  (`K_OXY_PROJECT`), built by `flatpak run --command=/app/dotnet/dotnet
  com.remobjects.earth /app/earth/EBuild.dll Ex.elements` and run by
  `flatpak run --command=/app/dotnet/dotnet com.remobjects.earth
  Bin/Ex.dll`, both in the work directory. A build is good when EBuild
  exits 0 **and** `Bin/Ex.dll` exists.
- **The work directory is under the home directory**
  (`<HOME>/.cache/qlora-verify/<n>`, `TEnvRec.OxyRoot`): the flatpak sees
  the host's files but has a `/tmp` of its own. It is deleted after the
  verification (`RemoveTree`), whatever the outcome; `AWorkRoot` is not
  used for Oxygene.
- EBuild's error lines start with `E: ` (`E: Unknown identifier "Foo"
  [<path> (7)]`); `FirstCompilerErrorLine` also takes them. The work
  directory is removed from the log (`StringReplace`), so a detail names
  `Program.pas`, never a path under the home directory.
- **One file per Oxygene record**: only the source itself is compiled
  (units next to it are not), so an Oxygene sample that has a class and its
  test program keeps both in one namespace file (the data specification
  says so).
- The flatpak is found with `ExeSearch('flatpak')`; without it the outcome
  is `skipped` (`compiler not found`), as for a missing FPC. The app id is
  `TEnvRec.Elements`, `com.remobjects.earth` from `ReadEnvRec`.
- `VS_STATIC_ONLY`, `OUT_STATIC_OK` and `OUT_STATIC_REJECT` stay as
  constants (other units name them) but are no longer produced.

Measured by the harness owner with a prototype written exactly as below:
- UT: every row of `DialectVerify_test` passes with `-Cr` (108 rows: the
  earlier ones, rows 013, 103, 104 changed as below, rows 301 to 305);
  `DialectVerify.pas` then has 100 % line coverage (551 lines); mutants
  are caught (EBuild's lines not taken: DialectVerify 103, 302, 303, 305, DatasetCli 105; Oxygene left without a compiler: DialectVerify 013, 103, 104, 305, DatasetCli 105, EvalCli 105; a failed build taken as good: DialectVerify 103, 104, 303, 305, DatasetCli 105, EvalCli 105; the work directory left in the log: DialectVerify 103, 303, 305; the work directory not removed: 305; RemoveTree not recursive: 304, 305; the build's time limit ignored: 303, 305; another app id: 103, 301, 303, 305, DatasetCli 105; another OxyRoot: 301, 305). About 0.9 s per Oxygene file (build and run).
- ITa (cases below): all 0 (0.9 s per file); with the code before T22,
  001 and 006 are NG.

These facts are final; do not investigate further or write probes.

## Interface (in the skeleton)

Already in `src/DialectVerify.pas`: the constant `K_OXY_PROJECT` (the
project text), the fields of `TEnvRec`

```pascal
  TEnvRec = record
    Fpc, FpcUnleashed: string;
    { T22: the Elements flatpak (app id) and the root of the Oxygene work
      directories (under the home directory) }
    Elements, OxyRoot: string;
  end;
```

and the declarations

```pascal
function OxygeneStep(AFlatpak, AApp, ASrcPath, AOutDir: string; ATimeoutMs: Integer;
  out ALog: string): TCompileState;
procedure RemoveTree(const ADir: string);
```

## Implementation (write it exactly so)

1. `ReadEnvRec`: two lines at its end:

```pascal
  Result.Elements := 'com.remobjects.earth';
  Result.OxyRoot := AHome + '/.cache/qlora-verify';
```

2. `ChooseCompiler`: the Oxygene branch becomes

```pascal
  else if D = D_OXYGENE then
  begin
    Result := ExeSearch('flatpak');  { T22: EBuild in the Elements flatpak }
    AClass := VS_COMPILED_RUN;
  end
```

3. `FirstCompilerErrorLine`: the condition becomes

```pascal
    if (Pos('error:', LowLine) > 0) or (Pos('fatal:', LowLine) > 0) or
       (Copy(Line, 1, 3) = 'E: ') then  { T22: EBuild's error lines }
      Result := Line;
```

4. `OxygeneStep` and `RemoveTree` (replace the skeleton's bodies):

```pascal
{ 2.13b (T22) Oxygene: copy the source as Program.pas into AOutDir (under
  the home directory: the flatpak's /tmp is its own), write the project
  Ex.elements (.NET Core, Echoes) and build it with EBuild inside the
  Elements flatpak. csOk when EBuild exits 0 and Bin/Ex.dll exists. }
function OxygeneStep(AFlatpak, AApp, ASrcPath, AOutDir: string; ATimeoutMs: Integer;
  out ALog: string): TCompileState;
var
  Text, Out: string;
  Code: Integer;
  TimedOut: Boolean;
  L: TStringList;
begin
  ALog := '';
  if (AFlatpak = '') or (not IsPathUsable(AFlatpak)) then
    Exit(csNoCompiler);
  ForceDirectories(AOutDir);
  ReadTextFile(ASrcPath, Text);
  L := TStringList.Create;
  try
    L.Text := Text;
    L.SaveToFile(AOutDir + '/Program.pas');
    L.Text := K_OXY_PROJECT;
    L.SaveToFile(AOutDir + '/Ex.elements');
  finally
    L.Free;
  end;
  RunLimited(AFlatpak, ['run', '--command=/app/dotnet/dotnet', AApp, '/app/earth/EBuild.dll',
    'Ex.elements'], AOutDir, ATimeoutMs, Out, Code, TimedOut);
  ALog := StringReplace(Out, AOutDir + '/', '', [rfReplaceAll]);  { no home path in a detail }
  if TimedOut then
    Exit(csTimeout);
  if (Code <> 0) or not FileExists(AOutDir + '/Bin/Ex.dll') then
    Exit(csFail);
  Result := csOk;
end;

{ 2.13c (T22) Deletes ADir and everything below it. }
procedure RemoveTree(const ADir: string);
var
  R: TSearchRec;
begin
  if FindFirst(ADir + '/*', faAnyFile or faDirectory, R) = 0 then
  begin
    repeat
      if (R.Name <> '.') and (R.Name <> '..') then
        if (R.Attr and faDirectory) <> 0 then
          RemoveTree(ADir + '/' + R.Name)
        else
          DeleteFile(ADir + '/' + R.Name);
    until FindNext(R) <> 0;
    FindClose(R);
  end;
  RemoveDir(ADir);
end;
```

5. `VerifyFile` as a whole (the static Oxygene branch at its start goes;
   the Oxygene build, run and clean-up come in):

```pascal
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
  { fresh out dir <workroot>/<n>, n unique per call (salt + per-process
    sequence), so a rerun never collides with a stale directory }
  GVerifySeq := GVerifySeq + 1;
  if Dialect = D_OXYGENE then
  begin
    W := AEnv.OxyRoot + '/' + IntToStr(GVerifyInit + GVerifySeq);
    State := OxygeneStep(Comp, AEnv.Elements, ASrcName, W, ATimeoutSec * 1000, Log);
  end
  else
  begin
    W := AWorkRoot + '/' + IntToStr(GVerifyInit + GVerifySeq);
    State := CompileStep(Comp, ASrcName, W, ATimeoutSec * 1000,
      (Dialect = D_DELPHI), Log, ExePath);
  end;
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
      if Dialect = D_OXYGENE then
        Ran := RunLimited(Comp, ['run', '--command=/app/dotnet/dotnet', AEnv.Elements,
          'Bin/Ex.dll'], W, ATimeoutSec * 1000, RunOut, Code, TimedOut)
      else
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
  if Dialect = D_OXYGENE then
    RemoveTree(W);
  AOK := IsGoodOutcome(Outcome);
  Result := ASrcName + #9 + Dialect + #9 + CClass + #9 + Outcome + #9 +
    NormalizeDetail(Detail);
end;
```

## Checks and messages

- No new message: an Oxygene build failure gives `compile-error` with
  EBuild's first `E: ` line as the detail (the work directory removed from
  it); `wrong-error` when an `expect-error:` marker's text is not in the
  log; the run's outcomes as for the other dialects (`run-error` with the
  exit code, `timeout`, `wrong-output`, `no-expected`, `accepted`); no
  flatpak: `skipped`, `compiler not found`; a build over the time limit:
  `skipped`, `timeout`.

## UT fixture and helpers (harness owner's; nobody changes them)

`tests/ut/DialectVerify_test.pas` (T06) with its helpers `WriteTestFile`,
`Cell(Line, Column)` (1-based TSV column). The T22 rows add, before row
103 (they are used there), these helpers (write them exactly so):

```pascal
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
```

Two earlier rows of other units change with T22 and are already changed
by the harness owner (in the skeleton): `UT-DatasetCli-105` (the Oxygene
record `namespace N; end.`: class `compiled-run`, outcome `compile-error`,
detail `E: Could not find a "Main" method in this project`) and
`UT-EvalCli-105` (the reply `namespace N;`: outcome `compile-error`, not
compiled, not passed). They fail until T22b is done.

## Unit tests (rows; every row has one method, every method belongs to a row)

"exact" means equal strings. Use fpcunit's checks (`AssertEquals`,
`AssertTrue(msg, cond)`); plain `Assert` is refused by the gate. The
Oxygene rows run EBuild and dotnet (about a second per build).

Rows 013, 103, 104 **change** (rewrite their methods):

| Row | Input | Expected |
|---|---|---|
| 013 | `ChooseCompiler('oxygene', ReadEnvRec('', '', '/h'), Cls)` | `ExeSearch('flatpak')`; Cls `compiled-run` |
| 103 | E := `OxyEnv('dv-103')`; `work/ut/dv-103/o.pas` = `namespace Demo; type T = class of object end; end.`; `VerifyFile(.., '', 10, E, '', OK)` | OK False; column 4 `compile-error`, 2 `oxygene`, 3 `compiled-run`, 5 `E: semicolon (;) expected, got "end" [Program.pas (1)]` |
| 104 | E := `OxyEnv('dv-104')`; `work/ut/dv-104/p.pas` = `program p; begin end.`; `VerifyFile(.., 'oxygene', 10, E, '', OK)` | OK False; column 4 `compile-error` |

New rows:

| Row | Input | Expected |
|---|---|---|
| 301 | `ReadEnvRec('', '', '/h')` | Elements `com.remobjects.earth`; OxyRoot `/h/.cache/qlora-verify` |
| 302 | `FirstCompilerErrorLine` of `Building`, `XE: not this`, `E: Unknown identifier "Foo" [Program.pas (7)]`#13, `E: second` (lines); of ` E: indented`, `e: lower` | `E: Unknown identifier "Foo" [Program.pas (7)]`; `''` |
| 303 | E := `OxyEnv('dv-303')`, Fp := `ExeSearch('flatpak')`; sources `OXY_OK` and `OxyWith('      writeLn(Foo);'#10)` under `work/ut/dv-303/src`; `OxygeneStep(Fp, E.Elements, ok, <OxyRoot>/a, 60000, Log)`; the bad one into `/b`; with `''` as the flatpak; with `/nonexistent/flatpak`; the good one with a time limit of 1 ms | `csOk`, and `/a` holds `Program.pas`, `Ex.elements`, `Bin/Ex.dll`; `csFail`, `FirstCompilerErrorLine(Log)` = `E: Unknown identifier "Foo" [Program.pas (8)]`, the directory not in Log; `csNoCompiler`; `csNoCompiler`; `csTimeout` |
| 304 | files `work/ut/dv-304/t/a.txt`, `t/s/b.txt`, `t/s/u/c.txt`; `RemoveTree('work/ut/dv-304/t')` | `t` gone, `work/ut/dv-304` still there |
| 305 | E := `OxyEnv('dv-305')`; under `work/ut/dv-305/s`: `ok.pas` = `OXY_OK` with `OXY_OUT`; `wrong.pas` = `OXY_OK` with `Hello`#10; `noexp.pas` = `OXY_OK` without an expected file; `code.pas` = `OxyWith` of `writeLn('x')` and `exit 3;` with `x`#10; `pit.pas` = `OxyWith` of `writeLn(Foo); // expect-error: Unknown identifier "Foo"`; `pit2.pas` = the same with `// expect-error: Type mismatch`; each `VerifyFile(.., '', 10, E, 'work/ut/dv-305/w', OK)` | ok: OK True, columns `oxygene`, `compiled-run`, `accepted`; `wrong-output`; `no-expected`; `run-error` with detail `3`; `accepted`; `wrong-error` with detail `E: Unknown identifier "Foo" [Program.pas (8)]`; afterwards no entry under E.OxyRoot and no `work/ut/dv-305/w` |

Build and run (the GPUs hidden as in the gate):
`fpc -Mobjfpc -Sh -vew -l- -Fu{proj}/src -Fu{proj}/../../src -Fu{proj}/tests/ut -FU{proj}/work/bin -Fl{proj}/../../build/libtch-rocm -k-rpath={proj}/../../build/libtch-rocm {proj}/tests/ut/DialectVerify_test.pas -o{proj}/work/bin/DialectVerify_test`
then, **from `{proj}`** (the rows use paths relative to it),
`HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 {proj}/work/bin/DialectVerify_test --all --format=plain`.

## ITa (prepared by the harness owner)

`QLORA_MODEL=<rehearsal GGUF> {proj}/tests/ita/T22-run.sh` builds
`src/qlora_verify.pas` and verifies `tests/ita/T22/files` (four correct
Oxygene programs and planted faults, made by `make-files.bb`;
`expected.tsv`) with `--dialect oxygene`; then runs the T19 ITa
(`tests/ita/T19-run.sh`) with the same `src/`, whose raw record
`x-oxygene` now has a planted wrong output (harness owner's change of
`tests/ita/T19/make-raw.bb` and `raw.jsonl`). The driver
`tests/ita/T22/check.bb` gives the cases. Writes
`{proj}/work/ita/T22/summary.tsv` (case, OK/NG, value, limit, note; then
the exit codes and the count of stderr lines other than the GPU warnings).
OK when value ≤ limit. About 20 seconds.

| Case | Compared | Limit |
|---|---|---|
| ITa-T22-001 | 11 Oxygene files: outcome as planted, class `compiled-run` (count of mismatches) | 0 |
| ITa-T22-002 | details free of the home directory and the work directory (count of lines) | 0 |
| ITa-T22-003 | Oxygene work directories left under `~/.cache/qlora-verify` by the run (count) | 0 |
| ITa-T22-004 | seconds per Oxygene file (build and run) | 5 |
| ITa-T22-005 | the T19 ITa with this `src/`: NG cases (6 OK expected) | 0 |
| ITa-T22-006 | T19's `x-oxygene` compiled, run and rejected: 1 if its reason is not `verify: wrong-output` | 0 |

`qlora_verify` exits 1 in this run (planted faults); that is expected.

## Stages and case IDs

- Stage 1 (task T22b): everything of "Implementation"; rows
  UT-DialectVerify-013, 103, 104 (changed), 301 to 305.
- ITa: ITa-T22-001 to 006.
